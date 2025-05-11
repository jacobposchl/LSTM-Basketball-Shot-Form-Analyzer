#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import json
from tensorflow.keras import regularizers
from sklearn.metrics import confusion_matrix, classification_report
from google.cloud import pubsub_v1
from google.cloud import firestore
import sys
import os
import io
from google.cloud import storage
from sklearn.model_selection import train_test_split
import logging
import sys

# NEW: configure a file + stdout logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("lstm_model_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_shared_dataset(bucket_name, prefix):
    client = storage.Client()
    blobs = list(client.list_blobs(bucket_name, prefix=prefix))
    raw_feats, raw_labels = [], []
    for blob in blobs:
        if not blob.name.endswith(".npz"):
            continue
        data = blob.download_as_bytes()
        with io.BytesIO(data) as f:
            npz = np.load(f)
            raw_feats.append(npz["features"])
            raw_labels.append(int(npz["label"]))
    # Determine the common seq_len
    seq_len      = max(f.shape[0] for f in raw_feats)
    feature_dim  = raw_feats[0].shape[1]
    feats, labs  = [], []
    for f, l in zip(raw_feats, raw_labels):
        if   f.shape[0] < seq_len:
            pad = np.zeros((seq_len - f.shape[0], feature_dim), dtype=np.float32)
            f   = np.vstack([pad, f])
        elif f.shape[0] > seq_len:
            f   = f[-seq_len:]
        feats.append(f)
        labs.append(l)
    X = np.stack(feats, axis=0)
    y = np.array(labs, dtype=np.float32)
    logger.info("Loaded %d sequences, all padded to %d timesteps", len(y), seq_len)
    return X, y

# -------------------------
# Model Building Function
# -------------------------
def build_lstm_model(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Masking(mask_value = 0.0, input_shape = input_shape),
        tf.keras.layers.LSTM(
            128,
            input_shape=input_shape,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.01)
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(
            64,
            kernel_regularizer=regularizers.l2(0.01)
        ),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.01)
        ),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    model.summary()
    return model

# -------------------------
# Custom Callback for Firestore Updates
# -------------------------
class FirestoreTrainingProgress(tf.keras.callbacks.Callback):
    def __init__(self, job_ref, total_epochs):
        super().__init__()
        self.job_ref = job_ref
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        # Update Firestore document with training progress and status.
        self.job_ref.update({
            "training_progress": progress,
            "training_status": "in_progress",
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        print(f"Epoch {epoch+1}/{self.total_epochs} - Updated training progress: {progress:.2f}")

# -------------------------
# Main Function
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", required=False, help="Session/job ID for Firestore updates")
    args = parser.parse_args()
    job_id = args.job_id
    logger.info("Starting lstm_model.py for job_id=%s", job_id)

    db = firestore.Client()
    job_ref = db.collection("jobs").document(job_id)
    logger.info("Initialized Firestore job_ref for job_id = %s", job_id)
    job_ref.update({
        "training_status":   "started",
        "training_progress": 0.0,
        "updated_at":        firestore.SERVER_TIMESTAMP
    })


    # Load data from the specified directory
    # ─── Load all shot sequences & labels from GCS shared store ───
    bucket    = os.environ["GCS_BUCKET"]   # injected via subscriber.py
    prefix    = os.environ["GCS_PREFIX"]

    try:
        X_all, y_all = load_shared_dataset(bucket, prefix)
    except Exception as e:
        logger.exception("Failed to load shared dataset")
        job_ref.update({
            "training_status": "error",
            "error_message":   f"Load error: {e}",
            "updated_at":      firestore.SERVER_TIMESTAMP
        })
        sys.exit(1)

    print(f"Loaded {X_all.shape[0]} total sequences; splitting for train/val/test…")

    # First carve off 15% for test
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_all, y_all,
            test_size=0.15,
            stratify=y_all,
            random_state=18
        )
        val_ratio = 0.15 / 0.85
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            stratify=y_temp,
            random_state=18
        )
        logger.info("Split sizes → train=%d, val=%d, test=%d",
                    len(y_train), len(y_val), len(y_test))
    except Exception as e:
        logger.exception("Failed during train/val/test split")
        job_ref.update({
            "training_status": "error",
            "error_message":   f"Split error: {e}",
            "updated_at":      firestore.SERVER_TIMESTAMP
        })
        sys.exit(1)

    print(f"Split sizes → train: {len(y_train)}, val: {len(y_val)}, test: {len(y_test)}")
    print(f"Training on {X_train.shape[0]} sequences.")

    if X_train.shape[1] == 0 or X_train.shape[2] == 0:
        job_ref.update({
        "training_progress": 0.0,
        "training_status": "error",
        "error_message": "Empty training sequences or zero features",
        "updated_at": firestore.SERVER_TIMESTAMP
        })
        sys.exit("Empty training data or features")

    # ——— dataset summary for the report ———
    total_shots     = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    total_makes     = int(y_train.sum()  + y_val.sum()  + y_test.sum())
    total_misses    = total_shots - total_makes

    dataset_summary = {
        "total_shots_detected": total_shots,
        "made_shots":            total_makes,
        "missed_shots":          total_misses
    }

    print("Dataset summary:", dataset_summary)
    
    # Determine input shape from training data
    num_frames = X_train.shape[1]
    num_features = X_train.shape[2]
    input_shape = (num_frames, num_features)
    print("Input shape for LSTM:", input_shape)
    
    model = build_lstm_model(input_shape=input_shape)
    

    
    # Initialize Firestore client and get reference to the existing job document
    db = firestore.Client()
    job_ref = db.collection("jobs").document(job_id)
    logger.info("Initialized Firestore job_ref")
    job_ref.update({
        "training_status":    "started",
        "training_progress":  0.0,
        "updated_at":         firestore.SERVER_TIMESTAMP
    })
    
    total_epochs = 50
    # Create callback that updates Firestore after each epoch.
    progress_callback = FirestoreTrainingProgress(job_ref, total_epochs)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='best_lstm_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    
    batch_size = min(32, X_train.shape[0])
    print("Batch size:", batch_size)
    try:
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=total_epochs,
            batch_size=batch_size,
            callbacks=[progress_callback],
            verbose=1
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.exception("Training failed")
        job_ref.update({
            "training_status": "error",
            "error_message":   f"Training error: {e}",
            "updated_at":      firestore.SERVER_TIMESTAMP
        })
        sys.exit(1)
    
    # After training completes, update Firestore with completion status.
    job_ref.update({
        "training_progress": 1.0,
        "training_status": "completed",
        "updated_at": firestore.SERVER_TIMESTAMP
    })
    
    # Plot training curves (optional)
    #plt.figure(figsize=(10, 4))
    #plt.plot(history.history['accuracy'], label='Train Accuracy')
    #plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    #plt.legend()
    #plt.title("Training Accuracy")
    #plt.show()
    
    try:
        results = model.evaluate(X_test, y_test, verbose=1, return_dict=True)
        logger.info("Evaluation results: %s", results)
    except Exception as e:
        logger.exception("Evaluation failed")
        job_ref.update({
            "training_status": "error",
            "error_message":   f"Eval error: {e}",
            "updated_at":      firestore.SERVER_TIMESTAMP
        })
        sys.exit(1)
    
    # Generate predictions, confusion matrix, and classification report.
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype('int32')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    
    print("Job ID:", job_id)
    
    # Build JSON summary.
    summary = {
        "job_id": job_id,
        "test_loss": results.get("loss"),
        "test_accuracy": results.get("accuracy"),
        "precision": results.get("precision"),
        "recall": results.get("recall"),
        "auc": results.get("auc"),
        "confusion_matrix": json.dumps(cm.tolist()),
        "classification_report": report,
        "training_history": history.history
    }

    summary.update({
        "dataset_summary":        dataset_summary,
        "train_split": {
            "sequences": X_train.shape[0],
            "makes":     int(y_train.sum()),
            "misses":    int((y_train == 0).sum())
        },
        "val_split": {
            "sequences": X_val.shape[0],
            "makes":     int(y_val.sum()),
            "misses":    int((y_val == 0).sum())
        },
        "test_split": {
            "sequences": X_test.shape[0],
            "makes":     int(y_test.sum()),
            "misses":    int((y_test == 0).sum())
        }
    })

    
    summary_json = json.dumps(summary)
    print("JSON Summary:")
    print(summary_json)
    
    # Publish the JSON summary to the Pub/Sub topic "lstm-results"
    try:
        publisher = pubsub_v1.PublisherClient()
        topic     = publisher.topic_path('basketballformai', 'lstm-results')
        publisher.publish(topic, json.dumps(summary).encode('utf-8')).result()
        logger.info("Published results to Pub/Sub")
        job_ref.update({
            "training_status":   "completed",
            "training_progress": 1.0,
            "updated_at":        firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        logger.exception("Failed to publish results")
        job_ref.update({
            "training_status": "error",
            "error_message":   f"Publish error: {e}",
            "updated_at":      firestore.SERVER_TIMESTAMP
        })
        sys.exit(1)

if __name__ == '__main__':
    main()
