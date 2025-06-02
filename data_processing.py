#!/usr/bin/env python3
import os
import sys
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import argparse
import json
import logging
from google.cloud import pubsub_v1
from google.cloud import firestore
from google.cloud import storage
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(datasets_folder, combined_filename):
    """Load the dataset from the specified path."""
    combined_dataset_path = os.path.join(datasets_folder, combined_filename)
    try:
        combined_df = pd.read_csv(combined_dataset_path)
        logger.info(f"Successfully loaded dataset: {combined_dataset_path}")
    except FileNotFoundError:
        logger.error(f"Combined dataset not found at: {combined_dataset_path}")
        raise FileNotFoundError(f"Combined dataset not found at: {combined_dataset_path}")
    except pd.errors.EmptyDataError:
        logger.error("Error: The file is empty.")
        sys.exit("Error: The file is empty.")
    except pd.errors.ParserError:
        logger.error("Error: The file could not be parsed.")
        sys.exit("Error: The file could not be parsed.")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(f"An unexpected error occurred: {e}")
    return combined_df

def preprocess_dataset(df):
    """Preprocess the dataset, checking for required columns and splitting complex fields."""
    required_columns = ['shot_id', 'is_shot', 'frame']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"Missing required columns in the dataset: {missing_columns}")
        raise ValueError(f"Missing required columns in the dataset: {missing_columns}")
    else:
        logger.info("All required columns are present.")

    if 'sports_ball_positions' in df.columns:
        df['sports_ball_positions'] = df['sports_ball_positions'].astype(str)
        df[['ball_pos_x', 'ball_pos_y']] = df['sports_ball_positions'].str.extract(r'([^,]+),\s*([^,]+)', expand=True)
        df['ball_pos_x'] = pd.to_numeric(df['ball_pos_x'], errors='coerce')
        df['ball_pos_y'] = pd.to_numeric(df['ball_pos_y'], errors='coerce')
        df.drop(columns=['sports_ball_positions'], inplace=True)
        logger.info("Split sports_ball_positions into ball_pos_x and ball_pos_y.")
    return df

def optimized_interpolate_zeros(data, columns):
    """Replace zeros with NaN and interpolate values in specified columns."""
    df_interpolated = data.copy()
    df_interpolated[columns] = df_interpolated[columns].replace(0, np.nan)
    df_interpolated[columns] = df_interpolated[columns].interpolate(method='linear', limit_direction='both')
    return df_interpolated

def interpolate_dataset(df):
    """Interpolate missing values in the dataset."""
    columns_to_exclude = ['frame', 'video', 'is_shot', 'shot_id', 'shot_invalid', 'make']
    columns_to_interpolate = [col for col in df.columns if col not in columns_to_exclude]
    if not columns_to_interpolate:
        logger.info("No columns available for interpolation after excluding specified columns.")
        return df.copy()
    else:
        df = optimized_interpolate_zeros(df, columns_to_interpolate)
        logger.info("Interpolation completed successfully.")
        return df

def compute_sequence_info(df):
    """Compute information about shot sequences in the dataset."""
    sequence_info = []
    unique_shot_ids = df['shot_id'].unique()
    for shot_id in unique_shot_ids:
        shot_df = df[df['shot_id'] == shot_id].copy().reset_index(drop=True)
        shot_df['is_shot_shift'] = shot_df['is_shot'].shift(1, fill_value=0)
        shot_df['is_shot_next'] = shot_df['is_shot'].shift(-1, fill_value=0)
        start_indices = shot_df[(shot_df['is_shot'] == 1) & (shot_df['is_shot_shift'] == 0)].index.tolist()
        end_indices = shot_df[(shot_df['is_shot'] == 1) & (shot_df['is_shot_next'] == 0)].index.tolist()
        if len(start_indices) != len(end_indices):
            if len(start_indices) > len(end_indices):
                end_indices.append(len(shot_df) - 1)
        for start_idx, end_idx in zip(start_indices, end_indices):
            start_frame = shot_df.loc[start_idx, 'frame']
            end_frame   = shot_df.loc[end_idx,   'frame']
            raw_length  = end_frame - start_frame

            # extend the start by 1.5× the raw length
            extension   = int(raw_length * 1.5)
            new_start   = max(0, start_frame - extension)

            # record the extended window
            extended_length = end_frame - new_start
            sequence_info.append((shot_id, new_start, end_frame, extended_length))
    return sequence_info

def compute_max_length(sequence_info):
    """Compute the maximum length of shot sequences after filtering outliers."""
    shot_id_lengths = np.array([info[3] for info in sequence_info if info is not None and len(info) > 0])
    Q1 = np.percentile(shot_id_lengths, 25)
    Q3 = np.percentile(shot_id_lengths, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 + 0.05 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_lengths = shot_id_lengths[(shot_id_lengths >= lower_bound) & (shot_id_lengths <= upper_bound)]
    max_length = int(filtered_lengths.max()) if filtered_lengths.size > 0 else 0
    return max_length

def compute_subdataset_info(df, sequence_info, max_length):
    """Compute subdataset information based on sequence info and max length."""
    subdataset_info = []
    for (shot_id, start_frame, end_frame, shot_id_length) in sequence_info:
        adjusted_start_frame = end_frame - max_length
        if adjusted_start_frame < 0:
            adjusted_start_frame = 0
        subdataset_length = end_frame - adjusted_start_frame + 1
        subdataset_info.append((shot_id, adjusted_start_frame, end_frame, subdataset_length))
    return subdataset_info

def create_subdatasets(df, subdataset_info):
    """Create subdatasets based on the subdataset info."""
    subdatasets = []
    for row in subdataset_info:
        shot_id, adjusted_start_frame, end_frame, sub_length = row
        sub_df = df[(df['frame'] >= adjusted_start_frame) & (df['frame'] <= end_frame)].copy()
        subdatasets.append(sub_df)
    for sub_df in subdatasets:
        sub_df.drop(columns=['video', 'is_shot', 'shot_invalid'], inplace=True, errors='ignore')
    return subdatasets

def assign_made_missed(subdatasets):
    """Assign shots to made or missed categories based on make value."""
    made = []
    missed = []
    for sub_df in subdatasets:
        if (sub_df["make"] == True).any():
            sub_df["make"] = 1
            max_shot_id = sub_df["shot_id"].max()
            sub_df["shot_id"] = max_shot_id
            made.append(sub_df)
        elif (sub_df["make"] == False).any():
            sub_df["make"] = 0
            max_shot_id = sub_df["shot_id"].max()
            sub_df["shot_id"] = max_shot_id
            missed.append(sub_df)
        else:
            sub_df["make"] = 0
            max_shot_id = sub_df["shot_id"].max()
            sub_df["shot_id"] = max_shot_id
            missed.append(sub_df)
    logger.info(f"Shots distribution: {len(made)} made shots, {len(missed)} missed shots")
    return made, missed

def combine_and_shuffle(made, missed):
    """Combine and shuffle the made and missed shots."""
    combined_list = made + missed
    if not combined_list:
        logger.error("No subdatasets available to combine. Exiting processing.")
        sys.exit(1)
    data_df = pd.concat(combined_list, ignore_index=True, sort=False)
    data_df = data_df.sort_values(by=['shot_id', 'frame']).reset_index(drop=True)
    unique_shot_ids = data_df['shot_id'].unique()
    np.random.seed(18)
    shuffled_shot_ids = np.random.permutation(unique_shot_ids)
    shuffled_dfs = [data_df[data_df['shot_id'] == shot_id] for shot_id in shuffled_shot_ids]
    shuffled_data_df = pd.concat(shuffled_dfs, ignore_index=True)
    return shuffled_data_df

def split_dataset(data_df):
    """Split into train/val/test, guaranteeing at least one shot per split if possible."""
    ids = data_df['shot_id'].unique().tolist()
    total = len(ids)
    np.random.seed(18)
    np.random.shuffle(ids)

    # At least one in each split if total >= 3
    n_train = max(1, int(0.7 * total))
    n_val   = max(1, int(0.15 * total))
    # Adjust so test also gets at least one
    n_train = min(n_train, total - 2)
    n_val   = min(n_val, total - n_train - 1)
    n_test  = total - n_train - n_val

    train_ids = ids[:n_train]
    val_ids   = ids[n_train:n_train + n_val]
    test_ids  = ids[n_train + n_val:]

    train_df = data_df[data_df['shot_id'].isin(train_ids)].reset_index(drop=True)
    val_df   = data_df[data_df['shot_id'].isin(val_ids)].reset_index(drop=True)
    test_df  = data_df[data_df['shot_id'].isin(test_ids)].reset_index(drop=True)

    logger.info(
        f"Split sizes: train {train_df['shot_id'].nunique()} shots, {len(train_df)} rows; "
        f"val {val_df['shot_id'].nunique()} shots, {len(val_df)} rows; "
        f"test {test_df['shot_id'].nunique()} shots, {len(test_df)} rows"
    )
    return train_df, val_df, test_df

def dataframe_to_tensor(df, feature_columns, seq_len):
    """Convert sequences to fixed-length tensors with pad/truncate."""
    data, labels = [], []
    for sid, grp in df.groupby('shot_id'):
        feat = grp.sort_values('frame')[feature_columns].values.astype(np.float32)
        # pad if too short
        if feat.shape[0] < seq_len:
            pad = np.zeros((seq_len - feat.shape[0], feat.shape[1]), dtype=np.float32)
            feat = np.vstack([pad, feat])
        # truncate if too long
        if feat.shape[0] > seq_len:
            feat = feat[-seq_len:]
        data.append(feat)
        labels.append(int(grp['make'].iloc[0]))
    if not data:
        return np.empty((0, seq_len, len(feature_columns)), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(data, axis=0), np.array(labels, dtype=np.int32)

def save_tensor_and_labels(data_tensor, labels, dataset_type, output_dir):
    """Save tensor data and labels to specified output directory."""
    data_path = os.path.join(output_dir, f"{dataset_type}_data.npy")
    labels_path = os.path.join(output_dir, f"{dataset_type}_labels.npy")
    np.save(data_path, data_tensor)
    np.save(labels_path, labels)
    logger.info(f"Saved {dataset_type} data to {data_path}")
    logger.info(f"Saved {dataset_type} labels to {labels_path}")

# -------------------------------------------------------------------
# Firestore Update Helper Function
# -------------------------------------------------------------------
def update_job_progress(job_ref, progress, status):
    """Update the job progress in Firestore."""
    try:
        job_ref.update({
            "data_processing_progress": progress,
            "data_processing_status": status,
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        logger.info(f"Updated job progress: {progress}, status: {status}")
    except Exception as e:
        logger.error(f"Failed to update job progress: {e}")

# -------------------------------------------------------------------
# Main Function for Data Processing with Firestore Updates
# -------------------------------------------------------------------
def main():
    """Main function to process the dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="Full path to the combined dataset CSV file")
    parser.add_argument("--job_id", required=True, help="Unique job ID for tracking")
    args = parser.parse_args()

    datasets_folder = os.path.dirname(args.dataset_path)
    combined_filename = os.path.basename(args.dataset_path)
    
    # Use the job_id from command line arguments
    job_id = args.job_id
    logger.info(f"Using job_id from arguments: {job_id}")

    # Initialize Firestore client and get reference to the existing job document
    db = firestore.Client()
    job_ref = db.collection("jobs").document(job_id)
    logger.info(f"Using job document ID: {job_id}")

    try:
        # -------------------------------
        # Stage 1: Load Dataset
        # -------------------------------
        update_job_progress(job_ref, 0.1, "loading_dataset")
        combined_df = load_dataset(datasets_folder, combined_filename)

        # -------------------------------
        # Stage 2: Preprocess Dataset
        # -------------------------------
        update_job_progress(job_ref, 0.3, "preprocessing")
        df = preprocess_dataset(combined_df)

        # -------------------------------
        # Stage 3: Interpolate Dataset
        # -------------------------------
        update_job_progress(job_ref, 0.5, "interpolating")
        df = interpolate_dataset(df)

        # -------------------------------
        # Stage 4: Compute Sequence Info and Create Subdatasets
        # -------------------------------
        update_job_progress(job_ref, 0.7, "computing_sequence")
        sequence_info = compute_sequence_info(df)

        if not sequence_info:
            msg = "No shots detected in dataset"
            logger.warning(msg)
            # Mark the job “complete but no shots” (or use “error” if you prefer)
            update_job_progress(job_ref, 1.0, "no_shots_detected")
            # Store the human‑readable reason
            job_ref.update({"data_processing_error_message": msg})
            sys.exit(0)

        max_length = compute_max_length(sequence_info)
        subdataset_info = compute_subdataset_info(df, sequence_info, max_length)
        seq_len = max_length + 1
        subdatasets = create_subdatasets(df, subdataset_info)
        made, missed = assign_made_missed(subdatasets)
        # ——————————————
        # Down-sample missed shots to match the number of makes
        # ——————————————
        n_made = len(made)
        if len(missed) > n_made:
            random.seed(18)                    # reproducible shuffle
            random.shuffle(missed)
            missed = missed[:n_made]
        logger.info(f"After downsampling: {n_made} made, {len(missed)} missed")
        data_df = combine_and_shuffle(made, missed)

        # ─── UPLOAD EVERY SHOT TO SHARED GCS STORE ───
        bucket_name = "basketball-ai-data"
        prefix      = "dev_shot_sequences"
        storage_client = storage.Client()
        bucket         = storage_client.bucket(bucket_name)

        # Determine which columns are actual features
        exclude_cols    = ["frame", "make", "shot_id"]
        feature_columns = [c for c in data_df.columns if c not in exclude_cols]

        # Write each shot to a .npz and upload
        with tempfile.TemporaryDirectory() as tmpdir:
            for shot_df in made + missed:
                sid   = int(shot_df["shot_id"].iloc[0])
                label = int(shot_df["make"].iloc[0])
                arr   = shot_df.sort_values("frame")[feature_columns].values.astype(np.float32)

                # ─── pad or truncate to fixed seq_len ───
                if arr.shape[0] < seq_len:
                    pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
                    arr = np.vstack([pad, arr])
                elif arr.shape[0] > seq_len:
                    arr = arr[-seq_len:]
                # now arr.shape == (seq_len, F)
                # ─────────────────────────────────────────

                local_path = os.path.join(tmpdir, f"{sid}.npz")
                np.savez(local_path, features=arr, label=label)

                blob_path = f"{prefix}/{job_id}_{sid}.npz"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(local_path)
                logger.info(f"Uploaded shot {sid} → gs://{bucket_name}/{blob_path}")
        # ──────────────────────────────────────────────

        train_df, validation_df, test_df = split_dataset(data_df)
        
        update_job_progress(job_ref, 0.9, "saving_data")

        columns_to_exclude = ['frame', 'make', 'shot_id']
        feature_columns = [col for col in train_df.columns if col not in columns_to_exclude]

        # Define output folder for processed .npy files as "Cleaned_Datasets"
        cleaned_folder = os.path.join(os.path.dirname(datasets_folder), "Cleaned_Datasets")
        os.makedirs(cleaned_folder, exist_ok=True)

        train_data_tensor, train_labels = dataframe_to_tensor(train_df, feature_columns, seq_len)
        logger.info(f"Training data tensor shape: {train_data_tensor.shape}")
        logger.info(f"Training labels shape: {train_labels.shape}")

        validation_data_tensor, validation_labels = dataframe_to_tensor(validation_df, feature_columns, seq_len)
        logger.info(f"Validation data tensor shape: {validation_data_tensor.shape}")
        logger.info(f"Validation labels shape: {validation_labels.shape}")

        test_data_tensor, test_labels = dataframe_to_tensor(test_df, feature_columns, seq_len)
        logger.info(f"Testing data tensor shape: {test_data_tensor.shape}")
        logger.info(f"Testing labels shape: {test_labels.shape}")
        logger.info(f"Feature Columns: {feature_columns}")

        # Save processed tensors
        save_tensor_and_labels(train_data_tensor, train_labels, 'train', output_dir=cleaned_folder)
        save_tensor_and_labels(validation_data_tensor, validation_labels, 'validation', output_dir=cleaned_folder)
        save_tensor_and_labels(test_data_tensor, test_labels, 'test', output_dir=cleaned_folder)

        # Mark as completed
        update_job_progress(job_ref, 1.0, "completed")
        logger.info("Data processing complete.")

        # Publish a Pub/Sub message to trigger model training
        message_payload = json.dumps({
            "data_dir": "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer/Cleaned_Datasets",
            "job_filename": combined_filename,
            "job_id": job_id  # Include the job_id in the message
        }).encode('utf-8')

        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path('basketballformai', 'dataset-processed')
        try:
            future = publisher.publish(topic_path, message_payload)
            result = future.result()  # force synchronous check
            logger.info(f"Published dataset processed message with job_id {job_id} to {topic_path}, message ID: {result}")
        except Exception as e:
            logger.error(f"Failed to publish dataset processed message: {e}")
            
    except Exception as e:
        err_msg = f"Error during data processing: {e}"
        logger.error(err_msg)
       # 0% progress, error state
        update_job_progress(job_ref, 0, "error")
       # write the actual Python exception into the document
        job_ref.update({"data_processing_error_message": str(e)})
        sys.exit(err_msg)

if __name__ == '__main__':
    main()