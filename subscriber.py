#!/usr/bin/env python3
import json
import subprocess
import logging
import threading
import os
from google.cloud import pubsub_v1
from google.cloud import firestore

# Set up logging\logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP project and subscription names
PROJECT_ID = "basketballformai"
SUB_VIDEO_PROCESSING = "video-processing-sub"
SUB_DATASET_CREATED   = "dataset-created-sub"
SUB_DATASET_PROCESSED = "dataset-processed-sub"

# Directory for logs
LOG_DIR = os.path.expanduser("~/LSTM-Basketball-Shot-Form-Analyzer/process_logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Initialize clients
subscriber = pubsub_v1.SubscriberClient()
db = firestore.Client()

def callback_main(message):
    """
    Launches the inference container when a new video is uploaded.
    """
    try:
        data = json.loads(message.data.decode("utf-8"))
        bucket = data.get("bucket")
        name   = data.get("name")
        job_id = data.get("job_id")

        raw_payload = data.get("raw_gesture_payload") or {}
        gesture_events = raw_payload.get("gesture_events", [])

        # If still empty, pull the same map from Firestore
        if not gesture_events:
            job_doc = db.collection("jobs").document(job_id).get().to_dict() or {}
            fb_raw = job_doc.get("raw_gesture_payload") or {}
            gesture_events = fb_raw.get("gesture_events", [])
            logger.info(f"🔄 Fallback to Firestore raw_gesture_payload: loaded {len(gesture_events)} events for job {job_id}")


        if not bucket or not name or not job_id:
            logger.error("[MAIN] Missing 'bucket', 'name', or 'job_id'.")
            message.ack()
            return

        video_path = f"gs://{bucket}/{name}"
        logger.info(f"Triggering inference for {video_path} (job_id={job_id})")

        # Write gestures to a temp file so Docker can mount it
        tmp_gest = f"/tmp/gesture_{job_id}.json"
        with open(tmp_gest, "w") as f:
            json.dump(gesture_events, f)

        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "--net", "host",
            "-e", f"DISPLAY={os.getenv('DISPLAY')}",
            "-e", "XAUTHORITY=/root/.Xauthority",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "-v", f"{tmp_gest}:{tmp_gest}:ro",  # Mount gestures file
            "-v", f"{os.path.expanduser('~')}/.Xauthority:/root/.Xauthority:ro",
            "-v", "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer:/workspace:rw",
            "-v", "/home/jakeposchl/keys/basketballformai-514fed041c9c.json:/workspace/firebase-key.json:ro",
            "-e", "GOOGLE_APPLICATION_CREDENTIALS=/workspace/firebase-key.json",
            "-e", "GCS_BUCKET=basketball-ai-data",
            "-e", "GCS_PREFIX=dev_shot_sequences",
            "basketball-inference:gpu",
            "--video", video_path,
            "--job_id", job_id,
            "--gesture_events_file", tmp_gest
        ]

        message.ack()
        stdout_log = os.path.join(LOG_DIR, f"inference_{job_id}_stdout.log")
        stderr_log = os.path.join(LOG_DIR, f"inference_{job_id}_stderr.log")

        subprocess.Popen(
            docker_cmd,
            stdout=open(stdout_log, 'w'),
            stderr=open(stderr_log, 'w'),
            close_fds=True
        )
        logger.info(f"Launched inference container for job {job_id}")
    except Exception as e:
        logger.error(f"Error in callback_main: {e}")
        message.ack()


def callback_data_processing(message):
    """
    Launches the data-processing container when the dataset CSV is ready.
    """
    try:
        data = json.loads(message.data.decode("utf-8"))
        container_dataset_path = data.get("dataset_path")
        job_id = data.get("job_id")

        if not container_dataset_path or not job_id:
            logger.error("[DATA] Missing 'dataset_path' or 'job_id'.")
            message.ack()
            return

        # Translate container path (/workspace/…) back to host path
        HOST_WS = os.path.expanduser("~/LSTM-Basketball-Shot-Form-Analyzer")
        if container_dataset_path.startswith("/workspace"):
            host_dataset_path = container_dataset_path.replace("/workspace", HOST_WS)
        else:
            host_dataset_path = container_dataset_path

        if not os.path.exists(host_dataset_path):
            logger.error(f"[DATA] Host file not found: {host_dataset_path}")
            message.ack()
            return

        host_dir = os.path.dirname(host_dataset_path)
        filename = os.path.basename(host_dataset_path)
        container_dir = "/data"
        container_dataset_path = f"{container_dir}/{filename}"

        logger.info(f"Triggering data-processing for job {job_id}")

        docker_cmd = [
            "docker", "run", "--rm",
            "-v", f"{host_dir}:{container_dir}:ro",
            "-v", "/home/jakeposchl/keys/basketballformai-514fed041c9c.json:/app/firebase-key.json:ro",
            "-e", "GOOGLE_APPLICATION_CREDENTIALS=/app/firebase-key.json",
            "formai-data-processing:latest",
            "--dataset_path", container_dataset_path,
            "--job_id", job_id
        ]

        message.ack()
        stdout_log = os.path.join(LOG_DIR, f"data_{job_id}_stdout.log")
        stderr_log = os.path.join(LOG_DIR, f"data_{job_id}_stderr.log")

        subprocess.Popen(
            docker_cmd,
            stdout=open(stdout_log, 'w'),
            stderr=open(stderr_log, 'w'),
            close_fds=True
        )
        logger.info(f"Launched data-processing container for job {job_id}")
    except Exception as e:
        logger.error(f"Error in callback_data_processing: {e}")
        message.ack()


def callback_lstm_model(message):
    """
    Launches the training container when cleaned data is ready.
    """
    try:
        data = json.loads(message.data.decode("utf-8"))
        job_id = data.get("job_id")

        if not job_id:
            logger.error("[TRAIN] Missing 'job_id'.")
            message.ack()
            return

        logger.info(f"Triggering training for job {job_id}")

        host_data_dir = os.path.expanduser("~/LSTM-Basketball-Shot-Form-Analyzer/Cleaned_Datasets")
        container_dir = "/data"

        docker_cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-e", "GCS_BUCKET=basketball-ai-data",
            "-e", "GCS_PREFIX=dev_shot_sequences",
            "-v", f"{host_data_dir}:{container_dir}:ro",
            "my_lstm_model",
            "--job_id", job_id
        ]

        message.ack()
        stdout_log = os.path.join(LOG_DIR, f"train_{job_id}_stdout.log")
        stderr_log = os.path.join(LOG_DIR, f"train_{job_id}_stderr.log")

        subprocess.Popen(
            docker_cmd,
            stdout=open(stdout_log, 'w'),
            stderr=open(stderr_log, 'w'),
            close_fds=True
        )
        logger.info(f"Launched training container for job {job_id}")
    except Exception as e:
        logger.error(f"Error in callback_lstm_model: {e}")
        message.ack()


def main():
    sub_main = subscriber.subscription_path(PROJECT_ID, SUB_VIDEO_PROCESSING)
    sub_data = subscriber.subscription_path(PROJECT_ID, SUB_DATASET_CREATED)
    sub_lstm = subscriber.subscription_path(PROJECT_ID, SUB_DATASET_PROCESSED)

    subscriber.subscribe(sub_main, callback=callback_main)
    subscriber.subscribe(sub_data, callback=callback_data_processing)
    subscriber.subscribe(sub_lstm, callback=callback_lstm_model)

    logger.info(f"Listening on subs: {SUB_VIDEO_PROCESSING}, {SUB_DATASET_CREATED}, {SUB_DATASET_PROCESSED}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        logger.info("Subscriber shutting down...")

if __name__ == "__main__":
    main()
