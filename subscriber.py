#!/usr/bin/env python3
import json
import subprocess
import logging
import threading
import os
from google.cloud import pubsub_v1
from google.cloud import firestore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "basketballformai"

# Subscription names
SUB_VIDEO_PROCESSING = "video-processing-sub"
SUB_DATASET_CREATED   = "dataset-created-sub"
SUB_DATASET_PROCESSED = "dataset-processed-sub"

# Ensure log directory exists
LOG_DIR = "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer/process_logs"
os.makedirs(LOG_DIR, exist_ok=True)

subscriber = pubsub_v1.SubscriberClient()

db = firestore.Client()

def callback_main(message):
    """
    Callback for the 'video-processing-sub' subscription.
    Expects message data with:
      {
        "bucket": "...",
        "name": "...",
        "job_id": "the-job-id-from-mobile"
      }
    Then triggers main.py with the --video and --job_id arguments.
    Uses a fire-and-forget approach to prevent message redelivery issues.
    """
    try:
        data = json.loads(message.data.decode("utf-8"))
        bucket = data.get("bucket")
        name = data.get("name")
        job_id = data.get("job_id")
        
        
        #first try to read them from the incoming Pub/Sub message
        gesture_events = data.get("gesture_events", [])
        logger.info(f"↪️ Loaded {len(gesture_events)} gesture_events from Pub/Sub message")

        #(optional) if you *also* eventually want to fallback to Firestore:
        if not gesture_events:
             job_doc = db.collection("jobs").document(job_id).get().to_dict() or {}
             gesture_events = job_doc.get("gesture_events", [])
             logger.info(f"↪️ Fallback loaded {len(gesture_events)} gesture_events from Firestore")

        if not bucket or not name or not job_id:
            logger.error("[MAIN] Message is missing required fields ('bucket', 'name', or 'job_id').")
            message.ack()
            return
            
        video_path = f"gs://{bucket}/{name}"
        logger.info(f"Triggering main.py for video: {video_path} with job_id: {job_id}")
        
        command = [
            "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer/myenv/bin/python3",
            "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer/main.py",
            "--video", video_path,
            "--job_id", job_id
        ]
        # safer: write to /tmp/gesture_<job_id>.json
        tmp = f"/tmp/gesture_{job_id}.json"
        with open(f"/tmp/gesture_{job_id}.json","w") as f:
            json.dump(gesture_events, f)
        command += ["--gesture_events_file", f"/tmp/gesture_{job_id}.json"]


        logger.info("Running command: " + " ".join(command))
        
        # Acknowledge the message immediately to prevent redelivery
        message.ack()
        
        # Create unique log files for this job
        stdout_log = os.path.join(LOG_DIR, f"main_py_{job_id}_stdout.log")
        stderr_log = os.path.join(LOG_DIR, f"main_py_{job_id}_stderr.log")
        
        # Launch the process in a completely detached manner
        subprocess.Popen(
            command,
            stdout=open(stdout_log, 'w'),
            stderr=open(stderr_log, 'w'),
            close_fds=True
        )
        
        logger.info(f"Successfully launched main.py process for job_id: {job_id}")
        logger.info(f"Output logs will be available at: stdout={stdout_log}, stderr={stderr_log}")
        
    except Exception as e:
        logger.error(f"Error in callback_main: {e}")
        # Try to acknowledge the message even if there was an error to prevent redelivery
        try:
            message.ack()
        except Exception:
            logger.error("Failed to acknowledge message after error")


def callback_data_processing(message):
    try:
        logger.info(f"Recieved message on dataset-created-sub: {message.data}")
        data = json.loads(message.data.decode("utf-8"))
        logger.info(f"Parsed message data: {data}")
        
        dataset_path = data.get("dataset_path")
        job_id = data.get("job_id")

        logger.info(f"Extraced dataset_path: {dataset_path}, job_id: {job_id}")
        
        if not dataset_path or not job_id:
            logger.error(f"[DATA PROCESSING] Message is missing required fields. Got: {data}")
            message.ack()
            return
        
        # Verify file existence before launching
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file does not exist at path: {dataset_path}")
            message.ack()
            return
            
        # Rest of the function...
            
        logger.info(f"Triggering data_processing.py with dataset: {dataset_path} and job_id: {job_id}")
        
        command = [
            "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer/myenv/bin/python3",
            "/home/jakeposchl/LSTM-Basketball-Shot-Form-Analyzer/data_processing.py",  # Use absolute path
            "--dataset_path", dataset_path,
            "--job_id", job_id
        ]
        
        # Acknowledge the message immediately
        message.ack()
        
        # Create unique log files for this job
        stdout_log = os.path.join(LOG_DIR, f"data_processing_{job_id}_stdout.log")
        stderr_log = os.path.join(LOG_DIR, f"data_processing_{job_id}_stderr.log")
        
        # Launch the process in a detached manner
        subprocess.Popen(
            command,
            stdout=open(stdout_log, 'w'),
            stderr=open(stderr_log, 'w'),
            close_fds=True
        )
        
        logger.info(f"Successfully launched data_processing.py for job_id: {job_id}")
        logger.info(f"Output logs will be available at: stdout={stdout_log}, stderr={stderr_log}")
        
    except Exception as e:
        logger.error(f"Error in callback_data_processing: {e}")
        # Try to acknowledge the message even if there was an error
        try:
            message.ack()
        except Exception:
            logger.error("Failed to acknowledge message after error")

def callback_lstm_model(message):
    """
    Callback for the 'dataset-processed-sub' subscription.
    Launches a Docker container for LSTM model processing.
    """
    try:
        data = json.loads(message.data.decode("utf-8"))

        # ─── OLD (expects data_dir & job_filename) ───
        # data_dir     = data.get("data_dir")
        # job_filename = data.get("job_filename", "unknown")
        # job_id       = data.get("job_id", "unknown")
        #
        # if not data_dir or not job_id:
        #     logger.error("Message is missing 'data_dir' or 'job_id' field.")
        #     message.ack()
        #     return
        # ─────────────────────────────────────────────

        # ─── NEW: only require job_id ───
        job_id = data.get("job_id")
        if not job_id:
            logger.error("Message is missing 'job_id' field.")
            message.ack()
            return
        logger.info(f"Triggering LSTM container for job_id: {job_id}")
        # ─────────────────────────────────

        # Build docker command without mounting Cleaned_Datasets,
        # and only passing --job_id to the container.
        docker_command = [
            "docker", "run", "--gpus", "all",
            "-e", "GCS_BUCKET=basketball-ai-data",
            "-e", "GCS_PREFIX=dev_shot_sequences",
            "my_lstm_model",
            "--job_id", job_id
        ]

        message.ack()

        stdout_log = os.path.join(LOG_DIR, f"lstm_model_{job_id}_stdout.log")
        stderr_log = os.path.join(LOG_DIR, f"lstm_model_{job_id}_stderr.log")

        subprocess.Popen(
            docker_command,
            stdout=open(stdout_log, 'w'),
            stderr=open(stderr_log, 'w'),
            close_fds=True
        )
        logger.info(f"Launched LSTM container for job_id {job_id}")
        logger.info(f"Logs → stdout: {stdout_log}, stderr: {stderr_log}")

    except Exception as e:
        logger.error(f"Error in callback_lstm_model: {e}")
        try:
            message.ack()
        except Exception:
            logger.error("Failed to acknowledge message after error")


def main():
    sub_path_main = subscriber.subscription_path(PROJECT_ID, SUB_VIDEO_PROCESSING)
    sub_path_data = subscriber.subscription_path(PROJECT_ID, SUB_DATASET_CREATED)
    sub_path_lstm = subscriber.subscription_path(PROJECT_ID, SUB_DATASET_PROCESSED)

    streaming_pull_main = subscriber.subscribe(sub_path_main, callback=callback_main)
    streaming_pull_data = subscriber.subscribe(sub_path_data, callback=callback_data_processing)
    streaming_pull_lstm = subscriber.subscribe(sub_path_lstm, callback=callback_lstm_model)

    logger.info(f"Subscriber started, listening to {SUB_VIDEO_PROCESSING}, {SUB_DATASET_CREATED}, and {SUB_DATASET_PROCESSED}")

    thread_main = threading.Thread(target=streaming_pull_main.result, daemon=True)
    thread_data = threading.Thread(target=streaming_pull_data.result, daemon=True)
    thread_lstm = threading.Thread(target=streaming_pull_lstm.result, daemon=True)

    thread_main.start()
    thread_data.start()
    thread_lstm.start()

    try:
        thread_main.join()
        thread_data.join()
        thread_lstm.join()
    except KeyboardInterrupt:
        streaming_pull_main.cancel()
        streaming_pull_data.cancel()
        streaming_pull_lstm.cancel()
        logger.info("Exiting subscriber...")

if __name__ == "__main__":
    main()