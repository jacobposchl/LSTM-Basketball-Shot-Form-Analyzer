from google.cloud import pubsub_v1
import json
import subprocess

PROJECT_ID = "basketballformai"
SUBSCRIPTION_NAME = "video-processing-sub"

def callback(message):
    print(f"Received message: {message.data}")
    
    # Parse the message
    try:
        event_data = json.loads(message.data.decode("utf-8"))
        bucket_name = event_data["bucket"]
        video_filename = event_data["name"]

        print(f"New video uploaded: {video_filename} in bucket {bucket_name}")

        # Run main.py to process the video
        command = ["python3", "main.py", "--video", f"gs://{bucket_name}/{video_filename}"]
        subprocess.Popen(command)

        # Acknowledge message to remove it from the queue
        message.ack()
        print("Processing started for:", video_filename)

    except Exception as e:
        print(f"Error processing message: {e}")
        message.ack()  # Still acknowledge the message to prevent re-processing

# Start the Pub/Sub subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(PROJECT_ID, SUBSCRIPTION_NAME)

print("Listening for new video uploads...")
subscriber.subscribe(subscription_path, callback=callback)

import time
while True:
    time.sleep(10)
