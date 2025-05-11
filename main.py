#!/usr/bin/env python3
# main.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import math
import numpy as np
import pandas as pd
import os
import sys
import torch
from collections import deque
import logging
import shutil  # For directory operations
import argparse
from google.cloud import storage  # For GCP interactions
from google.cloud import pubsub_v1 # For GCP Pub/Sub message sending
from google.cloud import firestore
from logging.handlers import RotatingFileHandler
import config
import uuid #for unique ID of dataset 
import json


# Import configurations from config.py
from config import (
    TARGET_WIDTH,
    TARGET_HEIGHT,
    SMOOTHING_WINDOW,
    joint_map,
    PAIRED_JOINTS,
    ANGLE_JOINTS,
    VELOCITY_THRESHOLD,
    CONSECUTIVE_FRAMES,
    DISTANCE_THRESHOLD,
    TIME_THRESHOLD,
    VERTICAL_DISPLACEMENT_THRESHOLD,
    BALL_WRIST_DISTANCE_THRESHOLD,
    STABLE_FRAMES_REQUIRED,
    STABLE_VELOCITY_THRESHOLD,
    WRIST_CLOSE_DISTANCE_THRESHOLD,
    MAX_BALL_INVISIBLE_FRAMES,
    SHOT_COOLDOWN_FRAMES,
    ShotState,
    YOLO_CONFIDENCE_THRESHOLD,
    DETECTION_THRESHOLD,
    THUMB_GESTURE_WINDOW,
    THUMB_GESTURE_THRESHOLD,
    GCP_BUCKET_NAME,
    GCP_PREFIX,
    GCP_DOWNLOAD_DIR,
    YOLO_WEIGHTS_PATH,
    DEV_MODE,
    DEV_VIDEOS,
    DATASETS_DIR,
    LOGS_DIR
)

from shot_detection import ShotDetector
from project_utils import calculate_angle, map_to_original  # Ensure these functions are correctly implemented

# Initialize logging
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILENAME = 'app.log'
LOG_FILEPATH = os.path.join(LOGS_DIR, LOG_FILENAME)

rotating_handler = RotatingFileHandler(LOG_FILEPATH, maxBytes=10*1024*1024, backupCount=5)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        rotating_handler,
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Logging initialized with rotation")

def download_video_from_gcs(bucket_name: str, video_filename: str) -> str:
    """Downloads a video from Google Cloud Storage to local storage."""
    local_dir = config.GCP_DOWNLOAD_DIR  # Use the same folder as the rest of your workflow
    os.makedirs(local_dir, exist_ok=True)  # Ensure the directory exists
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(video_filename)
    
    local_video_path = os.path.join(local_dir, os.path.basename(video_filename))
    blob.download_to_filename(local_video_path)
    
    logger.info(f"Downloaded {video_filename} to {local_video_path}")
    return local_video_path

def fetch_videos_from_gcp(bucket_name: str, prefix: str, download_dir: str, specific_videos: list = None) -> list:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    video_files = []
    available_videos = []
    for blob in blobs:
        if blob.name.endswith(('.mp4', '.mov', '.avi', '.mkv', '.MOV')):
            video_filename = os.path.basename(blob.name)
            available_videos.append(video_filename)

    logger.info(f"Available videos in bucket '{bucket_name}' with prefix '{prefix}': {available_videos}")

    for video_filename in available_videos:
        if specific_videos and video_filename not in specific_videos:
            continue  # Skip videos not in the specified list when in DEV_MODE
        blob = bucket.blob(os.path.join(prefix, video_filename))
        local_path = os.path.join(download_dir, video_filename)
        if not os.path.exists(local_path):
            logger.info(f"Downloading {blob.name} to {local_path}...")
            try:
                blob.download_to_filename(local_path)
                logger.info(f"Downloaded {blob.name} successfully.")
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")
                continue
        else: 
            logger.info(f"File {local_path} already exists. Skipping download.")
        video_files.append(local_path)
    return video_files

def draw_wrist_roi(frame: np.ndarray, x: float, y: float, roi_size: int, color: tuple, label: str) -> None:
    if x is not None and y is not None:
        x_min = max(int(x - roi_size / 2), 0)
        y_min = max(int(y - roi_size / 2), 0)
        x_max = min(int(x + roi_size / 2), frame.shape[1])
        y_max = min(int(y + roi_size / 2), frame.shape[0])
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        cv2.putText(frame, f"{label} ROI", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_pose_on_original(original_frame: np.ndarray, pose_landmarks, map_to_original_func, 
                          new_width: int, new_height: int, left_pad: int, top_pad: int) -> None:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    connections = mp_pose.POSE_CONNECTIONS
    landmarks = pose_landmarks.landmark
    landmark_coords_padded = [(lm.x * TARGET_WIDTH, lm.y * TARGET_HEIGHT) for lm in landmarks]
    landmark_coords_original = []
    for (x_padded, y_padded) in landmark_coords_padded:
        x_original, y_original = map_to_original_func(
            x_padded, y_padded,
            original_width=int(original_frame.shape[1]),
            original_height=int(original_frame.shape[0]),
            new_width=new_width,
            new_height=new_height,
            left_pad=left_pad,
            top_pad=top_pad
        )
        landmark_coords_original.append((x_original, y_original))
    for connection in connections:
        start_idx, end_idx = connection
        start_point = landmark_coords_original[start_idx]
        end_point = landmark_coords_original[end_idx]
        if None not in start_point and None not in end_point:
            cv2.line(original_frame, (int(start_point[0]), int(start_point[1])),
                             (int(end_point[0]), int(end_point[1])), (0, 255, 0), 1)
    for point in landmark_coords_original:
        if None not in point:
            cv2.circle(original_frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)

def main():

    # Parse CLI arguments; if --video is provided, process only that file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=False, help="GCS video path (gs://bucket-name/video.mp4)")
    parser.add_argument("--job_id", required=True, help="Unique job id provided by the mobile app")
    parser.add_argument("--gesture_events", help="JSON string of gesture events")
    parser.add_argument("--gesture_events_file", help="Path to JSON file with gesture_events")
    args = parser.parse_args()

    # load them:
    if args.gesture_events_file:
        with open(args.gesture_events_file) as f:
            gesture_events = json.load(f)
    elif args.gesture_events:
        gesture_events = json.loads(args.gesture_events)
    else:
        gesture_events = []
    
    logger.info(f"📦 Loaded gesture_events payload (count={len(gesture_events)}): {gesture_events}")


    job_id = args.job_id

    db = firestore.Client()
    job_ref = db.collection("jobs").document(job_id)
    logger.info(f"Using job document ID: {job_id}")\

    job_doc = job_ref.get().to_dict()
    # //ADD BACK ONCE DONE DEVELOPING
    #if job_doc.get("main_py_status") == "completed":
    #    logger.info("Job already processed, exiting.")
    #   sys.exit(0)
    # If a specific video is provided via CLI, download that file only.
    if args.video:
        if args.video.startswith("gs://"):
            bucket_name = args.video.split("/")[2]
            video_filename = "/".join(args.video.split("/")[3:])
            logger.info(f"Processing video from GCS path: {args.video}")
            INPUT_VIDEOS = [download_video_from_gcs(bucket_name, video_filename)]
            job_ref.update({
                "video_upload_progress": 1.0,
                "video_upload_status": "completed",
                "updated_at": firestore.SERVER_TIMESTAMP
            })
        else:
            logger.error("Provided video path must start with 'gs://'")
            sys.exit(1)
    else:
        # No CLI override; use DEV_MODE settings.
        # Clear the GCP_DOWNLOAD_DIR directory before starting.
        if os.path.exists(config.GCP_DOWNLOAD_DIR):
            logger.info(f"Clearing the directory: {config.GCP_DOWNLOAD_DIR}")
            shutil.rmtree(config.GCP_DOWNLOAD_DIR)
        os.makedirs(config.GCP_DOWNLOAD_DIR, exist_ok=True)
        logger.info(f"Created a fresh directory: {config.GCP_DOWNLOAD_DIR}")

        # Ensure the output datasets directory exists.
        os.makedirs(DATASETS_DIR, exist_ok=True)

        logger.info("Fetching videos from GCP bucket...")

        if config.DEV_MODE in [0, 2]:
            logger.info("Running in DEV_MODE: Fetching specified videos only.")
            gcp_videos = fetch_videos_from_gcp(
                bucket_name=config.GCP_BUCKET_NAME,
                prefix=config.GCP_PREFIX,
                download_dir=config.GCP_DOWNLOAD_DIR,
                specific_videos=config.DEV_VIDEOS
            )
            missing_videos = [video for video in config.DEV_VIDEOS if os.path.join(config.GCP_DOWNLOAD_DIR, video) not in gcp_videos]
            if missing_videos:
                logger.warning(f"The following DEV_VIDEOS were not found and will be skipped: {missing_videos}")
        else:
            logger.info("Running in production mode: Fetching all videos.")
            gcp_videos = fetch_videos_from_gcp(
                bucket_name=config.GCP_BUCKET_NAME,
                prefix=config.GCP_PREFIX,
                download_dir=config.GCP_DOWNLOAD_DIR
            )

        if config.DEV_MODE in [0, 2]:
            INPUT_VIDEOS = [os.path.join(config.GCP_DOWNLOAD_DIR, video) for video in config.DEV_VIDEOS if os.path.join(config.GCP_DOWNLOAD_DIR, video) in gcp_videos]
            logger.info(f"Selected DEV_MODE videos: {INPUT_VIDEOS}")
        else:
            INPUT_VIDEOS = gcp_videos
            logger.info(f"Total videos fetched from GCP: {len(INPUT_VIDEOS)}")

        # Update job: video upload completed if videos are fetched
        job_ref.update({
            "video_upload_progress": 1.0,
            "video_upload_status": "completed",
            "updated_at": firestore.SERVER_TIMESTAMP
        })
    
    if not INPUT_VIDEOS:
        logger.error("No video files found to process.")
        sys.exit(1)

    # Initialize detectors and models.
    shot_detector = ShotDetector()

    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_POSE_TRACKING_CONFIDENCE
    )
    mp_drawing = mp.solutions.drawing_utils

    try:
        model = torch.hub.load(
            'ultralytics/yolov5',
            'custom',
            path=config.YOLO_WEIGHTS_PATH,
            source='github'
        )
        model.eval()
        logger.info("YOLOv5 model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading YOLOv5 model: {e}")
        sys.exit(1)

    class_names = model.names
    SPORTS_BALL_CLASS_NAME = "basketball"
    SPORTS_BALL_CLASS_ID = None
    if isinstance(model.names, dict):
        for class_id, class_name in model.names.items():
            if str(class_name).lower() == SPORTS_BALL_CLASS_NAME.lower():
                SPORTS_BALL_CLASS_ID = class_id
                break
    else:
        logger.error(f"Error: model.names is of unexpected type: {type(model.names)}")
        sys.exit(1)
    if SPORTS_BALL_CLASS_ID is None:
        logger.error(f"Error: Class '{SPORTS_BALL_CLASS_NAME}' not found in model classes.")
        sys.exit(1)
    logger.info(f"Ball Class ID: {SPORTS_BALL_CLASS_ID}")

    last_shot_id_displayed = None
    last_shot_make_status = "N/A"
    gesture_history = deque(maxlen=config.THUMB_GESTURE_WINDOW)

    # Process each video in INPUT_VIDEOS.
    for idx, input_video_source in enumerate(INPUT_VIDEOS):
        logger.info(f"\nProcessing video {idx+1}/{len(INPUT_VIDEOS)}: {input_video_source}")
        if not os.path.isfile(input_video_source):
            logger.error(f"Video file does not exist: {input_video_source}")
            continue

        cap = cv2.VideoCapture(input_video_source)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {input_video_source}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Original Video Properties - FPS: {fps}, Width: {original_width}, Height: {original_height}")
        logger.info(f"Resizing all frames to {config.TARGET_WIDTH}x{config.TARGET_HEIGHT} for pose estimation.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prev_positions = {}
        prev_velocities = {}
        all_data = []
        frame_count = 0
        frames_with_landmarks = 0

        shot_detector.reset_shot_state()
        shot_num = 0

        joint_vel_history = {joint_name: deque(maxlen=config.SMOOTHING_WINDOW) for joint_name in joint_map.keys()}
        joint_acc_history = {joint_name: deque(maxlen=config.SMOOTHING_WINDOW) for joint_name in joint_map.keys()}
        detection_history = deque(maxlen=config.DETECTION_THRESHOLD)

        # Initialize variables for y-velocity calculation
        prev_left_wrist_y = None
        prev_right_wrist_y = None
        prev_ball_y = None
        prev_ball_pos = None
        prev_ball_frame = 0

        if config.DEV_MODE in [0, 1]:
            cv2.namedWindow("Detection with Orientation Vectors", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Feature Visualization", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Detection with Orientation Vectors", 600, 800)
            cv2.resizeWindow("Feature Visualization", 600, 800)
        else:
            logger.info("DEV_MODE is set to 2 or 3: Skipping visualization windows.")

        logger.info("Starting video processing...")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video file reached.")
                break
            frame_count += 1
            original_frame = frame.copy()

            # Resize frame to target resolution with aspect ratio preserved.
            aspect_ratio = original_width / original_height
            target_aspect_ratio = config.TARGET_WIDTH / config.TARGET_HEIGHT
            if aspect_ratio > target_aspect_ratio:
                new_width = config.TARGET_WIDTH
                new_height = int(config.TARGET_WIDTH / aspect_ratio)
            else:
                new_height = config.TARGET_HEIGHT
                new_width = int(config.TARGET_HEIGHT * aspect_ratio)
            frame_resized = cv2.resize(frame, (new_width, new_height))
            delta_w = config.TARGET_WIDTH - new_width
            delta_h = config.TARGET_HEIGHT - new_height
            top_pad, bottom_pad = delta_h // 2, delta_h - (delta_h // 2)
            left_pad, right_pad = delta_w // 2, delta_w - (delta_w // 2)
            color = [0, 0, 0]
            frame_padded = cv2.copyMakeBorder(frame_resized, top_pad, bottom_pad, left_pad, right_pad,
                                              cv2.BORDER_CONSTANT, value=color)
            rgb_frame = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)

            # Pose estimation and YOLO detection.
            results_pose = pose.process(rgb_frame)
            try:
                yolo_results = model(original_frame)
            except Exception as e:
                yolo_results = None
                logger.error(f"YOLO detection failed on frame {frame_count}: {e}")

            filtered_detections = []
            if yolo_results and hasattr(yolo_results, 'xyxy') and len(yolo_results.xyxy) > 0:
                for det in yolo_results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls_id = det
                    cls_id = int(cls_id)
                    conf = float(conf)
                    if cls_id == SPORTS_BALL_CLASS_ID and conf >= YOLO_CONFIDENCE_THRESHOLD:
                        filtered_detections.append(det)
                        if config.DEV_MODE in [0, 1]:
                            cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            label_text = f"Ball: {conf:.2f}"
                            cv2.putText(original_frame, label_text, (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            sports_ball_positions = []
            largest_ball = None
            largest_area = 0
            for det in filtered_detections:
                x1, y1, x2, y2, conf, cls_id = det.tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    if 0 <= x_center <= original_width and 0 <= y_center <= original_height:
                        largest_ball = (x_center, y_center)
            if largest_ball:
                sports_ball_positions = [largest_ball]
                valid_ball_detected = True
                prev_ball_pos = largest_ball           # <— store last-known
                prev_ball_frame = frame_count
            elif 'prev_ball_pos' in locals() and frame_count - prev_ball_frame <= MAX_BALL_INVISIBLE_FRAMES:
                sports_ball_positions = [prev_ball_pos]
                valid_ball_detected = True
            else:
                sports_ball_positions = []
                valid_ball_detected = False

            skeleton_center_x = None
            skeleton_center_y = None
            normalized_hip_center_y = None



            if results_pose and results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                left_hip = landmarks[joint_map["LEFT_HIP"]]
                right_hip = landmarks[joint_map["RIGHT_HIP"]]
                if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                    skeleton_center_padded_x = ((left_hip.x + right_hip.x) / 2) * config.TARGET_WIDTH
                    skeleton_center_padded_y = ((left_hip.y + right_hip.y) / 2) * config.TARGET_HEIGHT
                    skeleton_center_original_x, skeleton_center_original_y = map_to_original(
                        skeleton_center_padded_x, skeleton_center_padded_y,
                        original_width=int(original_frame.shape[1]),
                        original_height=int(original_frame.shape[0]),
                        new_width=new_width,
                        new_height=new_height,
                        left_pad=left_pad,
                        top_pad=top_pad
                    )
                    skeleton_center_x = skeleton_center_original_x
                    skeleton_center_y = skeleton_center_original_y
                    frames_with_landmarks += 1
                    normalized_hip_center_y = max(0.0, min(1.0, skeleton_center_y / original_height))
                else:
                    logger.warning(f"Frame {frame_count}: Hips not sufficiently visible.")
            else:
                logger.warning(f"Frame {frame_count}: No pose landmarks detected.")

            joint_data = {}
            if skeleton_center_x is not None and skeleton_center_y is not None and results_pose.pose_landmarks:
                for joint_name, joint_idx in joint_map.items():
                    j_lm = landmarks[joint_idx]
                    x_padded = j_lm.x * config.TARGET_WIDTH
                    y_padded = j_lm.y * config.TARGET_HEIGHT
                    x_original, y_original = map_to_original(
                        x_padded, y_padded,
                        original_width, original_height,
                        new_width, new_height,
                        left_pad, top_pad
                    )
                    relative_x = x_original - skeleton_center_x
                    relative_y = y_original - skeleton_center_y
                    current_pos = (relative_x, relative_y)
                    if joint_name in prev_positions and prev_positions[joint_name] is not None:
                        dx = relative_x - prev_positions[joint_name][0]
                        dy = relative_y - prev_positions[joint_name][1]
                        raw_velocity = round(math.sqrt(dx**2 + dy**2) * fps, 2)
                    else:
                        raw_velocity = None
                    if raw_velocity is not None and joint_name in prev_velocities and prev_velocities[joint_name] is not None:
                        dvel = raw_velocity - prev_velocities[joint_name]
                        raw_acc = round(dvel * fps, 2)
                    else:
                        raw_acc = None
                    joint_vel_history[joint_name].append(raw_velocity if raw_velocity is not None else 0)
                    joint_acc_history[joint_name].append(raw_acc if raw_acc is not None else 0)
                    smoothed_velocity = np.mean(joint_vel_history[joint_name])
                    smoothed_acc = np.mean(joint_acc_history[joint_name])
                    joint_data[joint_name] = {"pos": current_pos, "vel": round(smoothed_velocity, 2), "acc": round(smoothed_acc, 2)}
                    prev_positions[joint_name] = current_pos
                    prev_velocities[joint_name] = raw_velocity

            # --- Compute y-direction velocities for visualization ---
            if "LEFT_WRIST" in joint_data:
                current_left_wrist_y = joint_data["LEFT_WRIST"]["pos"][1]
                if prev_left_wrist_y is not None:
                    left_wrist_velocity_y = abs(current_left_wrist_y - prev_left_wrist_y) * fps
                else:
                    left_wrist_velocity_y = 0.0
                prev_left_wrist_y = current_left_wrist_y
            else:
                left_wrist_velocity_y = 0.0

            if "RIGHT_WRIST" in joint_data:
                current_right_wrist_y = joint_data["RIGHT_WRIST"]["pos"][1]
                if prev_right_wrist_y is not None:
                    right_wrist_velocity_y = abs(current_right_wrist_y - prev_right_wrist_y) * fps
                else:
                    right_wrist_velocity_y = 0.0
                prev_right_wrist_y = current_right_wrist_y
            else:
                right_wrist_velocity_y = 0.0

            if valid_ball_detected and skeleton_center_y is not None:
                current_ball_y = sports_ball_positions[0][1] - skeleton_center_y
                if prev_ball_y is not None:
                    ball_velocity_y = abs(current_ball_y - prev_ball_y) * fps
                else:
                    ball_velocity_y = 0.0
                prev_ball_y = current_ball_y
            else:
                ball_velocity_y = 0.0
            # -----------------------------------------------------------

            features_row = {'video': os.path.basename(input_video_source), 'frame': frame_count}
            if skeleton_center_x is not None and skeleton_center_y is not None and sports_ball_positions:
                ball_relative_pos = (
                    sports_ball_positions[0][0] - skeleton_center_x,
                    sports_ball_positions[0][1] - skeleton_center_y
                )
                features_row['sports_ball_positions'] = f"{round(ball_relative_pos[0],2)},{round(ball_relative_pos[1],2)}"
            else:
                features_row['sports_ball_positions'] = np.nan


            # ─── LOWER-BODY: Hip‐Center Y ──────────────────────────
            # WHAT: Raw & normalized vertical position of hip midpoint.
            # WHY: Reflects overall body rise (leg extension) during shot.
            features_row["hip_center_y"]      = round(skeleton_center_y, 2) if skeleton_center_y is not None else np.nan
            features_row["hip_center_y_norm"] = round(normalized_hip_center_y, 3) if normalized_hip_center_y is not None else np.nan
            # ───────────────────────────────────────────────────────


            # Shot detection based on available wrist data.
            if "LEFT_WRIST" in joint_data and "RIGHT_WRIST" in joint_data:
                shot_detector.update(
                    left_wrist_vel=joint_data["LEFT_WRIST"]["vel"],
                    right_wrist_vel=joint_data["RIGHT_WRIST"]["vel"],
                    left_wrist_abs_pos=joint_data["LEFT_WRIST"]["pos"],
                    right_wrist_abs_pos=joint_data["RIGHT_WRIST"]["pos"],
                    ball_pos=(sports_ball_positions[0][0] - skeleton_center_x, sports_ball_positions[0][1] - skeleton_center_y) if sports_ball_positions and skeleton_center_x is not None and skeleton_center_y is not None else None,
                    fps=fps,
                    frame_count=frame_count
                )
            elif "LEFT_WRIST" in joint_data:
                shot_detector.update(
                    left_wrist_vel=joint_data["LEFT_WRIST"]["vel"],
                    right_wrist_vel=None,
                    left_wrist_abs_pos=joint_data["LEFT_WRIST"]["pos"],
                    right_wrist_abs_pos=None,
                    ball_pos=(sports_ball_positions[0][0] - skeleton_center_x, sports_ball_positions[0][1] - skeleton_center_y) if sports_ball_positions and skeleton_center_x is not None and skeleton_center_y is not None else None,
                    fps=fps,
                    frame_count=frame_count
                )
            elif "RIGHT_WRIST" in joint_data:
                shot_detector.update(
                    left_wrist_vel=None,
                    right_wrist_vel=joint_data["RIGHT_WRIST"]["vel"],
                    left_wrist_abs_pos=None,
                    right_wrist_abs_pos=joint_data["RIGHT_WRIST"]["pos"],
                    ball_pos=(sports_ball_positions[0][0] - skeleton_center_x, sports_ball_positions[0][1] - skeleton_center_y) if sports_ball_positions and skeleton_center_x is not None and skeleton_center_y is not None else None,
                    fps=fps,
                    frame_count=frame_count
                )
            else:
                shot_detector.reset_shot_state()

            # ─── LOWER-BODY: Stance Width ─────────────────────────
            # WHAT: Horizontal distance between left & right ankle (pixels).
            # WHY: Wider stance often correlates with greater stability.
            if "LEFT_ANKLE" in joint_data and "RIGHT_ANKLE" in joint_data:
                lx, _ = joint_data["LEFT_ANKLE"]["pos"]
                rx, _ = joint_data["RIGHT_ANKLE"]["pos"]
                features_row["stance_width"] = round(abs(rx - lx), 2)
            else:
                features_row["stance_width"] = np.nan
            # ───────────────────────────────────────────────────────




            for left_joint, right_joint in PAIRED_JOINTS + [("LEFT_HIP","RIGHT_HIP"),
                                                            ("LEFT_KNEE","RIGHT_KNEE"),
                                                            ("LEFT_ANKLE","RIGHT_ANKLE")]:
                features_row[f"{left_joint}_vel"] = joint_data[left_joint]["vel"] if left_joint in joint_data else np.nan
                features_row[f"{right_joint}_vel"] = joint_data[right_joint]["vel"] if right_joint in joint_data else np.nan
                features_row[f"{left_joint}_acc"] = joint_data[left_joint]["acc"] if left_joint in joint_data else np.nan
                features_row[f"{right_joint}_acc"] = joint_data[right_joint]["acc"] if right_joint in joint_data else np.nan

            for joint_name in joint_map.keys():
                pos = joint_data[joint_name]["pos"] if joint_name in joint_data else (np.nan, np.nan)
                features_row[f"{joint_name}_pos_x"] = round(pos[0], 2) if pos[0] is not None else np.nan
                features_row[f"{joint_name}_pos_y"] = round(pos[1], 2) if pos[1] is not None else np.nan

            for triplet in ANGLE_JOINTS:
                joint_a, joint_b, joint_c = triplet
                if (joint_a in joint_data and joint_b in joint_data and joint_c in joint_data and
                    not any(np.isnan(joint_data[j]["pos"][0]) or np.isnan(joint_data[j]["pos"][1]) for j in [joint_a, joint_b, joint_c])):
                    angle = calculate_angle(
                        joint_data[joint_a]["pos"],
                        joint_data[joint_b]["pos"],
                        joint_data[joint_c]["pos"]
                    )
                    features_row[f"{joint_b}_{joint_a}_{joint_c}_angle"] = angle
                else:
                    features_row[f"{joint_b}_{joint_a}_{joint_c}_angle"] = np.nan

            if shot_detector.current_shot:
                features_row['is_shot'] = 1 if shot_detector.state == ShotState.SHOT_IN_PROGRESS else 0
                try:
                    current_shot_id = int(shot_detector.current_shot.get('shot_id', np.nan))
                except (ValueError, TypeError):
                    current_shot_id = np.nan
                features_row['shot_id'] = current_shot_id
                features_row['shot_invalid'] = 1 if shot_detector.current_shot.get('invalid', False) else 0
            else:
                features_row['is_shot'] = 0
                features_row['shot_id'] = np.nan
                features_row['shot_invalid'] = 0

            features_row['make'] = np.nan
            thumbs_up_count = 0
            thumbs_down_count = 0
            make_status = None

            

            all_data.append(features_row)

            if total_frames > 0 and frame_count % 10 == 0:
                progress_fraction = min(frame_count / total_frames, 1.0)

                job_ref.update({
                    "main_py_progress": progress_fraction,
                    "main_py_status": "in_progress",
                    "updated_at": firestore.SERVER_TIMESTAMP
                })
                logger.debug(f"Updated main_py_progress: {progress_fraction:.2f}")

            if config.DEV_MODE in [0, 1] and results_pose.pose_landmarks:
                draw_pose_on_original(original_frame, results_pose.pose_landmarks, map_to_original, new_width, new_height, left_pad, top_pad)

            feature_screen = np.zeros((800, 600, 3), dtype=np.uint8)
            y_offset = 30
            current_shot_id = shot_detector.current_shot['shot_id'] if shot_detector.current_shot else "N/A"
            cv2.putText(feature_screen, f"Current Shot ID: {current_shot_id}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            current_shot_state = shot_detector.state.name if shot_detector.state else "N/A"
            cv2.putText(feature_screen, f"Shot State: {current_shot_state}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            if shot_detector.shots:
                last_shot = shot_detector.shots[-1]
                last_shot_id = last_shot.get('shot_id', "N/A")
                last_shot_valid = last_shot.get('invalid', False)
                color_valid = (0, 255, 0) if not last_shot_valid else (0, 0, 255)
                validity_text = "Valid" if not last_shot_valid else "Invalid"
                cv2.putText(feature_screen, f"Last Shot ID: {last_shot_id} ({validity_text})", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_valid, 2)
            else:
                cv2.putText(feature_screen, "Last Shot ID: N/A", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            cv2.putText(feature_screen, f"Frame: {frame_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            if "LEFT_WRIST" in joint_data or "RIGHT_WRIST" in joint_data:
                if sports_ball_positions and skeleton_center_x is not None and skeleton_center_y is not None:
                    distance = shot_detector.distance
                    color_dist = (0, 255, 0) if shot_detector.is_ball_close else (0, 0, 255)
                    cv2.putText(feature_screen, f"Ball-Wrist Dist: {distance:.2f}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_dist, 2)
                else:
                    cv2.putText(feature_screen, "Ball-Wrist Dist: N/A", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

            if (shot_detector.wrist_distance):
                wrist_dist = shot_detector.wrist_distance
            else:
                wrist_dist = 0.0
            are_close = shot_detector.are_wrists_close

            cv2.putText(
                feature_screen,
                f"WristDist: {wrist_dist:.1f}/{WRIST_CLOSE_DISTANCE_THRESHOLD:.1f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 150), 2
            )
            y_offset += 40

            cv2.putText(
                feature_screen,
                f"WristsClose: {are_close}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 200, 150), 2
            )
            y_offset += 40

            # --- Display the new y-velocity features ---
            cv2.putText(feature_screen, f"Ball Vel (Y): {ball_velocity_y:.2f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            
            # Show left‐wrist Y‐velocity against the “stable” threshold
            cv2.putText(
                feature_screen,
                f"LVelY: {left_wrist_velocity_y:.1f}/{STABLE_VELOCITY_THRESHOLD:.1f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            y_offset += 40

            # Show right‐wrist Y‐velocity against the “stable” threshold
            cv2.putText(
                feature_screen,
                f"RVelY: {right_wrist_velocity_y:.1f}/{STABLE_VELOCITY_THRESHOLD:.1f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
            )
            y_offset += 40

            displacement = 0.0
            if shot_detector.baseline_wrist_y is not None:
                # choose dominant wrist
                dom_joint = "RIGHT_WRIST" if config.DOMINANT_HAND == "RIGHT" else "LEFT_WRIST"
                if dom_joint in joint_data:
                    current_y = joint_data[dom_joint]["pos"][1]
                    displacement = shot_detector.baseline_wrist_y - current_y

            cv2.putText(
                feature_screen,
                f"Disp: {displacement:.1f}/{VERTICAL_DISPLACEMENT_THRESHOLD:.1f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2
            )
            y_offset += 40

            # 2) Count how many of the last N velocities exceed your threshold
            # OLD: You never visualized velocity_history contents
            lh = list(shot_detector.velocity_history_left)
            rh = list(shot_detector.velocity_history_right)
            cnt_l = sum(1 for v in lh if v > VELOCITY_THRESHOLD)
            cnt_r = sum(1 for v in rh if v > VELOCITY_THRESHOLD)
            buf_len = len(lh)  # same as CONSECUTIVE_FRAMES once full

            cv2.putText(
                feature_screen,
                f"Vhist L:{cnt_l}/{buf_len} R:{cnt_r}/{buf_len}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 200), 2
            )
            y_offset += 40

            # 3) Display current stability probability
            # OLD: You didn’t show shot_detector.ball_stability_prob
            stability = shot_detector.ball_stability_prob

            cv2.putText(
                feature_screen,
                f"Stability: {stability:.2f}/1.00",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 200), 2
            )
            y_offset += 40

            if (shot_detector.state == ShotState.READY_TO_DETECT_SHOT):
                post_stability = shot_detector.post_ball_stability_prob
                cv2.putText(
                    feature_screen,
                    f"Post - Stability: {post_stability:.2f}/1.00",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 50, 200), 2
                )
                y_offset += 40

            cv2.putText(
                feature_screen,
                f"InvFrames: {shot_detector.consecutive_invisible_frames}/{MAX_BALL_INVISIBLE_FRAMES}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2
            )
            y_offset += 40

            if shot_detector.stability_frame is not None:
                cv2.putText(
                    feature_screen,
                    f"Stab@Frame: {shot_detector.stability_frame}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,200,100), 2
                )
                y_offset += 40

            # --- Display the ball_disappeared_during_shot variable ---
            cv2.putText(feature_screen, f"Ball Disappeared During Shot: {shot_detector.ball_disappeared_during_shot}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_offset += 40
            
            if shot_detector.shots:
                last_valid_shot = None
                for shot in reversed(shot_detector.shots):
                    if not shot.get('invalid', True):
                        last_valid_shot = shot
                        break
                if last_valid_shot:
                    make_value_of_last_shot = last_valid_shot.get('make', None)
                    if make_value_of_last_shot is True:
                        last_shot_make_status = "Make Detected"
                    elif make_value_of_last_shot is False:
                        last_shot_make_status = "No Make Detected"
                    elif make_value_of_last_shot is None:
                        last_shot_make_status = "Make Status Not Detected Yet"
                    else:
                        last_shot_make_status = "Error, Needs Fixing"
                    shot_id = last_valid_shot.get('shot_id', 'N/A')
                    shot_id_text = f"Last Shot ID: {shot_id}"
                else:
                    last_shot_make_status = "No valid shots detected."
                    shot_id_text = "Last Shot ID: N/A"


            # OLD: no visual cue at shot start
            if (shot_detector.state == ShotState.SHOT_IN_PROGRESS and
                shot_detector.current_shot and
                shot_detector.current_shot['start_frame'] == frame_count):

                cv2.putText(
                    original_frame,
                    "→ SHOT START ←",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
                )
            # --- END: highlight shot start on original_frame ---

            if config.DEV_MODE in [0, 1]:
                resized_detection_display = cv2.resize(original_frame, (config.TARGET_WIDTH // 2, config.TARGET_HEIGHT // 2))
                cv2.imshow("Detection with Orientation Vectors", resized_detection_display)
                cv2.imshow("Feature Visualization", feature_screen)
            else:
                pass

            if frame_count % 20 == 0:
                logger.info(f"Processed {frame_count} frames, {frames_with_landmarks} with landmarks detected.")
            if config.DEV_MODE in [0, 1]:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exiting processing loop as 'q' was pressed.")
                    break

        logger.info("FINISHED EXITING OUT OF WHILE LOOP.")
        
        logger.info("Finished processing video frames.")
        # Final update for main.py stage
        job_ref.update({
            "main_py_progress": 1.0,
            "main_py_status": "completed",
            "updated_at": firestore.SERVER_TIMESTAMP
        })
        logger.info("main_py stage marked as completed in Firestore.")

        cap.release()
        if config.DEV_MODE in [0, 1]:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        logger.info(f"Finished processing video: {input_video_source}")

        df = pd.DataFrame(all_data)
        df.replace("N/A", np.nan, inplace=True)
        df.fillna(0, inplace=True)

        output_csv_filename = (
            os.path.splitext(os.path.basename(input_video_source))[0]
            + f"_entire_dataset_{job_id}.csv"
        )

        if 'shot_id' in df.columns:
            # Ensure shot_id is integer
            df['shot_id'] = pd.to_numeric(df['shot_id'], errors='coerce').astype('Int64')

            # Build shot_make_map using the post‐shot gesture window
            shots = sorted(shot_detector.shots, key=lambda s: s['shot_id'])
            video_end_time = total_frames / fps

            shot_make_map = {}
            for idx, shot in enumerate(shots):
                s_id = shot.get('shot_id')
                start_window = shot.get('end_time', 0.0)
                if idx + 1 < len(shots):
                    end_window = shots[idx+1].get('start_time', start_window)
                else:
                    end_window = video_end_time

                # Pick gestures that occur after this shot ends and before the next one starts
                evs = [
                    e for e in gesture_events
                    if start_window <= e.get('timestamp', -1) <= end_window
                ]

                # Assign make/miss
                if any(e.get('gesture') == 'thumbsUp'   for e in evs):
                    make_flag = True
                elif any(e.get('gesture') == 'thumbsDown' for e in evs):
                    make_flag = False
                else:
                    make_flag = False

                shot_make_map[s_id] = make_flag
                logger.debug(f"Shot ID {s_id}: window [{start_window:.2f},{end_window:.2f}] → make={make_flag}")

            logger.info(f"🔑 Final shot_make_map: {shot_make_map}")

            # Map into DataFrame and convert booleans to 1/0
            df['make'] = (
                df['shot_id']
                .map(shot_make_map)          # True/False or NaN
                .map({True: 1, False: 0})    # 1=make, 0=miss
                .fillna(0)                   # any unmatched shot_id → 0
                .astype(int)
            )
            logger.debug(f"DataFrame after mapping 'make':\n{df[['shot_id','make']].head()}")
        else:
            logger.warning("'shot_id' column not found; 'make' column will not be assigned.")


        logger.info(
            "📝 make flags by shot_id:\n" +
            df[['shot_id','make']]
            .drop_duplicates()
            .sort_values('shot_id')
            .to_string(index=False)
        )

        output_csv_path = os.path.join(DATASETS_DIR, output_csv_filename)
        try:
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Complete dataset saved to {output_csv_path}")
        except Exception as e:
            logger.error(f"Error saving complete dataset: {e}")

        logger.info(f"Publishing message to Pub/Sub with job_id: {job_id} and dataset_path: {output_csv_path}")
        message_payload = json.dumps({
            "job_id": job_id,
            "dataset_path": output_csv_path
        }).encode('utf-8')
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path('basketballformai', 'dataset-created')
        future = publisher.publish(topic_path, message_payload)
        future.result()


    pose.close()
    if config.DEV_MODE in [0, 1]:
        cv2.destroyAllWindows()
    logger.info("\nAll videos have been processed and datasets have been created. Now sending message to pub/sub to activate data_processing.py and Exiting the Program!")
    
    sys.exit(0)

if __name__ == "__main__":
    main()
