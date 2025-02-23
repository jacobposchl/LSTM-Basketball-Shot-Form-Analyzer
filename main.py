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
import config

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
    SHOT_COOLDOWN_FRAMES,
    ShotState,
    YOLO_CONFIDENCE_THRESHOLD,
    DETECTION_THRESHOLD,
    THUMB_GESTURE_WINDOW,
    THUMB_GESTURE_THRESHOLD,
    GCP_BUCKET_NAME,
    GCP_PREFIX,
    GCP_DOWNLOAD_DIR,
    WEIGHTS_DIR,
    YOLO_WEIGHTS_PATH,
    DEV_MODE,
    DEV_VIDEOS,
    DATASETS_DIR,
    LOGS_DIR
)

from shot_detection import ShotDetector
from hand_detection import HandDetector  # Ensure this is correctly implemented
from project_utils import calculate_angle, map_to_original  # Ensure these functions are correctly implemented

# Initialize logging
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILENAME = 'app.log'
LOG_FILEPATH = os.path.join(LOGS_DIR, LOG_FILENAME)
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILEPATH, mode='w'),  # Overwrites the log file each run
        logging.StreamHandler(sys.stdout)             # Logs to console
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Using config file: {config.__file__}")

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
        if None not in start_point and None not in end_idx:
            cv2.line(original_frame, (int(start_point[0]), int(start_point[1])),
                             (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)
    for point in landmark_coords_original:
        if None not in point:
            cv2.circle(original_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

def main():
    # Parse CLI arguments; if --video is provided, process only that file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=False, help="GCS video path (gs://bucket-name/video.mp4)")
    args = parser.parse_args()

    # If a specific video is provided via CLI, download that file only.
    if args.video:
        if args.video.startswith("gs://"):
            bucket_name = args.video.split("/")[2]
            video_filename = "/".join(args.video.split("/")[3:])
            logger.info(f"Processing video from GCS path: {args.video}")
            INPUT_VIDEOS = [download_video_from_gcs(bucket_name, video_filename)]
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

    if not INPUT_VIDEOS:
        logger.error("No video files found to process.")
        sys.exit(1)

    # Initialize detectors and models.
    shot_detector = ShotDetector()
    hand_detector = HandDetector()

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
    SPORTS_BALL_CLASS_NAME = "ball"
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

        if config.DEV_MODE in [0, 1]:
            cv2.namedWindow("Detection with Orientation Vectors", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Feature Visualization", cv2.WINDOW_NORMAL)
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

            features_row = {'video': os.path.basename(input_video_source), 'frame': frame_count}
            if skeleton_center_x is not None and skeleton_center_y is not None and sports_ball_positions:
                ball_relative_pos = (
                    sports_ball_positions[0][0] - skeleton_center_x,
                    sports_ball_positions[0][1] - skeleton_center_y
                )
                features_row['sports_ball_positions'] = f"{round(ball_relative_pos[0],2)},{round(ball_relative_pos[1],2)}"
            else:
                features_row['sports_ball_positions'] = np.nan

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

            for left_joint, right_joint in PAIRED_JOINTS:
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

            if shot_detector.detect_hands:
                if "LEFT_WRIST" in joint_data and "RIGHT_WRIST" in joint_data:
                    left_wrist_rel_pos = joint_data["LEFT_WRIST"]["pos"]
                    right_wrist_rel_pos = joint_data["RIGHT_WRIST"]["pos"]
                    left_wrist_abs_pos = (skeleton_center_x + left_wrist_rel_pos[0], skeleton_center_y + left_wrist_rel_pos[1]) if skeleton_center_x is not None and skeleton_center_y is not None else (None, None)
                    right_wrist_abs_pos = (skeleton_center_x + right_wrist_rel_pos[0], skeleton_center_y + right_wrist_rel_pos[1]) if skeleton_center_x is not None and skeleton_center_y is not None else (None, None)
                    if left_wrist_abs_pos[0] is not None and left_wrist_abs_pos[1] is not None:
                        draw_wrist_roi(original_frame, left_wrist_abs_pos[0], left_wrist_abs_pos[1], config.ROI_SIZE, (255, 0, 0), "Left Wrist")
                    if right_wrist_abs_pos[0] is not None and right_wrist_abs_pos[1] is not None:
                        draw_wrist_roi(original_frame, right_wrist_abs_pos[0], right_wrist_abs_pos[1], config.ROI_SIZE, (0, 255, 0), "Right Wrist")
                    hands_detected = 0
                    thumbs_up_count = 0
                    thumbs_down_count = 0
                    hand_detector.clear_features()
                    # Process hands within ROIs; note the flipped labels as before.
                    for roi_pos, label in [((left_wrist_abs_pos[0], left_wrist_abs_pos[1]), "Right"), 
                                           ((right_wrist_abs_pos[0], right_wrist_abs_pos[1]), "Left")]:
                        x_center, y_center = roi_pos
                        if x_center is None or y_center is None:
                            continue
                        x_min = max(int(x_center - 75), 0)
                        y_min = max(int(y_center - 75), 0)
                        x_max = min(int(x_center + 75), original_width)
                        y_max = min(int(y_center + 75), original_height)
                        if x_min >= x_max or y_min >= y_max:
                            continue
                        roi_frame = original_frame[y_min:y_max, x_min:x_max]
                        if roi_frame.size == 0:
                            continue
                        roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                        if hand_detector:
                            try:
                                hand_results = hand_detector.process_frame(roi_rgb)
                            except Exception as e:
                                hand_results = None
                                logger.error(f"Hand detection failed in ROI: {e}")
                            if hand_results and hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                                if config.DEV_MODE in [0, 1]:
                                    hand_detector.draw_landmarks(original_frame, hand_results.multi_hand_landmarks, hand_results.multi_handedness, draw_orientation_vectors=True)
                                for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                                    detected_handedness = hand_handedness.classification[0].label
                                    if detected_handedness != label:
                                        continue
                                    hands_detected += 1
                                    is_up = hand_detector.is_thumbs_up(hand_landmarks, detected_handedness)
                                    is_down = hand_detector.is_thumbs_down(hand_landmarks, detected_handedness)
                                    if is_up:
                                        thumbs_up_count += 1
                                    if is_down:
                                        thumbs_down_count += 1

            if shot_detector.detect_hands:
                if thumbs_up_count >= 2:
                    current_gesture = 'up'
                elif thumbs_down_count >= 2:
                    current_gesture = 'down'
                elif thumbs_up_count == 1:
                    current_gesture = 'up'
                elif thumbs_down_count == 1:
                    current_gesture = 'down'
                else:
                    current_gesture = 'none'
                gesture_history.append(current_gesture)
                up_count = gesture_history.count('up')
                down_count = gesture_history.count('down')
                half_window = math.ceil(config.THUMB_GESTURE_WINDOW * config.THUMB_GESTURE_THRESHOLD)
                if up_count >= half_window:
                    make_status = "Make"
                    shot_detector.assign_make_shot(True)
                elif down_count >= half_window:
                    make_status = "No Make"
                    shot_detector.assign_make_shot(False)

            all_data.append(features_row)

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
            if shot_detector.detect_hands:
                color_thumbs = (0, 255, 0) if thumbs_up_count == 2 else ((0, 255, 255) if thumbs_up_count == 1 else (0, 0, 255))
                cv2.putText(feature_screen, f"Thumbs Ups: {thumbs_up_count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_thumbs, 2)
            else:
                cv2.putText(feature_screen, "Thumbs Ups: N/A", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            if shot_detector.detect_hands:
                color_thumbs = (0, 255, 0) if thumbs_down_count == 2 else ((0, 255, 255) if thumbs_down_count == 1 else (0, 0, 255))
                cv2.putText(feature_screen, f"Thumbs Downs: {thumbs_down_count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_thumbs, 2)
            else:
                cv2.putText(feature_screen, "Thumbs Downs: N/A", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40
            detect_hands_text = "Detect Hands: True" if shot_detector.detect_hands else "Detect Hands: False"
            color_detect_hands = (0, 255, 0) if shot_detector.detect_hands else (0, 0, 255)
            cv2.putText(feature_screen, detect_hands_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_detect_hands, 2)
            y_offset += 40
            if shot_detector.shots:
                if make_status in ["Make", "No Make"]:
                    make_status_text = f"Current Make Status: {make_status}"
                else:
                    make_status_text = "Make Status: No Thumb Gestures Detected Yet"
            else:
                make_status_text = "Make Status: No Shots Detected Yet"
            cv2.putText(feature_screen, make_status_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
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
            else:
                last_shot_make_status = "No Shots Detected Yet"
                shot_id_text = "Last Shot ID: N/A"
            last_shot_make_status_text = f"Last Shot Make Status: {last_shot_make_status}"
            cv2.putText(feature_screen, last_shot_make_status_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 40
            cv2.putText(feature_screen, shot_id_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

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
        cap.release()
        if config.DEV_MODE in [0, 1]:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        logger.info(f"Finished processing video: {input_video_source}")

        df = pd.DataFrame(all_data)
        df.replace("N/A", np.nan, inplace=True)
        df.fillna(0, inplace=True)
        df['make'] = np.nan
        output_csv_filename = os.path.splitext(os.path.basename(input_video_source))[0] + "_entire_dataset.csv"
        if 'shot_id' in df.columns:
            df['shot_id'] = pd.to_numeric(df['shot_id'], errors='coerce').astype('Int64')
            shot_make_map = {}
            for shot in shot_detector.shots:
                s_id = shot.get('shot_id', None)
                make_val = shot.get('make', None)
                if s_id is not None and not pd.isna(s_id):
                    shot_make_map[s_id] = make_val
            for s_id, make in shot_make_map.items():
                logger.debug(f"Shot ID {s_id}: Make={make}")
            df['make'] = df['shot_id'].map(shot_make_map)
            df['make'] = df['make'].map({True: 1, False: 0})
            logger.debug(f"DataFrame after mapping 'make':\n{df[['shot_id', 'make']].head()}")
        else:
            logger.warning("'shot_id' column not found in the dataframe. 'make' column will not be assigned.")

        output_csv_path = os.path.join(DATASETS_DIR, output_csv_filename)
        try:
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Complete dataset successfully saved to {output_csv_path}")
        except Exception as e:
            logger.error(f"Error saving complete dataset to {output_csv_path}: {e}")

    pose.close()
    if shot_detector.detect_hands and hand_detector:
        hand_detector.close()
    if config.DEV_MODE in [0, 1]:
        cv2.destroyAllWindows()
    logger.info("\nAll videos have been processed and datasets have been created. Now Exiting the Program!")
    sys.exit(0)

if __name__ == "__main__":
    main()
