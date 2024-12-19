# main.py

import cv2
import math
import numpy as np
import pandas as pd
import os
import sys
import torch
from collections import deque
import logging
import config

# Import configurations from config.py
from config import (
    INPUT_VIDEOS,
    LABEL,
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
    DETECTION_THRESHOLD  # Ensure this is defined in config.py
)

from shot_detection import ShotDetector
from hand_detection import HandDetector  # Ensure this is correctly implemented
from project_utils import calculate_angle, map_to_original  # Ensure these functions are correctly implemented

# Initialize logging
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILENAME = 'app.log'
LOG_FILEPATH = os.path.join(LOG_DIR, LOG_FILENAME)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILEPATH, mode='w'),  # Overwrites the log file each run
        logging.StreamHandler(sys.stdout)             # Logs to console
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Using config file: {config.__file__}")

def draw_wrist_roi(frame: np.ndarray, x: float, y: float, roi_size: int, color: tuple, label: str) -> None:
    """
    Draws an ROI rectangle around the specified wrist on the given frame.

    Args:
        frame (np.array): The image/frame to draw on.
        x (float): The x-coordinate of the wrist.
        y (float): The y-coordinate of the wrist.
        roi_size (int): The size of the ROI square.
        color (tuple): The color of the rectangle in BGR.
        label (str): The label to display.
    """
    if x is not None and y is not None:
        # Ensure ROI does not exceed frame boundaries
        x_min = max(int(x - roi_size / 2), 0)
        y_min = max(int(y - roi_size / 2), 0)
        x_max = min(int(x + roi_size / 2), frame.shape[1])
        y_max = min(int(y + roi_size / 2), frame.shape[0])

        # Draw rectangle
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

        # Put label text
        cv2.putText(frame, f"{label} ROI", (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_pose_on_original(original_frame: np.ndarray, pose_landmarks, map_to_original_func, 
                          new_width: int, new_height: int, left_pad: int, top_pad: int) -> None:
    """
    Draws pose landmarks on the original_frame by mapping them from the resized and padded frame.

    Args:
        original_frame (np.array): The original image/frame to draw on.
        pose_landmarks: Pose landmarks detected by Mediapipe.
        map_to_original_func (function): Function to map coordinates from padded frame to original frame.
        new_width (int): Width of the resized frame before padding.
        new_height (int): Height of the resized frame before padding.
        left_pad (int): Left padding added to the resized frame.
        top_pad (int): Top padding added to the resized frame.
    """
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    connections = mp_pose.POSE_CONNECTIONS

    # Extract landmark coordinates
    landmarks = pose_landmarks.landmark

    # Convert normalized landmarks to pixel coordinates in the padded frame
    landmark_coords_padded = []
    for lm in landmarks:
        x_padded = lm.x * TARGET_WIDTH
        y_padded = lm.y * TARGET_HEIGHT
        landmark_coords_padded.append((x_padded, y_padded))

    # Map landmarks to original frame coordinates
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

    # Draw connections
    for connection in connections:
        start_idx, end_idx = connection
        start_point = landmark_coords_original[start_idx]
        end_point = landmark_coords_original[end_idx]
        if None not in start_point and None not in end_point:
            cv2.line(original_frame, (int(start_point[0]), int(start_point[1])),
                             (int(end_point[0]), int(end_point[1])), (0, 255, 0), 2)

    # Draw landmarks
    for point in landmark_coords_original:
        if None not in point:
            cv2.circle(original_frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

def main():
    # Define output datasets directory from config.py
    DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datasets')
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # Initialize ShotDetector
    shot_detector = ShotDetector()

    # Initialize HandDetector if detect_hands is True
    if shot_detector.detect_hands:
        hand_detector = HandDetector()
    else:
        hand_detector = None

    # Initialize MediaPipe Pose
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_drawing = mp.solutions.drawing_utils

    # Initialize YOLOv5 model
    try:
        model = torch.hub.load(
            'ultralytics/yolov5',                # Repository name
            'custom',                            # Model variant
            path='Weights/best.pt',              # Path to custom weights (from config.py)
            source='github'                      # Load from GitHub repository
        )
        model.eval()  # Set model to evaluation mode
        logger.info("YOLOv5 model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading YOLOv5 model: {e}")
        sys.exit(1)

    # Get class names and ball class ID
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

    # Process each video
    for idx, input_video_path in enumerate(INPUT_VIDEOS):
        logger.info(f"\nProcessing video {idx+1}/{len(INPUT_VIDEOS)}: {input_video_path}")

        if not os.path.isfile(input_video_path):
            logger.error(f"Video file does not exist: {input_video_path}")
            continue

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video file: {input_video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 60.0  # Default FPS if unable to get from video

        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Original Video Properties - FPS: {fps}, Width: {original_width}, Height: {original_height}")
        logger.info(f"Resizing all frames to {TARGET_WIDTH}x{TARGET_HEIGHT} for pose estimation.")

        prev_positions = {}
        prev_velocities = {}
        all_data = []
        frame_count = 0
        frames_with_landmarks = 0

        # Reset ShotDetector
        shot_detector.reset_shot_state()
        shot_num = 0
        logger.info("Initialized shot_num to 0")

        # Initialize smoothing windows
        joint_vel_history = {joint_name: deque(maxlen=SMOOTHING_WINDOW) for joint_name in joint_map.keys()}
        joint_acc_history = {joint_name: deque(maxlen=SMOOTHING_WINDOW) for joint_name in joint_map.keys()}

        # Initialize detection history deque for temporal smoothing
        detection_history = deque(maxlen=DETECTION_THRESHOLD)

        # Initialize visualization windows for the first few videos
        if idx < 5:
            cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Feature Visualization", cv2.WINDOW_NORMAL)
        else:
            logger.info("Skipping visualization for videos beyond the first 5.")

        logger.info("Starting video processing...")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video file reached.")
                break

            frame_count += 1

            # Keep a copy of the original frame for YOLO
            original_frame = frame.copy()

            # Resize frame to target resolution while preserving aspect ratio
            aspect_ratio = original_width / original_height
            target_aspect_ratio = TARGET_WIDTH / TARGET_HEIGHT

            if aspect_ratio > target_aspect_ratio:
                # Fit to width
                new_width = TARGET_WIDTH
                new_height = int(TARGET_WIDTH / aspect_ratio)
            else:
                # Fit to height
                new_height = TARGET_HEIGHT
                new_width = int(TARGET_HEIGHT * aspect_ratio)

            # Resize while maintaining aspect ratio
            frame_resized = cv2.resize(frame, (new_width, new_height))

            # Add padding to reach TARGET_WIDTH x TARGET_HEIGHT
            delta_w = TARGET_WIDTH - new_width
            delta_h = TARGET_HEIGHT - new_height
            top_pad, bottom_pad = delta_h // 2, delta_h - (delta_h // 2)
            left_pad, right_pad = delta_w // 2, delta_w - (delta_w // 2)

            color = [0, 0, 0]  # Black padding
            frame_padded = cv2.copyMakeBorder(frame_resized, top_pad, bottom_pad, left_pad, right_pad, 
                                             cv2.BORDER_CONSTANT, value=color)

            rgb_frame = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)

            # Run pose processing synchronously on resized and padded frame
            results_pose = pose.process(rgb_frame)

            # Run YOLO on the original frame
            yolo_results = model(original_frame)
            filtered_detections = []

            if yolo_results and hasattr(yolo_results, 'xyxy') and len(yolo_results.xyxy) > 0:
                for det in yolo_results.xyxy[0]:
                    x1, y1, x2, y2, conf, cls_id = det
                    cls_id = int(cls_id)
                    conf = float(conf)
                    if cls_id == SPORTS_BALL_CLASS_ID and conf >= YOLO_CONFIDENCE_THRESHOLD:
                        filtered_detections.append(det)
                        # Draw bounding box on the original frame
                        cv2.rectangle(original_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label_text = f"Ball: {conf:.2f}"
                        cv2.putText(original_frame, label_text, (int(x1), int(y1) - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            sports_ball_positions = []
            largest_ball = None
            largest_area = 0

            # Find largest ball in the original frame coordinates
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

            if results_pose and results_pose.pose_landmarks:
                landmarks = results_pose.pose_landmarks.landmark
                left_hip = landmarks[joint_map["LEFT_HIP"]]
                right_hip = landmarks[joint_map["RIGHT_HIP"]]

                if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                    # Calculate the center of the hips in the padded frame
                    skeleton_center_padded_x = ((left_hip.x + right_hip.x) / 2) * TARGET_WIDTH
                    skeleton_center_padded_y = ((left_hip.y + right_hip.y) / 2) * TARGET_HEIGHT

                    # Map the skeleton center back to the original frame
                    skeleton_center_original_x, skeleton_center_original_y = map_to_original(
                        skeleton_center_padded_x, skeleton_center_padded_y,
                        original_width, original_height,
                        new_width, new_height,
                        left_pad, top_pad
                    )

                    skeleton_center_x = skeleton_center_original_x
                    skeleton_center_y = skeleton_center_original_y
                    frames_with_landmarks += 1
                else:
                    logger.warning(f"Frame {frame_count}: Hips not sufficiently visible.")
            else:
                logger.warning(f"Frame {frame_count}: No pose landmarks detected.")

            joint_data = {}
            if skeleton_center_x is not None and skeleton_center_y is not None and results_pose.pose_landmarks:
                for joint_name, joint_idx in joint_map.items():
                    j_lm = landmarks[joint_idx]
                    # Scale normalized landmark to actual pixel coordinates on padded frame
                    x_padded = j_lm.x * TARGET_WIDTH
                    y_padded = j_lm.y * TARGET_HEIGHT

                    # Map to original frame coordinates
                    x_original, y_original = map_to_original(
                        x_padded, y_padded,
                        original_width, original_height,
                        new_width, new_height,
                        left_pad, top_pad
                    )

                    # Calculate relative position to hip center
                    relative_x = x_original - skeleton_center_x
                    relative_y = y_original - skeleton_center_y

                    current_pos = (relative_x, relative_y)

                    # Calculate velocity
                    if joint_name in prev_positions and prev_positions[joint_name] is not None:
                        dx = relative_x - prev_positions[joint_name][0]
                        dy = relative_y - prev_positions[joint_name][1]
                        raw_velocity = math.sqrt(dx**2 + dy**2) * fps
                        raw_velocity = round(raw_velocity, 2)
                    else:
                        raw_velocity = None

                    # Acceleration computation
                    if raw_velocity is not None and joint_name in prev_velocities and prev_velocities[joint_name] is not None:
                        dvel = raw_velocity - prev_velocities[joint_name]
                        raw_acc = dvel * fps
                        raw_acc = round(raw_acc, 2)
                    else:
                        raw_acc = None

                    # Update histories
                    if raw_velocity is not None:
                        joint_vel_history[joint_name].append(raw_velocity)
                    else:
                        joint_vel_history[joint_name].append(0)

                    if raw_acc is not None:
                        joint_acc_history[joint_name].append(raw_acc)
                    else:
                        joint_acc_history[joint_name].append(0)

                    # Smoothed values
                    smoothed_velocity = np.mean(joint_vel_history[joint_name])
                    smoothed_acc = np.mean(joint_acc_history[joint_name])

                    joint_data[joint_name] = {
                        "pos": current_pos,  # Relative position
                        "vel": round(smoothed_velocity, 2),
                        "acc": round(smoothed_acc, 2)
                    }

                    prev_positions[joint_name] = current_pos
                    prev_velocities[joint_name] = raw_velocity

            features_row = {
                'video': os.path.basename(input_video_path),
                'frame': frame_count,
                'label': LABEL
            }

            # Safely compute sports_ball_positions
            if skeleton_center_x is not None and skeleton_center_y is not None and sports_ball_positions:
                # Adjust ball position to be relative to hip center
                ball_relative_pos = (
                    sports_ball_positions[0][0] - skeleton_center_x,
                    sports_ball_positions[0][1] - skeleton_center_y
                )
                positions_str = f"{round(ball_relative_pos[0],2)},{round(ball_relative_pos[1],2)}"
                features_row['sports_ball_positions'] = positions_str
            else:
                features_row['sports_ball_positions'] = np.nan

            # Shot Detection Logic
            if "LEFT_WRIST" in joint_data:
                wrist_vel = joint_data["LEFT_WRIST"]["vel"]
                wrist_pos = joint_data["LEFT_WRIST"]["pos"]  # Relative position
                wrist_abs_y = wrist_pos[1]
                # Ball position (relative)
                ball_pos = (
                    sports_ball_positions[0][0] - skeleton_center_x,
                    sports_ball_positions[0][1] - skeleton_center_y
                ) if sports_ball_positions and skeleton_center_x is not None and skeleton_center_y is not None else None

                # Update ShotDetector
                shot_detector.update(
                    wrist_vel=wrist_vel,
                    wrist_abs_y=wrist_abs_y,
                    wrist_pos=wrist_pos,
                    ball_pos=ball_pos,
                    fps=fps,
                    frame_count=frame_count
                )
            else:
                # No LEFT_WRIST found, possibly reset shot_state or handle accordingly
                logger.debug(f"Frame {frame_count}: LEFT_WRIST not detected. Resetting shot state.")
                shot_detector.reset_shot_state()

            # Add velocities and accelerations to features_row
            for left_joint, right_joint in PAIRED_JOINTS:
                left_vel = joint_data[left_joint]["vel"] if left_joint in joint_data else np.nan
                right_vel = joint_data[right_joint]["vel"] if right_joint in joint_data else np.nan
                left_acc = joint_data[left_joint]["acc"] if left_joint in joint_data else np.nan
                right_acc = joint_data[right_joint]["acc"] if right_joint in joint_data else np.nan

                features_row[f"{left_joint}_vel"] = left_vel
                features_row[f"{right_joint}_vel"] = right_vel
                features_row[f"{left_joint}_acc"] = left_acc
                features_row[f"{right_joint}_acc"] = right_acc

            # Add joint positions to features_row using relative coordinates
            for joint_name in joint_map.keys():
                pos_x, pos_y = joint_data[joint_name]["pos"] if joint_name in joint_data else (np.nan, np.nan)
                features_row[f"{joint_name}_pos_x"] = round(pos_x, 2) if pos_x is not None else np.nan
                features_row[f"{joint_name}_pos_y"] = round(pos_y, 2) if pos_y is not None else np.nan

            # Calculate angles using relative coordinates
            for triplet in ANGLE_JOINTS:
                joint_a, joint_b, joint_c = triplet
                if (joint_a in joint_data and joint_b in joint_data and joint_c in joint_data and
                    joint_data[joint_a]["pos"] != (np.nan, np.nan) and
                    joint_data[joint_b]["pos"] != (np.nan, np.nan) and
                    joint_data[joint_c]["pos"] != (np.nan, np.nan)):
                    angle = calculate_angle(
                        joint_data[joint_a]["pos"],
                        joint_data[joint_b]["pos"],
                        joint_data[joint_c]["pos"]
                    )
                    features_row[f"{joint_b}_{joint_a}_{joint_c}_angle"] = angle
                else:
                    features_row[f"{joint_b}_{joint_a}_{joint_c}_angle"] = np.nan

            # Update features_row with shot information
            features_row['is_shot'] = 0
            features_row['shot_id'] = np.nan
            features_row['shot_invalid'] = 0  # 0: valid, 1: invalid

            # Initialize thumbs_up_count
            thumbs_up_count = 0

            # Thumbs-Up Detection Logic
            if shot_detector.detect_hands and idx < 5:  # Only for the first 5 videos and if hand detection is enabled
                if "LEFT_WRIST" in joint_data and "RIGHT_WRIST" in joint_data:
                    # Process both wrists for thumbs-up
                    left_wrist_rel_pos = joint_data["LEFT_WRIST"]["pos"]
                    right_wrist_rel_pos = joint_data["RIGHT_WRIST"]["pos"]

                    # Convert relative positions to absolute positions
                    left_wrist_abs_pos = (skeleton_center_x + left_wrist_rel_pos[0], skeleton_center_y + left_wrist_rel_pos[1]) if skeleton_center_x is not None and skeleton_center_y is not None else (None, None)
                    right_wrist_abs_pos = (skeleton_center_x + right_wrist_rel_pos[0], skeleton_center_y + right_wrist_rel_pos[1]) if skeleton_center_x is not None and skeleton_center_y is not None else (None, None)

                    # Define ROIs around both wrists using absolute positions
                    if left_wrist_abs_pos[0] is not None and left_wrist_abs_pos[1] is not None:
                        draw_wrist_roi(original_frame, left_wrist_abs_pos[0], left_wrist_abs_pos[1], 150, (255, 0, 0), "Left Wrist")
                    if right_wrist_abs_pos[0] is not None and right_wrist_abs_pos[1] is not None:
                        draw_wrist_roi(original_frame, right_wrist_abs_pos[0], right_wrist_abs_pos[1], 150, (0, 255, 0), "Right Wrist")

                    # Process hands within ROIs
                    hands_detected = 0
                    thumbs_up_count = 0

                    for roi_pos, label in [((left_wrist_abs_pos[0], left_wrist_abs_pos[1]), "Left"), 
                                           ((right_wrist_abs_pos[0], right_wrist_abs_pos[1]), "Right")]:
                        x_center, y_center = roi_pos
                        if x_center is None or y_center is None:
                            continue  # Skip if wrist position is not available

                        x_min = max(int(x_center - 75), 0)
                        y_min = max(int(y_center - 75), 0)
                        x_max = min(int(x_center + 75), original_width)
                        y_max = min(int(y_center + 75), original_height)

                        # Validate ROI dimensions
                        if x_min >= x_max or y_min >= y_max:
                            logger.warning(f"Frame {frame_count}: Invalid ROI for {label} wrist. Skipping.")
                            continue

                        # Crop the ROI from the original frame
                        roi_frame = original_frame[y_min:y_max, x_min:x_max]
                        if roi_frame.size == 0:
                            logger.warning(f"Frame {frame_count}: Empty ROI for {label} wrist. Skipping.")
                            continue

                        roi_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)

                        # Process the ROI with HandDetector only if ShotState = Waiting for Thumbs-Up
                        if shot_detector.detect_hands and hand_detector:
                            hand_results = hand_detector.process_frame(roi_rgb)

                            if hand_results and hand_results.multi_hand_landmarks and hand_results.multi_handedness:
                                for hand_landmarks, hand_handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
                                    hands_detected += 1
                                    handedness_label = hand_handedness.classification[0].label  # 'Left' or 'Right'

                                    # *** Begin: Mapping and Drawing Landmarks ***
                                    # Convert normalized landmarks to ROI pixel coordinates
                                    landmarks_pixel = []
                                    for lm in hand_landmarks.landmark:
                                        x = int(lm.x * 150) + x_min  # ROI size is 150
                                        y = int(lm.y * 150) + y_min
                                        landmarks_pixel.append((x, y))

                                    # Draw connections
                                    for connection in mp.solutions.hands.HAND_CONNECTIONS:
                                        start_idx, end_idx = connection
                                        start_point = landmarks_pixel[start_idx]
                                        end_point = landmarks_pixel[end_idx]
                                        cv2.line(original_frame, start_point, end_point, (0, 255, 0), 2)

                                    # Draw landmarks
                                    for point in landmarks_pixel:
                                        cv2.circle(original_frame, point, 2, (0, 0, 255), -1)

                                    # Optionally, add text for handedness
                                    cv2.putText(original_frame, handedness_label, (x_min, y_max + 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                    # *** End: Mapping and Drawing Landmarks ***

                                    # Determine if the hand is upright
                                    if not hand_detector.is_hand_upright(hand_landmarks):
                                        continue  # Skip if hand is not upright

                                    # Determine if the hand is showing thumbs-up
                                    # Adjust the y-coordinate reference for relative positions
                                    hip_y_relative = 0  # Since positions are relative to hip center
                                    if hand_detector.is_thumbs_up(hand_landmarks, handedness_label, hip_y_relative):
                                        thumbs_up_count += 1
                                        # Annotate thumbs-up on the display frame
                                        cv2.putText(original_frame, f"Thumbs Up ({label})", 
                                                    (x_min, y_min - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                        logger.info(f"Frame {frame_count}: Thumbs Up detected on {label} hand.")
                                    else:
                                        logger.info(f"Frame {frame_count}: No thumbs up on {label} hand.")
                if shot_detector.detect_hands and idx < 5:
                    # Update detection history
                    if thumbs_up_count == 2 and hands_detected == 2:
                        detection_history.append(1)
                    else:
                        detection_history.append(0)

                    # Determine thumbs-up status based on detection history
                    if sum(detection_history) >= DETECTION_THRESHOLD:
                        # Display confirmation message
                        thumbs_up_status = "Both Thumbs Up!"
                        # Reset history after detection
                        detection_history.clear()
                    elif thumbs_up_count > 0:
                        thumbs_up_status = f"{thumbs_up_count} Hand(s) Showing Thumbs Up"
                    else:
                        thumbs_up_status = "No Thumbs Up Detected"

            # Add the current features_row to all_data
            all_data.append(features_row)

            # Draw pose landmarks on original_frame
            if results_pose.pose_landmarks:
                draw_pose_on_original(
                    original_frame,
                    results_pose.pose_landmarks,
                    map_to_original,
                    new_width,
                    new_height,
                    left_pad,
                    top_pad
                )

            # Simplified Feature Visualization
            feature_screen = np.zeros((400, 600, 3), dtype=np.uint8)
            y_offset = 30

            # Current Shot ID
            current_shot_id = shot_detector.current_shot['shot_id'] if shot_detector.current_shot else "N/A"
            cv2.putText(feature_screen, f"Current Shot ID: {current_shot_id}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

            # Current Shot State
            current_shot_state = shot_detector.state.name if shot_detector.state else "N/A"
            cv2.putText(feature_screen, f"Shot State: {current_shot_state}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

            # Last Shot ID and Validity
            if shot_detector.shots:
                last_shot = shot_detector.shots[-1]
                last_shot_id = last_shot['shot_id']
                last_shot_valid = last_shot['invalid']
                color_valid = (0, 255, 0) if not last_shot_valid else (0, 0, 255)
                validity_text = "Valid" if not last_shot_valid else "Invalid"
                cv2.putText(feature_screen, f"Last Shot ID: {last_shot_id} ({validity_text})", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_valid, 2)
            else:
                cv2.putText(feature_screen, "Last Shot ID: N/A", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

            # Current Frame
            cv2.putText(feature_screen, f"Frame: {frame_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

            # Distance between Ball and Wrist
            if "LEFT_WRIST" in joint_data and sports_ball_positions and skeleton_center_x is not None and skeleton_center_y is not None:
                distance = shot_detector.distance  # Assuming shot_detector calculates this
                color_dist = (0, 255, 0) if shot_detector.is_ball_close else (0, 0, 255)
                cv2.putText(feature_screen, f"Ball-Wrist Dist: {distance:.2f}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_dist, 2)
            else:
                cv2.putText(feature_screen, "Ball-Wrist Dist: N/A", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

            # Thumbs Up Count
            if shot_detector.detect_hands and idx < 5:
                if thumbs_up_count == 2:
                    color_thumbs = (0, 255, 0)  # Green
                elif thumbs_up_count == 1:
                    color_thumbs = (0, 255, 255)  # Yellow
                else:
                    color_thumbs = (0, 0, 255)  # Red
                cv2.putText(feature_screen, f"Thumbs Ups: {thumbs_up_count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_thumbs, 2)
            else:
                cv2.putText(feature_screen, "Thumbs Ups: N/A", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 40

            # Detect Hands Status
            detect_hands_text = "Detect Hands: True" if shot_detector.detect_hands else "Detect Hands: False"
            color_detect_hands = (0, 255, 0) if shot_detector.detect_hands else (0, 0, 255)
            cv2.putText(feature_screen, detect_hands_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_detect_hands, 2)
            y_offset += 40

            # Visualization
            if idx < 5:
                # Resize frame for display
                resized_detection_display = cv2.resize(original_frame, (TARGET_WIDTH // 2, TARGET_HEIGHT // 2))
                cv2.imshow("Detection", resized_detection_display)

                # Display Feature Visualization
                cv2.imshow("Feature Visualization", feature_screen)
            elif idx >= 5:
                # Display Feature Visualization
                cv2.imshow("Feature Visualization", feature_screen)

            # Log progress every 100 frames
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count} frames, {frames_with_landmarks} with landmarks detected.")

            # Allow exit on 'q' key press for the first 5 videos
            if idx < 5:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Exiting processing loop as 'q' was pressed.")
                    break

        # After exiting the loop
        cap.release()
        if idx < 5:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Ensure all windows are closed properly

        logger.info(f"Finished processing video: {input_video_path}")

        # Data Engineering: Save the entire dataset for the current video
        df = pd.DataFrame(all_data)
        df.replace("N/A", np.nan, inplace=True)
        df.fillna(0, inplace=True)

        output_csv_filename = os.path.splitext(os.path.basename(input_video_path))[0] + "_entire_dataset.csv"
        output_csv_path = os.path.join(DATASETS_DIR, output_csv_filename)
        df.to_csv(output_csv_path, index=False)
        logger.info(f"Complete dataset successfully saved to {output_csv_path}")

        # Save individual shots if needed
        if 'shot_id' in df.columns:
            unique_shots = df['shot_id'].dropna().unique()
            unique_shots = [int(s) for s in unique_shots if not pd.isna(s)]

            if len(unique_shots) > 0:
                for s_id in unique_shots:
                    # Filter shots list for valid shots with the current shot_id
                    shot_data = [s for s in shot_detector.shots if s.get('shot_id') == s_id and not s.get('invalid', False)]
                    if len(shot_data) == 1:
                        shot = shot_data[0]
                        if 'end_frame' in shot and shot['end_frame'] is not None:
                            # Filter df for this shot_id
                            shot_df = df[df['shot_id'] == s_id].copy()
                            if not shot_df.empty:
                                columns_to_drop = [
                                    'LEFT_HIP_pos_x', 'LEFT_HIP_pos_y',
                                    'RIGHT_HIP_pos_x', 'RIGHT_HIP_pos_y', 'LEFT_HIP_vel',
                                    'LEFT_HIP_acc', 'RIGHT_HIP_vel', 'RIGHT_HIP_acc',
                                    'is_shot', 'shot_invalid'
                                ]
                                shot_df = shot_df.drop(columns=columns_to_drop, errors='ignore')
                                shot_df = shot_df.reset_index(drop=True)

                                output_csv_filename = os.path.splitext(os.path.basename(input_video_path))[0] + f"_shot_{s_id}.csv"
                                output_csv_path = os.path.join(DATASETS_DIR, output_csv_filename)

                                shot_df.to_csv(output_csv_path, index=False)
                                logger.info(f"Feature data for shot {s_id} successfully saved to {output_csv_path}")
                    else:
                        logger.warning(f"No valid shot data found for shot_id {s_id}. No dataset saved.")
            else:
                logger.warning(f"No valid shots found in {input_video_path}")
        else:
            logger.warning(f"No shots detected or 'shot_id' column missing in {input_video_path}")

    # Cleanup after all videos are processed
    pose.close()
    if shot_detector.detect_hands and hand_detector:
        hand_detector.close()
    cv2.destroyAllWindows()
    logger.info("\nAll videos have been processed and datasets have been created.")

if __name__ == "__main__":
    main()
