import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import os
import torch
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from enum import Enum, auto

########################################
# Thresholds and Constants
########################################
VELOCITY_THRESHOLD = 150.0  # Velocity required to detect a shot
CONSECUTIVE_FRAMES = 4      # Number of consecutive frames with high velocity
DISTANCE_THRESHOLD = 150.0   # Distance the ball must move to validate a shot
TIME_THRESHOLD = 2.0         # Time within which the ball must move after shot initiation
YOLO_CONFIDENCE_THRESHOLD = 0.4  # YOLO detection confidence threshold

STABLE_FRAMES_REQUIRED = 5       # Number of consecutive stable frames required
STABLE_VELOCITY_THRESHOLD = 80.0 # Maximum velocity during stability check

VERTICAL_DISPLACEMENT_THRESHOLD = 50.0  # Minimum upward displacement for shot detection
BALL_WRIST_DISTANCE_THRESHOLD = 100.0    # Maximum distance between ball and wrist during stability check

SMOOTHING_WINDOW = 3  # Frames to average velocity and acceleration

# Target dimensions for resizing (for pose estimation)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# Cooldown period after shot detection to prevent immediate re-detection
SHOT_COOLDOWN_FRAMES = 30  # e.g., 0.5 seconds at 60 FPS

########################################
# Utility Functions
########################################
def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by points 'a', 'b', and 'c'.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return round(angle, 2)

def process_yolo(frame, model):
    """
    Runs YOLO inference on the given frame.
    """
    yolo_results = model(frame)
    return yolo_results

def extract_roi(frame, landmarks, width, height, padding=50):
    """
    Define a Region of Interest (ROI) based on the MediaPipe landmarks.
    This focuses YOLO detection within this area to improve performance.
    """
    # Example: Define ROI around the left wrist
    if "LEFT_WRIST" in joint_map:
        wrist = landmarks[joint_map["LEFT_WRIST"]]
        if wrist.visibility > 0.5:
            x = int(wrist.x * width)
            y = int(wrist.y * height)
            x1 = max(x - padding, 0)
            y1 = max(y - padding, 0)
            x2 = min(x + padding, width)
            y2 = min(y + padding, height)
            return frame[y1:y2, x1:x2], (x1, y1)
    # Default ROI if specific landmark not found
    return frame, (0, 0)

def map_to_original(x_padded, y_padded, original_width, original_height, new_width, new_height, left, top):
    """
    Maps coordinates from the resized and padded frame back to the original frame.
    
    Parameters:
    - x_padded, y_padded: Coordinates in the padded frame.
    - original_width, original_height: Dimensions of the original frame.
    - new_width, new_height: Dimensions of the resized frame (before padding).
    - left, top: Padding added to the left and top.
    
    Returns:
    - x_original, y_original: Mapped coordinates in the original frame.
    """
    # Remove padding
    x_resized = x_padded - left
    y_resized = y_padded - top
    
    # Avoid negative values
    x_resized = max(x_resized, 0)
    y_resized = max(y_resized, 0)
    
    # Calculate scaling factors
    scale_x = original_width / new_width
    scale_y = original_height / new_height
    
    # Map to original frame
    x_original = x_resized * scale_x
    y_original = y_resized * scale_y
    
    return x_original, y_original

########################################
# Shot Detection State Machine
########################################
class ShotState(Enum):
    WAITING_FOR_STABILITY = auto()
    READY_TO_DETECT_SHOT = auto()
    SHOT_IN_PROGRESS = auto()
    COOLDOWN = auto()

def reset_shot_state():
    """
    Resets the shot detection state to initial.
    """
    return {
        'state': ShotState.WAITING_FOR_STABILITY,
        'stable_frames': 0,
        'velocity_history': deque(maxlen=CONSECUTIVE_FRAMES),
        'baseline_wrist_y': None,
        'cooldown_counter': 0
    }

########################################
# Initialize MediaPipe Pose
########################################
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

USEFUL_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP"
]

joint_map = {j.name: j.value for j in mp_pose.PoseLandmark if j.name in USEFUL_LANDMARKS}

PAIRED_JOINTS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
    ("LEFT_HIP", "RIGHT_HIP"),
]

ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
    ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
]

########################################
# Initialize YOLOv5 Model
########################################
try:
    # Load the YOLOv5 model from the GitHub repository with custom weights
    model = torch.hub.load(
        'ultralytics/yolov5',                # Repository name
        'custom',                            # Model variant
        path='Weights/best.pt',              # Path to custom weights
        source='github'                      # Load from GitHub repository
    )
    model.eval()  # Set model to evaluation mode
    print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
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
    print(f"Error: model.names is of unexpected type: {type(model.names)}")
    sys.exit()

if SPORTS_BALL_CLASS_ID is None:
    print(f"Error: Class '{SPORTS_BALL_CLASS_NAME}' not found in model classes.")
    sys.exit()

print(f"Ball Class ID: {SPORTS_BALL_CLASS_ID}")

########################################
# Input Videos and Labels
########################################
input_videos = ['Videos/long_vid.mov']  # Ensure the correct video path
label = 'Good_Shots'
datasets_dir = "Datasets"
os.makedirs(datasets_dir, exist_ok=True)

########################################
# Process Each Video
########################################
for idx, input_video_path in enumerate(input_videos):
    print(f"\nProcessing video {idx+1}/{len(input_videos)}: {input_video_path}")

    if not os.path.isfile(input_video_path):
        print(f"Video file does not exist: {input_video_path}")
        continue

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {input_video_path}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 60.0  # Default FPS if unable to get from video

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original Video Properties - FPS: {fps}, Width: {original_width}, Height: {original_height}")
    print(f"Resizing all frames to {TARGET_WIDTH}x{TARGET_HEIGHT} for pose estimation.")

    prev_positions = {}
    prev_velocities = {}
    all_data = []
    shots = []
    current_shot = None

    frame_count = 0
    frames_with_landmarks = 0

    # Initialize shot detection state
    shot_state = reset_shot_state()

    # Initialize shot number counter
    shot_num = 0  # Define shot_num before the frame processing loop
    print(f"Initialized shot_num to {shot_num}")

    # For smoothing velocities and accelerations
    joint_vel_history = {joint_name: deque(maxlen=SMOOTHING_WINDOW) for joint_name in joint_map.keys()}
    joint_acc_history = {joint_name: deque(maxlen=SMOOTHING_WINDOW) for joint_name in joint_map.keys()}

    # Initialize visualization windows for the first few videos
    if idx < 5:
        feature_screen = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Feature Visualization", cv2.WINDOW_NORMAL)
    else:
        feature_screen = None

    print("Starting video processing...")

    executor = ThreadPoolExecutor(max_workers=1)  # YOLO in a separate thread

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached.")
            break

        frame_count += 1

        # Keep a copy of the original frame for YOLO
        original_frame = frame.copy()

        # Resize frame to target resolution while preserving aspect ratio
        # Calculate aspect ratio
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
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]  # Black padding
        frame_padded = cv2.copyMakeBorder(frame_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        rgb_frame = cv2.cvtColor(frame_padded, cv2.COLOR_BGR2RGB)

        # Run pose processing synchronously on resized and padded frame
        results_pose = pose.process(rgb_frame)

        # Run YOLO on the original frame in a separate thread
        future_yolo = executor.submit(process_yolo, original_frame, model)
        try:
            results = future_yolo.result(timeout=10)  # Add timeout to prevent hanging
        except Exception as e:
            print(f"YOLO inference failed at frame {frame_count}: {e}")
            results = None

        # YOLOv5 inference results filtering
        filtered_detections = []
        if results is not None and hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            for det in results.xyxy[0]:
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

        sports_ball_positions = [largest_ball] if largest_ball is not None else []
        valid_ball_detected = largest_ball is not None

        skeleton_center_x = None
        skeleton_center_y = None

        # Mapping parameters for coordinate transformation
        # Calculate scaling factors based on resizing
        scale_x = original_width / new_width
        scale_y = original_height / new_height

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            left_hip = landmarks[joint_map["LEFT_HIP"]]
            right_hip = landmarks[joint_map["RIGHT_HIP"]]

            # Since landmarks are normalized [0,1], scale them by TARGET_WIDTH and TARGET_HEIGHT
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                # Calculate the center of the hips in the padded frame
                skeleton_center_padded_x = ((left_hip.x + right_hip.x) / 2) * TARGET_WIDTH
                skeleton_center_padded_y = ((left_hip.y + right_hip.y) / 2) * TARGET_HEIGHT

                # Map the skeleton center back to the original frame
                skeleton_center_original_x, skeleton_center_original_y = map_to_original(
                    skeleton_center_padded_x, skeleton_center_padded_y,
                    original_width, original_height,
                    new_width, new_height,
                    left, top
                )

                skeleton_center_x = skeleton_center_original_x
                skeleton_center_y = skeleton_center_original_y
                frames_with_landmarks += 1
            else:
                print(f"Frame {frame_count}: Hips not sufficiently visible.")
        else:
            print(f"Frame {frame_count}: No pose landmarks detected.")

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
                    left, top
                )

                # Calculate velocity
                current_pos = (x_original, y_original)
                if joint_name in prev_positions and prev_positions[joint_name] is not None:
                    dx = x_original - prev_positions[joint_name][0]
                    dy = y_original - prev_positions[joint_name][1]
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
                    "pos": current_pos,
                    "vel": round(smoothed_velocity, 2),
                    "acc": round(smoothed_acc, 2)
                }

                prev_positions[joint_name] = current_pos
                prev_velocities[joint_name] = raw_velocity

            features_row = {
                'video': os.path.basename(input_video_path),
                'frame': frame_count,
                'label': label
            }

            if sports_ball_positions:
                positions_str = ';'.join([f"{round(pos[0],2)},{round(pos[1],2)}" for pos in sports_ball_positions])
                features_row['sports_ball_positions'] = positions_str
            else:
                features_row['sports_ball_positions'] = np.nan

            # Shot Detection Logic
            state = shot_state['state']

            if "LEFT_WRIST" in joint_data:
                wrist_vel = joint_data["LEFT_WRIST"]["vel"]
                wrist_pos_x, wrist_pos_y = joint_data["LEFT_WRIST"]["pos"]
                wrist_abs_y = wrist_pos_y
                wrist_abs_x = wrist_pos_x

                # Calculate distance between wrist and ball
                if sports_ball_positions:
                    ball_x, ball_y = sports_ball_positions[0]
                    distance = math.sqrt((wrist_pos_x - ball_x) ** 2 + (wrist_pos_y - ball_y) ** 2)
                    if distance <= BALL_WRIST_DISTANCE_THRESHOLD:
                        is_ball_close = True
                        print(f"Frame {frame_count}: Ball is within the valid distance ({BALL_WRIST_DISTANCE_THRESHOLD} units).")
                    else:
                        is_ball_close = False
                        print(f"Frame {frame_count}: Ball is too far from the wrist (Threshold: {BALL_WRIST_DISTANCE_THRESHOLD} units).")
                else:
                    is_ball_close = False
                    print(f"Frame {frame_count}: Ball not detected. Ball-Wrist proximity condition not met.")

                # Debugging: Print current state and key variables
                print(f"Frame {frame_count}: State={state}, Stable Frames={shot_state['stable_frames']}, Wrist Vel={wrist_vel}, Wrist Abs Y={wrist_abs_y}")

                if state == ShotState.WAITING_FOR_STABILITY:
                    # Check stability **and** ball proximity
                    if wrist_vel < STABLE_VELOCITY_THRESHOLD and is_ball_close:
                        shot_state['stable_frames'] += 1
                        print(f"Frame {frame_count}: Stable frames increased to {shot_state['stable_frames']} (wrist_vel={wrist_vel}, is_ball_close={is_ball_close})")
                    else:
                        if shot_state['stable_frames'] != 0:
                            print(f"Frame {frame_count}: Stability or ball proximity lost. Resetting stable_frames.")
                        shot_state['stable_frames'] = 0

                    # Transition to READY_TO_DETECT_SHOT if stable and ball is close
                    if shot_state['stable_frames'] >= STABLE_FRAMES_REQUIRED:
                        shot_state['baseline_wrist_y'] = wrist_abs_y
                        shot_state['state'] = ShotState.READY_TO_DETECT_SHOT
                        shot_state['velocity_history'].clear()
                        print(f"Frame {frame_count}: Baseline wrist Y set at {shot_state['baseline_wrist_y']}")
                        print(f"Frame {frame_count}: Transitioned to READY_TO_DETECT_SHOT")

                elif state == ShotState.READY_TO_DETECT_SHOT:
                    # Append to velocity_history
                    if wrist_vel is not None:
                        shot_state['velocity_history'].append(wrist_vel)
                        print(f"Frame {frame_count}: Appended wrist_vel={wrist_vel} to velocity_history")
                    else:
                        shot_state['velocity_history'].append(0)
                        print(f"Frame {frame_count}: Appended wrist_vel=0 to velocity_history")

                    # Check if enough frames for velocity history
                    if len(shot_state['velocity_history']) == CONSECUTIVE_FRAMES:
                        if all(v > VELOCITY_THRESHOLD for v in shot_state['velocity_history']):
                            # Check upward displacement
                            displacement = shot_state['baseline_wrist_y'] - wrist_abs_y
                            print(f"Frame {frame_count}: Displacement={displacement}")
                            if displacement > VERTICAL_DISPLACEMENT_THRESHOLD:
                                # Prepare to start a shot
                                current_shot = {
                                    'start_frame': frame_count,
                                    'end_frame': None,
                                    'start_time': frame_count / fps,
                                    'shot_id': None,  # To be assigned upon validation
                                    'invalid': False
                                }
                                shot_state['state'] = ShotState.SHOT_IN_PROGRESS
                                shot_state['velocity_history'].clear()
                                print(f"Frame {frame_count}: Shot detected. Transitioned to SHOT_IN_PROGRESS.")
                            else:
                                # High velocity but no upward movement
                                print("High velocity detected but no upward displacement. Not counting as shot start.")
                            shot_state['velocity_history'].clear()
                        else:
                            print("Velocity history does not exceed VELOCITY_THRESHOLD. Continuing READY_TO_DETECT_SHOT.")

                elif state == ShotState.SHOT_IN_PROGRESS:
                    # Detect shot end
                    if valid_ball_detected:
                        ball_x, ball_y = sports_ball_positions[0]
                        distance = math.sqrt((ball_x - wrist_abs_x)**2 + (ball_y - wrist_abs_y)**2)
                        current_time = frame_count / fps
                        shot_duration = current_time - current_shot['start_time']

                        print(f"Frame {frame_count}: Distance to ball={distance:.2f}, Duration={shot_duration:.2f}s")

                        # If ball far enough away within TIME_THRESHOLD and duration is acceptable
                        if distance > DISTANCE_THRESHOLD and shot_duration <= TIME_THRESHOLD:
                            # Valid shot
                            shot_state['state'] = ShotState.COOLDOWN
                            shot_state['cooldown_counter'] = SHOT_COOLDOWN_FRAMES
                            shot_num += 1  # Increment only for valid shots
                            current_shot['shot_id'] = shot_num
                            current_shot['end_frame'] = frame_count
                            current_shot['end_time'] = current_time
                            current_shot['duration'] = shot_duration
                            shots.append(current_shot)
                            print(f"Shot {current_shot['shot_id']} ended at frame {frame_count} (time {current_shot['end_time']:.2f}s) with duration {shot_duration:.2f}s")
                            current_shot = None
                        elif shot_duration > TIME_THRESHOLD:
                            # Shot duration exceeded without valid end
                            current_shot['invalid'] = True
                            shots.append(current_shot)
                            print(f"Shot at frame {current_shot['start_frame']} invalidated due to duration {shot_duration:.2f}s exceeding TIME_THRESHOLD.")
                            shot_state['state'] = ShotState.COOLDOWN
                            shot_state['cooldown_counter'] = SHOT_COOLDOWN_FRAMES
                            current_shot = None
                        else:
                            print("Shot end conditions not met. Continuing SHOT_IN_PROGRESS.")
                    else:
                        # If no ball detected, still check for TIME_THRESHOLD
                        current_time = frame_count / fps
                        shot_duration = current_time - current_shot['start_time']
                        if shot_duration > TIME_THRESHOLD:
                            current_shot['invalid'] = True
                            shots.append(current_shot)
                            print(f"Shot at frame {current_shot['start_frame']} invalidated due to exceeding TIME_THRESHOLD without detecting end.")
                            shot_state['state'] = ShotState.COOLDOWN
                            shot_state['cooldown_counter'] = SHOT_COOLDOWN_FRAMES
                            current_shot = None
                        else:
                            print("No ball detected and TIME_THRESHOLD not yet exceeded. Continuing SHOT_IN_PROGRESS.")

                elif state == ShotState.COOLDOWN:
                    # Decrement cooldown counter
                    shot_state['cooldown_counter'] -= 1
                    print(f"Frame {frame_count}: Cooldown counter at {shot_state['cooldown_counter']}")
                    if shot_state['cooldown_counter'] <= 0:
                        shot_state = reset_shot_state()
                        print(f"Frame {frame_count}: Cooldown complete. Transitioned to WAITING_FOR_STABILITY.")

            else:
                # No LEFT_WRIST found, possibly reset shot_state or handle accordingly
                if shot_state['state'] != ShotState.COOLDOWN:
                    print(f"Frame {frame_count}: LEFT_WRIST not detected. Resetting shot state.")
                    shot_state = reset_shot_state()

            # Add velocities, accelerations, angles to features_row
            for left_joint, right_joint in PAIRED_JOINTS:
                left_vel = joint_data[left_joint]["vel"] if left_joint in joint_data else np.nan
                right_vel = joint_data[right_joint]["vel"] if right_joint in joint_data else np.nan
                left_acc = joint_data[left_joint]["acc"] if left_joint in joint_data else np.nan
                right_acc = joint_data[right_joint]["acc"] if right_joint in joint_data else np.nan

                features_row[f"{left_joint}_vel"] = left_vel
                features_row[f"{right_joint}_vel"] = right_vel
                features_row[f"{left_joint}_acc"] = left_acc
                features_row[f"{right_joint}_acc"] = right_acc

            for joint_name in joint_map.keys():
                pos_x, pos_y = joint_data[joint_name]["pos"]
                features_row[f"{joint_name}_pos_x"] = round(pos_x, 2)
                features_row[f"{joint_name}_pos_y"] = round(pos_y, 2)

            for triplet in ANGLE_JOINTS:
                joint_a, joint_b, joint_c = triplet
                if (joint_a in joint_data and joint_b in joint_data and joint_c in joint_data and
                    joint_data[joint_a]["pos"] != (0, 0) and
                    joint_data[joint_b]["pos"] != (0, 0) and
                    joint_data[joint_c]["pos"] != (0, 0)):
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

            if current_shot:
                if current_shot['end_frame'] is None:
                    features_row['is_shot'] = 1
                    features_row['shot_id'] = current_shot['shot_id'] if current_shot['shot_id'] else np.nan
                    features_row['shot_invalid'] = 1 if current_shot['invalid'] else 0
            elif shots:
                last_shot = shots[-1]
                if last_shot['end_frame'] is not None:
                    if not last_shot.get('invalid', False):
                        shot_info = f"Shot {last_shot['shot_id']} Completed (Duration: {last_shot['duration']:.2f}s)"
                        cv2.putText(feature_screen, shot_info, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        shot_info = f"Shot at Frame {last_shot['start_frame']} Invalid"
                        cv2.putText(feature_screen, shot_info, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red for invalid
                    y_offset += 40
                else:
                    # Shot is in progress
                    if last_shot.get('invalid', False):
                        shot_info = f"Shot at Frame {last_shot['start_frame']} Invalid"
                        cv2.putText(feature_screen, shot_info, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red for invalid
                        y_offset += 40
                    else:
                        shot_info = f"Shot {last_shot['shot_id']} In Progress"
                        cv2.putText(feature_screen, shot_info, (10, y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        y_offset += 40

            all_data.append(features_row)
            # Draw landmarks on the resized and padded frame
            mp_drawing.draw_landmarks(frame_padded, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Visualization
            if idx < 5 and feature_screen is not None:
                feature_screen[:] = (0, 0, 0)  # Clear the screen

                cv2.putText(feature_screen, "Wrist Velocity & State:", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset = 60

                if "LEFT_WRIST" in joint_data:
                    left_wrist_vel = joint_data["LEFT_WRIST"]["vel"]
                    lw_vel_text = f"LEFT_WRIST Vel={left_wrist_vel if left_wrist_vel is not None else 'N/A'}"
                else:
                    lw_vel_text = "LEFT_WRIST Vel=N/A"

                cv2.putText(feature_screen, lw_vel_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 40

                # Display distance information
                if 'distance' in locals():
                    distance_text = f"Ball-Wrist Dist: {distance:.2f}"
                    color_dist = (0, 255, 0) if is_ball_close else (0, 0, 255)
                    cv2.putText(feature_screen, distance_text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_dist, 2)
                    y_offset += 40
                else:
                    cv2.putText(feature_screen, "Ball-Wrist Dist: N/A", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    y_offset += 40

                # Determine condition/state text
                condition_text = ""
                color_state = (255, 255, 255)  # Default white

                if shot_state['state'] == ShotState.WAITING_FOR_STABILITY:
                    condition_text = "Waiting for Stability"
                    color_state = (255, 255, 255)  # White
                elif shot_state['state'] == ShotState.READY_TO_DETECT_SHOT:
                    condition_text = "Ready to Detect Shot"
                    color_state = (0, 255, 255)  # Cyan
                elif shot_state['state'] == ShotState.SHOT_IN_PROGRESS:
                    condition_text = "Shot In Progress"
                    color_state = (0, 255, 255)  # Cyan
                elif shot_state['state'] == ShotState.COOLDOWN:
                    condition_text = "Cooldown Period"
                    color_state = (255, 0, 255)  # Magenta

                cv2.putText(feature_screen, condition_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_state, 2)
                y_offset += 40

                # Display shot information if any
                if shots:
                    last_shot = shots[-1]
                    if last_shot['end_frame'] is not None:
                        if not last_shot.get('invalid', False):
                            shot_info = f"Shot {last_shot['shot_id']} Completed (Duration: {last_shot['duration']:.2f}s)"
                            cv2.putText(feature_screen, shot_info, (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        else:
                            shot_info = f"Shot at Frame {last_shot['start_frame']} Invalid"
                            cv2.putText(feature_screen, shot_info, (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red for invalid
                        y_offset += 40
                    else:
                        # Shot is in progress
                        if last_shot.get('invalid', False):
                            shot_info = f"Shot at Frame {last_shot['start_frame']} Invalid"
                            cv2.putText(feature_screen, shot_info, (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red for invalid
                            y_offset += 40
                        else:
                            shot_info = f"Shot {last_shot['shot_id']} In Progress"
                            cv2.putText(feature_screen, shot_info, (10, y_offset),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                            y_offset += 40

                cv2.putText(feature_screen, f"Frame: {frame_count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 40

                # Overlay wrist velocity on feature_screen
                if "LEFT_WRIST" in joint_data:
                    cv2.putText(feature_screen, f"Wrist Vel: {wrist_vel}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                else:
                    cv2.putText(feature_screen, "Wrist Vel: N/A", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                y_offset += 40

                # Resize frames for display
                resized_pose_display = cv2.resize(frame_padded, (TARGET_WIDTH // 2, TARGET_HEIGHT // 2))
                resized_original_display = cv2.resize(original_frame, (original_width // 2, original_height // 2))

                cv2.imshow("Pose Detection", resized_pose_display)
                cv2.imshow("YOLO Detection", resized_original_display)
                cv2.imshow("Feature Visualization", feature_screen)

        # Log progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, {frames_with_landmarks} with landmarks detected.")

        # Allow exit on 'q' key press for the first 5 videos
        if idx < 5 and feature_screen is not None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting processing loop as 'q' was pressed.")
                break

    # Release video capture and destroy windows
    cap.release()
    if idx < 5 and feature_screen is not None:
        cv2.destroyWindow("Pose Detection")
        cv2.destroyWindow("YOLO Detection")
        cv2.destroyWindow("Feature Visualization")

    print(f"Finished processing video: {input_video_path}")

    # Data Engineering: Save the entire dataset for the current video
    df = pd.DataFrame(all_data)
    df.replace("N/A", np.nan, inplace=True)
    df.fillna(0, inplace=True)

    output_csv_filename = os.path.splitext(os.path.basename(input_video_path))[0] + "_entire_dataset.csv"
    output_csv_path = os.path.join(datasets_dir, output_csv_filename)
    df.to_csv(output_csv_path, index=False)
    print(f"Complete dataset successfully saved to {output_csv_path}")

    # Save individual shots if needed
    if 'shot_id' in df.columns:
        unique_shots = df['shot_id'].dropna().unique()
        unique_shots = [int(s) for s in unique_shots if not pd.isna(s)]

        if len(unique_shots) > 0:
            for s_id in unique_shots:
                # Filter shots list for valid shots with the current shot_id
                shot_data = [s for s in shots if s.get('shot_id') == s_id and not s.get('invalid', False)]
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
                                'is_shot', 'shot_id', 'shot_invalid'
                            ]
                            shot_df = shot_df.drop(columns=columns_to_drop, errors='ignore')
                            shot_df = shot_df.reset_index(drop=True)

                            output_csv_filename = os.path.splitext(os.path.basename(input_video_path))[0] + f"_shot_{s_id}.csv"
                            output_csv_path = os.path.join(datasets_dir, output_csv_filename)

                            shot_df.to_csv(output_csv_path, index=False)
                            print(f"Feature data for shot {s_id} successfully saved to {output_csv_path}")
                else:
                    print(f"No valid shot data found for shot_id {s_id}. No dataset saved.")
        else:
            print(f"No valid shots found in {input_video_path}")
    else:
        print(f"No shots detected or 'shot_id' column missing in {input_video_path}")

pose.close()
cv2.destroyAllWindows()
print("\nAll videos have been processed and datasets have been created.")
