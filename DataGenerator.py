import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import os
import torch
import sys
from collections import defaultdict, deque

########################################
# Thresholds and Constants (use your working values here)
########################################
VELOCITY_THRESHOLD = 200.0
CONSECUTIVE_FRAMES = 4
DISTANCE_THRESHOLD = 300.0
TIME_THRESHOLD = 2.0
YOLO_CONFIDENCE_THRESHOLD = 0.4

STABLE_FRAMES_REQUIRED = 5
STABLE_VELOCITY_THRESHOLD = 150
VERTICAL_DISPLACEMENT_THRESHOLD = 50

SMOOTHING_WINDOW = 3  # Number of frames to average for smoothing velocity/acc

########################################
# Function Definitions
########################################
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return round(angle, 2)

########################################
# Initialize MediaPipe Pose
########################################
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
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

# Define joints
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
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
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
input_videos = ['Videos/long_vid1.mov', 'Videos/long_vid2.mov', 'Videos/long_vid3.mov']
label = 'Good_Shots'
datasets_dir = "Datasets"
os.makedirs(datasets_dir, exist_ok=True)

########################################
# Helper function to reset state after shot completion
########################################
def reset_shot_state():
    global pre_shot_state, stable_frames, velocity_history, baseline_wrist_y, shot_invalid
    pre_shot_state = True
    stable_frames = 0
    velocity_history = []
    baseline_wrist_y = None
    shot_invalid = False

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
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Properties - FPS: {fps}, Width: {width}, Height: {height}")

    prev_positions = {}
    prev_velocities = {}
    all_data = []
    shots = []
    current_shot = None
    shot_detected = False
    shot_num = 0

    frame_count = 0
    frames_with_landmarks = 0

    # Initialize shot detection states
    pre_shot_state = True
    stable_frames = 0
    velocity_history = []
    baseline_wrist_y = None
    shot_invalid = False

    # For smoothing velocities and accelerations
    joint_vel_history = {joint_name: deque(maxlen=SMOOTHING_WINDOW) for joint_name in joint_map.keys()}
    joint_acc_history = {joint_name: deque(maxlen=SMOOTHING_WINDOW) for joint_name in joint_map.keys()}

    if idx < 5:
        feature_screen = np.zeros((600, 800, 3), dtype=np.uint8)
        cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Feature Visualization", cv2.WINDOW_NORMAL)
    else:
        feature_screen = None

    print("Starting video processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached.")
            break

        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb_frame)

        # YOLOv5 inference
        results = model(frame)
        filtered_detections = []
        for det in results.xyxy[0]:
            x1, y1, x2, y2, conf, cls_id = det
            cls_id = int(cls_id)
            conf = float(conf)
            if cls_id == SPORTS_BALL_CLASS_ID and conf >= YOLO_CONFIDENCE_THRESHOLD:
                filtered_detections.append(det)

        sports_ball_positions = []
        largest_ball = None
        largest_area = 0
        largest_ball_bbox = None

        # Find largest ball
        for det in filtered_detections:
            x1, y1, x2, y2, conf, cls_id = det.tolist()
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                x_center = (x1 + x2) / 2
                y_center = (y1 + y2) / 2
                if 0 <= x_center <= width and 0 <= y_center <= height:
                    largest_ball = (x_center, y_center)
                    largest_ball_bbox = (int(x1), int(y1), int(x2), int(y2))

        sports_ball_positions = [largest_ball] if largest_ball is not None else []
        valid_ball_detected = largest_ball is not None

        skeleton_center_x = None
        skeleton_center_y = None

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            left_hip = landmarks[joint_map["LEFT_HIP"]]
            right_hip = landmarks[joint_map["RIGHT_HIP"]]

            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                skeleton_center_x = ((left_hip.x + right_hip.x) / 2) * width
                skeleton_center_y = ((left_hip.y + right_hip.y) / 2) * height
                frames_with_landmarks += 1
            else:
                print(f"Frame {frame_count}: Hips not sufficiently visible.")
        else:
            print(f"Frame {frame_count}: No pose landmarks detected.")

        joint_data = {}

        if skeleton_center_x is not None and skeleton_center_y is not None and results_pose.pose_landmarks:
            for joint_name, joint_idx in joint_map.items():
                j_lm = landmarks[joint_idx]
                x = j_lm.x * width - skeleton_center_x
                y = j_lm.y * height - skeleton_center_y

                current_pos = (x, y)
                if joint_name in prev_positions and prev_positions[joint_name] is not None:
                    dx = x - prev_positions[joint_name][0]
                    dy = y - prev_positions[joint_name][1]
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

            # Shot Detection Logic with stable baseline & upward displacement
            if "LEFT_WRIST" in joint_data:
                wrist_vel = joint_data["LEFT_WRIST"]["vel"]  # Using smoothed velocity now
                wrist_pos_x, wrist_pos_y = joint_data["LEFT_WRIST"]["pos"]
                wrist_abs_y = wrist_pos_y + skeleton_center_y

                # Check stability: if wrist velocity is low, count stable frames
                if wrist_vel is not None and wrist_vel < STABLE_VELOCITY_THRESHOLD:
                    stable_frames += 1
                else:
                    stable_frames = 0

                # Once stable enough and we are in pre_shot_state, set baseline
                if stable_frames >= STABLE_FRAMES_REQUIRED and pre_shot_state:
                    baseline_wrist_y = wrist_abs_y
                    pre_shot_state = False
                    print(f"Baseline wrist Y set at frame {frame_count}: {baseline_wrist_y}")

                # Only attempt shot start detection if baseline is set
                if not pre_shot_state:
                    velocity_history.append(wrist_vel if wrist_vel is not None else 0)
                    if len(velocity_history) > CONSECUTIVE_FRAMES:
                        velocity_history.pop(0)

                    if current_shot is None:
                        # Check if all velocities in last frames exceed threshold
                        if (len(velocity_history) == CONSECUTIVE_FRAMES and
                            all(v > VELOCITY_THRESHOLD for v in velocity_history)):

                            # Check upward displacement from baseline
                            if baseline_wrist_y is not None and (baseline_wrist_y - wrist_abs_y) > VERTICAL_DISPLACEMENT_THRESHOLD:
                                # Confirm shot start
                                shot_num += 1
                                start_frame = frame_count
                                current_shot = {
                                    'start_frame': start_frame,
                                    'end_frame': None,
                                    'start_time': start_frame / fps,
                                    'shot_id': shot_num
                                }
                                shots.append(current_shot)
                                print(f"Shot {shot_num} started at frame {start_frame} (time {current_shot['start_time']:.2f}s)")
                            else:
                                # High velocity but no significant upward movement
                                velocity_history = []
                                print("High velocity detected but no upward displacement. Not counting as shot start.")
                    else:
                        # Detect Shot End
                        if valid_ball_detected:
                            ball_x, ball_y = sports_ball_positions[0]
                            wrist_abs_x = wrist_pos_x + skeleton_center_x
                            distance = math.sqrt((ball_x - wrist_abs_x)**2 + (ball_y - wrist_abs_y)**2)
                            print(f"Frame {frame_count}: Distance between ball and wrist is {distance:.2f} pixels.")

                            current_time = frame_count / fps
                            # If ball is far enough away within TIME_THRESHOLD, end shot
                            if not np.isnan(distance) and distance > DISTANCE_THRESHOLD:
                                end_frame = frame_count
                                end_time = end_frame / fps
                                shot_duration = end_time - current_shot['start_time']
                                if shot_duration <= TIME_THRESHOLD:
                                    current_shot['end_frame'] = end_frame
                                    current_shot['end_time'] = end_time
                                    current_shot['duration'] = shot_duration
                                    print(f"Shot {current_shot['shot_id']} ended at frame {end_frame} (time {end_time:.2f}s) with duration {shot_duration:.2f}s")
                                    shot_detected = True
                                    current_shot = None
                                    # Reset states for next shot
                                    reset_shot_state()
                                else:
                                    print(f"Shot {current_shot['shot_id']} duration {shot_duration:.2f}s exceeded TIME_THRESHOLD. Invalidated.")
                                    shots.pop()  # Remove this shot
                                    shot_invalid = True
                                    shot_detected = True
                                    current_shot = None
                                    # Reset states
                                    reset_shot_state()
                            else:
                                # If time threshold exceeded without ball moving away
                                if (current_time - current_shot['start_time']) > TIME_THRESHOLD:
                                    print(f"Shot {current_shot['shot_id']} duration exceeded TIME_THRESHOLD without detecting end. Invalidated.")
                                    shots.pop()  # Remove this shot
                                    shot_invalid = True
                                    shot_detected = True
                                    current_shot = None
                                    # Reset states
                                    reset_shot_state()
            else:
                # No LEFT_WRIST found, reset states
                stable_frames = 0
                pre_shot_state = True
                velocity_history = []
                baseline_wrist_y = None
                shot_invalid = False

            # Add velocities, accelerations, angles to features_row
            for left_joint, right_joint in PAIRED_JOINTS:
                left_vel = joint_data[left_joint]["vel"] if joint_data[left_joint]["vel"] is not None else np.nan
                right_vel = joint_data[right_joint]["vel"] if joint_data[right_joint]["vel"] is not None else np.nan
                features_row[f"{left_joint}_vel"] = left_vel
                features_row[f"{right_joint}_vel"] = right_vel

                left_acc = joint_data[left_joint]["acc"] if joint_data[left_joint]["acc"] is not None else np.nan
                right_acc = joint_data[right_joint]["acc"] if joint_data[right_joint]["acc"] is not None else np.nan
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

            # Annotate shot info
            features_row['is_shot'] = 0
            features_row['shot_id'] = np.nan

            if current_shot and current_shot['end_frame'] is None:
                features_row['is_shot'] = 1
                features_row['shot_id'] = current_shot['shot_id']

            all_data.append(features_row)
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Visualization changes:
            if idx < 5 and feature_screen is not None:
                feature_screen[:] = (0, 0, 0)  # Clear the screen

                cv2.putText(feature_screen, "Wrist Velocity & State:", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                y_offset = 60

                # LEFT_WRIST velocity display
                if "LEFT_WRIST" in joint_data:
                    left_wrist_vel = joint_data["LEFT_WRIST"]["vel"]
                    lw_vel_text = f"LEFT_WRIST Vel={left_wrist_vel if left_wrist_vel is not None else 'N/A'}"
                else:
                    lw_vel_text = "LEFT_WRIST Vel=N/A"

                cv2.putText(feature_screen, lw_vel_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 40

                # Determine condition/state text
                if pre_shot_state:
                    condition_text = "Stability Check in Progress"
                elif not pre_shot_state and current_shot is None and not shot_detected:
                    condition_text = "Checking Upward Displacement & Velocity"
                elif current_shot and current_shot['end_frame'] is None:
                    condition_text = "Shot in Progress"
                elif shot_detected:
                    # If shot detected, either completed or invalid
                    if shot_invalid:
                        condition_text = "Shot Invalid"
                    else:
                        last_shot = shots[-1] if len(shots) > 0 else None
                        if last_shot and 'end_frame' in last_shot and last_shot['end_frame'] is not None:
                            condition_text = "Shot Completed"
                        else:
                            condition_text = "Shot Invalid"
                else:
                    condition_text = "No Condition"

                # Display condition text
                color = (255, 255, 0) if "Check" in condition_text else (0, 255, 255)
                if "Invalid" in condition_text:
                    color = (0, 0, 255)
                elif "Completed" in condition_text:
                    color = (255, 255, 0)
                elif "Progress" in condition_text:
                    color = (255, 255, 255)

                cv2.putText(feature_screen, condition_text, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_offset += 40

                # If shot invalid, explicitly show it
                if shot_invalid:
                    cv2.putText(feature_screen, "Shot Invalid", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    y_offset += 40

                # Display current frame on feature visualization
                cv2.putText(feature_screen, f"Frame: {frame_count}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 40

                resized_frame = cv2.resize(frame, (width // 3, height // 3))
                cv2.imshow("Pose Detection", resized_frame)
                cv2.imshow("Feature Visualization", feature_screen)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, {frames_with_landmarks} with landmarks detected.")

        if idx < 5 and feature_screen is not None:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting processing loop as 'q' was pressed.")
                break

    cap.release()
    if idx < 5 and feature_screen is not None:
        cv2.destroyWindow("Pose Detection")
        cv2.destroyWindow("Feature Visualization")

    print(f"Finished processing video: {input_video_path}")

    # Data Engineering
    df = pd.DataFrame(all_data)
    df.replace("N/A", np.nan, inplace=True)
    df.fillna(0, inplace=True)

    if 'shot_id' in df.columns:
        unique_shots = df['shot_id'].dropna().unique()
        unique_shots = [int(s) for s in unique_shots if not pd.isna(s)]

        if len(unique_shots) > 0:
            for s_id in unique_shots:
                # Find the corresponding shot dictionary
                shot_data = [s for s in shots if s.get('shot_id') == s_id]
                if len(shot_data) == 1:
                    shot = shot_data[0]
                    # Only save if shot completed (has valid end_frame)
                    if 'end_frame' in shot and shot['end_frame'] is not None:
                        # Filter df for this shot_id (frames from shot start to end)
                        shot_df = df[df['shot_id'] == s_id].copy()
                        if not shot_df.empty:
                            columns_to_drop = [
                                'LEFT_HIP_pos_x', 'LEFT_HIP_pos_y',
                                'RIGHT_HIP_pos_x', 'RIGHT_HIP_pos_y', 'LEFT_HIP_vel',
                                'LEFT_HIP_acc', 'RIGHT_HIP_vel', 'RIGHT_HIP_acc',
                                'is_shot', 'shot_id'
                            ]
                            shot_df = shot_df.drop(columns=columns_to_drop, errors='ignore')
                            shot_df = shot_df.reset_index(drop=True)

                            output_csv_filename = os.path.splitext(os.path.basename(input_video_path))[0] + f"_shot_{s_id}.csv"
                            output_csv_path = os.path.join(datasets_dir, output_csv_filename)

                            shot_df.to_csv(output_csv_path, index=False)
                            print(f"Feature data for shot {s_id} successfully saved to {output_csv_path}")
                    else:
                        print(f"Shot {s_id} not completed. No dataset saved.")
                else:
                    print(f"No valid shot data found for shot_id {s_id}. No dataset saved.")
        else:
            print(f"No valid shots found in {input_video_path}")
    else:
        print(f"No shots detected or 'shot_id' column missing in {input_video_path}")

pose.close()
cv2.destroyAllWindows()
print("\nAll videos have been processed and datasets have been created.")
