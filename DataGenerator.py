import cv2
import mediapipe as mp
import math
import numpy as np
import pandas as pd
import os
from ultralytics import YOLO  # Import YOLOv5 via ultralytics

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    """
    Calculates the angle at point b given three points a, b, and c.

    Parameters:
    - a, b, c: Tuples representing the (x, y) coordinates of the points.

    Returns:
    - Angle in degrees.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    # Vectors BA and BC
    ba = a - b
    bc = c - b

    # Compute the cosine of the angle using dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)  # Added epsilon to avoid division by zero

    # Clip the cosine to the valid range [-1, 1] to avoid numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(cosine_angle)
    angle = np.degrees(angle)

    return round(angle, 2)

# Initialize MediaPipe Pose with optimized parameters
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,           # Highest complexity for better accuracy
    smooth_landmarks=True,
    min_detection_confidence=0.7, # Higher detection confidence
    min_tracking_confidence=0.7   # Higher tracking confidence
)
mp_drawing = mp.solutions.drawing_utils

# Define the joints to include (paired joints)
PAIRED_JOINTS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
    ("LEFT_HIP", "RIGHT_HIP"),
]

# Define joint triplets for angle calculation (e.g., Shoulder-Elbow-Wrist)
ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
    ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
]

# **2. Filter Pose Landmarks: Define Useful Landmarks**
# Exclude facial landmarks and those below the waist
USEFUL_LANDMARKS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP"
]

# Update joint_map to only include useful landmarks
joint_map = {j.name: j.value for j in mp_pose.PoseLandmark if j.name in USEFUL_LANDMARKS}

# **1. Integrate YOLOv5 for Sports Ball Detection**

# Load the YOLOv5 model trained to detect sports balls
# Replace 'yolov5su.pt' with the actual path if it's located elsewhere
yolo_model_path = 'yolov5su.pt'
if not os.path.isfile(yolo_model_path):
    print(f"YOLOv5 model file does not exist: {yolo_model_path}")
    exit()

yolo_model = YOLO(yolo_model_path)

# **Determine Class ID for Sports Ball**
# Retrieve class names from the YOLOv5 model
model_classes = yolo_model.model.names  # Dictionary: {class_id: class_name}
print("YOLOv5 Model Classes:")
for class_id, class_name in model_classes.items():
    print(f"Class ID {class_id}: {class_name}")

# Define the class name for sports ball based on the model's classes
# Update this if the class name differs (e.g., "sports_ball" vs. "sports ball")
SPORTS_BALL_CLASS_NAME = "sports ball"  # Must match exactly with the model's class name
SPORTS_BALL_CLASS_ID = None

# Find the class ID for sports ball
for cls_id, cls_name in model_classes.items():
    if cls_name.lower() == SPORTS_BALL_CLASS_NAME.lower():
        SPORTS_BALL_CLASS_ID = cls_id
        break

if SPORTS_BALL_CLASS_ID is None:
    print(f"Sports ball class not found in YOLOv5 model classes. Available classes: {model_classes}")
    exit()

print(f"Sports Ball Class ID: {SPORTS_BALL_CLASS_ID}")

# Input video files and labels
input_videos = ['IMG_8184.MOV', 'IMG_8185.MOV', 'IMG_8185(1).MOV', 'IMG_8185(2).MOV']
label = 'Good_Shots'  # Assuming all videos have the same label; modify as needed

# Create "Datasets" directory if it doesn't exist
datasets_dir = "Datasets"
os.makedirs(datasets_dir, exist_ok=True)

# Loop through each video in the input_videos list
for idx, input_video_path in enumerate(input_videos):
    print(f"\nProcessing video {idx + 1}/{len(input_videos)}: {input_video_path}")

    # Check if the input video file exists
    if not os.path.isfile(input_video_path):
        print(f"Video file does not exist: {input_video_path}")
        continue  # Skip to the next video

    # Initialize Video Capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Could not open video file: {input_video_path}")
        continue  # Skip to the next video

    # Retrieve video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0  # Default fallback if FPS not detected

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video Properties - FPS: {fps}, Width: {width}, Height: {height}")

    # Dictionaries to store previous positions and velocities for each joint
    prev_positions = {}
    prev_velocities = {}

    # List to store all feature data
    all_data = []

    frame_count = 0
    frames_with_landmarks = 0

    # Create a window for feature visualization only for the first two videos
    if idx < 2:
        feature_screen = np.zeros((600, 800, 3), dtype=np.uint8)  # Black background
        cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Feature Visualization", cv2.WINDOW_NORMAL)
    else:
        feature_screen = None  # No rendering for other videos

    print("Starting video processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video file reached.")
            break  # Exit loop if no frame is returned

        frame_count += 1

        # Convert the frame to RGB as MediaPipe expects RGB images
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(rgb_frame)

        # **1. Detect Sports Ball Position using YOLOv5**
        # Run YOLOv5 detection on the current frame
        yolo_results = yolo_model.predict(source=frame, verbose=False)
        sports_ball_positions = []

        for result in yolo_results:
            # Each result corresponds to detections in one image/frame
            for box in result.boxes:
                cls_id = int(box.cls[0])  # Class ID
                if cls_id == SPORTS_BALL_CLASS_ID:
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    # Calculate center coordinates
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    sports_ball_positions.append((x_center, y_center))

        # Initialize skeleton center
        skeleton_center_x = None
        skeleton_center_y = None

        if results_pose.pose_landmarks:
            landmarks = results_pose.pose_landmarks.landmark
            left_hip = landmarks[joint_map["LEFT_HIP"]]
            right_hip = landmarks[joint_map["RIGHT_HIP"]]

            # Check if both hips are visible to compute skeleton center
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                skeleton_center_x = ((left_hip.x + right_hip.x) / 2) * width
                skeleton_center_y = ((left_hip.y + right_hip.y) / 2) * height
                frames_with_landmarks += 1
                # Debug: Print skeleton center
            else:
                print(f"Frame {frame_count}: Hips not sufficiently visible.")
        else:
            print(f"Frame {frame_count}: No pose landmarks detected.")

        joint_data = {}

        if skeleton_center_x is not None and skeleton_center_y is not None:
            for joint_name, joint_idx in joint_map.items():
                j_lm = landmarks[joint_idx]
                x = j_lm.x * width - skeleton_center_x
                y = j_lm.y * height - skeleton_center_y

                current_pos = (x, y)

                # Compute velocity
                if joint_name in prev_positions and prev_positions[joint_name] is not None:
                    dx = x - prev_positions[joint_name][0]
                    dy = y - prev_positions[joint_name][1]
                    velocity = math.sqrt(dx ** 2 + dy ** 2) * fps
                    velocity = round(velocity, 2)
                else:
                    velocity = None

                # Compute acceleration
                if velocity is not None and joint_name in prev_velocities and prev_velocities[joint_name] is not None:
                    dvel = velocity - prev_velocities[joint_name]
                    acceleration = dvel * fps
                    acceleration = round(acceleration, 2)
                else:
                    acceleration = None

                joint_data[joint_name] = {
                    "pos": current_pos,
                    "vel": velocity,
                    "acc": acceleration
                }

                # Update previous positions and velocities
                prev_positions[joint_name] = current_pos
                prev_velocities[joint_name] = velocity

            # Compile features for the current frame
            features_row = {
                'video': os.path.basename(input_video_path),
                'frame': frame_count,
                'label': label
            }

            # **Add Sports Ball Positions to the Dataset**
            if sports_ball_positions:
                # If multiple sports balls are detected, store all positions
                # Here, we'll concatenate positions as a string (e.g., "x1,y1;x2,y2;...")
                positions_str = ';'.join([f"{round(pos[0],2)},{round(pos[1],2)}" for pos in sports_ball_positions])
                features_row['sports_ball_positions'] = positions_str
            else:
                features_row['sports_ball_positions'] = np.nan

            # Add velocities and accelerations for paired joints
            for left_joint, right_joint in PAIRED_JOINTS:
                # Velocities
                left_vel = joint_data[left_joint]["vel"] if joint_data[left_joint]["vel"] is not None else np.nan
                right_vel = joint_data[right_joint]["vel"] if joint_data[right_joint]["vel"] is not None else np.nan
                features_row[f"{left_joint}_vel"] = left_vel
                features_row[f"{right_joint}_vel"] = right_vel

                # Accelerations
                left_acc = joint_data[left_joint]["acc"] if joint_data[left_joint]["acc"] is not None else np.nan
                right_acc = joint_data[right_joint]["acc"] if joint_data[right_joint]["acc"] is not None else np.nan
                features_row[f"{left_joint}_acc"] = left_acc
                features_row[f"{right_joint}_acc"] = right_acc

            # Add joint positions relative to skeleton center
            for joint_name in joint_map.keys():
                pos_x, pos_y = joint_data[joint_name]["pos"]
                features_row[f"{joint_name}_pos_x"] = round(pos_x, 2)
                features_row[f"{joint_name}_pos_y"] = round(pos_y, 2)

            # Calculate angles for defined joint triplets
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

            all_data.append(features_row)

            # Draw pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if idx < 2 and feature_screen is not None:
                # Optionally, highlight joints of interest
                for joint in joint_map.keys():
                    pos_x = joint_data[joint]["pos"][0]
                    pos_y = joint_data[joint]["pos"][1]
                    # Convert back to absolute coordinates for visualization
                    abs_x = int(pos_x + skeleton_center_x)
                    abs_y = int(pos_y + skeleton_center_y)
                    # Ensure the coordinates are within frame boundaries
                    abs_x = max(0, min(abs_x, width - 1))
                    abs_y = max(0, min(abs_y, height - 1))
                    cv2.circle(frame, (abs_x, abs_y), 5, (0, 255, 0), -1)  # Green dots for joints

                # **Draw Bounding Boxes for Detected Sports Balls**
                for pos in sports_ball_positions:
                    x_center, y_center = pos
                    # Optional: Draw a small circle at the ball's position
                    cv2.circle(frame, (int(x_center), int(y_center)), 10, (255, 0, 0), 2)  # Blue circles for sports balls

                # Prepare feature visualization screen
                feature_screen[:] = (0, 0, 0)  # Reset to black
                cv2.putText(feature_screen, "Features (Velocities, Accelerations, Angles):", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset = 50
                for left_joint, right_joint in PAIRED_JOINTS:
                    # Velocities
                    left_vel = joint_data[left_joint]["vel"] if joint_data[left_joint]["vel"] is not None else "N/A"
                    right_vel = joint_data[right_joint]["vel"] if joint_data[right_joint]["vel"] is not None else "N/A"

                    # Determine colors based on velocity comparison
                    if left_vel != "N/A" and (right_vel == "N/A" or left_vel > right_vel):
                        left_color = (255, 0, 0)  # Blue for higher velocity
                    else:
                        left_color = (0, 0, 255)  # Red for lower or unavailable

                    if right_vel != "N/A" and (left_vel == "N/A" or right_vel > left_vel):
                        right_color = (255, 0, 0)  # Blue for higher velocity
                    else:
                        right_color = (0, 0, 255)  # Red for lower or unavailable

                    cv2.putText(feature_screen, f"{left_joint} Vel={left_vel}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
                    cv2.putText(feature_screen, f"{right_joint} Vel={right_vel}", (400, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                    y_offset += 25

                    # Accelerations
                    left_acc = joint_data[left_joint]["acc"] if joint_data[left_joint]["acc"] is not None else "N/A"
                    right_acc = joint_data[right_joint]["acc"] if joint_data[right_joint]["acc"] is not None else "N/A"

                    cv2.putText(feature_screen, f"{left_joint} Acc={left_acc}", (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, left_color, 1)
                    cv2.putText(feature_screen, f"{right_joint} Acc={right_acc}", (400, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, right_color, 1)
                    y_offset += 25

                # **Display Angles**
                for triplet in ANGLE_JOINTS:
                    joint_a, joint_b, joint_c = triplet
                    angle = features_row.get(f"{joint_b}_{joint_a}_{joint_c}_angle", "N/A")
                    angle_text = f"{joint_b}-{joint_a}-{joint_c} Angle={angle}"
                    angle_color = (0, 255, 0) if angle != "N/A" else (0, 0, 255)
                    cv2.putText(feature_screen, angle_text, (10, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 1)
                    y_offset += 25

                # **Show Both Screens**
                resized_frame = cv2.resize(frame, (width // 3, height // 3))
                cv2.imshow("Pose Detection", resized_frame)
                cv2.imshow("Feature Visualization", feature_screen)

            # Optional: Print progress every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, {frames_with_landmarks} with landmarks detected.")

            # Exit if 'q' is pressed and rendering is active
            if idx < 2 and feature_screen is not None:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting processing loop as 'q' was pressed.")
                    break

    # Release Video Capture and close windows for the current video
    cap.release()
    if idx < 2 and feature_screen is not None:
        cv2.destroyWindow("Pose Detection")
        cv2.destroyWindow("Feature Visualization")

    print(f"Finished processing video: {input_video_path}")

    # BEGIN DATA ENGINEERING

    # Convert all_data to a Pandas DataFrame
    df = pd.DataFrame(all_data)

    # Replace "N/A" with NaN for easier handling
    df.replace("N/A", np.nan, inplace=True)

    # Optionally, fill NaN with zeros or another appropriate value
    df.fillna(0, inplace=True)

    # Define the output CSV path within the "Datasets" folder
    output_csv_filename = os.path.splitext(os.path.basename(input_video_path))[0] + ".csv"
    output_csv_path = os.path.join(datasets_dir, output_csv_filename)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Feature data successfully saved to {output_csv_path}")

# Close the MediaPipe Pose instance after all videos are processed
pose.close()
cv2.destroyAllWindows()
print("\nAll videos have been processed and datasets have been created.")
