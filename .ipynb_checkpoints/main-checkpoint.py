import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (vertex)
    c = np.array(c)  # Third point

    ba = a - b
    bc = c - b

    # Compute cosine angle and convert to degrees
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# Initialize variables for velocities and accelerations
previous_keypoints = None
previous_time = None
velocities = {}
accelerations = {}

# Video title
video_title = input("Enter the video title: ")

# Prepare dataframe for storing results
data = []
columns = [
    "Frame", "Time (s)", "Joint", "Position (x,y,z)", "Angle (degrees)", "Velocity (m/s)", "Acceleration (m/s^2)", "Video Title"
]

# Access webcam
cap = cv2.VideoCapture(0)
start_time = time.time()
frame_count = 0

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty frame.")
            break

        frame_count += 1
        # Convert BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image
        results = pose.process(image)

        # Convert image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        height, width, _ = image.shape
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Extract landmarks if available
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract relevant landmarks: shoulders, elbows, and wrists
            keypoints = {
                "left_shoulder": (landmarks[11].x, landmarks[11].y, landmarks[11].z),
                "right_shoulder": (landmarks[12].x, landmarks[12].y, landmarks[12].z),
                "left_elbow": (landmarks[13].x, landmarks[13].y, landmarks[13].z),
                "right_elbow": (landmarks[14].x, landmarks[14].y, landmarks[14].z),
                "left_wrist": (landmarks[15].x, landmarks[15].y, landmarks[15].z),
                "right_wrist": (landmarks[16].x, landmarks[16].y, landmarks[16].z),
            }

            # Calculate angles
            angles = {
                "left_elbow": calculate_angle(keypoints["left_shoulder"], keypoints["left_elbow"], keypoints["left_wrist"]),
                "right_elbow": calculate_angle(keypoints["right_shoulder"], keypoints["right_elbow"], keypoints["right_wrist"]),
                "shoulder": calculate_angle(keypoints["left_elbow"], keypoints["left_shoulder"], keypoints["right_shoulder"]),
            }

            # Calculate velocities and accelerations
            if previous_keypoints is not None and previous_time is not None:
                delta_time = elapsed_time - previous_time
                for joint, position in keypoints.items():
                    if joint in previous_keypoints:
                        prev_position = previous_keypoints[joint]
                        velocity = np.linalg.norm(np.array(position) - np.array(prev_position)) / delta_time
                        velocities[joint] = velocity

                        if joint in velocities:
                            acceleration = (velocity - velocities[joint]) / delta_time
                            accelerations[joint] = acceleration
                        else:
                            accelerations[joint] = 0.0
                    else:
                        velocities[joint] = 0.0
                        accelerations[joint] = 0.0
            
            # Update previous keypoints and time
            previous_keypoints = keypoints
            previous_time = elapsed_time

            # Store data for the current frame
            for joint, position in keypoints.items():
                data.append([
                    frame_count, elapsed_time, joint,
                    position, angles.get(joint, "N/A"),
                    velocities.get(joint, "N/A"), accelerations.get(joint, "N/A"),
                    video_title
                ])

        # Display the output
        cv2.imshow('Pose Estimation', image)

        # Break loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Save the data to a CSV file
df = pd.DataFrame(data, columns=columns)
csv_filename = f"{video_title.replace(' ', '_')}_pose_data.csv"
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}")
