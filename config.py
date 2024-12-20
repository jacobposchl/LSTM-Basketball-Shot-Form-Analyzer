# config.py

import os
from enum import Enum, auto

# ===================================
# Thresholds and Constants
# ===================================

# Shot Detection Thresholds
VELOCITY_THRESHOLD = 150.0        # Velocity required to detect a shot
CONSECUTIVE_FRAMES = 4            # Number of consecutive frames with high velocity
DISTANCE_THRESHOLD = 150.0         # Distance the ball must move to validate a shot
TIME_THRESHOLD = 2.0               # Time within which the ball must move after shot initiation (in seconds)
VERTICAL_DISPLACEMENT_THRESHOLD = 50.0  # Minimum upward displacement for shot detection

# Stability Thresholds
STABLE_FRAMES_REQUIRED = 3            # Number of consecutive stable frames required
STABLE_VELOCITY_THRESHOLD = 150.0       # Maximum velocity during stability check
BALL_WRIST_DISTANCE_THRESHOLD = 100.0   # Maximum distance between ball and wrist during stability check

# YOLO Detection Threshold
YOLO_CONFIDENCE_THRESHOLD = 0.4        # YOLO detection confidence threshold

# Thumbs-Up Detection Threshold
DETECTION_THRESHOLD = 3                 # Number of consecutive frames required for thumbs-up detection

# Smoothing Parameters
SMOOTHING_WINDOW = 3                    # Frames to average velocity and acceleration

# Frame Resizing Parameters
TARGET_HEIGHT = 1920                    # Target height for resizing frames
TARGET_WIDTH = 1080                     # Target width for resizing frames

# Cooldown Parameters
SHOT_COOLDOWN_FRAMES = 30               # Cooldown period after shot detection to prevent immediate re-detection

# Pose Detection Thresholds
MIN_POSE_DETECTION_CONFIDENCE = 0.7      # Minimum confidence for pose detection
MIN_POSE_TRACKING_CONFIDENCE = 0.7       # Minimum confidence for pose tracking

# Hand Detection Thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.6      # Minimum confidence for hand detection
MIN_HAND_TRACKING_CONFIDENCE = 0.7       # Minimum confidence for hand tracking

ROI_SIZE = 150

# ===================================
# Directory Paths
# ===================================

# Get the absolute path of the current file's directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the project directory
VIDEOS_DIR = os.path.join(PROJECT_DIR, 'Videos')
DATASETS_DIR = os.path.join(PROJECT_DIR, 'Datasets')
WEIGHTS_DIR = os.path.join(PROJECT_DIR, 'Weights')
YOLOV5_DIR = os.path.join(PROJECT_DIR, 'yolov5')
NOTES_DIR = os.path.join(PROJECT_DIR, 'Notes')

# Ensure that directories exist
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(YOLOV5_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

# ===================================
# Input Videos and Labels Configuration
# ===================================

# Webcam Configuration
USE_WEBCAM = False  # Set to True to use the webcam instead of input videos

if USE_WEBCAM:
    INPUT_VIDEOS = [0]  # 0 is the default webcam index. Change if you have multiple webcams.
else:
    INPUT_VIDEOS = ['Videos/thumbs_up.mov']  # Add more video paths as needed
    #Videos/thumbs_up.mov <- Thumbs Up Video
    #Videos/long_vid.mov <- Long Video
    #new_long_vid <- new long video

LABEL = 'Good_Shots'

# YOLO Weights Path
YOLO_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'best.pt')

# ===================================
# MediaPipe Pose Configuration
# ===================================

import mediapipe as mp

mp_pose = mp.solutions.pose

# Useful Landmarks for Pose Estimation
USEFUL_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST",
    "LEFT_HIP", "RIGHT_HIP"
]

# Create a joint_map dictionary mapping landmark names to their indices
joint_map = {j.name: j.value for j in mp_pose.PoseLandmark if j.name in USEFUL_LANDMARKS}

# Paired Joints for Velocity and Acceleration Calculations
PAIRED_JOINTS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
    ("LEFT_HIP", "RIGHT_HIP"),
]

# Triplet Joints for Angle Calculations
ANGLE_JOINTS = [
    ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_HIP", "LEFT_SHOULDER", "LEFT_ELBOW"),
    ("RIGHT_HIP", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
]

# ===================================
# Shot Detection States
# ===================================

class ShotState(Enum):
    """
    Enumeration of possible states in the shot detection state machine.
    """
    WAITING_FOR_STABILITY = auto()
    READY_TO_DETECT_SHOT = auto()
    SHOT_IN_PROGRESS = auto()
    COOLDOWN = auto()
