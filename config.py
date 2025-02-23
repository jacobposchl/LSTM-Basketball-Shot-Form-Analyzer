# config.py

import os
from enum import Enum, auto


# ===================================
# Development Mode Configuration
# ===================================

DEV_MODE = 2
#Visualize processing of specific videos in bucket -> 0
#Visualize processing of all videos in bucket      -> 1
#Only render specific videos in bucket             -> 2
#Only render all videos in bucket                  -> 3

DOMINANT_HAND = "RIGHT"


DEV_VIDEOS = [
    "latest_vid.MOV"
]
"""
Available videos:
- latest_vid.MOV -> Newest video, hard for yolo to detect balls
- long_vid.mov -> uncut long vid, easy for yolo to detect balls
- long_vid_final.mov -> trimmed long vid, easy for yolo to detect balls
- thumbs_up_short.mov -> short video for thumbs up count development
"""


# ===================================
# GCP Configuration
# ===================================

GCP_BUCKET_NAME = "basketball-ai-data"
GCP_PREFIX = "project/basketball-ai-data/files/Videos/"  # Path to videos in the bucket

# Directory to temporarily store downloaded videos
GCP_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GCP_Downloads')
os.makedirs(GCP_DOWNLOAD_DIR, exist_ok=True)

# YOLO Weights Path (ensure this points to the correct location on the VM)
YOLO_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weights', 'old_best_v2.pt')
"""
Available weights:
- old_best.pt
- old_best_v1.pt
- old_best_v2.pt
- best.pt
"""

# ===================================
# Thresholds and Constants
# ===================================

# Shot Detection Thresholds
VELOCITY_THRESHOLD = 150.0        # Velocity required to detect a shot
CONSECUTIVE_FRAMES = 4            # Number of consecutive frames with high velocity
DISTANCE_THRESHOLD = 150.0         # Distance the ball must move to validate a shot
TIME_THRESHOLD = 2.0               # Time within which the ball must move after shot initiation (in seconds)
VERTICAL_DISPLACEMENT_THRESHOLD = 50.0  # Minimum upward displacement for shot detection


WRIST_CLOSE_DISTANCE_THRESHOLD = 100.0
PROBABILITY_INCREMENT_BALL_CLOSE = 0.20
MIN_PROBABILITY = 0.00
MAX_PROBABILITY = 1.00
PROBABILITY_INCREMENT_WRISTS_CLOSE = 0.10
PROBABILITY_DECREMENT_UNSTABLE = 0.15



# Stability Thresholds
STABLE_FRAMES_REQUIRED = 2            # Number of consecutive stable frames required
STABLE_VELOCITY_THRESHOLD = 120.0       # Maximum velocity during stability check
BALL_WRIST_DISTANCE_THRESHOLD = 100.0   # Maximum distance between ball and wrist during stability check

# YOLO Detection Threshold
YOLO_CONFIDENCE_THRESHOLD = 0.3      # Normal is 0.4

# Thumbs-Up Detection Threshold
DETECTION_THRESHOLD = 3                 # Number of consecutive frames required for thumbs-up detection

# Smoothing Parameters
SMOOTHING_WINDOW = 3                    # Frames to average velocity and acceleration

# Frame Resizing Parameters
TARGET_HEIGHT = 1280                    # Target height for resizing frames (Small = 1280 x 720)
TARGET_WIDTH = 720                     # Target width for resizing frames

# Cooldown Parameters
SHOT_COOLDOWN_FRAMES = 30               # Cooldown period after shot detection to prevent immediate re-detection

# Pose Detection Thresholds
MIN_POSE_DETECTION_CONFIDENCE = 0.7      # Minimum confidence for pose detection
MIN_POSE_TRACKING_CONFIDENCE = 0.7       # Minimum confidence for pose tracking

# Hand Detection Thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.7      # Minimum confidence for hand detection
MIN_HAND_TRACKING_CONFIDENCE = 0.5       # Minimum confidence for hand tracking

THUMB_GESTURE_WINDOW = 20                # Number of frames to track for thumbs-up gesture
THUMB_GESTURE_THRESHOLD = 0.8            # Minimum confidence for thumbs-up gesture

ROI_SIZE = 150

# ===================================
# Directory Paths
# ===================================

# Update directories to point to VM's local paths
VIDEOS_DIR = GCP_DOWNLOAD_DIR  # Videos will be downloaded here
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datasets')
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weights')
YOLOV5_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov5')

# Ensure all directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(YOLOV5_DIR, exist_ok=True)

# ===================================
# Input Videos and Labels Configuration
# ===================================

# Since webcam is no longer used, remove all webcam configurations.
# INPUT_VIDEOS will be managed based on DEV_MODE and DEV_VIDEOS.

INPUT_VIDEOS = []  # This will be populated dynamically based on DEV_MODE

# YOLO Weights Path
YOLO_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'old_best_v1.pt')
# Best currently -> old_best_v1
# Currently training -> best

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
# Test Write
