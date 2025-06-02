# config.py

import os
from enum import Enum, auto
from google.cloud import storage

#
# ===================================
# Development Mode Configuration
# ===================================

DATA_MODE = 0
#0 -> Only Angle Based Features
#1 -> Include Positional Based Features

DEV_MODE = 0
#Visualize processing of specific videos in bucket          -> 0 USED FOR VISULIZATION
#Visualize processing of all videos in bucket               -> 1
#DONT Visualize and process specific videos in bucket       -> 2 USED FOR PRODUCTION
#DONT Visualize and process all videos in bucket            -> 3
 
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
GCP_PREFIX = "jobs/"  # Path to videos in the bucket

YOLO_WEIGHTS_BLOB     = "yolo-weights/Weights/new_best.pt"

# Directory to temporarily store downloaded videos
GCP_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'GCP_Downloads')
os.makedirs(GCP_DOWNLOAD_DIR, exist_ok=True)

# YOLO Weights Path (ensure this points to the correct location on the VM)
YOLO_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Weights', 'new_best.pt')
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
VELOCITY_THRESHOLD = 150.0        # Velocity required to detect a shot -> INITIALLY (150)
CONSECUTIVE_FRAMES = 8            # Number of consecutive frames with high velocity
DISTANCE_THRESHOLD = 150.0         # Distance the ball must move to validate a shot -> INITIALLY (150)
TIME_THRESHOLD = 3.5               # Time within which the ball must move after shot initiation (in seconds) -> INITIALLY (2)
VERTICAL_DISPLACEMENT_THRESHOLD = 80.0  # Minimum upward displacement for shot detection -> INITIALLY (50)


WRIST_CLOSE_DISTANCE_THRESHOLD = 50.0
PROBABILITY_INCREMENT_BALL_CLOSE = 0.25
MIN_PROBABILITY = 0.00
MAX_PROBABILITY = 1.00
PROBABILITY_INCREMENT_WRISTS_CLOSE = 0.15
PROBABILITY_DECREMENT_UNSTABLE = 0.3

# how many frames of continuous stability to ARM the detector
ARM_STABLE_FRAMES = 8

# once armed, how many more stable frames until we REBASELINE
REBASELINE_STABLE_FRAMES = 60

STABILITY_THRESHOLD = .8
MAX_UNSTABLE_FRAMES = 14
MAX_WAITING_UNSTABLE_FRAMES = 8




# Stability Thresholds
STABLE_FRAMES_REQUIRED = 10            # Number of consecutive stable frames required
STABLE_VELOCITY_THRESHOLD = 350.0       # Maximum velocity during stability check
BALL_WRIST_DISTANCE_THRESHOLD = 50.0   # Maximum distance between ball and wrist during stability check


# YOLO Detection Threshold
YOLO_CONFIDENCE_THRESHOLD = 0.2      # Normal is 0.4
MAX_BALL_INVISIBLE_FRAMES = 14   # Maximum frames the ball can be invisible before detection is reset
# Thumbs-Up Detection Threshold
DETECTION_THRESHOLD = 6                 # Number of consecutive frames required for thumbs-up detection

# Smoothing Parameters
SMOOTHING_WINDOW = 8                    # Frames to average velocity and acceleration

# Frame Resizing Parameters
TARGET_HEIGHT = 1280                    # Target height for resizing frames (Small = 1280 x 720)
TARGET_WIDTH = 720                     # Target width for resizing frames

# Cooldown Parameters
SHOT_COOLDOWN_FRAMES = 16               # Cooldown period after shot detection to prevent immediate re-detection

# Pose Detection Thresholds
MIN_POSE_DETECTION_CONFIDENCE = 0.7      # Minimum confidence for pose detection
MIN_POSE_TRACKING_CONFIDENCE = 0.7       # Minimum confidence for pose tracking

# Hand Detection Thresholds
MIN_HAND_DETECTION_CONFIDENCE = 0.7      # Minimum confidence for hand detection
MIN_HAND_TRACKING_CONFIDENCE = 0.5       # Minimum confidence for hand tracking

THUMB_GESTURE_WINDOW = 40                # Number of frames to track for thumbs-up gesture
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
YOLO_WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'new_best.pt')
# Best currently -> old_best_v1
# Currently training -> best

def ensure_yolo_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    if os.path.exists(YOLO_WEIGHTS_PATH):
        return
    client = storage.Client()
    bucket = client.bucket(GCP_BUCKET_NAME)
    blob   = bucket.blob(YOLO_WEIGHTS_BLOB)
    print(f"[config] downloading YOLO weights to {YOLO_WEIGHTS_PATH} …")
    blob.download_to_filename(YOLO_WEIGHTS_PATH)
    print("[config] download complete.")

ensure_yolo_weights()

# ===================================
# MediaPipe Pose Configuration
# ===================================



# COCO 17‐point keypoint indices for MMPose “td‐hm_hrnet‐w32_8xb64”
joint_map = {
    "NOSE": 0,
    "LEFT_SHOULDER": 5, "RIGHT_SHOULDER": 6,
    "LEFT_ELBOW": 7,    "RIGHT_ELBOW": 8,
    "LEFT_WRIST": 9,    "RIGHT_WRIST": 10,
    "LEFT_HIP": 11,     "RIGHT_HIP": 12,
    "LEFT_KNEE": 13,    "RIGHT_KNEE": 14,
    "LEFT_ANKLE": 15,   "RIGHT_ANKLE": 16
}


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
    ("LEFT_HIP",      "LEFT_KNEE",     "LEFT_ANKLE"),
    ("RIGHT_HIP",     "RIGHT_KNEE",    "RIGHT_ANKLE"),
    ("LEFT_KNEE",     "LEFT_ANKLE",    "LEFT_FOOT_INDEX"),
    ("RIGHT_KNEE",    "RIGHT_ANKLE",   "RIGHT_FOOT_INDEX"),
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
