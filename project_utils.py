# project_utils.py

import math
import numpy as np
from collections import deque
import mediapipe as mp
from config import joint_map, TARGET_WIDTH, TARGET_HEIGHT, BALL_WRIST_DISTANCE_THRESHOLD, STABLE_VELOCITY_THRESHOLD
import logging

def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by points 'a', 'b', and 'c'.
    
    Args:
        a (tuple): Coordinates of point 'a' (x, y, z).
        b (tuple): Coordinates of point 'b' (x, y, z).
        c (tuple): Coordinates of point 'c' (x, y, z).
    
    Returns:
        float: Angle at point 'b' in degrees.
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    magnitude_ab = np.linalg.norm(ba)
    magnitude_cb = np.linalg.norm(bc)
    if magnitude_ab * magnitude_cb == 0:
        return 0.0
    angle = np.arccos(dot_product / (magnitude_ab * magnitude_cb))
    return round(math.degrees(angle), 2)

def calculate_angle_3d(a, b, c):
    """
    Calculates the 3D angle at point 'b' formed by points 'a', 'b', and 'c'.
    
    Args:
        a (tuple or list): Coordinates of point 'a' (x, y, z).
        b (tuple or list): Coordinates of point 'b' (x, y, z).
        c (tuple or list): Coordinates of point 'c' (x, y, z).
    
    Returns:
        float: Angle at point 'b' in degrees. Returns 0.0 if calculation is not possible.
    """
    try:
        # Convert points to numpy arrays
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        c = np.array(c, dtype=np.float64)
        
        # Compute vectors BA and BC
        ba = a - b
        bc = c - b
        
        # Calculate dot product and magnitudes
        dot_product = np.dot(ba, bc)
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)
        
        if magnitude_ba == 0 or magnitude_bc == 0:
            return 0.0
        
        # Compute the cosine of the angle using the dot product formula
        cos_angle = dot_product / (magnitude_ba * magnitude_bc)
        
        # Clamp the cosine value to the valid range to avoid numerical issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Calculate the angle in radians and then convert to degrees
        angle_rad = np.arccos(cos_angle)
        angle_deg = math.degrees(angle_rad)
        
        return round(angle_deg, 2)
    
    except Exception as e:
        return 0.0


def extract_roi(frame, landmarks, width, height, padding=50):
    """
    Define a Region of Interest (ROI) based on the MediaPipe landmarks.
    This focuses YOLO detection within this area to improve performance.
    
    Args:
        frame (numpy.ndarray): The input frame.
        landmarks (mediapipe.framework.formats.pose_pb2.NormalizedLandmarkList): Detected pose landmarks.
        width (int): Width of the frame.
        height (int): Height of the frame.
        padding (int, optional): Padding around the ROI. Defaults to 50.
    
    Returns:
        tuple: (roi_frame, (x1, y1)) where roi_frame is the cropped region and (x1, y1) are the top-left coordinates.
    """
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

# project_utils.py

import numpy as np

# project_utils.py

def map_to_original(x_padded, y_padded, original_width, original_height, new_width, new_height, left_pad, top_pad):
    try:
        # Remove padding
        x_resized = x_padded - left_pad
        y_resized = y_padded - top_pad

        # Scale to original size
        scale_x = original_width / new_width
        scale_y = original_height / new_height

        x_original = x_resized * scale_x
        y_original = y_resized * scale_y

        # Clamp values to be within the original frame
        x_original = max(0, min(original_width - 1, x_original))
        y_original = max(0, min(original_height - 1, y_original))


        return (x_original, y_original)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error in map_to_original: {e}")
        return (None, None)





def reset_detection_history(maxlen=3):
    """
    Resets the detection history deque.
    
    Args:
        maxlen (int): Maximum length of the deque.
    
    Returns:
        deque: A deque initialized with the specified maximum length.
    """
    return deque(maxlen=maxlen)
