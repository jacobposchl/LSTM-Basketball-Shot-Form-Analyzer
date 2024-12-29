# hand_detection.py

import mediapipe as mp
import cv2
import math
import logging
from collections import deque
import numpy as np

class HandDetector:
    """
    A class to handle hand detection and thumbs-up/thumbs-down gesture recognition using MediaPipe Hands.
    """
    def __init__(self, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.7, buffer_size=5):
        """
        Initializes the MediaPipe Hands detector.
        
        Args:
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand detection.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be tracked.
            buffer_size (int): Size of the buffer for temporal smoothing of gesture detection.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,  # Higher for better accuracy
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.DEBUG)
        self.thumbs_up_buffer = deque(maxlen=buffer_size)
        self.thumbs_down_buffer = deque(maxlen=buffer_size)  # Initialize thumbs_down_buffer
        self.latest_features = []  # Initialize feature storage

    def clear_features(self):
        """
        Clears the latest_features list. Should be called once per frame before processing hands.
        """
        self.latest_features = []

    def process_frame(self, frame_rgb):
        """
        Processes an RGB frame to detect hand landmarks.
        
        Args:
            frame_rgb (np.array): The RGB image/frame to process.
        
        Returns:
            results: The detection results containing hand landmarks.
        """
        if frame_rgb is None or not hasattr(frame_rgb, 'shape'):
            self.logger.error("Invalid frame provided for processing.")
            return None
        try:
            results = self.hands.process(frame_rgb)
            return results
        except Exception as e:
            self.logger.error(f"Error processing frame for hand detection: {e}")
            return None

    def draw_landmarks(self, frame, hand_landmarks, handedness=None, draw_orientation_vectors=False):
        """
        Draws hand landmarks and optionally orientation vectors on the given frame.

        Args:
            frame (np.array): The image/frame to draw on.
            hand_landmarks: Detected hand landmarks.
            handedness (list): List indicating 'Left' or 'Right' for each detected hand.
            draw_orientation_vectors (bool): Flag to draw orientation vectors for debugging.
        """
        try:
            if hand_landmarks:
                for idx, hand_landmark in enumerate(hand_landmarks):
                    if handedness and idx < len(handedness):
                        hand_label = handedness[idx].classification[0].label  # 'Left' or 'Right'
                        if hand_label == 'Left':
                            color = (0, 0, 225)  # Red for Left Hand
                        else:
                            color = (225, 0, 0)    # Blue for Right Hand
                        self.logger.debug(f"Drawing {hand_label} hand with color {color}")
                    else:
                        color = (121, 22, 76)  # Purple as default color
                        self.logger.debug(f"Drawing hand with default color {color}")
                    
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmark,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)  # Use the same color for connections
                    )
                    
                    if draw_orientation_vectors:
                        self.draw_orientation_vectors(frame, hand_landmark)
        except Exception as e:
            self.logger.error(f"Error drawing hand landmarks: {e}")

    def draw_orientation_vectors(self, frame, hand_landmark):
        """
        Draws the orientation vectors on the frame for debugging purposes.

        Args:
            frame (np.array): The image/frame to draw on.
            hand_landmark: The detected hand landmarks.
        """
        try:
            # Get wrist and middle finger MCP landmarks
            wrist = hand_landmark.landmark[self.mp_hands.HandLandmark.WRIST]
            mcp = hand_landmark.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

            # Convert normalized coordinates to pixel values
            wrist_x = int(wrist.x * frame.shape[1])
            wrist_y = int(wrist.y * frame.shape[0])
            mcp_x = int(mcp.x * frame.shape[1])
            mcp_y = int(mcp.y * frame.shape[0])

            # Draw vector from wrist to MCP
            cv2.line(frame, (wrist_x, wrist_y), (mcp_x, mcp_y), (255, 0, 0), 2)
            cv2.circle(frame, (wrist_x, wrist_y), 5, (255, 0, 0), -1)  # Wrist
            cv2.circle(frame, (mcp_x, mcp_y), 5, (255, 0, 0), -1)      # MCP

            # Draw vertical y-axis from wrist for reference
            cv2.line(frame, (wrist_x, wrist_y - 50), (wrist_x, wrist_y + 50), (0, 255, 0), 2)
        except Exception as e:
            self.logger.error(f"Error drawing orientation vectors: {e}")

    def calculate_angle(self, a, b, c):
        """
        Calculates the angle (in degrees) at point b given three points a, b, and c.

        Args:
            a (tuple): (x, y, z) coordinates of point a.
            b (tuple): (x, y, z) coordinates of point b (vertex of the angle).
            c (tuple): (x, y, z) coordinates of point c.

        Returns:
            float: Angle in degrees at point b.
        """
        try:
            # Convert to numpy arrays
            a = np.array([a.x, a.y, a.z])
            b = np.array([b.x, b.y, b.z])
            c = np.array([c.x, c.y, c.z])

            # Vectors
            ba = a - b
            bc = c - b

            # Calculate the cosine of the angle using the dot product formula
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)  # Add small epsilon to avoid division by zero
            angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip to handle numerical errors

            return np.degrees(angle)
        except Exception as e:
            self.logger.error(f"Error calculating angle: {e}")
            return 180.0  # Return a default value indicating a straight angle

    def is_finger_curled(self, hand_landmarks, finger_name, angle_threshold=170):
        """
        Determines if a specific finger is curled based on the angle at the PIP joint.

        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            finger_name (str): Name of the finger ('INDEX', 'MIDDLE', 'RING', 'PINKY').
            angle_threshold (float): Angle in degrees above which the finger is considered extended.

        Returns:
            bool: True if the finger is curled, False otherwise.
        """
        try:
            # Define the landmarks for each finger
            finger_landmarks = {
                'INDEX': self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
                'MIDDLE': self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                'RING': self.mp_hands.HandLandmark.RING_FINGER_TIP,
                'PINKY': self.mp_hands.HandLandmark.PINKY_TIP
            }

            pip_landmarks = {
                'INDEX': self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                'MIDDLE': self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                'RING': self.mp_hands.HandLandmark.RING_FINGER_PIP,
                'PINKY': self.mp_hands.HandLandmark.PINKY_PIP
            }

            dip_landmarks = {
                'INDEX': self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
                'MIDDLE': self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                'RING': self.mp_hands.HandLandmark.RING_FINGER_DIP,
                'PINKY': self.mp_hands.HandLandmark.PINKY_DIP
            }

            mcp_landmarks = {
                'INDEX': self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                'MIDDLE': self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                'RING': self.mp_hands.HandLandmark.RING_FINGER_MCP,
                'PINKY': self.mp_hands.HandLandmark.PINKY_MCP
            }

            if finger_name.upper() not in finger_landmarks:
                self.logger.error(f"Invalid finger name: {finger_name}")
                return False

            tip = hand_landmarks.landmark[finger_landmarks[finger_name.upper()]]
            pip = hand_landmarks.landmark[pip_landmarks[finger_name.upper()]]
            mcp = hand_landmarks.landmark[mcp_landmarks[finger_name.upper()]]
            dip = hand_landmarks.landmark[dip_landmarks[finger_name.upper()]]

            # Calculate the angle at the PIP joint using MCP-PIP-DIP
            angle = self.calculate_angle(mcp, pip, dip)

            # If the angle is less than the threshold, consider the finger as curled
            is_curled = angle < angle_threshold


            return is_curled

        except Exception as e:
            self.logger.error(f"Error determining if {finger_name} finger is curled: {e}")
            return False


    def is_thumbs_up(self, hand_landmarks, handedness):
        thumb_threshold = 0.15
        """
        Determines if the detected hand gesture is a thumbs-up based on thumb position
        relative to other fingers and other fingers being curled.
        
        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            handedness (str): 'Left' or 'Right' indicating the detected hand's side.
        
        Returns:
            bool: True if thumbs-up gesture is detected, False otherwise.
        """
        try:
            # Extract necessary landmarks for thumb
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
            thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
            thumb_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]

            # Extract landmarks for other fingers' tips
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

            # 1. Check if thumb tip is higher than other fingers' tips with threshold
            thumb_is_above = (
                thumb_tip.y + thumb_threshold < index_tip.y and
                thumb_tip.y + thumb_threshold < middle_tip.y and
                thumb_tip.y + thumb_threshold < ring_tip.y and
                thumb_tip.y + thumb_threshold < pinky_tip.y
            )

            #self.logger.debug(f"Thumb is above other fingers: {thumb_is_above} : {handedness}")

            # 2. Check if other fingers are curled
            fingers = ['INDEX', 'MIDDLE', 'RING', 'PINKY']
            fingers_curled = all([self.is_finger_curled(hand_landmarks, finger) for finger in fingers])

            #self.logger.debug(f"All other fingers curled: {fingers_curled}")

            # 3. Combine both conditions
            if thumb_is_above and fingers_curled:
                self.logger.debug("Thumbs-up gesture detected.")
                return True
            else:
                return False

            # Store features
            feature_dict = {
                'thumb_is_above': thumb_is_above,
                'fingers_curled': fingers_curled,
            }
            self.latest_features.append(feature_dict)

        except Exception as e:
            self.logger.error(f"Error determining thumbs-up gesture: {e}")
            self.thumbs_up_buffer.append(False)
            feature_dict = {
                'thumb_is_above': False,
                'fingers_curled': False,
            }
            self.latest_features.append(feature_dict)
            return False

    def is_thumbs_down(self, hand_landmarks, handedness):
        thumb_threshold = 0.1  # Optional: Define a threshold for consistency
        """
        Determines if the detected hand gesture is a thumbs-down based on thumb position
        relative to other fingers and other fingers being curled.
        
        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            handedness (str): 'Left' or 'Right' indicating the detected hand's side.
        
        Returns:
            bool: True if thumbs-down gesture is detected, False otherwise.
        """
        try:
            # Extract necessary landmarks for thumb
            thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
            thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
            thumb_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC]

            # Extract landmarks for other fingers' tips
            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

            # 1. Check if thumb tip is below other fingers' tips with threshold
            thumb_is_below = (
                thumb_tip.y - thumb_threshold > index_tip.y and
                thumb_tip.y - thumb_threshold > middle_tip.y and
                thumb_tip.y - thumb_threshold > ring_tip.y and
                thumb_tip.y - thumb_threshold > pinky_tip.y
            )

            #self.logger.debug(f"Thumb is below other fingers: {thumb_is_below} : {handedness}")

            # 2. Check if other fingers are curled
            fingers = ['INDEX', 'MIDDLE', 'RING', 'PINKY']
            fingers_curled = all([self.is_finger_curled(hand_landmarks, finger) for finger in fingers])

            #self.logger.debug(f"All other fingers curled: {fingers_curled}")

            # 3. Combine both conditions
            if thumb_is_below and fingers_curled:
                self.logger.debug("Thumbs-down gesture detected.")
                return True
            else:
                return False

            # Store features
            feature_dict = {
                'thumb_is_below': thumb_is_below,
                'fingers_curled': fingers_curled,
            }
            self.latest_features.append(feature_dict)

            
        except Exception as e:
            self.logger.error(f"Error determining thumbs-down gesture: {e}")
            feature_dict = {
                'thumb_is_below': False,
                'fingers_curled': False,
            }
            self.latest_features.append(feature_dict)
            self.thumbs_down_buffer.append(False)  # Ensure buffer is updated even on error
            return False

    def get_latest_features(self):
        """
        Returns the latest detected features for each hand in the current frame.
        
        Returns:
            list: A list of dictionaries containing feature data.
        """
        return self.latest_features

    def close(self):
        """
        Closes the MediaPipe Hands detector and releases resources.
        """
        try:
            self.hands.close()
            self.logger.debug("MediaPipe Hands closed successfully.")
        except Exception as e:
            self.logger.error(f"Error closing MediaPipe Hands: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
