# hand_detection.py

import mediapipe as mp
import cv2
import math
import logging
from collections import deque

class HandDetector:
    """
    A class to handle hand detection and thumbs-up gesture recognition using MediaPipe Hands.
    """
    def __init__(self, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.7):
        """
        Initializes the MediaPipe Hands detector.

        Args:
            max_num_hands (int): Maximum number of hands to detect.
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for hand detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for the hand landmarks to be considered tracked successfully.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.logger = logging.getLogger(__name__)

    def process_frame(self, frame_rgb):
        """
        Processes an RGB frame to detect hand landmarks.

        Args:
            frame_rgb (numpy.ndarray): The input frame in RGB format.

        Returns:
            mediapipe.framework.formats.hand_landmark_pb2.NormalizedLandmarkList: The detected hand landmarks.
        """
        try:
            results = self.hands.process(frame_rgb)
            return results
        except Exception as e:
            self.logger.error(f"Error processing frame for hand detection: {e}")
            return None

    def draw_landmarks(self, frame, hand_landmarks):
        """
        Draws hand landmarks on the given frame.

        Args:
            frame (numpy.ndarray): The input frame in BGR format.
            hand_landmarks (mediapipe.framework.formats.hand_landmark_pb2.NormalizedLandmarkList): The detected hand landmarks.
        """
        try:
            if hand_landmarks:
                for hand_landmark in hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmark,
                        self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=2)
                    )
        except Exception as e:
            self.logger.error(f"Error drawing hand landmarks: {e}")

    def calculate_angle(self, a, b, c):
        """
        Calculates the angle between three points (in degrees).

        Args:
            a, b, c: Each is a landmark with x, y, z coordinates.

        Returns:
            Angle in degrees.
        """
        ab = [b.x - a.x, b.y - a.y]
        cb = [b.x - c.x, b.y - c.y]

        dot_product = ab[0]*cb[0] + ab[1]*cb[1]
        magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
        magnitude_cb = math.sqrt(cb[0]**2 + cb[1]**2)

        if magnitude_ab * magnitude_cb == 0:
            return 0
        angle = math.acos(dot_product / (magnitude_ab * magnitude_cb))
        return math.degrees(angle)

    def calculate_finger_pip_angle(self, hand_landmarks, finger):
        """
        Calculates the angle at the PIP joint for a given finger.

        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            finger: String indicating the finger name ('index', 'middle', 'ring', 'pinky').

        Returns:
            Angle at PIP joint in degrees.
        """
        if finger == 'index':
            mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
            pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
            dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_DIP]
        elif finger == 'middle':
            mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
        elif finger == 'ring':
            mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
            pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
            dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_DIP]
        elif finger == 'pinky':
            mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
            pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
            dip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_DIP]
        else:
            return 180  # Neutral angle if finger not recognized

        angle = self.calculate_angle(mcp, pip, dip)
        return angle

    def get_hand_orientation(self, hand_landmarks):
        """
        Estimates the hand's orientation based on landmarks.

        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.

        Returns:
            A tuple (pitch, roll) in degrees.
        """
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

        # Vector from wrist to MCP
        vector_x = mcp.x - wrist.x
        vector_y = mcp.y - wrist.y
        vector_z = mcp.z - wrist.z

        # Calculate pitch and roll angles
        pitch = math.degrees(math.atan2(vector_y, vector_z)) if vector_z != 0 else 0
        roll = math.degrees(math.atan2(vector_x, vector_z)) if vector_z != 0 else 0

        return pitch, roll

    def is_hand_upright(self, hand_landmarks, threshold=30):
        """
        Checks if the hand is upright based on pitch and roll angles.

        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            threshold: Maximum allowed deviation in degrees from upright position.

        Returns:
            True if the hand is upright, False otherwise.
        """
        pitch, roll = self.get_hand_orientation(hand_landmarks)
        return abs(pitch) < threshold and abs(roll) < threshold

    def is_thumbs_up(self, hand_landmarks, handedness, hip_center_y):
        """
        Determines if the detected hand gesture is a thumbs-up based on the thumb's position relative to other fingers
        and the hip center.

        Args:
            hand_landmarks: The hand landmarks detected by MediaPipe.
            handedness: 'Left' or 'Right' indicating the detected hand's side.
            hip_center_y: The y-coordinate of the hip center.

        Returns:
            True if thumbs-up gesture is detected, False otherwise.
        """
        # Extract necessary landmarks
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Check if thumb is above other fingers
        thumb_is_above = thumb_tip.y < index_tip.y and thumb_tip.y < middle_tip.y and thumb_tip.y < ring_tip.y and thumb_tip.y < pinky_tip.y

        # Check if thumb is above hip center
        thumb_above_hip = thumb_tip.y < hip_center_y

        # Calculate angles at PIP joints for other fingers
        index_angle = self.calculate_finger_pip_angle(hand_landmarks, 'index')
        middle_angle = self.calculate_finger_pip_angle(hand_landmarks, 'middle')
        ring_angle = self.calculate_finger_pip_angle(hand_landmarks, 'ring')
        pinky_angle = self.calculate_finger_pip_angle(hand_landmarks, 'pinky')

        # Define threshold for curled fingers
        curled_angle_threshold = 90  # Degrees

        # Determine if fingers are curled based on angles
        index_curled = index_angle < curled_angle_threshold
        middle_curled = middle_angle < curled_angle_threshold
        ring_curled = ring_angle < curled_angle_threshold
        pinky_curled = pinky_angle < curled_angle_threshold

        fingers_curled = index_curled and middle_curled and ring_curled and pinky_curled

        return thumb_is_above and thumb_above_hip and fingers_curled

    def close(self):
        """
        Closes the MediaPipe Hands detector and releases resources.
        """
        self.hands.close()
