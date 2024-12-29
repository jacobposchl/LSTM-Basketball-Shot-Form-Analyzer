# pose_estimation.py

import mediapipe as mp
import cv2
import logging

class PoseEstimator:
    """
    A class to handle pose estimation using MediaPipe Pose.
    """
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        """
        Initializes the MediaPipe Pose estimator.

        Args:
            min_detection_confidence (float): Minimum confidence value ([0.0, 1.0]) for pose detection to be considered successful.
            min_tracking_confidence (float): Minimum confidence value ([0.0, 1.0]) for the pose landmarks to be considered tracked successfully.
        """
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.logger = logging.getLogger(__name__)

    def process_frame(self, rgb_frame):
        """
        Processes an RGB frame to detect pose landmarks.

        Args:
            rgb_frame (numpy.ndarray): The input frame in RGB format.

        Returns:
            mediapipe.framework.formats.pose_pb2.NormalizedLandmarkList: The detected pose landmarks.
        """
        try:
            results = self.pose.process(rgb_frame)
            return results
        except Exception as e:
            self.logger.error(f"Error processing frame for pose estimation: {e}")
            return None

    def draw_landmarks(self, frame, pose_landmarks):
        """
        Draws pose landmarks on the given frame.

        Args:
            frame (numpy.ndarray): The input frame in BGR format.
            pose_landmarks (mediapipe.framework.formats.pose_pb2.NormalizedLandmarkList): The detected pose landmarks.
        """
        try:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
        except Exception as e:
            self.logger.error(f"Error drawing pose landmarks: {e}")

    def close(self):
        """
        Closes the MediaPipe Pose estimator and releases resources.
        """
        self.pose.close()
