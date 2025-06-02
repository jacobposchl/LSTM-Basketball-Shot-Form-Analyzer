# yolo_detection.py

import torch
import cv2
import logging
import os
from config import YOLO_WEIGHTS_PATH, YOLOV5_DIR, YOLO_CONFIDENCE_THRESHOLD

class YOLODetector:
    """
    A class to perform YOLOv5 object detection within specified Regions of Interest (ROIs).
    """

    def __init__(self, weights_path=YOLO_WEIGHTS_PATH, confidence_threshold=YOLO_CONFIDENCE_THRESHOLD, repo_path=YOLOV5_DIR):
        """
        Initializes the YOLODetector with the specified weights and confidence threshold.

        Args:
            weights_path (str): Path to the YOLOv5 weights file (e.g., 'Weights/best.pt').
            confidence_threshold (float): Minimum confidence threshold for detections.
            repo_path (str): Path to the cloned YOLOv5 repository.
        """
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

        # Check if the YOLOv5 repository exists
        if not os.path.isdir(repo_path):
            self.logger.error(f"YOLOv5 repository not found at {repo_path}. Please clone it first.")
            raise FileNotFoundError(f"YOLOv5 repository not found at {repo_path}.")

        # Load the YOLOv5 model using torch.hub
        try:
            self.model = torch.hub.load(
                repo_or_dir=repo_path,  # Path to the cloned repository
                model='custom',
                path=self.weights_path,  # Path to custom weights
                source='local'  # Specify that the source is local
            )
            self.model.to(torch.device("cuda:0")) 
            self.model.eval()
            self.logger.info(f"Successfully loaded YOLOv5 model from {self.weights_path}")
        except Exception as e:
            self.logger.error(f"Error loading YOLOv5 model: {e}")
            raise e

    def run_inference(self, frame):
        """
        Runs YOLOv5 inference on the provided frame.

        Args:
            frame (numpy.ndarray): The image/frame to run detection on.

        Returns:
            list: A list of detections, each represented as a dictionary with keys:
                  'bbox' (list of [x1, y1, x2, y2]),
                  'confidence' (float),
                  'class_id' (int),
                  'class_name' (str)
        """
        try:
            # Perform inference
            results = self.model(frame)

            detections = []
            for det in results.xyxy[0]:  # x1, y1, x2, y2, confidence, class
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)
                conf = float(conf)
                if conf < self.confidence_threshold:
                    continue  # Skip detections below the confidence threshold
                class_name = self.model.names[cls_id] if cls_id < len(self.model.names) else 'Unknown'

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': class_name
                })

            return detections

        except Exception as e:
            self.logger.error(f"Error during YOLOv5 inference: {e}")
            return []

    def draw_detections(self, frame, detections):
        """
        Draws bounding boxes and labels on the frame for each detection.

        Args:
            frame (numpy.ndarray): The image/frame to draw on.
            detections (list): List of detection dictionaries.
        """
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class_id']
            class_name = detection['class_name']

            # Define colors for different classes (optional)
            color = (0, 0, 255)  # Red for ball; modify as needed

            # Draw bounding box
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Prepare label with class name and confidence
            label = f"{class_name}: {confidence:.2f}"

            # Calculate label size
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_y_min = max(bbox[1] - label_size[1], 0)
            label_x_min = bbox[0]
            label_background_top_left = (label_x_min, label_y_min - label_size[1])
            label_background_bottom_right = (label_x_min + label_size[0], label_y_min + 5)

            # Draw label background
            cv2.rectangle(frame, label_background_top_left, label_background_bottom_right, color, cv2.FILLED)

            # Put label text above the bounding box
            cv2.putText(frame, label, (bbox[0], label_y_min),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
