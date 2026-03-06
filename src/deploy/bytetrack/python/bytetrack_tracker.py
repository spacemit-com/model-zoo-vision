# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
ByteTrack Tracker with YOLOv8 Integration
"""

import cv2
import numpy as np
from typing import List, Dict, Optional
from deploy.yolov8 import YOLOv8Detector
from .tracking.byte_tracker import BYTETracker
from core import BaseModel


class ByteTrackTracker(BaseModel):
    """
    ByteTrack Tracker integrated with YOLOv8 detector.

    Combines object detection and tracking for video analysis.
    Inherits from BaseModel for consistent interface.
    """

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 frame_rate: int = 30,
                 num_threads: int = 4,
                 lazy_load: bool = False,
                 **kwargs):
        """
        Initialize ByteTrack tracker.

        Args:
            model_path: Path to YOLOv8 ONNX model
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            frame_rate: Video frame rate
            num_threads: Number of threads for ONNX Runtime
            lazy_load: If True, delay model loading until first inference call
            **kwargs: Additional parameters (e.g., providers)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.frame_rate = frame_rate
        self.num_threads = num_threads
        self._detector_kwargs = kwargs

        # Initialize BaseModel
        super().__init__(model_path, lazy_load=lazy_load, **kwargs)

    def _load_model(self, **kwargs):
        """
        Load the YOLOv8 detector model and initialize ByteTrack tracker.

        Args:
            **kwargs: Additional parameters for model loading
        """
        # Initialize YOLOv8 detector with num_threads and providers
        self.detector = YOLOv8Detector(
            self.model_path,
            self.conf_threshold,
            self.iou_threshold,
            num_threads=self.num_threads,
            lazy_load=False,  # Load immediately since we're in _load_model
            **self._detector_kwargs
        )

        # Initialize ByteTrack tracker
        self.tracker = BYTETracker(frame_rate=self.frame_rate)

        # Save input shape for coordinate transformation
        self.input_shape = self.detector.input_shape

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for inference.

        For ByteTrack, preprocessing is handled by the detector,
        so we just return the image as-is here.

        Args:
            image: Input image in BGR format

        Returns:
            Preprocessed image (same as input for ByteTrack)
        """
        return image

    def infer(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection and tracking on the input image.

        Args:
            image: Input image in BGR format

        Returns:
            List of tracking results, each containing:
            - tlbr: [x1, y1, x2, y2] bounding box
            - track_id: Unique tracking ID
            - score: Detection confidence
            - class_id: Object class ID
        """
        # Ensure model is loaded
        self._ensure_model_loaded()
        # Get original image size
        img_h, img_w = image.shape[:2]

        # Run detection - YOLOv8Detector.infer() returns detections (list of dicts)
        detections = self.detector.infer(image)

        if detections is None or len(detections) == 0:
            # No detections: update tracker with empty to keep state, return empty list
            # No detections, update tracker with empty results
            output_results = np.empty((0, 6), dtype=np.float32)
            boxes = None
            classes = None
            scores = None
        else:
            # Extract boxes, classes, scores from detections
            boxes = np.array([det["bbox"] for det in detections], dtype=np.float32)
            classes = np.array([det["class_id"] for det in detections], dtype=np.int32)
            scores = np.array([det["confidence"] for det in detections], dtype=np.float32)

            # Prepare detection results format [x1, y1, x2, y2, conf, cls]
            output_results = np.zeros((len(boxes), 6), dtype=np.float32)
            output_results[:, :4] = boxes
            output_results[:, 4] = scores
            output_results[:, 5] = classes

        # Update tracker
        img_info = [img_h, img_w]
        img_size = [img_h, img_w]
        class_ids = classes if boxes is not None and len(boxes) > 0 else None

        tracked_stracks = self.tracker.update(output_results, img_info, img_size, class_ids=class_ids)

        # Format output
        results = []
        for track in tracked_stracks:
            tlbr = track.tlbr

            # Skip tracks with invalid (NaN/Inf) coordinates
            if np.any(np.isnan(tlbr)) or np.any(np.isinf(tlbr)):
                continue

            results.append({
                'tlbr': tlbr.tolist(),
                'track_id': track.track_id,
                'score': track.score,
                'class_id': track.class_id if hasattr(track, 'class_id') else 0
            })

        return results

    def track(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and track objects in the image.

        This is a convenience method that calls infer().
        Kept for backward compatibility.

        Args:
            image: Input image in BGR format

        Returns:
            List of tracking results, each containing:
            - tlbr: [x1, y1, x2, y2] bounding box
            - track_id: Unique tracking ID
            - score: Detection confidence
            - class_id: Object class ID
        """
        return self.infer(image)

    def draw_results(self, image: np.ndarray, results: List[Dict],
                    labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Draw tracking results on the image.

        Args:
            image: Input image
            results: Tracking results from track() or infer()
            labels: Optional list of class labels

        Returns:
            Image with drawn tracking results
        """
        result_image = image.copy()
        if not results:
            return result_image
        # Generate colors for different track IDs
        colors = self._generate_colors(100)

        for result in results:
            tlbr = result['tlbr']
            track_id = result['track_id']
            score = result['score']
            class_id = result['class_id']

            # Skip if any coordinate is NaN or Inf
            if any(np.isnan(tlbr)) or any(np.isinf(tlbr)):
                continue

            x1, y1, x2, y2 = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])

            # Get color based on track_id
            color = colors[track_id % len(colors)]

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # Prepare label text
            if labels and class_id < len(labels):
                label_text = f"{labels[class_id]} ID:{track_id} {score:.2f}"
            else:
                label_text = f"ID:{track_id} {score:.2f}"

            # Draw label background
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                result_image,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        return result_image

    def _generate_colors(self, num_colors: int) -> List[tuple]:
        """Generate distinct colors for visualization."""
        np.random.seed(42)
        colors = []
        for i in range(num_colors):
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            colors.append(color)
        return colors
