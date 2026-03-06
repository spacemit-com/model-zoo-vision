# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
OCSort Tracker (BaseModel style)

This wraps the existing OC-SORT python implementation and a YOLOv8 detector
into a factory-loadable model, similar to ByteTrack.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core import BaseModel
from deploy.yolov8 import YOLOv8Detector
from core.python.vision_model_exceptions import (
    ModelLoadError,
    ModelInferenceError,
)
from .tracking.ocsort import OCSort


class OCSortTracker(BaseModel):
    """
    OC-SORT tracker integrated with YOLOv8 detector.

    `track(image)` returns a list of dicts:
      - tlbr: [x1,y1,x2,y2] - bounding box coordinates
      - track_id: int - unique tracking ID
      - score: float - detection confidence score
      - class_id: int - object class ID
    """

    def __init__(
        self,
        model_path: str,
        # detector params
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        num_threads: int = 4,
        # tracker params
        track_thresh: float = 0.6,  # maps to det_thresh
        track_iou_thresh: float = 0.3,
        track_buffer: int = 60,  # maps to max_age
        min_hits: int = 3,
        delta_t: int = 3,
        asso_func: str = "iou",
        inertia: float = 0.2,
        use_byte: bool = False,
        **kwargs,
    ):
        # Store parameters for detector initialization
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_threads = num_threads

        # Store tracker parameters
        self.track_thresh = float(track_thresh)
        self.track_iou_thresh = float(track_iou_thresh)
        self.track_buffer = int(track_buffer)
        self.min_hits = int(min_hits)
        self.delta_t = int(delta_t)
        self.asso_func = str(asso_func)
        self.inertia = float(inertia)
        self.use_byte = bool(use_byte)

        # Detector will be initialized in _load_model
        self.detector: Optional[YOLOv8Detector] = None

        # OC-SORT tracker instance (pure python, no model loading needed)
        self.tracker = OCSort(
            det_thresh=self.track_thresh,
            max_age=self.track_buffer,
            min_hits=self.min_hits,
            iou_threshold=self.track_iou_thresh,
            delta_t=self.delta_t,
            asso_func=self.asso_func,
            inertia=self.inertia,
            use_byte=self.use_byte,
        )

        # Initialize BaseModel (will call _load_model if not lazy_load)
        super().__init__(model_path=model_path, lazy_load=kwargs.get('lazy_load', False), **kwargs)

    def _load_model(self, **kwargs):
        """
        Load YOLOv8 detection model.

        This method initializes the YOLOv8 detector which will be used for detection.
        """
        try:
            # Initialize YOLOv8 detector
            self.detector = YOLOv8Detector(
                model_path=self.model_path,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
                num_threads=self.num_threads,
                **kwargs
            )

            # Save input shape from detector
            self.input_shape = self.detector.input_shape

        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=str(e),
                message=f"OCSort detector (YOLOv8) 加载失败: {e}"
            ) from e

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess image using YOLOv8 detector's preprocessing.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (preprocessed_image, ratio, pad) as returned by YOLOv8Detector.preprocess
        """
        self._ensure_model_loaded()
        return self.detector.preprocess(image)

    def postprocess(self, outputs: Any, **kwargs) -> Any:
        """
        Postprocess is not used directly in tracking.
        Detection postprocessing is handled by YOLOv8Detector.
        """
        # This method is required by BaseModel but not used in tracking flow
        # Detection postprocessing is done by detector.infer()
        pass

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run detection + OC-SORT tracking on a single frame.

        Args:
            image: Input image in BGR format

        Returns:
            List of tracking results, each containing:
            - tlbr: [x1, y1, x2, y2] bounding box coordinates
            - track_id: Unique tracking ID
            - score: Detection confidence score
            - class_id: Object class ID
        """
        self._ensure_model_loaded()

        # Run YOLOv8 detection
        try:
            detections = self.detector.infer(image)
        except Exception as e:
            raise ModelInferenceError(
                model_name="OCSortTracker",
                reason=f"检测失败: {e}",
                message=f"OCSort 检测错误: {e}"
            ) from e

        results: List[Dict[str, Any]] = []

        if detections is None or len(detections) == 0:
            # 没有检测结果，使用空数组更新追踪器以保持状态
            self.tracker.update_public(
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32)
            )
            return results

        # Extract boxes, classes, scores from detections
        boxes = np.array([det["bbox"] for det in detections], dtype=np.float32)
        classes = np.array([det["class_id"] for det in detections], dtype=np.int32)
        scores = np.array([det["confidence"] for det in detections], dtype=np.float32)

        # Update OC-SORT tracker
        try:
            online = self.tracker.update_public(boxes, classes, scores)
        except Exception as e:
            raise ModelInferenceError(
                model_name="OCSortTracker",
                reason=f"追踪失败: {e}",
                message=f"OCSort 追踪错误: {e}"
            ) from e

        if online is None or len(online) == 0:
            return results

        # online: [x1,y1,x2,y2,id,cate,pad]
        for t in online:
            x1, y1, x2, y2 = map(float, t[:4])
            track_id = int(t[4])
            class_id = int(t[5]) if len(t) > 5 else 0

            # Find corresponding score from original detections
            score = 1.0
            for det in detections:
                if det["class_id"] == class_id:
                    # Try to match by position (simple heuristic)
                    det_bbox = det["bbox"]
                    if abs(det_bbox[0] - x1) < 5 and abs(det_bbox[1] - y1) < 5:
                        score = det["confidence"]
                        break

            results.append({
                "tlbr": [x1, y1, x2, y2],
                "track_id": track_id,
                "score": score,
                "class_id": class_id,
            })

        return results

    def track(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Alias for infer() method for consistency with ByteTrackTracker.

        Args:
            image: Input image in BGR format

        Returns:
            List of tracking results (same format as infer())
        """
        return self.infer(image)

    def draw_results(
        self,
        image: np.ndarray,
        results: List[Dict[str, Any]],
        labels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Simple visualization similar to ByteTrackTracker.draw_results()."""
        out = image.copy()
        if not results:
            return out
        colors = self._generate_colors(100)
        for r in results:
            tlbr = r.get("tlbr")
            if not tlbr or len(tlbr) != 4:
                continue
            x1, y1, x2, y2 = map(int, tlbr)
            track_id = int(r.get("track_id", -1))
            class_id = int(r.get("class_id", 0))
            score = float(r.get("score", 0.0))
            color = colors[track_id % len(colors)] if track_id >= 0 else (0, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            if labels and 0 <= class_id < len(labels):
                text = f"{labels[class_id]} ID:{track_id} {score:.2f}"
            else:
                text = f"ID:{track_id} {score:.2f}"
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, max(0, y1 - th - bl - 6)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return out

    @staticmethod
    def _generate_colors(num_colors: int) -> List[tuple]:
        np.random.seed(42)
        return [tuple(map(int, np.random.randint(0, 255, 3))) for _ in range(num_colors)]
