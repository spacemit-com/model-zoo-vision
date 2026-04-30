# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv5 Object Detector Implementation
"""

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from core import BaseModel
from common import letterbox, nms, xywh2xyxy, scale_coords
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class YOLOv5Detector(BaseModel):
    """
    YOLOv5 object detection model.

    Supports detection with various input sizes and configurations.
    Handles both sigmoid and non-sigmoid outputs automatically.
    """

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize YOLOv5 detector.

        Args:
            model_path: Path to ONNX model file
            conf_threshold: Confidence threshold for filtering detections
            iou_threshold: IoU threshold for NMS
            num_threads: Number of threads for ONNX Runtime
            **kwargs: Additional parameters
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_threads = num_threads
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs):
        """Load YOLOv5 ONNX model."""
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"YOLOv5 模型文件不存在: {self.model_path}"
            )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads

        try:
            providers = kwargs.get('providers', ['SpaceMITExecutionProvider'])
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=str(e),
                message=f"YOLOv5 模型加载失败: {self.model_path} - {e}"
            )

        try:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = tuple(self.session.get_inputs()[0].shape[2:4])
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=f"无法获取模型输入/输出信息: {e}",
                message=f"YOLOv5 模型初始化失败: {self.model_path} - {e}"
            )

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess image for YOLOv5 inference.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (preprocessed_tensor, ratio, pad)
        """
        img_resized, r, (dw, dh) = letterbox(image, self.input_shape)

        # BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # HWC to CHW
        img_chw = np.transpose(img_normalized, (2, 0, 1))

        # Add batch dimension
        img_batch = np.expand_dims(img_chw, axis=0)

        return img_batch, r, (dw, dh)

    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Postprocess YOLOv5 outputs.

        Args:
            outputs: Raw model outputs
            orig_shape: Original image shape (H, W)
            ratio: Scale ratio from letterbox
            pad: Padding (dw, dh) from letterbox
            **kwargs: Additional parameters

        Returns:
            List of detection dictionaries
        """
        if not outputs or len(outputs) == 0:
            return []

        output = outputs[0]

        # Handle different output shapes
        if output.ndim == 3:
            # Shape: [1, num_boxes, features] or [1, features, num_boxes]
            if output.shape[1] <= 256 and output.shape[2] > 256:
                # Layout: [1, features, num_boxes] -> transpose
                output = output.transpose(0, 2, 1)
            output = output[0]  # Remove batch dimension
        elif output.ndim == 2:
            # Shape: [num_boxes, features]
            pass
        else:
            raise ValueError(f"Unexpected YOLOv5 output shape: {output.shape}")

        num_boxes, features = output.shape

        if features < 5:
            raise ValueError(f"Unexpected YOLOv5 output features: {features}")

        # Detect if model has objectness score (85+ features for COCO)
        has_objectness = (features >= 85)
        cls_start = 5 if has_objectness else 4
        num_classes = features - cls_start

        # Auto-detect if sigmoid is needed by probing first 200 boxes
        probe_n = min(num_boxes, 200)
        probe_data = output[:probe_n]

        need_sigmoid = False
        need_scale_xywh = False

        # Check confidence range
        if has_objectness:
            obj_vals = probe_data[:, 4]
            if obj_vals.max() > 1.5 or obj_vals.min() < -0.1:
                need_sigmoid = True

        if num_classes > 0:
            cls_vals = probe_data[:, cls_start:]
            if cls_vals.max() > 1.5 or cls_vals.min() < -0.1:
                need_sigmoid = True

        # Check if xywh needs scaling
        xywh_vals = probe_data[:, :4]
        if np.abs(xywh_vals).max() <= 1.5:
            need_scale_xywh = True

        # Extract objectness scores
        if has_objectness:
            obj_scores = output[:, 4]
            if need_sigmoid:
                obj_scores = self._sigmoid(obj_scores)
        else:
            obj_scores = np.ones(num_boxes, dtype=np.float32)

        # Extract class scores
        class_scores = output[:, cls_start:]
        if need_sigmoid:
            class_scores = self._sigmoid(class_scores)

        # Compute final confidence scores
        if has_objectness:
            confidences = class_scores * obj_scores[:, np.newaxis]
        else:
            confidences = class_scores

        # Get best class for each box
        class_ids = np.argmax(confidences, axis=1)
        scores = confidences[np.arange(num_boxes), class_ids]

        # Filter by confidence threshold
        mask = scores > self.conf_threshold
        if not mask.any():
            return []

        scores = scores[mask]
        class_ids = class_ids[mask]
        boxes_xywh = output[mask, :4]

        # Scale xywh if needed
        if need_scale_xywh:
            boxes_xywh[:, 0] *= self.input_shape[1]  # x * width
            boxes_xywh[:, 1] *= self.input_shape[0]  # y * height
            boxes_xywh[:, 2] *= self.input_shape[1]  # w * width
            boxes_xywh[:, 3] *= self.input_shape[0]  # h * height

        # Convert xywh to xyxy
        boxes_xyxy = xywh2xyxy(boxes_xywh)

        # Apply NMS
        keep_indices = nms(boxes_xyxy, scores, self.iou_threshold)

        if len(keep_indices) == 0:
            return []

        boxes_xyxy = boxes_xyxy[keep_indices]
        scores = scores[keep_indices]
        class_ids = class_ids[keep_indices]

        # Scale boxes back to original image size
        input_size = (self.input_shape[1], self.input_shape[0])  # (W, H)
        boxes_xyxy = scale_coords(input_size, boxes_xyxy, orig_shape)

        # Format results
        results = []
        for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
            results.append({
                'bbox': box.tolist(),
                'confidence': float(score),
                'class_id': int(class_id)
            })

        return results

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv5 inference on an image.

        Args:
            image: Input image in BGR format

        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2] in original image coordinates
            - confidence: detection confidence score
            - class_id: class index
        """
        self._ensure_model_loaded()

        orig_shape = image.shape[:2]

        try:
            input_tensor, ratio, pad = self.preprocess(image)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5",
                reason=f"图像预处理失败: {e}",
                message=f"YOLOv5 预处理错误: {e}"
            )

        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5",
                reason=f"模型推理失败: {e}",
                message=f"YOLOv5 推理错误: {e}"
            )

        try:
            detections = self.postprocess(outputs, orig_shape, ratio, pad)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5",
                reason=f"结果后处理失败: {e}",
                message=f"YOLOv5 后处理错误: {e}"
            )

        return detections
