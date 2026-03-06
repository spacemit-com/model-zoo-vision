# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv8-Pose Keypoint Detection Implementation
"""

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from typing import List, Dict, Any, Tuple
from pathlib import Path

from core import BaseModel
from common import (
    letterbox, nms
)
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)

class YOLOv8PoseDetector(BaseModel):
    """
    YOLOv8-Pose model for human pose estimation.
    Detects persons and their 17 keypoints.
    """

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize YOLOv8-Pose detector.

        Args:
            model_path: Path to ONNX model file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            num_threads: Number of threads
            **kwargs: Additional parameters
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_threads = num_threads
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs):
        """Load YOLOv8-Pose ONNX model."""
        # Check if model file exists
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"YOLOv8-Pose 模型文件不存在: {self.model_path}"
            )

        # Create session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads

        # Load model
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
                message=f"YOLOv8-Pose 模型加载失败: {self.model_path} - {e}"
            )

        # Get input/output info
        try:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = tuple(self.session.get_inputs()[0].shape[2:4])
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=f"无法获取模型输入/输出信息: {e}",
                message=f"YOLOv8-Pose 模型初始化失败: {self.model_path} - {e}"
            )

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess image for YOLOv8-Pose inference.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (preprocessed_tensor, ratio, pad):
            - preprocessed_tensor: Preprocessed tensor ready for inference
            - ratio: Scale ratio used for resizing
            - pad: Padding (dw, dh) applied
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

    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Postprocess YOLOv8-Pose outputs.

        Args:
            outputs: Raw model outputs
            orig_shape: Original image shape (height, width)
            ratio: Scale ratio used in preprocessing
            pad: Padding (dw, dh) used in preprocessing
            **kwargs: Additional parameters

        Returns:
            List of detections, each containing:
            - box: [x1, y1, x2, y2] coordinates (scaled to original image size)
            - score: Confidence score
            - keypoints: List of 17 keypoints, each as (x, y, visibility)
        """
        output = outputs[0].squeeze()
        input_H, input_W = self.input_shape
        dw, dh = pad

        num_detections = output.shape[1]
        if num_detections == 0:
            return []

        cx = output[0, :]
        cy = output[1, :]
        w = output[2, :]
        h = output[3, :]

        x1 = np.maximum(0, (cx - w / 2).astype(np.int32))
        y1 = np.maximum(0, (cy - h / 2).astype(np.int32))
        x2 = np.minimum(input_W, (cx + w / 2).astype(np.int32))
        y2 = np.minimum(input_H, (cy + h / 2).astype(np.int32))

        scores = output[4, :]
        mask = scores >= self.conf_threshold

        if np.sum(mask) == 0:
            return []

        valid_indices = np.where(mask)[0]
        x1 = x1[valid_indices]
        y1 = y1[valid_indices]
        x2 = x2[valid_indices]
        y2 = y2[valid_indices]
        scores = scores[valid_indices]

        keypoints = output[5:, valid_indices].T
        keypoints = keypoints.reshape(-1, 17, 3)

        boxes = np.column_stack((x1, y1, x2, y2))
        keep_indices = nms(boxes, scores, self.iou_threshold)

        if len(keep_indices) == 0:
            return []

        x1 = ((x1[keep_indices] - dw) / ratio).astype(np.int32)
        y1 = ((y1[keep_indices] - dh) / ratio).astype(np.int32)
        x2 = ((x2[keep_indices] - dw) / ratio).astype(np.int32)
        y2 = ((y2[keep_indices] - dh) / ratio).astype(np.int32)

        adjusted_keypoints = []
        for kp in keypoints[keep_indices]:
            adjusted_kp = []
            for point in kp:
                x = int((point[0] - dw) / ratio)
                y = int((point[1] - dh) / ratio)
                vis = point[2]
                adjusted_kp.append((x, y, vis))
            adjusted_keypoints.append(adjusted_kp)

        detections = []
        for i in range(len(keep_indices)):
            detections.append({
                'box': [x1[i], y1[i], x2[i], y2[i]],
                'score': float(scores[keep_indices[i]]),
                'keypoints': adjusted_keypoints[i]
            })

        return detections

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8-Pose inference on an image.

        Args:
            image: Input image in BGR format

        Returns:
            List of detections, each containing:
            - box: [x1, y1, x2, y2] coordinates (scaled to original image size)
            - score: Confidence score
            - keypoints: List of 17 keypoints, each as (x, y, visibility)
        """
        # Save original image shape
        orig_shape = image.shape[:2]

        # Preprocess
        try:
            input_tensor, ratio, pad = self.preprocess(image)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8-Pose",
                reason=f"图像预处理失败: {e}",
                message=f"YOLOv8-Pose 预处理错误: {e}"
            )

        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8-Pose",
                reason=f"模型推理失败: {e}",
                message=f"YOLOv8-Pose 推理错误: {e}"
            )

        # Postprocess
        try:
            detections = self.postprocess(outputs, orig_shape, ratio, pad)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8-Pose",
                reason=f"结果后处理失败: {e}",
                message=f"YOLOv8-Pose 后处理错误: {e}"
            )
        return detections

