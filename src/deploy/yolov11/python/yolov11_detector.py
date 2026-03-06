# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv11 Object Detector Implementation
"""

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from typing import List, Tuple, Dict, Any
from pathlib import Path

from core import BaseModel
from common import (
    letterbox, multiclass_nms,
    flatten_yolo_tensor, filter_boxes_by_confidence,
    scale_boxes_letterbox, process_box_dfl
)
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class YOLOv11Detector(BaseModel):
    """
    YOLOv11 object detection model.
    Similar to YOLOv8 but with updated architecture.
    """

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize YOLOv11 detector.

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
        """Load YOLOv11 ONNX model."""
        # Check if model file exists
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"YOLOv11 模型文件不存在: {self.model_path}"
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
                message=f"YOLOv11 模型加载失败: {self.model_path} - {e}"
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
                message=f"YOLOv11 模型初始化失败: {self.model_path} - {e}"
            )

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess image for YOLOv11 inference.

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
        Postprocess YOLOv11 outputs.

        Args:
            outputs: Raw model outputs
            orig_shape: Original image shape (height, width)
            ratio: Scale ratio used in preprocessing
            pad: Padding (dw, dh) used in preprocessing
            **kwargs: Additional parameters

        Returns:
            List of detections
        """
        boxes_list, scores_list, classes_conf_list = [], [], []
        default_branch = 3
        pair_per_branch = len(outputs) // default_branch

        # Process each output branch
        for i in range(default_branch):
            boxes_list.append(process_box_dfl(outputs[pair_per_branch * i], self.input_shape))
            classes_conf_list.append(outputs[pair_per_branch * i + 1])
            scores_list.append(
                np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32)
            )

        # Flatten all outputs
        all_boxes = np.concatenate([flatten_yolo_tensor(b) for b in boxes_list])
        all_classes_conf = np.concatenate([flatten_yolo_tensor(cc) for cc in classes_conf_list])
        all_scores = np.concatenate([flatten_yolo_tensor(s) for s in scores_list])

        # Filter by confidence threshold
        boxes, classes, scores = filter_boxes_by_confidence(
            all_boxes, all_scores, all_classes_conf, self.conf_threshold
        )

        if len(boxes) == 0:
            return []

        # Apply NMS per class
        final_boxes, final_classes, final_scores = multiclass_nms(boxes, classes, scores, self.iou_threshold)
        if final_boxes is not None and len(final_boxes) > 0:
            final_boxes = scale_boxes_letterbox(final_boxes, ratio, pad, orig_shape)

        detections = []
        for i in range(len(final_boxes)):
            box = final_boxes[i]
            detections.append({
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "class_id": int(final_classes[i]),
                "confidence": float(final_scores[i])
            })

        return detections

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv11 inference on an image.

        Args:
            image: Input image in BGR format

        Returns:
            List of detections, each containing:
            - bbox: [x1, y1, x2, y2] coordinates (scaled to original image size)
            - class_id: Class index
            - confidence: Confidence score
        """
        # 确保模型已加载（延迟加载支持）
        self._ensure_model_loaded()

        # Save original image shape
        orig_shape = image.shape[:2]

        # Preprocess
        try:
            input_tensor, ratio, pad = self.preprocess(image)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv11",
                reason=f"图像预处理失败: {e}",
                message=f"YOLOv11 预处理错误: {e}"
            )

        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv11",
                reason=f"模型推理失败: {e}",
                message=f"YOLOv11 推理错误: {e}"
            )

        # Postprocess
        try:
            detections = self.postprocess(outputs, orig_shape, ratio, pad)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv11",
                reason=f"结果后处理失败: {e}",
                message=f"YOLOv11 后处理错误: {e}"
            )

        return detections
