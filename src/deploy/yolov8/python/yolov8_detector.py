# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv8 Object Detector Implementation
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
    scale_boxes_letterbox, process_box_dfl, xywh2xyxy
)
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class YOLOv8Detector(BaseModel):
    """
    YOLOv8 object detection model.

    Supports detection with various input sizes and configurations.
    """

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize YOLOv8 detector.

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
        """Load YOLOv8 ONNX model."""
        # Check if model file exists
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"YOLOv8 模型文件不存在: {self.model_path}"
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
                message=f"YOLOv8 模型加载失败: {self.model_path} - {e}"
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
                message=f"YOLOv8 模型初始化失败: {self.model_path} - {e}"
            )

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess image for YOLOv8 inference.

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

    @staticmethod
    def _postprocess_single_output(
        output: np.ndarray,
        input_shape: Tuple[int, int],
        conf_threshold: float,
        iou_threshold: float,
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        """Ultralytics single-output layout, e.g. [1, 84, 8400]."""
        if output.ndim == 3:
            # cn layout = [features, anchors]: features sit on the smaller trailing dim.
            # Pure size comparison, matching the C++ detector and yolov8_seg. Equal dims
            # are ambiguous — refuse rather than silently mis-parsing box/score channels.
            dim1, dim2 = output.shape[1], output.shape[2]
            if dim1 == dim2:
                raise ModelInferenceError(
                    model_name="YOLOv8",
                    reason=f"ambiguous output shape: {output.shape}",
                    message=f"YOLOv8 后处理错误: 输出维度歧义 {output.shape}",
                )
            data = output[0]
            layout_cn = dim1 < dim2
        elif output.ndim == 2:
            data = output
            layout_cn = False
        else:
            raise ModelInferenceError(
                model_name="YOLOv8",
                reason=f"unexpected output rank: {output.ndim}",
                message=f"YOLOv8 后处理错误: 不支持的输出维度 {output.ndim}",
            )

        # Ultralytics exports box xywh in input-pixel scale, so no normalization rescale.
        if layout_cn:
            num_anchors = data.shape[1]

            cls_scores = data[4:]
            class_ids = np.argmax(cls_scores, axis=0)
            scores = cls_scores[class_ids, np.arange(num_anchors)]
            mask = scores >= conf_threshold
            if not mask.any():
                return []

            cx = data[0, mask]
            cy = data[1, mask]
            w = data[2, mask]
            h = data[3, mask]
            boxes_xywh = np.stack([cx, cy, w, h], axis=1)
            scores = scores[mask]
            class_ids = class_ids[mask]
        else:
            num_anchors, num_features = data.shape
            if num_features < 5:
                raise ModelInferenceError(
                    model_name="YOLOv8",
                    reason=f"unexpected feature size: {num_features}",
                    message=f"YOLOv8 后处理错误: 特征维度 {num_features} 无效",
                )

            cls_scores = data[:, 4:]
            class_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(num_anchors), class_ids]
            mask = scores >= conf_threshold
            if not mask.any():
                return []

            boxes_xywh = data[mask, :4].copy()
            scores = scores[mask]
            class_ids = class_ids[mask]

        boxes_xyxy = xywh2xyxy(boxes_xywh)
        final_boxes, final_classes, final_scores = multiclass_nms(
            boxes_xyxy, class_ids, scores, iou_threshold
        )
        if final_boxes is None or len(final_boxes) == 0:
            return []

        final_boxes = scale_boxes_letterbox(final_boxes, ratio, pad, orig_shape)
        return [
            {
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "class_id": int(cls_id),
                "confidence": float(score),
            }
            for box, cls_id, score in zip(final_boxes, final_classes, final_scores)
        ]

    def _postprocess_dfl_outputs(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        """Multi-branch DFL heads (boxes + class conf per scale)."""
        boxes_list, scores_list, classes_conf_list = [], [], []
        default_branch = 3
        pair_per_branch = len(outputs) // default_branch

        for i in range(default_branch):
            boxes_list.append(process_box_dfl(outputs[pair_per_branch * i], self.input_shape))
            classes_conf_list.append(outputs[pair_per_branch * i + 1])
            scores_list.append(
                np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32)
            )

        all_boxes = np.concatenate([flatten_yolo_tensor(b) for b in boxes_list])
        all_classes_conf = np.concatenate([flatten_yolo_tensor(cc) for cc in classes_conf_list])
        all_scores = np.concatenate([flatten_yolo_tensor(s) for s in scores_list])

        boxes, classes, scores = filter_boxes_by_confidence(
            all_boxes, all_scores, all_classes_conf, self.conf_threshold
        )
        if len(boxes) == 0:
            return []

        final_boxes, final_classes, final_scores = multiclass_nms(
            boxes, classes, scores, self.iou_threshold
        )
        if final_boxes is None or len(final_boxes) == 0:
            return []

        final_boxes = scale_boxes_letterbox(final_boxes, ratio, pad, orig_shape)
        return [
            {
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                "class_id": int(cls_id),
                "confidence": float(score),
            }
            for box, cls_id, score in zip(final_boxes, final_classes, final_scores)
        ]

    def postprocess(
        self,
        outputs: List[np.ndarray],
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Postprocess YOLOv8 outputs.

        Auto-selects by output count:
        - 1 tensor: Ultralytics export [1, 84, 8400]
        - 6+ tensors: multi-branch DFL heads
        """
        if len(outputs) == 1:
            return self._postprocess_single_output(
                outputs[0],
                self.input_shape,
                self.conf_threshold,
                self.iou_threshold,
                orig_shape,
                ratio,
                pad,
            )
        return self._postprocess_dfl_outputs(outputs, orig_shape, ratio, pad)

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLOv8 inference on an image.

        Args:
            image: Input image in BGR format

        Returns:
            List of detections with bbox, class_id, confidence (empty list if none)
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
                model_name="YOLOv8",
                reason=f"图像预处理失败: {e}",
                message=f"YOLOv8 预处理错误: {e}"
            )

        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8",
                reason=f"模型推理失败: {e}",
                message=f"YOLOv8 推理错误: {e}"
            )

        # Postprocess
        try:
            detections = self.postprocess(outputs, orig_shape, ratio, pad)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8",
                reason=f"结果后处理失败: {e}",
                message=f"YOLOv8 后处理错误: {e}"
            )

        return detections




