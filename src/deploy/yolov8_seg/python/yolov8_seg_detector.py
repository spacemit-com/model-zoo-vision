# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv8-Seg Instance Segmentation Implementation
"""

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from core import BaseModel
from common import (
    letterbox, nms, sigmoid,
    flatten_yolo_tensor, scale_boxes_letterbox, process_box_dfl
)
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class YOLOv8SegDetector(BaseModel):
    """
    YOLOv8-Seg model for instance segmentation.
    Detects objects and generates segmentation masks.
    """

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize YOLOv8-Seg detector.

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
        """Load YOLOv8-Seg ONNX model."""
        # Check if model file exists
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"YOLOv8-Seg 模型文件不存在: {self.model_path}"
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
                message=f"YOLOv8-Seg 模型加载失败: {self.model_path} - {e}"
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
                message=f"YOLOv8-Seg 模型初始化失败: {self.model_path} - {e}"
            )

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Preprocess image for YOLOv8-Seg inference.

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
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Postprocess YOLOv8-Seg outputs.

        Args:
            outputs: Raw model outputs
            orig_shape: Original image shape (height, width)
            ratio: Scale ratio used in preprocessing
            pad: Padding (dw, dh) used in preprocessing
            **kwargs: Additional parameters

        Returns:
            Tuple of (detections, masks):
            - detections: List of detections, each containing bbox, class_id, confidence
            - masks: Optional array of segmentation masks
        """
        boxes, scores, classes_conf, seg_part = [], [], [], []
        default_branch = 3
        branch_element = 3
        output_proto = outputs[-1]
        for i in range(default_branch):
            boxes.append(process_box_dfl(outputs[branch_element * i], self.input_shape))
            classes_conf.append(outputs[branch_element * i + 1])
            scores.append(np.ones_like(outputs[branch_element * i + 1][:, :1, :, :], dtype=np.float32))
            seg_part.append(outputs[branch_element * default_branch + i])

        all_boxes = np.concatenate([flatten_yolo_tensor(b) for b in boxes])
        all_classes_conf = np.concatenate([flatten_yolo_tensor(c) for c in classes_conf])
        all_seg_part = np.concatenate([flatten_yolo_tensor(p) for p in seg_part])

        boxes, classes, scores, seg_part = self._filter_boxes(all_boxes, all_classes_conf, all_seg_part)

        if boxes.size == 0:
            return [], None

        nboxes, nclasses, nscores, nseg_part = [], [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b = boxes[inds]
            c_arr = classes[inds]
            s = scores[inds]
            sp = seg_part[inds]
            keep = nms(b, s, self.iou_threshold)

            if len(keep) != 0:
                nboxes.append(b[keep])
                nclasses.append(c_arr[keep])
                nscores.append(s[keep])
                nseg_part.append(sp[keep])

        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)
        seg_part = np.concatenate(nseg_part)

        boxes = scale_boxes_letterbox(boxes, ratio, pad, orig_shape)

        masks = None
        if output_proto is not None and len(seg_part) > 0:
            masks = self._process_masks(output_proto, seg_part, boxes, orig_shape)

        detections = []
        for i in range(len(boxes)):
            detections.append({
                "bbox": [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])],
                "class_id": int(classes[i]),
                "confidence": float(scores[i])
            })

        return detections, masks

    def infer(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Run YOLOv8-Seg inference on an image.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (detections, masks):
            - detections: List of detections, each containing bbox, class_id, confidence
            - masks: Optional array of segmentation masks
        """
        # Save original image shape
        orig_shape = image.shape[:2]

        # Preprocess
        try:
            input_tensor, ratio, pad = self.preprocess(image)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8-Seg",
                reason=f"图像预处理失败: {e}",
                message=f"YOLOv8-Seg 预处理错误: {e}"
            )

        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8-Seg",
                reason=f"模型推理失败: {e}",
                message=f"YOLOv8-Seg 推理错误: {e}"
            )

        # Postprocess
        try:
            detections, masks = self.postprocess(outputs, orig_shape, ratio, pad)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv8-Seg",
                reason=f"结果后处理失败: {e}",
                message=f"YOLOv8-Seg 后处理错误: {e}"
            )

        return detections, masks

    def _filter_boxes(self, boxes: np.ndarray, classes_conf: np.ndarray,
                     seg_part: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter boxes by confidence.

        Args:
            boxes: Box coordinates, shape (N, 4)
            classes_conf: Class probabilities, shape (N, num_classes)
            seg_part: Segmentation part, shape (N, seg_dim)

        Returns:
            Filtered (boxes, classes, scores, seg_part)
        """
        # Get max class score and class index
        class_max_score = np.max(classes_conf, axis=-1)
        classes = np.argmax(classes_conf, axis=-1)

        # Filter by confidence threshold
        pred_box_pos = np.where(class_max_score >= self.conf_threshold)
        filtered_scores = class_max_score[pred_box_pos]
        filtered_boxes = boxes[pred_box_pos]
        filtered_classes = classes[pred_box_pos]
        filtered_seg_part = seg_part[pred_box_pos]

        return filtered_boxes, filtered_classes, filtered_scores, filtered_seg_part

    def _process_masks(self, protos: np.ndarray, mask_coeffs: np.ndarray,
                      boxes: np.ndarray, orig_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Process segmentation masks."""
        if len(boxes) == 0 or len(mask_coeffs) == 0:
            return None

        protos = protos.squeeze(0)
        mask_dim, mask_h, mask_w = protos.shape
        protos_flat = protos.reshape(mask_dim, -1)
        masks = np.dot(mask_coeffs, protos_flat)
        masks = masks.reshape(-1, mask_h, mask_w)
        masks = sigmoid(masks)

        orig_h, orig_w = orig_shape
        gain = min(mask_h / orig_h, mask_w / orig_w)
        pad_w = mask_w - orig_w * gain
        pad_h = mask_h - orig_h * gain
        pad_w /= 2
        pad_h /= 2

        top = int(round(pad_h - 0.1)) if pad_h > 0 else 0
        left = int(round(pad_w - 0.1)) if pad_w > 0 else 0
        bottom = mask_h - int(round(pad_h + 0.1)) if pad_h > 0 else mask_h
        right = mask_w - int(round(pad_w + 0.1)) if pad_w > 0 else mask_w

        top = max(0, min(top, mask_h - 1))
        left = max(0, min(left, mask_w - 1))
        bottom = max(top + 1, min(bottom, mask_h))
        right = max(left + 1, min(right, mask_w))

        masks_scaled = []
        for i in range(len(masks)):
            mask = masks[i]
            if top < bottom and left < right:
                mask_cropped = mask[top:bottom, left:right]
            else:
                mask_cropped = mask
            mask_resized = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            masks_scaled.append(mask_resized)

        masks_scaled = np.array(masks_scaled)

        masks_cropped = []
        for i in range(len(masks_scaled)):
            mask = masks_scaled[i]
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, min(int(x1), orig_w - 1))
            y1 = max(0, min(int(y1), orig_h - 1))
            x2 = max(x1 + 1, min(int(x2), orig_w))
            y2 = max(y1 + 1, min(int(y2), orig_h))

            mask_cropped_arr = np.zeros((orig_h, orig_w), dtype=np.float32)
            mask_cropped_arr[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
            masks_cropped.append(mask_cropped_arr)

        masks_cropped = np.array(masks_cropped)
        masks_cropped = (masks_cropped > 0.5).astype(np.float32)

        return masks_cropped

