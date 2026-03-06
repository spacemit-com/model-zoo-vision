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
from typing import List, Tuple, Any, Dict
from pathlib import Path

from core import BaseModel
from common import (
    xywh2xyxy, scale_coords, box_iou, letterbox
)
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class YOLOv5_GestureDetector(BaseModel):
    """
    YOLOv5 gesture detection model.
    """

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.4,
                 iou_threshold: float = 0.5,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize YOLOv5 gesture detector.

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
        # Check if model file exists
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"YOLOv5 模型文件不存在: {self.model_path}"
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
                message=f"YOLOv5 模型加载失败: {self.model_path} - {e}"
            )

        # Get input/output info
        try:
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            # Same as original code: input_shape is (height, width)
            self.input_shape = self.session.get_inputs()[0].shape[2:4]
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=f"无法获取模型输入/输出信息: {e}",
                message=f"YOLOv5 模型初始化失败: {self.model_path} - {e}"
            )

    def infer(self, image: np.ndarray) -> List[Dict[str, Any]]:
        # 确保模型已加载（延迟加载支持）
        self._ensure_model_loaded()

        image_shape = image.shape[:2]

        # Preprocess image
        try:
            input_tensor, ratio, pad = self.preprocess(image)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5",
                reason=f"图像预处理失败: {e}",
                message=f"YOLOv5 预处理错误: {e}"
            )

        # Run inference
        try:
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5",
                reason=f"模型推理失败: {e}",
                message=f"YOLOv5 推理错误: {e}"
            )

        # Postprocess
        try:
            detections = self.postprocess(outputs, self.conf_threshold, self.iou_threshold, image_shape, ratio, pad)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5",
                reason=f"结果后处理失败: {e}",
                message=f"YOLOv5 后处理错误: {e}"
            )

        return detections


    # Preprocess image
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
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
        outputs: np.ndarray,
        conf_threshold: float,
        iou_threshold: float,
        orig_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> List[Dict[str, Any]]:
        # non_max_suppression 返回 List[np.ndarray]，需要取出第一个元素
        detections_list = self.non_max_suppression(outputs[0], conf_threshold, iou_threshold)

        if len(detections_list) == 0 or detections_list[0].size == 0:
            return []

        detections = detections_list[0]  # 取出第一个数组

        # scale_coords 需要 ratio_pad 参数，格式为 ((ratio, ratio), (pad[0], pad[1]))
        ratio_pad = ((ratio, ratio), pad)
        # 缩放坐标到原始图像尺寸
        detections[:, :4] = scale_coords(
            self.input_shape, detections[:, :4], orig_shape, ratio_pad=ratio_pad
        )
        outputs = []
        for i in range(len(detections)):
            d = detections[i]
            outputs.append({
                "bbox": [float(d[0]), float(d[1]), float(d[2]), float(d[3])],
                "class_id": int(d[5]),
                "confidence": float(d[4])
            })
        return outputs

    def non_max_suppression(
        self,
        prediction: np.ndarray,
        conf_thres: float = 0.1,
        iou_thres: float = 0.6,
    ) -> List[np.ndarray]:
        """
        纯 NumPy 实现的非极大值抑制 (NMS)，适用于 ONNX 推理输出。

        参数:
            prediction: 模型的原始输出 (1, N, 85) 或 (1, N, 24) 等。
            conf_thres: 置信度阈值。
            iou_thres: IoU 阈值。
        返回:
            一个包含处理后 NumPy 数组的列表 [detections]，格式为 [x1, y1, x2, y2, conf, cls]。
        """
        if prediction.ndim != 3:
            # 确保输入是 (Batch, N_proposals, N_features)
            return [np.array([])]

        prediction = prediction[0]  # 取出 batch 1

        # 1. 置信度过滤
        # box confidence (第5列, index 4) > conf_thres
        xc = prediction[:, 4] > conf_thres
        x = prediction[xc]

        if x.shape[0] == 0:
            return [np.array([])]

        # 2. 类别置信度 * box置信度（原始 YOLOv5 逻辑）
        x[:, 5:] = x[:, 5:] * x[:, 4:5]

        # 3. XYWH -> XYXY
        box = xywh2xyxy(x[:, :4])

        # 4. 多标签处理 (简化版: 找到每个框的最佳类别)
        # conf: 最佳类别分数 (np.max); j: 最佳类别索引 (np.argmax)
        conf = np.amax(x[:, 5:], axis=1, keepdims=True)
        j = np.argmax(x[:, 5:], axis=1, keepdims=True)

        # 5. 组合结果 [x1, y1, x2, y2, conf, cls]
        x = np.concatenate((box, conf, j.astype(np.float32)), axis=1)

        # 6. 再次过滤 (确保 conf > conf_thres)
        x = x[x[:, 4] > conf_thres]
        n = x.shape[0]

        if n == 0:
            return [np.array([])]

        # 7. 纯 NumPy NMS（手势检测默认使用 class-agnostic NMS，避免不同类别在同一只手上“叠框”）
        boxes = x[:, :4]
        scores = x[:, 4]

        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            iou = box_iou(boxes[i][None, :], boxes[order[1:]])[0]
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]

        return [x[keep]]

