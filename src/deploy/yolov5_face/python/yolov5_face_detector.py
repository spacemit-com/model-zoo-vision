# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv5-Face Detector Implementation
Aligned with original code from k3/ai_web/model_zoo/examples/CV/yolov5-face/python/utils.py
"""

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
import time
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from core import BaseModel
from common import (
    sigmoid, letterbox, xywh2xyxy, nms, scale_coords
)
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class YOLOv5FaceDetector(BaseModel):
    """
    YOLOv5-Face detector for face detection.
    Same interface as original Yolov5FaceDetection class.
    """

    def __init__(self, model_path: str,
                 conf_thres: float = 0.25,
                 iou_thres: float = 0.45,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize YOLOv5-Face detector.
        Same as original code.

        Args:
            model_path: Path to ONNX model file
            conf_thres: Confidence threshold for filtering detections
            iou_thres: IoU threshold for NMS
            num_threads: Number of threads for ONNX Runtime
            **kwargs: Additional parameters
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.num_threads = num_threads
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs):
        """Load YOLOv5-Face ONNX model."""
        # Check if model file exists
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"YOLOv5-Face 模型文件不存在: {self.model_path}"
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
                message=f"YOLOv5-Face 模型加载失败: {self.model_path} - {e}"
            )

        # Get input/output info - same as original: shape[2:4] not tuple
        try:
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape[2:4]
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=f"无法获取模型输入/输出信息: {e}",
                message=f"YOLOv5-Face 模型初始化失败: {self.model_path} - {e}"
            )


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLOv5-Face inference.
        Same as original code: process_img method.

        Args:
            img: Input image in BGR format

        Returns:
            Preprocessed tensor ready for inference
        """
        img = letterbox(image, self.input_shape)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1).copy()

        # Same as original code: cv2.normalize instead of direct division
        img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img = img[None]
        return img


    def postprocess(self, outputs: List[np.ndarray], **kwargs) -> np.ndarray:
        """
        Postprocess YOLOv5-Face outputs.
        Same as original code: post_process method.

        Args:
            output: Raw model outputs

        Returns:
            Processed predictions
        """
        init_anchors = np.array([
            4., 5., 8., 10., 13., 16., 23., 29.,
            43., 55., 73., 105., 146., 217.,
            231., 300., 335., 433.
        ]).astype(np.float32).reshape(3, 1, 3, 1, 1, 2)

        strides = [8, 16, 32]
        pred_list = []

        for idx, pred in enumerate(outputs):
            y = np.full_like(pred[..., :6], 0)
            bs, _, ny, nx, _ = pred.shape
            grid = self._make_grid(nx, ny)

            y_tmp = sigmoid(np.concatenate([pred[..., :5], pred[..., 15:]], axis=-1))
            y[..., :2] = (y_tmp[..., :2] * 2. - 0.5 + grid) * strides[idx]
            y[..., 2:4] = (y_tmp[..., 2:4] * 2) ** 2 * init_anchors[idx]
            y[..., 4:] = y_tmp[..., 4:]
            pred_list.append(y.reshape(bs, -1, 6))

        pred_result = np.concatenate(pred_list, 1)
        return pred_result

    def infer(self, img_src: np.ndarray) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """
        Run YOLOv5-Face inference on an image.
        Same as original code: returns detections and face_images.

        Args:
            img_src: Input image in BGR format

        Returns:
            Tuple of (detections, face_images):
            - detections: List of dicts with "bbox" and "confidence"
            - face_images: List of cropped face images
        """
        img = img_src.copy()

        # Preprocess
        try:
            img = self.preprocess(img)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5-Face",
                reason=f"图像预处理失败: {e}",
                message=f"YOLOv5-Face 预处理错误: {e}"
            )

        # Run inference
        try:
            model_infer = self.session.run(None, {self.input_name: img})
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5-Face",
                reason=f"模型推理失败: {e}",
                message=f"YOLOv5-Face 推理错误: {e}"
            )

        # Postprocess
        try:
            pred = self.postprocess(model_infer)
            pred = self.non_max_suppression_face(pred, self.conf_thres, self.iou_thres)
        except Exception as e:
            raise ModelInferenceError(
                model_name="YOLOv5-Face",
                reason=f"结果后处理失败: {e}",
                message=f"YOLOv5-Face 后处理错误: {e}"
            )

        detections = []
        face_images = []
        if pred is not None and len(pred) > 0 and len(pred[0]) > 0:
            for i in range(len(pred[0])):
                # Same as original: use img_src.shape (not shape[:2])
                bbox = scale_coords(self.input_shape, pred[0][i][:4], img_src.shape).round()
                conf = float(pred[0][i][4]) if len(pred[0][i]) > 4 else 1.0
                face_images.append(img_src[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])])
                detections.append({
                    "bbox": bbox.tolist(),
                    "confidence": conf
                })

        return detections, face_images

    def _make_grid(self, nx: int, ny: int) -> np.ndarray:
        """Create grid for anchor processing. Same as original _make_grid function."""
        xv, yv = np.meshgrid(np.arange(ny), np.arange(nx))
        return np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype(np.float32)

    def non_max_suppression_face(self, prediction: np.ndarray,
                                  conf_thres: float = 0.25,
                                  iou_thres: float = 0.45,
                                  classes: Optional[List[int]] = None,
                                  agnostic: bool = False,
                                  labels: Tuple[str, ...] = ()) -> List[np.ndarray]:
        """
        Apply NMS for face detection.
        Same as original code: non_max_suppression_face static method.
        """
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Settings
        max_wh = 4096  # (pixels) maximum box width and height
        time_limit = 10.0  # seconds to quit after

        t = time.time()
        output = [np.zeros((0, 6))] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lbl = labels[xi]
                v = np.zeros((len(lbl), nc + 5))
                v[:, :4] = lbl[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lbl)), lbl[:, 0].astype(int) + 5] = 1.0  # cls
                x = np.concatenate((x, v), 0)
            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            box = xywh2xyxy(x[:, :4])
            conf = x[:, 5:].max(1).reshape(-1, 1)
            class_idx = np.argmax(x[:, 5:], axis=1).reshape(-1, 1).astype(np.float32)
            x = np.concatenate((box, conf, class_idx), 1)[conf.reshape(-1) > conf_thres]

            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = nms(boxes, scores, iou_thres)  # NMS

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break  # time limit exceeded

        return output
