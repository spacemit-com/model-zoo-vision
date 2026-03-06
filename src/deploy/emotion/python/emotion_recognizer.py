# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Emotion Recognition Implementation
"""

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from typing import List, Dict, Any, Tuple, Sequence, Union
from pathlib import Path
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class EmotionRecognizer:
    """
    Emotion Recognition model.

    Combines face detection (YOLOv5-Face) and emotion classification to
    recognize emotions in facial images.
    """


    def __init__(self, model_path: str,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize Emotion Recognizer.

        Args:
            model_path: Path to emotion recognition ONNX model（与 C++ 统一用 model_path）
            num_threads: Number of threads for ONNX Runtime
            **kwargs: Additional parameters
        """
        self.emotion_model_path = model_path
        self.num_threads = num_threads
        # 兼容 factory 的多路径参数：这里不再初始化人脸检测器，所以丢弃该参数，避免误传给 ORT 配置
        kwargs.pop("face_detector_path", None)

        # Load emotion model
        self._load_emotion_model(**kwargs)

    def _load_emotion_model(self, **kwargs):
        """Load emotion recognition ONNX model."""
        # Check if model file exists
        model_path_obj = Path(self.emotion_model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.emotion_model_path,
                message=f"Emotion 模型文件不存在: {self.emotion_model_path}"
            )

        # Create session options
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads

        # Load model
        try:
            providers = kwargs.get('providers', ['SpaceMITExecutionProvider'])
            self.session = ort.InferenceSession(
                self.emotion_model_path,
                sess_options=session_options,
                providers=providers
            )
        except Exception as e:
            raise ModelLoadError(
                model_path=self.emotion_model_path,
                reason=str(e),
                message=f"Emotion 模型加载失败: {self.emotion_model_path} - {e}"
            )

        # Get input/output info
        try:
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = tuple(self.session.get_inputs()[0].shape[2:4])
        except Exception as e:
            raise ModelLoadError(
                model_path=self.emotion_model_path,
                reason=f"无法获取模型输入/输出信息: {e}",
                message=f"Emotion 模型初始化失败: {self.emotion_model_path} - {e}"
            )

    def preprocess(self, face_image: np.ndarray,
                       target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocess face image for emotion recognition.

        Args:
            face_image: Cropped face image
            target_size: Target size for the model

        Returns:
            Preprocessed tensor
        """
        # Resize
        img_resized = cv2.resize(face_image, target_size, interpolation=cv2.INTER_NEAREST)

        # Preprocess (mean subtraction)
        img_preprocessed = np.copy(img_resized).astype(np.float32)
        img_preprocessed[..., 0] -= 91.4953  # B
        img_preprocessed[..., 1] -= 103.8827  # G
        img_preprocessed[..., 2] -= 131.0912  # R

        # HWC to CHW
        img_tensor = np.transpose(img_preprocessed, (2, 0, 1))

        # Add batch dimension
        img_tensor = np.expand_dims(img_tensor, axis=0)

        return img_tensor

    def infer(self, face_images: Union[np.ndarray, Sequence[np.ndarray]]) -> List[Dict[str, Any]]:
        """
        Run emotion recognition (classification only).

        注意：本方法**只做情绪分类**，输入必须是人脸裁剪图（单张或多张）。
        人脸检测、裁剪、以及 bbox/confidence 的维护应在外部完成（例如 `example_emotion.py`）。

        Args:
            face_images: np.ndarray (单张 BGR 人脸图) 或 Sequence[np.ndarray]（多张 BGR 人脸图）

        Returns:
            List[Dict]，每个元素至少包含：
            - emotion: Emotion class index
        """
        if isinstance(face_images, np.ndarray):
            face_images_list = [face_images]
        else:
            face_images_list = list(face_images)

        if not face_images_list:
            return []

        results: List[Dict[str, Any]] = []

        for face_img_bgr in face_images_list:
            try:
                face_tensor = self.preprocess(face_img_bgr)
            except Exception as e:
                raise ModelInferenceError(
                    model_name="EmotionRecognizer",
                    reason=f"图像预处理失败: {e}",
                    message=f"Emotion 识别 - 预处理错误: {e}",
                )

            try:
                output = self.session.run(None, {self.input_name: face_tensor})
            except Exception as e:
                raise ModelInferenceError(
                    model_name="EmotionRecognizer",
                    reason=f"模型推理失败: {e}",
                    message=f"Emotion 识别 - 推理错误: {e}",
                )

            try:
                output_tensor = output[0] if isinstance(output, list) else output
                emotion_id = int(np.argmax(output_tensor))
            except Exception as e:
                raise ModelInferenceError(
                    model_name="EmotionRecognizer",
                    reason=f"结果后处理失败: {e}",
                    message=f"Emotion 识别 - 后处理错误: {e}",
                )

            results.append({"emotion": emotion_id})

        return results

    def recognize_emotions_from_faces(
        self,
        image: np.ndarray,
        face_boxes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Recognize emotions from pre-detected face boxes.

        Args:
            image: Input image in BGR format
            face_boxes: List of face detection dictionaries with keys:
                       - bbox: [x1, y1, x2, y2]
                       - confidence: Face detection confidence

        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - confidence: Face detection confidence
            - emotion: Emotion class index
        """
        if not face_boxes:
            return []

        img = image.copy()
        detections: List[Dict[str, Any]] = []

        for box in face_boxes:
            try:
                face_img = img[int(box["bbox"][1]):int(box["bbox"][3]), int(box["bbox"][0]):int(box["bbox"][2])]
                face_img = self.preprocess(face_img)
            except Exception as e:
                raise ModelInferenceError(
                    model_name="EmotionRecognizer",
                    reason=f"图像预处理失败: {e}",
                    message=f"Emotion 识别 - 预处理错误: {e}"
                )

            try:
                output = self.session.run(None, {self.input_name: face_img})
            except Exception as e:
                raise ModelInferenceError(
                    model_name="EmotionRecognizer",
                    reason=f"模型推理失败: {e}",
                    message=f"Emotion 识别 - 推理错误: {e}"
                )

            try:
                output_tensor = output[0] if isinstance(output, list) else output
                emotion_id = int(np.argmax(output_tensor))
                detections.append({
                    "bbox": box["bbox"],
                    "confidence": box.get("confidence", 1.0),
                    "emotion": emotion_id
                })
            except Exception as e:
                raise ModelInferenceError(
                    model_name="EmotionRecognizer",
                    reason=f"结果后处理失败: {e}",
                    message=f"Emotion 识别 - 后处理错误: {e}"
                )

        return detections

