# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
ArcFace Face Recognition Implementation
"""

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from PIL import Image
from typing import Tuple
from pathlib import Path
from core import BaseModel
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)


class ArcFaceRecognizer(BaseModel):
    """
    ArcFace face recognition model.

    Extracts face embeddings for face recognition tasks.
    The model should receive aligned face images as input.
    """

    def __init__(self, model_path: str,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize ArcFace recognizer.

        Args:
            model_path: Path to ONNX model file
            num_threads: Number of threads for ONNX Runtime
            **kwargs: Additional parameters
        """
        self.num_threads = num_threads
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs):
        """Load ArcFace ONNX model."""
        # Check if model file exists
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"ArcFace 模型文件不存在: {self.model_path}"
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
                message=f"ArcFace 模型加载失败: {self.model_path} - {e}"
            )

        # Get input/output info
        try:
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = tuple(self.session.get_inputs()[0].shape[2:])
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=f"无法获取模型输入/输出信息: {e}",
                message=f"ArcFace 模型初始化失败: {self.model_path} - {e}"
            )

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for ArcFace inference.

        Args:
            image: Input face image in BGR format

        Returns:
            Preprocessed tensor ready for inference
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        img = Image.fromarray(img)

        # Resize with letterbox
        img = self._resize_image(img, self.input_shape, letterbox_image=True)

        # Convert back to numpy
        img = np.array(img, dtype=np.float32)

        # Normalize: (img / 255.0 - 0.5) / 0.5
        img = (img / 255.0 - 0.5) / 0.5

        # HWC to CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, outputs: np.ndarray, **kwargs) -> np.ndarray:
        """
        Postprocess ArcFace outputs.

        Args:
            outputs: Raw model outputs
            **kwargs: Additional parameters

        Returns:
            Normalized face embedding
        """
        # Extract embedding from list
        if isinstance(outputs, list):
            embedding = outputs[0]
        else:
            embedding = outputs

        # L2 normalize
        embedding = self._normalize_embedding(embedding)

        return embedding

    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        Extract face embedding from an image.

        Args:
            image: Input face image in BGR format

        Returns:
            Normalized face embedding vector
        """
        # 确保模型已加载（延迟加载支持）
        self._ensure_model_loaded()

        # Preprocess
        try:
            input_tensor = self.preprocess(image)
        except Exception as e:
            raise ModelInferenceError(
                model_name="ArcFace",
                reason=f"图像预处理失败: {e}",
                message=f"ArcFace 预处理错误: {e}"
            )

        # Run inference
        try:
            outputs = self.session.run(None, {self.input_name: input_tensor})
        except Exception as e:
            raise ModelInferenceError(
                model_name="ArcFace",
                reason=f"模型推理失败: {e}",
                message=f"ArcFace 推理错误: {e}"
            )

        # Postprocess
        try:
            embedding = self.postprocess(outputs)
        except Exception as e:
            raise ModelInferenceError(
                model_name="ArcFace",
                reason=f"结果后处理失败: {e}",
                message=f"ArcFace 后处理错误: {e}"
            )

        return embedding

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity between two face embeddings.

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Similarity score (-1 to 1, higher means more similar)
            The embeddings are already L2-normalized, so the result is cosine similarity.
        """
        # Use matrix multiplication for cosine similarity (embeddings are already normalized)
        similarity_matrix = embedding1 @ embedding2.T

        # Extract scalar value
        if similarity_matrix.ndim > 0:
            similarity = float(similarity_matrix.flatten()[0])
        else:
            similarity = float(similarity_matrix)

        return similarity

    @staticmethod
    def _resize_image(image: Image.Image,
                     size: Tuple[int, int] = (112, 112),
                     letterbox_image: bool = True) -> Image.Image:
        """
        Resize image with optional letterbox mode.

        Args:
            image: PIL Image
            size: Target size (width, height)
            letterbox_image: If True, maintain aspect ratio with padding

        Returns:
            Resized PIL Image
        """
        iw, ih = image.size
        w, h = size

        if letterbox_image:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)

        return new_image

    @staticmethod
    def _normalize_embedding(embedding: np.ndarray) -> np.ndarray:
        """
        L2 normalize embedding vector.

        Args:
            embedding: Input embedding

        Returns:
            Normalized embedding
        """
        norm = np.sqrt(np.sum(np.power(embedding, 2)))
        return embedding / (norm + 1e-8)

