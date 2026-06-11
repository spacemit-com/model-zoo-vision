# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
ResNet Image Classifier Implementation
"""

import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
import cv2
from typing import List

from core import BaseModel
from common import preprocess_classification


# Map human-readable interpolation names (from YAML) to cv2 flags.
_INTERPOLATION_MAP = {
    "bilinear": cv2.INTER_LINEAR,
    "linear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "cubic": cv2.INTER_CUBIC,
    "nearest": cv2.INTER_NEAREST,
    "area": cv2.INTER_AREA,
}


class ResNetClassifier(BaseModel):
    """
    ResNet Image Classifier.
    """

    def __init__(self, model_path: str,
                 num_threads: int = 4,
                 **kwargs):
        """
        Initialize ResNet classifier.

        Args:
            model_path: Path to ONNX model
            num_threads: Number of threads
            **kwargs: Additional parameters. Recognized preprocessing keys:
                resize_size: Size to resize before center crop (default (256, 256))
                mean: Per-channel normalization mean (default ImageNet)
                std: Per-channel normalization std (default ImageNet)
                center_crop: Whether to center crop after resize (default True)
                interpolation: Resize interpolation name (default "bilinear")
        """
        resize_size = kwargs.get('resize_size', (256, 256))
        self.resize_size = tuple(resize_size) if resize_size is not None else None
        self.mean = tuple(kwargs.get('mean', (0.485, 0.456, 0.406)))
        self.std = tuple(kwargs.get('std', (0.229, 0.224, 0.225)))
        self.center_crop = kwargs.get('center_crop', True)
        interp_name = str(kwargs.get('interpolation', 'bilinear')).lower()
        self.interpolation = _INTERPOLATION_MAP.get(interp_name, cv2.INTER_LINEAR)
        self.num_threads = num_threads
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs):
        """Load ResNet ONNX model."""
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads

        providers = kwargs.get('providers', ['SpaceMITExecutionProvider'])
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=session_options,
            providers=providers
        )

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        input_shape = input_meta.shape
        self.input_shape = input_shape[2:4]
        batch_dim = input_shape[0] if len(input_shape) > 0 else 1
        if isinstance(batch_dim, int) and batch_dim > 0:
            self.batch_size = batch_dim
        else:
            # Dynamic/unknown batch dim (e.g. "batch_size" or None): default to 1.
            # A single-image NCHW tensor is valid for dynamic-batch models, so this
            # is safe; we note it so a genuinely fixed batch>1 export isn't masked.
            print(f"警告: ResNet 模型 batch 维为动态/未知 ({batch_dim!r})，按 batch=1 处理")
            self.batch_size = 1


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ResNet."""
        img_batch = preprocess_classification(
            image,
            self.input_shape,
            mean=self.mean,
            std=self.std,
            resize_size=self.resize_size,
            center_crop=self.center_crop,
            interpolation=self.interpolation,
        )
        if self.batch_size > 1:
            img_batch = np.tile(img_batch, (self.batch_size, 1, 1, 1))

        return img_batch

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run classification inference."""
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]

    def postprocess(self, outputs, **kwargs) -> np.ndarray:
        """Postprocess outputs."""
        out = np.squeeze(outputs)
        if out.ndim == 2:
            out = out[0]
        return out

    def predict_top_k(self, image: np.ndarray, labels: List[str],
                     k: int = 5) -> List[tuple]:
        """Predict top-K classes (softmax probabilities)."""
        outputs = self.postprocess(self.infer(image)).astype(np.float64)
        outputs = outputs - np.max(outputs)
        exp_scores = np.exp(outputs)
        probs = exp_scores / np.sum(exp_scores)

        top_indices = np.argsort(probs)[-k:][::-1]
        top_scores = probs[top_indices]

        results = []
        for idx, score in zip(top_indices, top_scores):
            if labels and idx < len(labels):
                results.append((labels[idx], float(score)))
            else:
                results.append((f"Class {idx}", float(score)))

        return results

