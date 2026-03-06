# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
ResNet Image Classifier Implementation
"""

import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from typing import List

from core import BaseModel
from common import preprocess_classification


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
            resize_size: Size to resize before center crop
            num_threads: Number of threads
            **kwargs: Additional parameters
        """
        self.resize_size = (256,256)
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

        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape[2:4]


    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ResNet."""
        img_batch = preprocess_classification(image, self.input_shape, resize_size=self.resize_size)

        return img_batch

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Run classification inference."""
        input_tensor = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]

    def postprocess(self, outputs, **kwargs) -> np.ndarray:
        """Postprocess outputs."""
        return np.squeeze(outputs)

    def predict_top_k(self, image: np.ndarray, labels: List[str],
                     k: int = 5) -> List[tuple]:
        """Predict top-K classes."""
        outputs = self.infer(image)
        outputs = np.squeeze(outputs)

        top_indices = np.argsort(outputs)[-k:][::-1]
        top_scores = outputs[top_indices]

        results = []
        for idx, score in zip(top_indices, top_scores):
            if idx < len(labels):
                results.append((labels[idx], float(score)))
            else:
                results.append((f"Class {idx}", float(score)))

        return results

