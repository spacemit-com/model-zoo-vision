# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Stateless dynamic-emotion LSTM classifier.

Input:  feature sequence (10, 512) or (1, 10, 512)
Output: 7-class emotion probabilities

The 10-frame sliding window of ResNet50 features is maintained by the
caller (application layer), not by this class. The ResNet50 feature
extraction is done separately by EmotionRecognizer(feature_mode=True).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

try:
    import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
    _DEFAULT_PROVIDERS = ["SpaceMITExecutionProvider", "CPUExecutionProvider"]
except ImportError:
    _DEFAULT_PROVIDERS = ["CPUExecutionProvider"]

from common.python.image_processing import load_labels
from core.python.vision_model_exceptions import (
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)

SEQ_LEN = 10
FEATURE_DIM = 512

_DEFAULT_LABELS = [
    "neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger",
]


def _resolve_labels(label_file_path: Optional[str]) -> List[str]:
    root = Path(__file__).resolve().parents[4]
    if label_file_path:
        path = Path(label_file_path).expanduser()
        if not path.is_absolute():
            path = root / path
        if path.is_file():
            return load_labels(str(path))
    default = root / "assets" / "labels" / "emotion.txt"
    if default.is_file():
        return load_labels(str(default))
    return list(_DEFAULT_LABELS)


class EmotionLstm:
    """
    Stateless LSTM emotion classifier for model_zoo.

    Args:
        model_path:  Path to the LSTM ONNX (maps to YAML ``model_path``).
        num_threads: ORT intra-op threads.
        providers:   ORT execution providers (from YAML ``default_params.providers``).

    Usage::

        lstm = EmotionLstm(model_path=...)
        result = lstm.infer(features_seq)   # features_seq: (10, 512) or (1, 10, 512)
        print(result["emotion_label"], result["emotion_probs"])
    """

    def __init__(self, model_path: str, num_threads: int = 4, **kwargs) -> None:
        self.model_path = model_path
        self.num_threads = num_threads
        providers: List[str] = kwargs.get("providers", _DEFAULT_PROVIDERS)
        self.labels = _resolve_labels(kwargs.get("label_file_path"))

        p = Path(model_path).expanduser()
        if not p.is_file():
            raise ModelNotFoundError(model_path=str(p), message=f"LSTM 模型文件不存在: {p}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            self.session = ort.InferenceSession(str(p), sess_options=opts, providers=providers)
        except Exception as e:
            raise ModelLoadError(model_path=str(p), reason=str(e), message=str(e))

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def infer(self, features_seq: np.ndarray) -> Dict:
        """
        Classify a sequence of features.

        Args:
            features_seq: (SEQ_LEN, FEATURE_DIM) or (1, SEQ_LEN, FEATURE_DIM).

        Returns:
            dict with keys:
              - emotion:       int class index
              - emotion_label: str
              - emotion_probs: np.ndarray shape (7,)
        """
        arr = np.asarray(features_seq, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        if arr.shape != (1, SEQ_LEN, FEATURE_DIM):
            raise ValueError(
                f"期望特征序列 ({SEQ_LEN}, {FEATURE_DIM}) 或 (1, {SEQ_LEN}, {FEATURE_DIM})，"
                f"当前 {np.asarray(features_seq).shape}"
            )
        arr = np.ascontiguousarray(arr)

        try:
            probs = self.session.run([self.output_name], {self.input_name: arr})[0].reshape(-1)
        except Exception as e:
            raise ModelInferenceError(
                model_name="EmotionLstm",
                reason=str(e),
                message=f"LSTM 推理失败: {e}",
            )

        idx = int(np.argmax(probs))
        label = self.labels[idx] if 0 <= idx < len(self.labels) else str(idx)
        return {
            "emotion": idx,
            "emotion_label": label,
            "emotion_probs": probs,
        }
