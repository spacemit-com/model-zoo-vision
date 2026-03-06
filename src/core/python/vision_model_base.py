# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Base Model Class for All Vision Models
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import numpy as np


def _get_vision_root() -> Path:
    # This file lives at: <...>/cv/src/core/python/vision_model_base.py
    # Path(__file__).parents[3] == <...>/cv (project root)
    return Path(__file__).resolve().parents[3]


def _download_script_hint() -> str:
    return str(_get_vision_root() / "examples" / "<model>" / "scripts" / "download_models.sh")


def _resolve_and_check_model_path(model_path: str) -> str:
    if not model_path:
        raise FileNotFoundError("Model path is empty")
    if model_path.startswith("http://") or model_path.startswith("https://"):
        raise FileNotFoundError(
            f"Model URL is not supported: {model_path}. Please download the model first, e.g. run: "
            f"{_download_script_hint()}"
        )
    p = Path(model_path).expanduser()
    if not p.is_absolute():
        p = (_get_vision_root() / p).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {p}. Please download the model first, e.g. run: {_download_script_hint()}"
        )
    return str(p)


class BaseModel(ABC):
    """
    Abstract base class for all vision models.

    All vision models should inherit from this class and implement the required methods.
    """

    def __init__(self, model_path: str, lazy_load: bool = False, **kwargs):
        """
        Initialize the model.

        Args:
            model_path: Path to the model file (e.g., ONNX, TorchScript)
            lazy_load: If True, delay model loading until first inference call
            **kwargs: Additional model-specific parameters
        """
        self.model_path = _resolve_and_check_model_path(model_path)
        self.session = None
        self.input_shape = None
        self._model_loaded = False
        self._lazy_load = lazy_load
        self._load_kwargs = kwargs

        if not lazy_load:
            self._load_model(**kwargs)
            self._model_loaded = True

    @abstractmethod
    def _load_model(self, **kwargs):
        """
        Load the model from file.

        This method should:
        1. Initialize the inference session
        2. Get input/output information
        3. Store model metadata
        """
        pass

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> Any:
        """
        Preprocess the input image for inference.

        Args:
            image: Input image in BGR format (OpenCV default)

        Returns:
            Preprocessed input ready for model inference
        """
        pass

    def _ensure_model_loaded(self):
        """Ensure model is loaded (for lazy loading)."""
        if not self._model_loaded:
            self._load_model(**self._load_kwargs)
            self._model_loaded = True

    @abstractmethod
    def infer(self, image: np.ndarray) -> Any:
        """
        Run inference on the input image.

        Args:
            image: Input image in BGR format

        Returns:
            Model output (format depends on specific model type)
        """
        pass

    def postprocess(self, outputs: Any, **kwargs) -> Any:
        """
        Postprocess the model outputs.

        Args:
            outputs: Raw model outputs
            **kwargs: Additional parameters for postprocessing

        Returns:
            Processed results in a usable format
        """
        pass

    def get_input_shape(self) -> Optional[Tuple[int, ...]]:
        """
        Get the input shape of the model.

        Returns:
            Input shape tuple (C, H, W) or None if not available
        """
        return self.input_shape

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dictionary containing model metadata
        """
        return {
            'model_path': self.model_path,
            'input_shape': self.input_shape,
        }
