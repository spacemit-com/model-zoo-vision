# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Call the C++ :cpp:class:`VisionService` from Python via pybind11 module ``_vision_service_cpp``.

Build the extension::

    pip install pybind11
    cmake -S . -B build -DBUILD_PYTHON_BINDINGS=ON
    cmake --build build

Run with the repository's ``src`` on ``PYTHONPATH`` and CMake ``build/python`` (where
``_vision_service_cpp*.so`` is placed) on ``PYTHONPATH``. Ensure ``libvision.so`` is loadable
(``LD_LIBRARY_PATH`` including the build tree root, or install layout).

Example::

    import cv2
    from core.python.vision_service_native import VisionServiceNative, VisionServiceStatus

    svc = VisionServiceNative.create("examples/yolov8/config/yolov8.yaml")
    img = cv2.imread("test.jpg")
    status, results = svc.infer_image(img)
    assert status == VisionServiceStatus.OK
"""

from __future__ import annotations

from typing import List, Tuple, Union

import numpy as np
import numpy.typing as npt

try:
    import _vision_service_cpp as _ext
except ImportError as _import_error:  # pragma: no cover - extension optional at import time
    _ext = None
    _IMPORT_ERROR = _import_error
else:
    _IMPORT_ERROR = None

__all__ = [
    "VisionServiceNative",
    "VisionServiceResult",
    "VisionServiceKeypoint",
    "VisionServiceTiming",
    "VisionServiceTimingOptions",
    "VisionServiceStatus",
    "extension_available",
    "extension_import_error",
]


def extension_available() -> bool:
    return _ext is not None


def extension_import_error() -> ImportError | None:
    return _IMPORT_ERROR


def _require_ext() -> None:
    if _ext is None:
        raise ImportError(
            "Missing native module _vision_service_cpp. Build with "
            "-DBUILD_PYTHON_BINDINGS=ON, install pybind11, add CMake build directory "
            "'python' subfolder to PYTHONPATH, and ensure libvision.so is on the loader path."
        ) from _IMPORT_ERROR


VisionServiceStatus = None  # type: ignore[assignment,misc]
VisionServiceResult = None  # type: ignore[assignment,misc]
VisionServiceKeypoint = None  # type: ignore[assignment,misc]
VisionServiceTiming = None  # type: ignore[assignment,misc]
VisionServiceTimingOptions = None  # type: ignore[assignment,misc]

if _ext is not None:
    VisionServiceStatus = _ext.VisionServiceStatus
    VisionServiceResult = _ext.VisionServiceResult
    VisionServiceKeypoint = _ext.VisionServiceKeypoint
    VisionServiceTiming = _ext.VisionServiceTiming
    VisionServiceTimingOptions = _ext.VisionServiceTimingOptions


class VisionServiceNative:
    """
    Thin wrapper around :cpp:class:`VisionService` with typed hints for NumPy images.

    Images must be ``uint8``, shape ``(H, W, 3)``, BGR, C-contiguous (OpenCV default).
    """

    __slots__ = ("_svc",)

    def __init__(self, impl: object) -> None:
        _require_ext()
        self._svc = impl

    @staticmethod
    def create(
        config_path: str,
        model_path_override: str = "",
        lazy_load: bool = False,
    ) -> VisionServiceNative:
        _require_ext()
        impl = _ext.VisionService.create(config_path, model_path_override, lazy_load)
        return VisionServiceNative(impl)

    @staticmethod
    def last_create_error() -> str:
        _require_ext()
        return _ext.VisionService.last_create_error()

    def infer_image(
        self,
        image_or_path: Union[str, npt.NDArray[np.uint8]],
    ) -> Tuple[object, List]:
        if isinstance(image_or_path, str):
            return self._svc.infer_image(image_path=image_or_path)
        arr = np.ascontiguousarray(image_or_path)
        if arr.dtype != np.uint8:
            raise TypeError("image must be uint8 BGR (HxWx3)")
        return self._svc.infer_image(image_bgr_uint8=arr)

    def infer_embedding(
        self,
        image_or_path: Union[str, npt.NDArray[np.uint8]],
    ) -> Tuple[object, List[float]]:
        if isinstance(image_or_path, str):
            return self._svc.infer_embedding(image_path=image_or_path)
        arr = np.ascontiguousarray(image_or_path)
        if arr.dtype != np.uint8:
            raise TypeError("image must be uint8 BGR (HxWx3)")
        return self._svc.infer_embedding(image_bgr_uint8=arr)

    @staticmethod
    def embedding_similarity(a: List[float], b: List[float]) -> float:
        _require_ext()
        return _ext.VisionService.embedding_similarity(embedding_a=a, embedding_b=b)

    def infer_sequence(
        self,
        pts: npt.NDArray[np.float32],
        image_width: int,
        image_height: int,
    ) -> Tuple[object, List[float]]:
        arr = np.ascontiguousarray(pts, dtype=np.float32)
        return self._svc.infer_sequence(pts=arr, image_width=image_width, image_height=image_height)

    def get_sequence_class_names(self) -> List[str]:
        return self._svc.get_sequence_class_names()

    def get_fall_down_class_index(self) -> int:
        return self._svc.get_fall_down_class_index()

    def draw(self, image_bgr_uint8: npt.NDArray[np.uint8]) -> Tuple[object, npt.NDArray[np.uint8]]:
        arr = np.ascontiguousarray(image_bgr_uint8)
        if arr.dtype != np.uint8:
            raise TypeError("image must be uint8 BGR (HxWx3)")
        return self._svc.draw(image_bgr_uint8=arr)

    def get_last_keypoints(self, result_index: int) -> Tuple[object, List]:
        return self._svc.get_last_keypoints(result_index=result_index)

    def supports_draw(self) -> bool:
        return self._svc.supports_draw()

    def set_timing_options(self, options: object) -> None:
        self._svc.set_timing_options(options)

    def get_last_timing(self) -> object:
        return self._svc.get_last_timing()

    def get_default_image(self) -> str:
        return self._svc.get_default_image()

    def get_config_path_value(self, config_key: str) -> str:
        return self._svc.get_config_path_value(config_key)

    def last_error(self) -> str:
        return self._svc.last_error()
