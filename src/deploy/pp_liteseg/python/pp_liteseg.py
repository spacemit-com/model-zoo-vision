# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
PP-LiteSeg semantic segmentation implementation.
"""

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider

from core import BaseModel
from core.python.vision_model_exceptions import (
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
)


class PPLiteSeg(BaseModel):
    """PP-LiteSeg semantic segmentor."""

    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        num_classes: int = 19,
        **kwargs,
    ):
        self.num_threads = num_threads
        self.num_classes = num_classes
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        super().__init__(model_path, **kwargs)

    def _load_model(self, **kwargs):
        model_path_obj = Path(self.model_path)
        if not model_path_obj.exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"PP-LiteSeg model file not found: {self.model_path}",
            )

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = self.num_threads

        try:
            providers = kwargs.get("providers", ["SpaceMITExecutionProvider"])
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=session_options,
                providers=providers,
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            self.input_shape = tuple(self.session.get_inputs()[0].shape[2:4])
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=str(e),
                message=f"PP-LiteSeg model load failed: {e}",
            ) from e

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        scale = min(self.input_shape[0] / h, self.input_shape[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        img_padded = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        img_padded[:new_h, :new_w, :] = img_resized

        img_float = img_padded.astype(np.float32) / 255.0
        img_norm = (img_float - self.mean) / self.std
        img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32)

        return img_tensor, new_h, new_w

    def postprocess(
        self,
        outputs,
        origin_h: int,
        origin_w: int,
        valid_h: int,
        valid_w: int,
        **kwargs,
    ) -> np.ndarray:
        raw_output = outputs[0]

        if raw_output.ndim == 4:
            raw_output = raw_output[0]
        elif raw_output.ndim == 3 and raw_output.shape[0] == 1:
            raw_output = raw_output[0]

        if raw_output.ndim == 3:
            pred_mask = np.argmax(raw_output, axis=0).astype(np.uint8)
        elif raw_output.ndim == 2:
            pred_mask = raw_output.astype(np.uint8)
        else:
            raise ModelInferenceError(
                model_name="PPLiteSeg",
                reason=f"Unsupported output shape: {raw_output.shape}",
                message=f"PP-LiteSeg unsupported output shape: {raw_output.shape}",
            )

        pred_mask = pred_mask[:valid_h, :valid_w]
        pred_mask_origin = cv2.resize(
            pred_mask,
            (origin_w, origin_h),
            interpolation=cv2.INTER_NEAREST,
        )
        return pred_mask_origin

    def infer(self, image: np.ndarray) -> np.ndarray:
        try:
            origin_h, origin_w = image.shape[:2]
            input_tensor, valid_h, valid_w = self.preprocess(image)
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            return self.postprocess(outputs, origin_h, origin_w, valid_h, valid_w)
        except ModelInferenceError:
            raise
        except Exception as e:
            raise ModelInferenceError(
                model_name="PPLiteSeg",
                reason=str(e),
                message=f"PP-LiteSeg inference failed: {e}",
            ) from e
