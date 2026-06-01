# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Python invalid-input and pure-logic smoke tests (no spacemit_ort backend)."""

import numpy as np
import pytest

from common.python.image_processing import letterbox
from common.python.nms import nms
from core.python.vision_model_exceptions import (
    ModelConfigError,
    ModelImportError,
    ModelNotFoundError,
)
from core.python.vision_model_factory import create_model


class TestModelFactoryInvalidInput:
    def test_nonexistent_config_raises_model_config_error(self, data_dir):
        with pytest.raises(ModelConfigError) as exc_info:
            create_model("nonexistent_model_xyz", config_dir=data_dir, cache=False)
        msg = str(exc_info.value)
        assert "nonexistent_model_xyz" in msg

    def test_missing_class_field_raises_model_config_error(self, data_dir):
        with pytest.raises(ModelConfigError) as exc_info:
            create_model("invalid_model", config_dir=data_dir, cache=False)
        msg = str(exc_info.value)
        assert "class" in msg.lower() or "invalid_model" in msg

    def test_bad_class_path_raises_model_import_error(self, data_dir):
        with pytest.raises(ModelImportError) as exc_info:
            create_model("bad_class", config_dir=data_dir, cache=False)
        msg = str(exc_info.value)
        assert "totally_fake_module_xyz" in msg or "NonExistentDetector" in msg

    def test_http_model_url_rejected(self, data_dir):
        # create_model catches the underlying FileNotFoundError ("URL is not
        # supported") and re-raises it as ModelNotFoundError, so the public
        # API surface for a URL model_path is ModelNotFoundError.
        with pytest.raises(ModelNotFoundError) as exc_info:
            create_model(
                "missing_model",
                config_dir=data_dir,
                cache=False,
                model_path="https://example.com/model.onnx",
            )
        msg = str(exc_info.value)
        assert "http" in msg.lower() or "url" in msg.lower()

    def test_missing_model_file_raises_model_not_found(self, data_dir):
        with pytest.raises(ModelNotFoundError) as exc_info:
            create_model("missing_model", config_dir=data_dir, cache=False)
        msg = str(exc_info.value)
        assert "vision_test_missing_model_xyz" in msg or "not found" in msg.lower()


class TestPureLogicSmoke:
    def test_nms_keeps_non_overlapping_and_best_overlap(self):
        boxes = np.array(
            [
                [10, 10, 50, 50],
                [15, 15, 55, 55],
                [200, 200, 250, 250],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.7, 0.8], dtype=np.float32)
        keep = nms(boxes, scores, iou_threshold=0.45)
        assert list(keep) == [0, 2]

    def test_nms_empty_boxes_returns_empty_array(self):
        keep = nms(np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.float32))
        assert keep.size == 0

    def test_letterbox_480x640_to_640x640(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        out, ratio, pad = letterbox(image, new_shape=(640, 640))
        assert out.shape == (640, 640, 3)
        assert abs(ratio - 1.0) < 1e-5
        assert abs(pad[0] - 0.0) < 1e-5
        assert abs(pad[1] - 80.0) < 1e-5
