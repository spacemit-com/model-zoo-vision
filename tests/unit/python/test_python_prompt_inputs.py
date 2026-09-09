# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Exercise wrapper, pybind, service forwarding and MobileSAM validation."""

import numpy as np
import pytest


@pytest.fixture
def prompt_service(tmp_path):
    module = pytest.importorskip("spacemit_vision")
    if not module.extension_available():
        pytest.skip("spacemit_vision native extension not built/installed")
    config = tmp_path / "config.yaml"
    config.write_text(
        "class: deploy.mobilesam1.MobileSAMSegmentor\n"
        "model_path: missing-encoder.onnx\n"
        "default_params:\n"
        "  decoder_model_path: missing-decoder.onnx\n"
        "  providers: [CPUExecutionProvider]\n",
        encoding="utf-8",
    )
    return module, module.VisionServiceNative.create(str(config), lazy_load=True)


@pytest.mark.parametrize(
    "points,labels,error",
    [
        ([], [], "requires matching points and labels"),
        ([[5, 5]], [], "requires matching points and labels"),
        ([[5, 5]], [1, 0], "requires matching points and labels"),
        ([], [1], "requires matching points and labels"),
        ([[5, 5]], [4], "invalid point or label"),
        ([[5, 5]], [-2], "invalid point or label"),
        ([[float("nan"), 5]], [1], "invalid point or label"),
        ([[5, float("inf")]], [1], "invalid point or label"),
        ([[5, 5]], [2], "invalid box corners"),
        ([[5, 5]], [3], "missing top-left box corner"),
        ([[5, 5], [10, 10]], [3, 2], "missing top-left box corner"),
        ([[5, 5], [10, 10]], [2, 1], "invalid box corners"),
        ([[5, 5], [5, 10]], [2, 3], "invalid box corners"),
        ([[5, 5], [10, 5]], [2, 3], "invalid box corners"),
        ([[10, 10], [5, 5]], [2, 3], "invalid box corners"),
    ],
)
def test_prompt_contract(prompt_service, points, labels, error):
    module, service = prompt_service
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    # Repeat on the same instance: validation must not mutate prompts or state.
    for _ in range(2):
        status, results = service.infer_image_points(image, points, labels)
        assert status == module.VisionServiceStatus.INFER_FAILED
        assert results == []
        assert error in service.last_error()


@pytest.mark.parametrize("point", [[], [5], [5, 5, 5]])
def test_point_requires_two_coordinates(prompt_service, point):
    _, service = prompt_service
    with pytest.raises(ValueError, match="Each point must contain x,y"):
        service.infer_image_points(
            np.zeros((32, 32, 3), dtype=np.uint8), [point], [1]
        )


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((32, 32, 3), dtype=np.float32),
        np.zeros((32, 32), dtype=np.uint8),
        np.zeros((32, 32, 4), dtype=np.uint8),
    ],
)
def test_prompt_image_requires_bgr8(prompt_service, image):
    _, service = prompt_service
    with pytest.raises(TypeError, match="uint8 BGR"):
        service.infer_image_points(image, [[5, 5]], [1])
