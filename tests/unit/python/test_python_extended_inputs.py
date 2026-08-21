# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Public Python surface tests for stereo/features/matching/tracking additions."""

import pytest


def _extension():
    spacemit_vision = pytest.importorskip(
        "spacemit_vision",
        reason="spacemit_vision wheel not installed",
    )
    if not spacemit_vision.extension_available():
        pytest.skip("spacemit_vision native extension not built/installed")
    return spacemit_vision


def test_extended_types_and_methods_are_exported():
    module = _extension()
    assert module.VisionServiceLocalFeatures is not None
    for method in (
        "infer_stereo",
        "infer_depth",
        "extract_local_features",
        "match_local_features",
        "track",
    ):
        assert hasattr(module.VisionServiceNative, method)


def test_local_features_round_trip():
    module = _extension()
    features = module.VisionServiceLocalFeatures()
    point = module.VisionServiceKeypoint()
    point.x = 10.0
    point.y = 20.0
    point.visibility = 0.9
    features.keypoints = [point]
    features.descriptors = [1.0, 2.0]
    features.descriptor_dim = 2
    features.image_width = 100
    features.image_height = 80
    features.feature_type = "unit"

    assert len(features.keypoints) == 1
    assert features.descriptors == [1.0, 2.0]
    assert features.descriptor_dim == 2
    assert features.image_width == 100
    assert features.image_height == 80
    assert features.feature_type == "unit"


def test_flat_result_has_extended_fields():
    module = _extension()
    result_type = module.VisionServiceResult
    for field in (
        "disparity",
        "depth",
        "descriptors",
        "descriptor_dim",
        "image_width",
        "image_height",
        "feature_type",
        "query_index",
        "train_index",
        "query_point",
        "train_point",
    ):
        assert hasattr(result_type, field)
