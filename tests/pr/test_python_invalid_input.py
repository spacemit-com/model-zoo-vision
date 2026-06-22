# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Invalid-input tests for the spacemit_vision wheel API (create() error surface)."""

import pytest

_SVC = None  # module-level handle, populated lazily


def _native():
    """Return VisionServiceNative, skipping if the extension is unavailable."""
    global _SVC
    if _SVC is None:
        spacemit_vision = pytest.importorskip(
            "spacemit_vision",
            reason="spacemit_vision wheel not installed",
        )
        if not spacemit_vision.extension_available():
            pytest.skip("spacemit_vision native extension not built/installed")
        _SVC = spacemit_vision.VisionServiceNative
    return _SVC


class TestCreateInvalidInput:
    def test_nonexistent_config_fails(self, data_dir):
        svc = _native()
        with pytest.raises(Exception):
            svc.create(str(data_dir / "nonexistent_model_xyz.yaml"))

    def test_missing_class_field_fails(self, data_dir):
        svc = _native()
        with pytest.raises(Exception):
            svc.create(str(data_dir / "invalid_model.yaml"))

    def test_bad_class_path_fails(self, data_dir):
        svc = _native()
        with pytest.raises(Exception):
            svc.create(str(data_dir / "bad_class.yaml"))

    def test_missing_model_file_fails(self, data_dir):
        svc = _native()
        with pytest.raises(Exception):
            svc.create(str(data_dir / "missing_model.yaml"))
