# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight stub detector for path-check tests (no onnxruntime import)."""


class StubDetector:
    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
