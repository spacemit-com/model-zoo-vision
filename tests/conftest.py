# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for vision module tests."""

import sys
from pathlib import Path

import pytest

MODULE_ROOT = Path(__file__).resolve().parent.parent
SRC = MODULE_ROOT / "src"
PR_DIR = Path(__file__).resolve().parent / "pr"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(PR_DIR))


@pytest.fixture
def models_dir():
    return Path.home() / ".cache" / "models" / "vision"


@pytest.fixture
def assets_dir():
    return Path.home() / ".cache" / "assets"


@pytest.fixture
def data_dir():
    return Path(__file__).resolve().parent / "data"
