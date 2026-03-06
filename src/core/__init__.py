# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Vision Core Module - Base Model Class and Core Utilities
"""

from .python import BaseModel
from .python.vision_model_exceptions import (
    VisionModelError,
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
    ModelConfigError,
    ImageProcessingError,
    UnsupportedTaskError,
    UnsupportedModelError,
    ModelImportError,
    DependencyError,
)
from common.python.config_loader import (
    load_model_config,
    get_default_params,
    merge_params,
)
from .python.vision_model_factory import (
    ModelFactory,
    create_model,
    create_model_by_name,
    list_models,
    reload_config,
    find_model_task,
)

CVModelError = VisionModelError  # backward compatibility

__all__ = [
    'BaseModel',
    # Exceptions
    'VisionModelError',
    'CVModelError',  # alias
    'ModelNotFoundError',
    'ModelLoadError',
    'ModelInferenceError',
    'ModelConfigError',
    'ImageProcessingError',
    'UnsupportedTaskError',
    'UnsupportedModelError',
    'ModelImportError',
    'DependencyError',
    # Config loader
    'load_model_config',
    'get_default_params',
    'merge_params',
    # Factory
    'ModelFactory',
    'create_model',
    'create_model_by_name',
    'list_models',
    'reload_config',
    'find_model_task',
]

