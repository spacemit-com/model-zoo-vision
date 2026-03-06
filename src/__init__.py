# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
ModelZoo - Vision Model Deployment Library
"""

__version__ = "0.1.0"

# Re-export core modules for backward compatibility
from .core import (
    BaseModel,
    ModelFactory,
    create_model,
    create_model_by_name,
    list_models,
    reload_config,
    find_model_task,
    # Exceptions
    CVModelError,
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

# Re-export common utilities
from .common import (
    # Config loader
    load_model_config,
    get_default_params,
    merge_params,
    # Common utilities (will be imported as needed)
)

__all__ = [
    # Version
    '__version__',
    # Core
    'BaseModel',
    'ModelFactory',
    'create_model',
    'create_model_by_name',
    'list_models',
    'reload_config',
    'find_model_task',
    # Exceptions
    'CVModelError',  # backward compat (alias of VisionModelError)
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
]
