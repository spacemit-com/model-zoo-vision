# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified exception types for vision model library.
"""


class VisionModelError(Exception):
    """
    Base exception class for vision model library.

    All other exceptions should inherit from this class.
    """

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ModelNotFoundError(VisionModelError):
    """Raised when model file path is invalid or file does not exist."""

    def __init__(self, model_path: str, message: str = None):
        if message is None:
            message = f"模型文件不存在: {model_path}"
        super().__init__(message, {"model_path": model_path})
        self.model_path = model_path


class ModelLoadError(VisionModelError):
    """Raised when model file exists but cannot be loaded."""

    def __init__(self, model_path: str, reason: str = None, message: str = None):
        if message is None:
            message = f"模型加载失败: {model_path}"
            if reason:
                message += f" - {reason}"
        super().__init__(message, {
            "model_path": model_path,
            "reason": reason or "未知原因"
        })
        self.model_path = model_path
        self.reason = reason


class ModelInferenceError(VisionModelError):
    """Raised when model inference fails."""

    def __init__(self, model_name: str = None, reason: str = None, message: str = None):
        if message is None:
            message = "模型推理失败"
            if model_name:
                message = f"模型推理失败 [{model_name}]"
            if reason:
                message += f": {reason}"
        super().__init__(message, {
            "model_name": model_name or "未知模型",
            "reason": reason or "未知原因"
        })
        self.model_name = model_name
        self.reason = reason


class ModelConfigError(VisionModelError):
    """Raised when model config is invalid, missing or malformed."""

    def __init__(self, config_key: str = None, reason: str = None, message: str = None):
        if message is None:
            message = "模型配置错误"
            if config_key:
                message = f"模型配置错误 [{config_key}]"
            if reason:
                message += f": {reason}"
        super().__init__(message, {
            "config_key": config_key or "未知配置",
            "reason": reason or "未知原因"
        })
        self.config_key = config_key
        self.reason = reason


class ImageProcessingError(VisionModelError):
    """Raised when image decode, preprocess or postprocess fails."""

    def __init__(self, operation: str = None, reason: str = None, message: str = None):
        if message is None:
            message = "图像处理失败"
            if operation:
                message = f"图像处理失败 [{operation}]"
            if reason:
                message += f": {reason}"
        super().__init__(message, {
            "operation": operation or "未知操作",
            "reason": reason or "未知原因"
        })
        self.operation = operation
        self.reason = reason


class UnsupportedTaskError(VisionModelError):
    """Raised when requested task type is not supported."""

    def __init__(self, task: str, supported_tasks: list = None, message: str = None):
        if message is None:
            message = f"不支持的任务类型: {task}"
            if supported_tasks:
                message += f"，支持的任务类型: {', '.join(supported_tasks)}"
        super().__init__(message, {
            "task": task,
            "supported_tasks": supported_tasks or []
        })
        self.task = task
        self.supported_tasks = supported_tasks


class UnsupportedModelError(VisionModelError):
    """Raised when requested model name is not supported."""

    def __init__(self, task: str, model_name: str, supported_models: list = None, message: str = None):
        if message is None:
            message = f"不支持的模型: {task}/{model_name}"
            if supported_models:
                message += f"，支持的模型: {', '.join(supported_models)}"
        super().__init__(message, {
            "task": task,
            "model_name": model_name,
            "supported_models": supported_models or []
        })
        self.task = task
        self.model_name = model_name
        self.supported_models = supported_models


class ModelImportError(VisionModelError):
    """Raised when model class cannot be imported or dependency is missing."""

    def __init__(self, class_path: str, reason: str = None, message: str = None):
        if message is None:
            message = f"无法导入模型类: {class_path}"
            if reason:
                message += f" - {reason}"
        super().__init__(message, {
            "class_path": class_path,
            "reason": reason or "未知原因"
        })
        self.class_path = class_path
        self.reason = reason


class DependencyError(VisionModelError):
    """Raised when required dependency is not installed."""

    def __init__(self, dependency: str, install_command: str = None, message: str = None):
        if message is None:
            message = f"缺少必需的依赖库: {dependency}"
            if install_command:
                message += f"。请运行: {install_command}"
        super().__init__(message, {
            "dependency": dependency,
            "install_command": install_command or ""
        })
        self.dependency = dependency
        self.install_command = install_command
