# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
配置加载器 - 从 yaml 文件加载模型默认配置
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def _get_vision_root() -> Path:
    """Return project root (config_loader is at cv/src/common/python/)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def load_app_config(app_name: str) -> Dict[str, Any]:
    """
    按应用名加载应用配置 yaml，路径固定为：
    applications/<app_name>/config/<app_name>.yaml

    只负责读文件并返回字典，不解析具体字段。各示例在各自 py 里按需用
    app_config.get("xxx") 等显式取用自己需要的参数名，便于维护和扩展。

    Args:
        app_name: 应用名，与目录名一致，如 'emotion_detection', 'fall_detection'

    Returns:
        配置字典，文件不存在或解析失败时返回空字典
    """
    root = _get_vision_root()
    config_file = root / "applications" / app_name / "config" / f"{app_name}.yaml"
    if not config_file.exists():
        logger.debug(f"应用配置文件不存在: {config_file}")
        return {}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config or {}
    except Exception as e:
        logger.error(f"加载应用配置文件失败 {config_file}: {e}")
        return {}


def load_model_config(model_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    加载指定模型的配置文件

    Args:
        model_name: 模型名称（如 'yolov8', 'yolov5' 等）
        config_dir: 配置文件目录，默认为 model_zoo/config

    Returns:
        模型配置字典，包含 'class' 和 'default_params'
        如果配置文件不存在，返回空字典
    """
    if config_dir is None:
        config_dir = _get_vision_root() / "config"

    config_file = config_dir / f"{model_name}.yaml"

    if not config_file.exists():
        logger.warning(f"配置文件不存在: {config_file}")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        logger.debug(f"已加载模型配置: {config_file}")
        return config or {}
    except Exception as e:
        logger.error(f"加载配置文件失败 {config_file}: {e}")
        return {}


def get_default_params(model_name: str, config_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    获取模型的默认参数

    Args:
        model_name: 模型名称
        config_dir: 配置文件目录

    Returns:
        默认参数字典
    """
    config = load_model_config(model_name, config_dir)
    return config.get("default_params", {})


def merge_params(default_params: Dict[str, Any], override_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并默认参数和覆盖参数

    Args:
        default_params: 默认参数
        override_params: 覆盖参数（优先级更高）

    Returns:
        合并后的参数字典
    """
    merged = default_params.copy()
    merged.update(override_params)
    return merged
