# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
模型工厂 - 统一的模型创建和管理
"""

import importlib
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

from core.python.vision_model_exceptions import (
    ModelConfigError,
    ModelImportError,
    ModelLoadError,
    ModelNotFoundError,
)
from common.python.config_loader import load_model_config, merge_params

logger = logging.getLogger(__name__)

def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _get_vision_root() -> Path:
    # This file lives at: <...>/cv/src/core/python/vision_model_factory.py
    # Path(__file__).parents[3] == project root
    return Path(__file__).resolve().parents[3]


def _download_script_hint(config_dir: Optional[Path]) -> str:
    """
    Best-effort hint to the example download script.

    When called from examples, config_dir is typically:
      <cv>/examples/<name>/config
    and the script lives at:
      <cv>/examples/<name>/scripts/download_models.sh
    """
    try:
        if config_dir:
            return str(Path(config_dir).resolve().parent / "scripts" / "download_models.sh")
    except Exception:
        pass
    return str(_get_vision_root() / "examples" / "<model>" / "scripts" / "download_models.sh")


def _resolve_and_check_model_path(path_value: Any, config_dir: Optional[Path]) -> str:
    """
    Resolve model path to an absolute local file path and ensure it exists.

    NOTE: model auto-downloading has been removed. If the file is missing,
    raise a clear error telling the user to run the example download script.
    """
    raw = str(path_value)
    if not raw:
        raise FileNotFoundError("Model path is empty")
    if _is_url(raw):
        raise FileNotFoundError(
            f"Model URL is not supported: {raw}. Please download the model first, e.g. run: "
            f"{_download_script_hint(config_dir)}"
        )

    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (_get_vision_root() / p).resolve()

    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {p}. Please download the model first, e.g. run: "
            f"{_download_script_hint(config_dir)}"
        )
    return str(p)


class ModelFactory:
    """
    模型工厂类

    负责根据配置动态创建模型实例，实现配置与代码的解耦。
    """

    _instance = None
    _config = None
    _model_cache = {}

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化工厂"""
        # 不自动加载配置，按需加载
        pass

    def load_config(self, config_path: Optional[str] = None):
        """
        加载 web_models.yaml 配置文件

        Args:
            config_path: 配置文件路径，默认使用 config/web_models.yaml
        """
        if config_path is None:
            # 默认配置路径：web_models.yaml（用于 web 项目）
            # 从 src/core/python/factory.py 到 model_zoo/config/web_models.yaml 需要 4 层 parent
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "web_models.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"配置文件不存在: {config_path}")
            self._config = {"models": {}}
            return

        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        logger.info(f"已加载模型配置: {config_path}")

    def reload_config(self, config_path: Optional[str] = None):
        """
        重新加载配置（支持热更新）

        Args:
            config_path: 配置文件路径
        """
        self._config = None
        self._model_cache.clear()
        self.load_config(config_path)
        logger.info("配置已重新加载，模型缓存已清空")

    def get_model_config(self, task: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        获取模型配置

        Args:
            task: 任务类型（detect, pose, segment, face等）
            model_name: 模型名称

        Returns:
            模型配置字典，如果不存在返回 None
        """
        if self._config is None:
            self.load_config()

        models = self._config.get("models", {})
        task_models = models.get(task, {})
        return task_models.get(model_name)

    def find_model_task(self, model_name: str) -> Optional[str]:
        """
        根据模型名称查找对应的任务类型

        Args:
            model_name: 模型名称（如 'yolov8', 'yolov8_pose' 等）

        Returns:
            任务类型（如 'detect', 'pose' 等），如果未找到返回 None
        """
        if self._config is None:
            self.load_config()

        models = self._config.get("models", {})
        for task, task_models in models.items():
            if model_name in task_models:
                return task
        return None

    def list_models(self, task: Optional[str] = None) -> Dict[str, Any]:
        """
        列出所有可用的模型

        Args:
            task: 任务类型，如果为 None 则返回所有任务的模型

        Returns:
            模型列表字典
        """
        if self._config is None:
            self.load_config()

        models = self._config.get("models", {})

        if task:
            return {task: models.get(task, {})}
        return models


    def _import_model_class(self, class_path: str):
        """
        动态导入模型类

        Args:
            class_path: 模型类的完整路径

        Returns:
            模型类

        Raises:
            ModelImportError: 如果无法导入模型类
        """
        try:
            module_path, class_name = class_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ModelImportError(
                class_path=class_path,
                reason=str(e),
                message=f"无法导入模型类 {class_path}: {e}"
            )

    def _create_model_instance(self, model_class, model_params: Dict[str, Any],
                              cache_key: str, cache: bool):
        """
        创建并缓存模型实例

        Args:
            model_class: 模型类
            model_params: 模型参数字典
            cache_key: 缓存键
            cache: 是否使用缓存

        Returns:
            模型实例

        Raises:
            ModelNotFoundError: 模型文件不存在
            ModelLoadError: 模型加载失败
        """

        try:
            model_instance = model_class(**model_params)
            logger.info(f"成功创建模型: {cache_key}")

            if cache:
                self._model_cache[cache_key] = model_instance

            return model_instance
        except ModelNotFoundError:
            raise
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(
                model_path=model_params.get("model_path", "未知路径"),
                reason=str(e),
                message=f"创建模型失败 {cache_key}: {e}"
            )

    def create_model(self,
                    model_name: str,
                    base_dir: Optional[Path] = None,
                    cache: bool = True,
                    config_dir: Optional[Path] = None,
                    **override_params):
        """
        创建模型实例（从单个模型的 yaml 文件加载）

        默认从 config/{model_name}.yaml 加载配置。
        model_path 直接从 yaml 文件中读取，不做任何处理。

        Args:
            model_name: 模型名称（如 'yolov8', 'yolov5' 等）
            base_dir: 已废弃，不再使用
            cache: 是否使用缓存
            config_dir: 配置文件目录，为 None 时使用默认 model_zoo/cv/config
            **override_params: 覆盖默认参数（包括 model_path 等）

        Returns:
            模型实例

        Raises:
            ModelConfigError: 如果模型配置不存在
            ModelImportError: 如果无法导入模型类
            ModelLoadError: 如果模型加载失败

        Examples:
            >>> from core.python.vision_model_factory import create_model
            >>> detector = create_model("yolov8")
        """
        # 缓存键直接使用传入的模型名称
        cache_key = model_name

        # 检查缓存
        if cache and cache_key in self._model_cache:
            logger.debug(f"使用缓存的模型: {cache_key}")
            return self._model_cache[cache_key]

        # 从单个模型的 yaml 文件加载配置
        # 要求 config 目录下存在同名的 {model_name}.yaml
        config = load_model_config(model_name, config_dir=config_dir)
        if not config:
            raise ModelConfigError(
                config_key=model_name,
                reason="配置文件不存在",
                message=f"未找到模型配置文件: {model_name}.yaml"
            )

        # 获取模型类路径
        class_path = config.get("class")
        if not class_path:
            raise ModelConfigError(
                config_key=model_name,
                reason="缺少 'class' 字段",
                message=f"模型配置缺少 'class' 字段: {model_name}"
            )

        # 导入模型类
        model_class = self._import_model_class(class_path)

        # 准备模型参数
        default_params = config.get("default_params", {})
        model_params = merge_params(default_params, override_params)

        # 直接从 config 顶层获取所有路径参数（如果存在且未被覆盖）
        # 支持 model_path 以及多路径模型的路径参数（emotion_model_path, face_detector_path 等）
        path_keys = [
            "model_path",
            "emotion_model_path",
            "face_detector_path",
            "vit_model_path",
            "decoder_model_path",
            "decoder_path",
            "encoder_path",
            "detector_model_path",
            "recognizer_model_path",
        ]
        for key in path_keys:
            if key in config and key not in model_params:
                model_params[key] = config[key]

        # 确保所有路径类参数对应的模型文件存在（不再自动下载）
        for key in path_keys:
            if key in model_params and model_params[key]:
                try:
                    model_params[key] = _resolve_and_check_model_path(model_params[key], config_dir=config_dir)
                except (FileNotFoundError, OSError) as e:
                    raise ModelNotFoundError(
                        model_path=str(model_params[key]),
                        message=str(e),
                    ) from e

        # 创建模型实例
        return self._create_model_instance(model_class, model_params, cache_key, cache)

    def create_model_from_list(self,
                        model_config: Optional[Dict[str, Any]] = None,
                        model_name: Optional[str] = None,
                        task: Optional[str] = None,
                        base_dir: Optional[Path] = None,
                        cache: bool = True,
                        **override_params):
        """
        从 web_models.yaml 创建模型实例（专用于 web 项目）

        支持两种方式：
        1. 传入完整的模型配置字典（推荐，从模型列表中查找后传入）
        2. 传入 model_name 和 task（向后兼容）

        Args:
            model_config: 完整的模型配置字典（包含 task, model_name, class, path 等）
            model_name: 模型名称（如 'yolov8', 'yolov8_pose' 等），如果提供了 model_config 则忽略
            task: 任务类型（可选），如果提供了 model_config 则忽略
            base_dir: 已废弃，不再使用
            cache: 是否使用缓存
            **override_params: 覆盖默认参数（包括 model_path 等）

        Note:
            model_path 直接从 web_models.yaml 中读取，不做任何处理。

        Returns:
            模型实例

        Raises:
            ModelConfigError: 如果模型配置不存在
            ModelImportError: 如果无法导入模型类
            ModelLoadError: 如果模型加载失败
        """


        # 如果提供了完整的模型配置，直接使用
        if model_config is not None:
            task = model_config.get("task")
            model_name = model_config.get("model_name")
            config = model_config.get("config", {})
        else:
            # 向后兼容：从 web_models.yaml 查找配置
            # 确保已加载 web_models.yaml
            if self._config is None or not self._config.get("models"):
                self.load_config()

            # 如果没有提供 task，自动查找
            if task is None:
                task = self.find_model_task(model_name)
                if task is None:
                    raise ModelConfigError(
                        config_key=model_name,
                        reason="模型未找到",
                        message=f"未找到模型配置: {model_name}，请检查 web_models.yaml"
                    )

            # 从 web_models.yaml 获取配置
            config = self.get_model_config(task, model_name)
            if config is None:
                raise ModelConfigError(
                    config_key=f"{task}/{model_name}",
                    reason="配置不存在",
                    message=f"未找到模型配置: {task}/{model_name}，请检查 web_models.yaml"
                )

        if not task or not model_name:
            raise ModelConfigError(
                config_key="model_config",
                reason="缺少必要字段",
                message="模型配置必须包含 task 和 model_name 字段"
            )

        # 确定缓存键
        cache_key = f"{task}/{model_name}"

        # 检查缓存
        if cache and cache_key in self._model_cache:
            logger.debug(f"使用缓存的模型: {cache_key}")
            return self._model_cache[cache_key]

        # 获取模型类路径
        class_path = config.get("class")
        if not class_path:
            raise ModelConfigError(
                config_key=f"{task}/{model_name}",
                reason="缺少 'class' 字段",
                message=f"模型配置缺少 'class' 字段: {task}/{model_name}"
            )

        # 导入模型类
        model_class = self._import_model_class(class_path)

        # 准备模型参数
        default_params = config.get("default_params", {}).copy()
        model_params = merge_params(default_params, override_params)

        # 处理 path 字段（web_models.yaml 使用 path，需要转换为 model_path）
        if "path" in config and "model_path" not in model_params:
            model_params["model_path"] = config["path"]

        # 直接从 config 顶层获取所有路径参数（如果存在且未被覆盖）
        # 支持 model_path 以及多路径模型的路径参数（emotion_model_path, face_detector_path 等）
        path_keys = [
            "model_path",
            "emotion_model_path",
            "face_detector_path",
            "vit_model_path",
            "decoder_model_path",
            "decoder_path",
            "encoder_path",
            "detector_model_path",
            "recognizer_model_path",
        ]
        for key in path_keys:
            if key in config and key not in model_params:
                model_params[key] = config[key]

        # 确保所有路径类参数对应的模型文件存在（不再自动下载）
        for key in path_keys:
            if key in model_params and model_params[key]:
                try:
                    model_params[key] = _resolve_and_check_model_path(model_params[key], config_dir=None)
                except (FileNotFoundError, OSError) as e:
                    raise ModelNotFoundError(
                        model_path=str(model_params[key]),
                        message=str(e),
                    ) from e

        # 创建模型实例
        return self._create_model_instance(model_class, model_params, cache_key, cache)

    def get_model_list(self) -> List[Dict[str, Any]]:
        """
        获取所有模型的列表（用于前端选择）

        Returns:
            模型列表，每个元素包含：
            - task: 任务类型
            - model_name: 模型名称
            - config: 模型配置字典（包含 class, path, default_params 等）
        """
        if self._config is None or not self._config.get("models"):
            self.load_config()

        model_list = []
        models = self._config.get("models", {})

        for task, task_models in models.items():
            for model_name, config in task_models.items():
                model_list.append({
                    "task": task,
                    "model_name": model_name,
                    "config": config
                })

        return model_list

    def find_model_by_name(self, task: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        根据 task 和 model_name 查找模型配置

        Args:
            task: 任务类型
            model_name: 模型名称

        Returns:
            模型配置字典，包含 task, model_name, config，如果未找到返回 None
        """
        if self._config is None or not self._config.get("models"):
            self.load_config()

        config = self.get_model_config(task, model_name)
        if config is None:
            return None

        return {
            "task": task,
            "model_name": model_name,
            "config": config
        }

    def clear_cache(self, task: Optional[str] = None, model_name: Optional[str] = None):
        """
        清除模型缓存

        Args:
            task: 任务类型，如果为 None 则清除所有缓存
            model_name: 模型名称，如果为 None 则清除该任务的所有模型
        """
        if task is None:
            self._model_cache.clear()
            logger.info("已清除所有模型缓存")
        elif model_name is None:
            # 清除特定任务的所有模型
            keys_to_remove = [k for k in self._model_cache.keys() if k.startswith(f"{task}/")]
            for key in keys_to_remove:
                del self._model_cache[key]
            logger.info(f"已清除任务 {task} 的所有模型缓存")
        else:
            # 清除特定模型
            cache_key = f"{task}/{model_name}"
            if cache_key in self._model_cache:
                del self._model_cache[cache_key]
                logger.info(f"已清除模型缓存: {cache_key}")


# 全局工厂实例
_factory = ModelFactory()


def create_model(model_name: str, **kwargs):
    """
    便捷函数：从单个模型的 yaml 文件创建模型实例

    默认从 config/{model_name}.yaml 加载配置。

    Args:
        model_name: 模型名称（如 'yolov8', 'yolov5' 等）
        **kwargs: 其他参数（必须包含 model_path 等必需参数）

    Returns:
        模型实例

        Examples:
            >>> from core.python.vision_model_factory import create_model
        >>> detector = create_model("yolov8", model_path="path/to/model.onnx")
    """
    return _factory.create_model(model_name, **kwargs)


def list_models(task: Optional[str] = None):
    """
    便捷函数：列出可用模型

    Args:
        task: 任务类型

    Returns:
        模型列表
    """
    return _factory.list_models(task)


def reload_config(config_path: Optional[str] = None):
    """
    便捷函数：重新加载配置

    Args:
        config_path: 配置文件路径
    """
    _factory.reload_config(config_path)


def find_model_task(model_name: str) -> Optional[str]:
    """
    便捷函数：根据模型名称查找对应的任务类型

    Args:
        model_name: 模型名称（如 'yolov8', 'yolov8_pose' 等）

    Returns:
        任务类型（如 'detect', 'pose' 等），如果未找到返回 None
    """
    return _factory.find_model_task(model_name)


def create_model_by_name(model_name: str, base_dir: Optional[Path] = None, **override_params):
    """
    便捷函数：根据模型名称自动创建模型实例（已废弃，请使用 create_model_from_list）

    这个函数现在调用 create_model_from_list，保留用于向后兼容。
    建议直接使用 create_model_from_list。

    Args:
        model_name: 模型名称（如 'yolov8', 'yolov8_pose' 等）
        base_dir: 已废弃，不再使用
        **override_params: 覆盖默认参数

    Returns:
        模型实例

        Examples:
            >>> from core.python.vision_model_factory import create_model_by_name
        >>> detector = create_model_by_name("yolov8")
    """
    return _factory.create_model_from_list(
        model_name=model_name,
        base_dir=base_dir,
        **override_params
    )

