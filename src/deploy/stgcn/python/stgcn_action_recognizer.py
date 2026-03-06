# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
STGCN / TSSTG Action Recognition Implementation

基于 TSSTG 的 30 帧骨架动作识别（ONNX），用于跌倒等动作分类。
输入为骨架点序列 (t, 13, 3)，输出 7 类概率（含 Fall Down）。
"""

import numpy as np
import onnxruntime as ort
import spacemit_ort  # noqa: F401 - register SpaceMITExecutionProvider
from pathlib import Path
from typing import Tuple, Optional, List

from core.python.vision_model_base import _get_vision_root
from core.python.vision_model_exceptions import ModelNotFoundError, ModelLoadError

# TSSTG：30 帧，13 个关键点（COCO 子集 + center 在模型内加）
SEQUENCE_LENGTH = 30
NUM_KEYPOINTS = 13
CLASS_NAMES = [
    'Standing', 'Walking', 'Sitting', 'Lying Down',
    'Stand up', 'Sit down', 'Fall Down',
]
FALL_DOWN_CLASS_INDEX = 6


def _normalize_points_with_size(pts_xy: np.ndarray, width: float, height: float) -> np.ndarray:
    """将像素坐标归一化到 [0, 1]。"""
    pts_xy = np.asarray(pts_xy, dtype=np.float32)
    if width <= 0 or height <= 0:
        return pts_xy
    pts_xy[..., 0] = pts_xy[..., 0] / width
    pts_xy[..., 1] = pts_xy[..., 1] / height
    return pts_xy


def _scale_pose(pts_xy: np.ndarray) -> np.ndarray:
    """将 [0,1] 坐标缩放到约 [-1, 1]。"""
    pts_xy = np.asarray(pts_xy, dtype=np.float32)
    pts_xy = (pts_xy - 0.5) * 2.0
    return pts_xy


class StgcnActionRecognizer:
    """
    STGCN/TSSTG 动作识别器（ONNX）。

    输入骨架序列 (t, 13, 3)，t=30，13 关键点，3 通道 (x, y, score)；
    输出 7 类动作概率，class 6 为 Fall Down。
    """

    def __init__(
        self,
        model_path: str,
        num_threads: int = 4,
        lazy_load: bool = False,
        **kwargs,
    ):
        """
        Args:
            model_path: ONNX 模型路径（相对路径时相对于 cv 根目录解析）
            num_threads: ONNX Runtime 线程数
            lazy_load: 是否延迟加载模型
            **kwargs: 如 providers 等，供 _load_model 使用
        """
        p = Path(model_path)
        if not p.is_absolute():
            p = (_get_vision_root() / model_path).resolve()
        self.model_path = str(p)
        self.num_threads = num_threads
        self._load_kwargs = kwargs
        self.session: Optional[ort.InferenceSession] = None
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        self.class_names = list(CLASS_NAMES)
        self.sequence_length = SEQUENCE_LENGTH
        self.num_keypoints = NUM_KEYPOINTS
        self.fall_down_class_index = FALL_DOWN_CLASS_INDEX

        if not lazy_load:
            self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """加载 ONNX 模型。"""
        if not Path(self.model_path).exists():
            raise ModelNotFoundError(
                model_path=self.model_path,
                message=f"STGCN/TSSTG 模型文件不存在: {self.model_path}",
            )
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.num_threads
        providers = kwargs.get("providers", ["CPUExecutionProvider"])
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=opts,
                providers=providers,
            )
        except Exception as e:
            raise ModelLoadError(
                model_path=self.model_path,
                reason=str(e),
                message=f"STGCN/TSSTG 模型加载失败: {self.model_path} - {e}",
            )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def get_class_name(self, pred_class: int) -> str:
        """根据预测类别索引返回动作名称。"""
        if 0 <= pred_class < len(self.class_names):
            return self.class_names[pred_class]
        return str(pred_class)

    def predict(self, pts: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """
        对骨架序列做动作识别。

        Args:
            pts: (t, v, c)，t=30 帧，v=13 关键点，c=3 (x, y, score)，像素坐标
            image_size: (width, height) 图像尺寸，用于归一化

        Returns:
            概率数组 shape (1, 7)，索引 6 为 Fall Down
        """
        if self.session is None:
            self._load_model(**self._load_kwargs)
        pts = np.asarray(pts, dtype=np.float32).copy()
        w, h = image_size[0], image_size[1]
        pts[:, :, :2] = _normalize_points_with_size(pts[:, :, :2].copy(), w, h)
        pts[:, :, :2] = _scale_pose(pts[:, :, :2])
        # 增加 shoulder 中心点（索引 1、2 为左右肩）
        center = (pts[:, 1, :] + pts[:, 2, :]) / 2
        pts = np.concatenate((pts, np.expand_dims(center, 1)), axis=1)
        # (t, v+1, c) -> (1, c, t, v+1)
        pts_tensor = np.transpose(pts.astype(np.float32), (2, 0, 1))[np.newaxis, :]
        mot = pts_tensor[:, :2, 1:, :] - pts_tensor[:, :2, :-1, :]
        mot = mot.astype(np.float32)
        batch_size, channels, time_steps, num_nodes = mot.shape
        mot_padded = np.zeros((batch_size, channels, time_steps + 1, num_nodes), dtype=np.float32)
        mot_padded[:, :, 1:, :] = mot
        mot = mot_padded
        input_dict = {
            self.input_names[0]: pts_tensor,
            self.input_names[1]: mot,
        }
        outputs = self.session.run(self.output_names, input_dict)
        return outputs[0]
