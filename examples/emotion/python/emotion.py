#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Emotion recognition example using spacemit_vision.

运行方式：通过 --config 指定 yaml 路径
→ VisionServiceNative.create(config_path) 创建模型
→ 从 yaml 读 test_image、label_file_path 等参数做推理。
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def load_labels(label_file: str) -> list:
    """读取标签文件：每行一个类别名。"""
    labels = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(line)
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Recognition Example")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config yaml path (default: examples/emotion/config/emotion.yaml)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Input image path (if not provided, uses config default)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result_emotion.jpg",
        help="Output image path",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model_path in yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        default_config = Path(__file__).parent.parent / "config" / "emotion.yaml"
        config_path = Path(args.config) if args.config else default_config
        project_root = Path(__file__).parent.parent.parent.parent
        model_name = config_path.stem

        print(f"创建 {model_name} 情绪识别器...")
        model_path_override = ""
        if args.model_path:
            p = Path(args.model_path).expanduser()
            model_path_override = str(
                p if p.is_absolute() else (project_root / p).resolve()
            )
        recognizer = VisionServiceNative.create(
            str(config_path), model_path_override=model_path_override
        )

        # Load config from specified yaml
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Determine image path
        image_path = args.image or config.get("test_image", "~/.cache/assets/image/003_face0.png")
        image_path = resolve_path(image_path, project_root)

        print(f"加载图像: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")

        print(f"图像尺寸: {image.shape}")

        # Load labels
        labels = None
        label_file_path = config.get("label_file_path")
        if label_file_path:
            label_file_path = resolve_path(label_file_path, project_root)
            try:
                labels = load_labels(str(label_file_path))
                print(f"加载标签文件: {label_file_path} ({len(labels)} 个标签)")
            except Exception as e:
                print(f"警告: 无法加载标签文件 {label_file_path}: {e}")

        # Use default labels if not loaded
        if not labels:
            labels = [
                "neutral", "happiness", "sadness", "surprise",
                "fear", "disgust", "anger",
            ]

        print("运行情绪识别...")
        status, results = recognizer.infer_image(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(recognizer.last_error())

        if not results:
            print("未得到情绪识别结果。")
            return 0

        scores = list(results[0].class_scores)
        if not scores:
            print("情绪识别结果的类别分数为空。")
            return 0

        emotion_class = int(np.argmax(scores))
        emotion_score = float(scores[emotion_class])

        emotion_name = (
            labels[emotion_class]
            if 0 <= emotion_class < len(labels)
            else "unknown"
        )
        print(f"Emotion: {emotion_name} (class {emotion_class}, score: {emotion_score:.4f})")

        result_image = image.copy()
        label_text = f"{emotion_name} {emotion_score:.2f}"
        cv2.putText(
            result_image,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(args.output, result_image)
        print(f"结果已保存到: {args.output}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
