#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
MobileNetV1 Image Classification Example using CV Model Factory

MobileNetV1 输出 1001 类（含 background），使用 deploy.mobilenet.MobileNetV1Classifier
截掉背景类后对齐 1000 类 ImageNet 标签。

运行方式：通过 --config 指定 yaml 路径（与 yolov8.py 一致）。
"""

import sys
import argparse
from pathlib import Path
import cv2
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from core.python.vision_model_factory import ModelFactory


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def load_imagenet_labels(label_file: Path) -> list:
    """ImageNet 标签文件：每行「WordNet_ID 类别名」，只取类别名。"""
    labels = []
    with open(label_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, _, name = line.partition(" ")
            labels.append(name if name else line)
    return labels


def find_label_file(path_value: str, project_root: Path, config_dir: Path):
    p = Path(path_value).expanduser()
    if p.is_absolute():
        return p if p.is_file() else None
    for base in (project_root, config_dir.parent.parent):
        candidate = (base / p).resolve()
        if candidate.is_file():
            return candidate
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="MobileNetV1 Classification Example")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config yaml path (default: examples/mobilenetv1/config/mobilenetv1.yaml)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Input image path (if not provided, uses config default)",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Show top-k predictions")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Override model_path in yaml"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        default_config = Path(__file__).parent.parent / "config" / "mobilenetv1.yaml"
        config_path = Path(args.config) if args.config else default_config
        config_dir = config_path.parent
        project_root = Path(__file__).parent.parent.parent.parent  # model_zoo/cv
        model_name = config_path.stem

        # Create model factory
        factory = ModelFactory()

        # Create MobileNet classifier
        print(f"创建 {model_name} 分类器...")
        override_params = {}
        if args.model_path:
            p = Path(args.model_path).expanduser()
            override_params["model_path"] = str(
                p if p.is_absolute() else (project_root / p).resolve()
            )
        classifier = factory.create_model(
            model_name,
            config_dir=config_dir,
            **override_params,
        )

        # Get model config from specified yaml (test_image, label_file_path)
        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Use provided image or default from config
        if args.image:
            image_path = resolve_path(args.image, project_root)
        else:
            image_path = config.get("test_image", "test_data/images/cat.jpg")
            image_path = resolve_path(image_path, project_root)

        # Load image
        print(f"加载图像: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")

        print(f"图像尺寸: {image.shape}")

        labels = []
        label_rel = config.get("label_file_path")
        if label_rel:
            label_file = find_label_file(label_rel, project_root, config_dir)
            if label_file:
                labels = load_imagenet_labels(label_file)
                print(f"加载标签文件: {label_file} ({len(labels)} 个)")
            else:
                print(f"警告: 未找到标签文件 {label_rel}")

        # Run classification（ResNetClassifier 使用 predict_top_k，返回 [(class_name, score), ...]）
        print("运行图像分类...")
        predictions = classifier.predict_top_k(image, labels or [], k=args.top_k)

        print(f"\nTop-{args.top_k} 预测结果:")
        for i, (class_name, confidence) in enumerate(predictions):
            print(f"  {i + 1}: {class_name} ({confidence:.4f})")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
