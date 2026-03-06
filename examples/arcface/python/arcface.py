#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
ArcFace Face Recognition Example using CV Model Factory

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


def parse_args():
    parser = argparse.ArgumentParser(description="ArcFace Face Recognition Example")
    parser.add_argument("--config", type=str, default=None,
                       help="Config yaml path (default: examples/arcface/config/arcface.yaml)")
    parser.add_argument("--image1", type=str, default=None,
                       help="First face image path (default: from config test_image1)")
    parser.add_argument("--image2", type=str, default=None,
                       help="Second face image path (default: from config test_image2)")
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Similarity threshold for face matching")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override model_path in yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        default_config = Path(__file__).parent.parent / "config" / "arcface.yaml"
        config_path = Path(args.config) if args.config else default_config
        config_dir = config_path.parent
        project_root = Path(__file__).parent.parent.parent.parent  # model_zoo/cv
        model_name = config_path.stem

        # 未指定 image1/image2 时从 config 读取默认路径
        if args.image1 is None or args.image2 is None:
            config = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            image1_path = args.image1 or config.get("test_image1")
            image2_path = args.image2 or config.get("test_image2")
            if image1_path and not Path(image1_path).is_absolute():
                image1_path = project_root / image1_path
            if image2_path and not Path(image2_path).is_absolute():
                image2_path = project_root / image2_path
            if not image1_path or not image2_path:
                raise ValueError("请指定 --image1 和 --image2，或在 config 的 yaml 中配置 test_image1、test_image2")
        else:
            image1_path = args.image1
            image2_path = args.image2

        # Create model factory
        factory = ModelFactory()
        print(f"创建 {model_name} 人脸识别器...")
        override_params = {}
        if args.model_path:
            p = Path(args.model_path).expanduser()
            override_params["model_path"] = str(
                p if p.is_absolute() else (project_root / p).resolve()
            )
        recognizer = factory.create_model(
            model_name,
            config_dir=config_dir,
            **override_params,
        )

        # Load images
        print(f"加载第一张图像: {image1_path}")
        image1 = cv2.imread(str(image1_path))
        if image1 is None:
            raise FileNotFoundError(f"无法加载图像: {image1_path}")

        print(f"加载第二张图像: {image2_path}")
        image2 = cv2.imread(str(image2_path))
        if image2 is None:
            raise FileNotFoundError(f"无法加载图像: {image2_path}")

        print(f"图像1尺寸: {image1.shape}")
        print(f"图像2尺寸: {image2.shape}")

        # Extract face embeddings
        print("提取人脸特征...")

        # For ArcFace, we assume the input images are already cropped faces
        # In a real application, you would first detect faces and then crop them

        embedding1 = recognizer.infer(image1)
        embedding2 = recognizer.infer(image2)
        similarity = recognizer.compute_similarity(embedding1, embedding2)

        print(f"相似度: {similarity:.4f}")
        if similarity >= args.threshold:
            print("判断: 同一人")
        else:
            print("判断: 非同一人")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
