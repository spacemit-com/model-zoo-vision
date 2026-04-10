#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
PP-LiteSeg semantic segmentation example using CV Model Factory.
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from core.python.vision_model_factory import ModelFactory


def parse_args():
    parser = argparse.ArgumentParser(description="PP-LiteSeg Semantic Segmentation Example")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config yaml path (default: examples/pp_liteseg/config/pp_liteseg.yaml)",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Override model_path in yaml")
    parser.add_argument("--image", type=str, help="Input image path (if not provided, uses config default)")
    parser.add_argument("--output", type=str, default="pp_liteseg_result.jpg", help="Output image path")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay alpha")
    return parser.parse_args()


def build_color_map(num_classes: int) -> np.ndarray:
    np.random.seed(42)
    return np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)


def color_encode(mask: np.ndarray, colors: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if 0 <= int(label) < len(colors):
            color_img[mask == label] = colors[int(label)]
    return color_img


def main():
    args = parse_args()
    try:
        default_config = Path(__file__).parent.parent / "config" / "pp_liteseg.yaml"
        config_path = Path(args.config) if args.config else default_config
        config_dir = config_path.parent
        project_root = Path(__file__).parent.parent.parent.parent
        model_name = config_path.stem

        factory = ModelFactory()
        print(f"创建 {model_name} 分割器...")
        override_params = {}
        if args.model_path:
            p = Path(args.model_path).expanduser()
            override_params["model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
        segmentor = factory.create_model(model_name, config_dir=config_dir, **override_params)

        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        if args.image:
            image_path = Path(args.image).expanduser()
        else:
            image_path = config.get("test_image", "")
            if not image_path:
                raise ValueError("No --image provided and test_image missing in config")
            image_path = Path(image_path).expanduser()
            if not image_path.is_absolute():
                image_path = (project_root / image_path).resolve()

        print(f"加载图像: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")

        print("运行语义分割...")
        pred_mask = segmentor.infer(image)

        num_classes = int(config.get("default_params", {}).get("num_classes", 19))
        colors = build_color_map(num_classes)
        color_mask = color_encode(pred_mask, colors)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
        result_image = cv2.addWeighted(image, 1.0 - args.alpha, color_mask_bgr, args.alpha, 0)

        cv2.imwrite(args.output, result_image)
        print(f"结果已保存到: {args.output}")
    except Exception as e:
        print(f"错误: {e}")
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
