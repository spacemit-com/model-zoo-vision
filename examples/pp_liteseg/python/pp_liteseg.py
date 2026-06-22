#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
PP-LiteSeg semantic segmentation example using CV Model Factory.
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
        project_root = Path(__file__).parent.parent.parent.parent
        model_name = config_path.stem

        print(f"创建 {model_name} 分割器...")
        segmentor = VisionServiceNative.create(
            str(config_path), model_path_override=args.model_path or ""
        )

        config = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        if args.image:
            image_path = resolve_path(args.image, project_root)
        else:
            image_path = config.get("test_image", "")
            if not image_path:
                raise ValueError("No --image provided and test_image missing in config")
            image_path = resolve_path(image_path, project_root)

        print(f"加载图像: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")

        print("运行语义分割...")
        status, results = segmentor.infer_image(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(segmentor.last_error())

        # Draw results using C++ side drawing API when available
        if segmentor.supports_draw():
            st, out = segmentor.draw(image)
            if st != VisionServiceStatus.OK:
                raise RuntimeError(segmentor.last_error())
            result_image = out
        else:
            # Fallback: color-encode the predicted mask (r.mask, numpy HxW) ourselves
            pred_mask = results[0].mask if results else None
            if pred_mask is None:
                raise RuntimeError("分割结果为空，未获得掩码")
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
