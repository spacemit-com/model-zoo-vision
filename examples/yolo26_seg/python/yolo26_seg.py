#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""YOLO26 instance segmentation through VisionServiceNative."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26-Seg example")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output", default="yolo26_seg_result.jpg")
    parser.add_argument("--conf-threshold", type=float, default=None)
    return parser.parse_args()


def resolve_path(value, project_root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def main():
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    config_path = (
        resolve_path(args.config, project_root)
        if args.config
        else script_path.parent.parent / "config" / "yolo26_seg.yaml"
    )
    service = None
    try:
        service = VisionServiceNative.create(
            str(config_path), model_path_override=args.model_path or ""
        )
        image_path = (
            str(resolve_path(args.image, project_root))
            if args.image
            else service.get_default_image()
        )
        print(f"加载图像: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        print(f"图像尺寸: {image.shape}")
        print("运行实例分割...")
        status, instances = service.infer_image(
            image, conf=args.conf_threshold
        )
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        print(f"检测到 {len(instances)} 个实例:")
        names = service.get_class_names()
        for index, instance in enumerate(instances, start=1):
            mask = instance.mask
            if (
                not isinstance(mask, np.ndarray)
                or mask.dtype != np.uint8
                or mask.shape != image.shape[:2]
            ):
                raise RuntimeError("Invalid instance mask")
            label = int(instance.label)
            name = names[label] if 0 <= label < len(names) else str(label)
            x1, y1, x2, y2 = (
                int(instance.x1), int(instance.y1),
                int(instance.x2), int(instance.y2),
            )
            print(
                f"  {index}: {name} ({instance.score:.3f}) "
                f"at [{x1}, {y1}, {x2}, {y2}]"
            )
            print(f"    掩码尺寸: {mask.shape}")
        draw_status, visualization = service.draw(image)
        if draw_status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        if not cv2.imwrite(args.output, visualization):
            raise RuntimeError(f"Cannot save visualization: {args.output}")
        print(f"结果已保存到: {args.output}")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.release()


if __name__ == "__main__":
    sys.exit(main())
