#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""YOLOP multi-task inference through the unified VisionService binding."""

import argparse
from pathlib import Path

import cv2

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOP example")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config" / "yolop.yaml"),
    )
    parser.add_argument("--model-path", default="")
    parser.add_argument("--image")
    parser.add_argument("--output", default="yolop_result.jpg")
    parser.add_argument("--conf-threshold", type=float, default=-1.0)
    parser.add_argument("--iou-threshold", type=float, default=-1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    service = VisionServiceNative.create(
        args.config, model_path_override=args.model_path
    )
    try:
        image_path = args.image or service.get_default_image()
        if not image_path:
            raise ValueError("No image; use --image or set test_image in config")
        image = cv2.imread(str(Path(image_path).expanduser()))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        status, results = service.infer_image(
            image, conf=args.conf_threshold, iou=args.iou_threshold
        )
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        detections = [result for result in results if result.mask is None]
        masks = [result for result in results if result.mask is not None]
        if len(masks) != 2:
            raise RuntimeError(f"Expected two semantic masks, got {len(masks)}")
        for result in masks:
            if result.mask.shape != image.shape[:2]:
                raise RuntimeError(
                    f"Unexpected mask shape {result.mask.shape}, expected {image.shape[:2]}"
                )
        if not service.supports_draw():
            raise RuntimeError("YOLOP must support draw")
        draw_status, output = service.draw(image)
        if draw_status != VisionServiceStatus.OK or output is None:
            raise RuntimeError(service.last_error())
        if not cv2.imwrite(args.output, output):
            raise RuntimeError(f"Could not write output: {args.output}")
        print(f"Detections: {len(detections)}, masks: {len(masks)}")
        print(f"Saved: {args.output}")
        return 0
    finally:
        service.release()


if __name__ == "__main__":
    raise SystemExit(main())
