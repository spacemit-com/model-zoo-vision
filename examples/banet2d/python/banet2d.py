#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Estimate stereo disparity through VisionServiceNative."""

import argparse
from pathlib import Path
import sys

import cv2

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "config/banet2d.yaml",
    )
    parser.add_argument("--left", type=Path, help="Left image; defaults to test_image1 in YAML")
    parser.add_argument("--right", type=Path, help="Right image; defaults to test_image2 in YAML")
    parser.add_argument("--output", type=Path, default=Path("banet2d_disparity.png"))
    parser.add_argument("--model-path", default="", help="Override the ONNX model path")
    args = parser.parse_args()
    if (args.left is None) != (args.right is None):
        parser.error("--left and --right must be provided together")
    return args


def main():
    args = parse_args()
    service = VisionServiceNative.create(
        str(args.config.expanduser()), model_path_override=args.model_path, lazy_load=True,
    )
    try:
        left_path = str(args.left.expanduser()) if args.left else service.get_config_path_value("test_image1")
        right_path = str(args.right.expanduser()) if args.right else service.get_config_path_value("test_image2")
        if not left_path or not right_path:
            raise ValueError("Set test_image1/test_image2 in YAML or pass --left and --right")
        left = cv2.imread(left_path)
        right = cv2.imread(right_path)
        if left is None or right is None:
            raise ValueError(f"Failed to load stereo images: {left_path}, {right_path}")
        status, disparity = service.infer_stereo(left, right)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(f"Stereo inference failed: {service.last_error()}")
        if disparity is None or disparity.shape != left.shape[:2]:
            raise RuntimeError("BANet2D returned no valid full-size disparity map")
        normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        if not cv2.imwrite(str(args.output.expanduser()), color):
            raise RuntimeError(f"Failed to write output: {args.output}")
        print(f"Disparity {disparity.shape[1]}x{disparity.shape[0]}, output: {args.output}")
        return 0
    finally:
        service.release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
