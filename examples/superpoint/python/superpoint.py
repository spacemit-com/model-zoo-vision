#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Extract and draw local features through VisionServiceNative."""

import argparse
from pathlib import Path
import sys

import cv2

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "config/superpoint.yaml",
    )
    parser.add_argument("--image", type=Path, help="Input image; defaults to test_image in YAML")
    parser.add_argument("--output", type=Path, default=Path("superpoint_keypoints.jpg"))
    parser.add_argument("--model-path", default="", help="Override the ONNX model path")
    return parser.parse_args()


def main():
    args = parse_args()
    service = VisionServiceNative.create(
        str(args.config.expanduser()), model_path_override=args.model_path, lazy_load=True,
    )
    try:
        image_path = str(args.image.expanduser()) if args.image else service.get_default_image()
        if not image_path:
            raise ValueError("Set test_image in YAML or pass --image")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        status, features = service.extract_local_features(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(f"Feature extraction failed: {service.last_error()}")
        status, output = service.draw(image)
        if status != VisionServiceStatus.OK or output is None:
            raise RuntimeError(f"Drawing failed: {service.last_error()}")
        if not cv2.imwrite(str(args.output.expanduser()), output):
            raise RuntimeError(f"Failed to write output: {args.output}")
        print(
            f"Features: {len(features.keypoints)}, descriptor dim: {features.descriptor_dim}, "
            f"output: {args.output}"
        )
        return 0
    finally:
        service.release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
