#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Extract SuperPoint features and match them through VisionServiceNative."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "config/lightglue.yaml",
    )
    parser.add_argument("--image1", type=Path, help="First image; defaults to test_image1 in YAML")
    parser.add_argument("--image2", type=Path, help="Second image; defaults to test_image2 in YAML")
    parser.add_argument("--extractor-config", type=Path, help="Override superpoint_config_path")
    parser.add_argument("--output", type=Path, default=Path("lightglue_matches.jpg"))
    parser.add_argument("--model-path", default="", help="Override the LightGlue ONNX model path")
    args = parser.parse_args()
    if (args.image1 is None) != (args.image2 is None):
        parser.error("--image1 and --image2 must be provided together")
    return args


def main():
    args = parse_args()
    matcher = VisionServiceNative.create(
        str(args.config.expanduser()), model_path_override=args.model_path, lazy_load=True,
    )
    extractor = None
    try:
        extractor_config = (
            str(args.extractor_config.expanduser()) if args.extractor_config
            else matcher.get_config_path_value("superpoint_config_path")
        )
        if not extractor_config:
            raise ValueError("Set superpoint_config_path in YAML or pass --extractor-config")
        extractor = VisionServiceNative.create(extractor_config, lazy_load=True)
        image1_path = str(args.image1.expanduser()) if args.image1 else matcher.get_config_path_value("test_image1")
        image2_path = str(args.image2.expanduser()) if args.image2 else matcher.get_config_path_value("test_image2")
        if not image1_path or not image2_path:
            raise ValueError("Set test_image1/test_image2 in YAML or pass --image1 and --image2")
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)
        if image1 is None or image2 is None:
            raise ValueError(f"Failed to load image pair: {image1_path}, {image2_path}")
        status, features1 = extractor.extract_local_features(image1)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(f"First image feature extraction failed: {extractor.last_error()}")
        status, features2 = extractor.extract_local_features(image2)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(f"Second image feature extraction failed: {extractor.last_error()}")
        status, matches = matcher.match_local_features(features1, features2)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(f"Feature matching failed: {matcher.last_error()}")

        height = max(image1.shape[0], image2.shape[0])
        left_width = image1.shape[1]
        canvas = np.zeros((height, left_width + image2.shape[1], 3), dtype=np.uint8)
        canvas[:image1.shape[0], :left_width] = image1
        canvas[:image2.shape[0], left_width:] = image2
        for match in matches:
            cv2.line(
                canvas,
                (round(match.query_point.x), round(match.query_point.y)),
                (left_width + round(match.train_point.x), round(match.train_point.y)),
                (0, 255, 0), 1, cv2.LINE_AA,
            )
        if not cv2.imwrite(str(args.output.expanduser()), canvas):
            raise RuntimeError(f"Failed to write output: {args.output}")
        print(f"Matches: {len(matches)}, output: {args.output}")
        return 0
    finally:
        if extractor is not None:
            extractor.release()
        matcher.release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
