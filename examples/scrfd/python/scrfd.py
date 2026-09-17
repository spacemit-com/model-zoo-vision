#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""SCRFD face detection via the unified native service."""

import argparse
from pathlib import Path
import sys

import cv2

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def main():
    parser = argparse.ArgumentParser(description="SCRFD face detection example")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config/scrfd.yaml"),
    )
    parser.add_argument("--model-path", default="", help="Override model_path")
    parser.add_argument("--image", help="Default: test_image from YAML")
    parser.add_argument("--output", default="scrfd_result.jpg")
    args = parser.parse_args()
    service = None
    try:
        service = VisionServiceNative.create(
            str(Path(args.config).expanduser()), model_path_override=args.model_path
        )
        image_path = args.image or service.get_default_image()
        if not image_path:
            raise ValueError("No image; use --image or set test_image in YAML")
        image = cv2.imread(str(Path(image_path).expanduser()))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        status, results = service.infer_image(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        print(f"Faces: {len(results)}")
        for face in results:
            if len(face.keypoints) != 5:
                raise RuntimeError("Expected a face with five landmarks")
            print(
                f"  score: {face.score:.4f}, box: ["
                f"{face.x1:.4f}, {face.y1:.4f}, "
                f"{face.x2:.4f}, {face.y2:.4f}], landmarks: 5"
            )
        output = image
        if results:
            status, output = service.draw(image)
            if status != VisionServiceStatus.OK or output is None:
                raise RuntimeError(service.last_error())
        if not cv2.imwrite(args.output, output):
            raise RuntimeError(f"Could not write output: {args.output}")
        print(f"Saved: {args.output}")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.release()


if __name__ == "__main__":
    sys.exit(main())
