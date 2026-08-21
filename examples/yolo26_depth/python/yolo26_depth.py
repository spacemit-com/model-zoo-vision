#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""YOLO26-Depth monocular metric depth through VisionServiceNative."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26-Depth monocular metric-depth example"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config yaml path "
            "(default: examples/yolo26_depth/config/yolo26_depth.yaml)"
        ),
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Override model_path in yaml",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Input image path (default: test_image from yaml)",
    )
    parser.add_argument(
        "--output",
        default="yolo26_depth_result.jpg",
        help="Output visualization path",
    )
    return parser.parse_args()


def resolve_cli_path(path_value, project_root):
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def main():
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    default_config = (
        script_path.parent.parent / "config" / "yolo26_depth.yaml"
    )
    config_path = (
        resolve_cli_path(args.config, project_root)
        if args.config
        else default_config
    )

    service = None
    try:
        service = VisionServiceNative.create(
            str(config_path),
            model_path_override=args.model_path or "",
        )
        image_path = (
            str(resolve_cli_path(args.image, project_root))
            if args.image
            else service.get_default_image()
        )
        if not image_path:
            raise RuntimeError(
                "No --image provided and test_image is missing in yaml"
            )
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        status, depth = service.infer_depth(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        if (
            not isinstance(depth, np.ndarray)
            or depth.dtype != np.float32
            or depth.shape != image.shape[:2]
        ):
            raise RuntimeError(
                "YOLO26-Depth returned an invalid float32 depth map"
            )
        valid = np.isfinite(depth) & (depth > 0.0)
        if not np.any(valid):
            raise RuntimeError("Depth map has no positive finite values")

        if not service.supports_draw():
            raise RuntimeError(
                "YOLO26-Depth does not expose the required Draw capability"
            )
        draw_status, visualization = service.draw(image)
        if draw_status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        if (
            not isinstance(visualization, np.ndarray)
            or visualization.dtype != np.uint8
            or visualization.shape != image.shape
        ):
            raise RuntimeError("Draw returned an invalid visualization")
        if not cv2.imwrite(args.output, visualization):
            raise RuntimeError(f"Could not save output: {args.output}")

        values = depth[valid]
        print(
            f"Depth: min={float(values.min()):.4f} m, "
            f"max={float(values.max()):.4f} m, "
            f"mean={float(values.mean()):.4f} m, "
            f"valid={values.size}"
        )
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
