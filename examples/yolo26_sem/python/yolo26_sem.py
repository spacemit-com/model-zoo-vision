#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""YOLO26-Sem semantic segmentation through VisionServiceNative."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26-Sem semantic segmentation example"
    )
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Config yaml path "
            "(default: examples/yolo26_sem/config/yolo26_sem.yaml)"
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
        default="yolo26_sem_result.jpg",
        help="Output visualization path",
    )
    return parser.parse_args()


def resolve_cli_path(path_value, project_root):
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (project_root / path).resolve()


def validate_results(results, image_shape, class_names):
    if not results:
        raise RuntimeError("YOLO26-Sem returned no semantic masks")

    expected_shape = image_shape[:2]
    previous_label = -1
    for result in results:
        label = int(result.label)
        mask = result.mask
        if label <= previous_label:
            raise RuntimeError(
                "YOLO26-Sem masks are not ordered by class id"
            )
        if label < 0 or label >= len(class_names):
            raise RuntimeError(
                f"YOLO26-Sem returned out-of-range class id {label}"
            )
        if not isinstance(mask, np.ndarray):
            raise RuntimeError(
                f"YOLO26-Sem class {label} has no numpy mask"
            )
        if mask.dtype != np.uint8 or mask.shape != expected_shape:
            raise RuntimeError(
                f"YOLO26-Sem class {label} returned invalid mask "
                f"{mask.dtype} {mask.shape}, expected uint8 {expected_shape}"
            )
        if not np.any(mask):
            raise RuntimeError(
                f"YOLO26-Sem class {label} returned an empty mask"
            )
        print(
            f"  {class_names[label]} (class {label}): "
            f"{int(np.count_nonzero(mask))} pixels"
        )
        previous_label = label


def main():
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[3]
    default_config = (
        script_path.parent.parent / "config" / "yolo26_sem.yaml"
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
        class_names = service.get_class_names()
        if len(class_names) != 19:
            raise RuntimeError(
                "YOLO26-Sem expects exactly 19 Cityscapes class names, "
                f"got {len(class_names)}"
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
            raise FileNotFoundError(
                f"Could not load image: {image_path}"
            )

        status, results = service.infer_image(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        validate_results(results, image.shape, class_names)

        if not service.supports_draw():
            raise RuntimeError(
                "YOLO26-Sem does not expose the required Draw capability"
            )
        draw_status, visualization = service.draw(image)
        if draw_status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        if (
            not isinstance(visualization, np.ndarray)
            or visualization.dtype != np.uint8
            or visualization.shape != image.shape
        ):
            raise RuntimeError(
                "YOLO26-Sem Draw returned an invalid visualization"
            )
        if not cv2.imwrite(args.output, visualization):
            raise RuntimeError(
                f"Could not save output: {args.output}"
            )

        print(
            f"Saved: {args.output} "
            f"({len(results)} semantic class mask(s), "
            f"{image.shape[1]}x{image.shape[0]})"
        )
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.release()


if __name__ == "__main__":
    sys.exit(main())
