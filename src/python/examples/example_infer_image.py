#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""CLI: C++ VisionService from Python.

Build & install wheel::
    pip install pybind11 build setuptools wheel
    cmake -S . -B build && cmake --build build -j        # 仓库根，编出扩展
    cd src/python && ./build_wheel.sh                     # 打包 wheel
    pip install --force-reinstall dist/spacemit_vision-*.whl

Run from repo root (after installing the wheel)::
    python3 src/python/examples/example_infer_image.py --config examples/yolov8/config/yolov8.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VisionService.infer_image from Python.")
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--image", default="", help="Image path (default: model default test image)")
    parser.add_argument("--lazy-load", action="store_true", help="Pass lazy_load=True to Create()")
    parser.add_argument(
        "--output",
        default="",
        help="Output image path to save visualization (default: not saved)",
    )
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv-python is required.", file=sys.stderr)
        return 2

    try:
        from spacemit_vision import VisionServiceNative, VisionServiceStatus
    except ImportError as e:
        print(str(e), file=sys.stderr)
        return 2

    # 切到仓库根，使 yaml 中相对路径（model_path / test_image / label_file_path）可解析。
    # src/python/examples/example_infer_image.py -> 仓库根为 parents[3]
    os.chdir(Path(__file__).resolve().parents[3])

    try:
        svc = VisionServiceNative.create(args.config, "", args.lazy_load)
    except ValueError as e:
        print(f"Create failed: {e}", file=sys.stderr)
        return 1

    image_path = args.image or svc.get_default_image()
    if not image_path:
        print("No --image and no test_image in config.", file=sys.stderr)
        return 1

    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to read image: {image_path}", file=sys.stderr)
        return 1

    status, results = svc.infer_image(img)
    if status != VisionServiceStatus.OK:
        print(f"Infer failed status={status} err={svc.last_error()}", file=sys.stderr)
        return 1

    print(f"image={image_path} detections={len(results)}")
    for i, r in enumerate(results):
        print(f"  [{i}] xyxy=({r.x1:.1f},{r.y1:.1f})-({r.x2:.1f},{r.y2:.1f}) score={r.score:.3f} label={r.label}")

    if args.output:
        if not svc.supports_draw():
            print("This model does not support draw(); skip saving output image.", file=sys.stderr)
            return 1
        draw_status, drawn = svc.draw(img)
        if draw_status != VisionServiceStatus.OK:
            print(f"Draw failed status={draw_status} err={svc.last_error()}", file=sys.stderr)
            return 1
        if not cv2.imwrite(args.output, drawn):
            print(f"Failed to save output image: {args.output}", file=sys.stderr)
            return 1
        print(f"saved={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
