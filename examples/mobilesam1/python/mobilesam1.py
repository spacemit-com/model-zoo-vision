#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
"""MobileSAM prompt segmentation through the native service."""

import argparse

import cv2
import numpy as np
from spacemit_vision import VisionServiceNative, VisionServiceStatus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", help="Default: test_image from YAML")
    parser.add_argument("--box", nargs=4, type=float, default=[190, 70, 460, 280])
    parser.add_argument("--output", default="mobilesam1_result.jpg")
    args = parser.parse_args()
    service = VisionServiceNative.create(args.config)
    image_path = args.image or service.get_default_image()
    if not image_path:
        raise ValueError("Set test_image in YAML or pass --image")
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    x1, y1, x2, y2 = args.box
    status, results = service.infer_image_points(
        image, [[x1, y1], [x2, y2]], [2, 3]
    )
    if status != VisionServiceStatus.OK:
        raise RuntimeError(str(status))
    mask = results[0].mask
    overlay = image.copy()
    overlay[mask != 0] = (30, 144, 144)
    output = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    cv2.rectangle(
        output, (round(x1), round(y1)), (round(x2), round(y2)), (0, 255, 0), 2
    )
    if not cv2.imwrite(args.output, output):
        raise RuntimeError("Could not save image")
    print(f"Score: {results[0].score}, mask pixels: {np.count_nonzero(mask)}")
    print(f"Result: {args.output}")


if __name__ == "__main__":
    main()
