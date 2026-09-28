#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run one-person RTMPose inference through VisionServiceNative."""

import argparse
import math
from pathlib import Path
import sys

import cv2
import numpy as np
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus


INPUT_WIDTH = 192
INPUT_HEIGHT = 256
BOX_PADDING = np.float32(1.25)
SKELETON = (
    (16, 14), (14, 12), (15, 13), (13, 11), (12, 11),
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
    (6, 12), (11, 13), (12, 14), (0, 1), (0, 2),
    (1, 3), (2, 4), (0, 5), (0, 6), (3, 5), (4, 6),
)


def warp_person(image, box):
    x1, y1, x2, y2 = map(np.float32, box)
    center_x = np.float32((x1 + x2) * np.float32(0.5))
    center_y = np.float32((y1 + y2) * np.float32(0.5))
    width = np.float32((x2 - x1) * BOX_PADDING)
    height = np.float32((y2 - y1) * BOX_PADDING)
    aspect = np.float32(INPUT_WIDTH / INPUT_HEIGHT)
    if width > height * aspect:
        height = np.float32(width / aspect)
    else:
        width = np.float32(height * aspect)
    scale_x = np.float32(INPUT_WIDTH / width)
    scale_y = np.float32(INPUT_HEIGHT / height)
    matrix = np.array(
        [
            [scale_x, 0, np.float32(INPUT_WIDTH * 0.5 - center_x * scale_x)],
            [0, scale_y, np.float32(INPUT_HEIGHT * 0.5 - center_y * scale_y)],
        ],
        dtype=np.float32,
    )
    warped = cv2.warpAffine(image, matrix, (INPUT_WIDTH, INPUT_HEIGHT))
    return warped, (center_x, center_y, width, height)


def to_image_keypoints(keypoints, affine):
    center_x, center_y, width, height = affine
    points = []
    for point in keypoints:
        x = np.float32(
            (np.float32(point.x) - np.float32(INPUT_WIDTH * 0.5))
            * width / np.float32(INPUT_WIDTH) + center_x
        )
        y = np.float32(
            (np.float32(point.y) - np.float32(INPUT_HEIGHT * 0.5))
            * height / np.float32(INPUT_HEIGHT) + center_y
        )
        points.append((float(x), float(y), point.visibility))
    return points


def draw_pose(image, points):
    for x, y, visibility in points:
        if visibility >= 0.2:
            cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
    for start, end in SKELETON:
        if start < len(points) and end < len(points):
            if points[start][2] >= 0.2 and points[end][2] >= 0.2:
                cv2.line(
                    image,
                    (int(points[start][0]), int(points[start][1])),
                    (int(points[end][0]), int(points[end][1])),
                    (255, 0, 0), 2,
                )


def main():
    parser = argparse.ArgumentParser(description="RTMPose top-down example")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config/rtmpose.yaml"))
    parser.add_argument("--model-path", default="")
    parser.add_argument("--image", help="Defaults to test_image in config")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--output", default="rtmpose_result.jpg")
    args = parser.parse_args()
    service = None
    try:
        service = VisionServiceNative.create(args.config, model_path_override=args.model_path)
        default_image = args.image is None
        image_path = args.image or service.get_default_image()
        if not image_path:
            raise ValueError("No image; use --image or set test_image in config")
        image = cv2.imread(str(Path(image_path).expanduser()))
        if image is None:
            raise FileNotFoundError(image_path)
        box = args.bbox
        if default_image and box is None:
            with open(args.config, encoding="utf-8") as stream:
                box = (yaml.safe_load(stream) or {}).get("test_bbox")
        if box is not None:
            if len(box) != 4:
                raise ValueError("bbox requires four coordinates")
            if not all(math.isfinite(value) for value in box):
                raise ValueError("bbox must be finite")
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                raise ValueError("bbox must be positive-area and inside image")
        if box is None:
            box = (0, 0, image.shape[1], image.shape[0])
        warped, affine = warp_person(image, box)
        status, results = service.infer_image(warped)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        print(f"Poses: {len(results)}")
        for pose in results:
            points = to_image_keypoints(pose.keypoints, affine)
            print(
                f"  score={pose.score:.4f}, keypoints={len(pose.keypoints)}, "
                f"bbox=[{box[0]:.1f},{box[1]:.1f},{box[2]:.1f},{box[3]:.1f}]"
            )
            draw_pose(image, points)
        if not cv2.imwrite(args.output, image):
            raise RuntimeError(f"Could not save {args.output}")
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
