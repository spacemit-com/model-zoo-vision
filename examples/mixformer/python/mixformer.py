#!/usr/bin/env python3
# Copyright (C) 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Track one object in a video with MixFormer through VisionServiceNative."""

import argparse
import math
from pathlib import Path
import sys

import cv2
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "config/mixformer.yaml",
    )
    parser.add_argument("--video", type=Path, help="Input video; defaults to test_video in YAML")
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"),
        help="Initial box in original-image pixels; defaults to initial_bbox in YAML",
    )
    parser.add_argument("--output", type=Path, default=Path("mixformer_tracking.mp4"))
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 processes the full video")
    parser.add_argument("--model-path", default="", help="Override the primary ONNX model path")
    args = parser.parse_args()
    if args.max_frames < 0:
        parser.error("--max-frames must be nonnegative")
    return args


def main():
    args = parse_args()
    config_path = args.config.expanduser()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    if args.bbox is not None:
        initial_bbox = args.bbox
    else:
        box = config.get("initial_bbox")
        if not isinstance(box, dict):
            raise ValueError("Set initial_bbox in YAML or pass --bbox X1 Y1 X2 Y2")
        x, y = float(box["x"]), float(box["y"])
        initial_bbox = [x, y, x + float(box["w"]), y + float(box["h"])]

    service = VisionServiceNative.create(
        str(config_path), model_path_override=args.model_path, lazy_load=True,
    )
    capture = None
    writer = None
    try:
        video_path = str(args.video.expanduser()) if args.video else service.get_config_path_value("test_video")
        if not video_path:
            raise ValueError("Set test_video in YAML or pass --video")
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        ok, frame = capture.read()
        if not ok:
            raise ValueError("Input video contains no readable frames")
        height, width = frame.shape[:2]
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = max(1.0, fps) if math.isfinite(fps) else 1.0
        writer = cv2.VideoWriter(
            str(args.output.expanduser()), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to create output video: {args.output}")

        frame_index = 0
        while True:
            status, _ = service.track(frame, initial_bbox if frame_index == 0 else None)
            if status != VisionServiceStatus.OK:
                raise RuntimeError(f"Frame {frame_index}: {service.last_error()}")
            status, output = service.draw(frame)
            if status != VisionServiceStatus.OK or output is None:
                raise RuntimeError(f"Drawing frame {frame_index} failed: {service.last_error()}")
            writer.write(output)
            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break
            ok, frame = capture.read()
            if not ok:
                break
        print(f"Processed {frame_index} frames, output: {args.output}")
        return 0
    finally:
        if writer is not None:
            writer.release()
        if capture is not None:
            capture.release()
        service.release()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
