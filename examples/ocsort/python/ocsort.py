#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
OC-SORT Tracker Example using CV Model Factory

运行方式：通过 --config 指定 yaml 路径（与 yolov8.py 一致）。
"""

import sys
import argparse
from pathlib import Path
import time
import cv2
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from core.python.vision_model_factory import ModelFactory
from common import load_labels


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="OC-SORT Tracker Example")
    parser.add_argument("--config", type=str, default=None,
                       help="Config yaml path (default: examples/ocsort/config/ocsort.yaml)")
    parser.add_argument("--video", type=str,
                       help="Input video path (if not provided, uses config default)")
    parser.add_argument("--conf-threshold", type=float, default=None,
                       help="Confidence threshold (default: from config yaml)")
    parser.add_argument("--iou-threshold", type=float, default=None,
                       help="IoU threshold for NMS (default: from config yaml)")
    parser.add_argument("--track-thresh", type=float, default=None,
                       help="Track threshold (default: from config yaml)")
    parser.add_argument("--track-iou-thresh", type=float, default=None,
                       help="Track IoU threshold (default: from config yaml)")
    parser.add_argument("--track-buffer", type=int, default=None,
                       help="Track buffer size (default: from config yaml)")
    parser.add_argument("--min-hits", type=int, default=None,
                       help="Minimum hits for track confirmation (default: from config yaml)")
    parser.add_argument("--delta-t", type=int, default=None,
                       help="Delta t for velocity estimation (default: from config yaml)")
    parser.add_argument("--use-camera", action="store_true",
                       help="Use camera input instead of video file")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera device ID (default: 0)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override model_path in yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        default_config = Path(__file__).parent.parent / "config" / "ocsort.yaml"
        config_path = Path(args.config) if args.config else default_config
        config_dir = config_path.parent
        project_root = Path(__file__).parent.parent.parent.parent  # model_zoo/cv
        model_name = config_path.stem

        # Create model factory
        factory = ModelFactory()

        # Create OC-SORT tracker using factory
        print(f"创建 {model_name} 追踪器...")
        override_params = {}
        if args.model_path:
            p = Path(args.model_path).expanduser()
            override_params["model_path"] = str(
                p if p.is_absolute() else (project_root / p).resolve()
            )
        if args.conf_threshold is not None:
            override_params["conf_threshold"] = args.conf_threshold
        if args.iou_threshold is not None:
            override_params["iou_threshold"] = args.iou_threshold
        if args.track_thresh is not None:
            override_params["track_thresh"] = args.track_thresh
        if args.track_iou_thresh is not None:
            override_params["track_iou_thresh"] = args.track_iou_thresh
        if args.track_buffer is not None:
            override_params["track_buffer"] = args.track_buffer
        if args.min_hits is not None:
            override_params["min_hits"] = args.min_hits
        if args.delta_t is not None:
            override_params["delta_t"] = args.delta_t
        tracker = factory.create_model(
            model_name,
            config_dir=config_dir,
            **override_params
        )

        # Load labels and video path from specified yaml
        labels = None
        video_path = None

        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

                # Load labels (路径相对 model_zoo/cv)
                label_file_path = config.get('label_file_path')
                if label_file_path:
                    label_file_path = resolve_path(label_file_path, project_root)
                    try:
                        labels = load_labels(str(label_file_path))
                        print(f"加载标签文件: {label_file_path} ({len(labels)} 个标签)")
                    except Exception as e:
                        print(f"警告: 无法加载标签文件 {label_file_path}: {e}")
                        labels = None

                # Get test video if not provided
                if not args.video:
                    test_video = config.get('test_video')
                    if test_video:
                        video_path = resolve_path(test_video, project_root)
                        print(f"从 yaml 配置读取视频路径: {video_path}")

        # Determine input source
        if args.use_camera:
            video_path = args.camera_id
            print(f"使用摄像头 {args.camera_id}...")
        elif args.video:
            video_path = resolve_path(args.video, project_root)
        elif video_path is None:
            raise ValueError("No test video specified in config, --video not provided, and --use-camera not set")

        # Open video or camera
        if args.use_camera:
            print(f"打开摄像头: {video_path}")
        else:
            print(f"打开视频: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            if args.use_camera:
                raise ValueError(f"无法打开摄像头: {video_path}")
            else:
                raise ValueError(f"无法打开视频: {video_path}")

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        print("实时显示中，按 'q' 退出...")

        frame_count = 0
        delay = max(1, int(1000 / fps)) if fps > 0 else 33
        t_prev = time.perf_counter() if args.use_camera else None
        current_fps = 0.0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            tracks = tracker.track(frame)

            if tracks:
                result_frame = tracker.draw_results(frame, tracks, labels)
            else:
                result_frame = frame

            if args.use_camera:
                cv2.putText(result_frame, f"FPS: {current_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("OC-SORT", result_frame)
            if cv2.waitKey(delay) & 0xFF == ord('q'):
                break
            if args.use_camera:
                t_now = time.perf_counter()
                current_fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 1e-6 else 0.0
                t_prev = t_now

        cap.release()
        cv2.destroyAllWindows()
        print(f"已处理 {frame_count} 帧")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
