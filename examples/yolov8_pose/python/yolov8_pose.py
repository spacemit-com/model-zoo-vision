#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv8 Pose Estimation Example using CV Model Factory

运行方式（与其它 examples 一致）：
  1. 通过 --config 指定 yaml 路径
  2. 通过 VisionServiceNative.create(config_path) 创建模型
  3. 从 yaml 读取参数（test_image 等）进行推理
"""

import argparse
from pathlib import Path
import time
import cv2
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Pose Estimation Example")
    parser.add_argument("--config", type=str, default=None,
                       help="Config yaml path (default: examples/yolov8_pose/config/yolov8_pose.yaml)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override model_path in yaml")
    parser.add_argument("--image", type=str,
                       help="Input image path (if not provided, uses config default)")
    parser.add_argument("--output", type=str, default="yolov8_pose_result.jpg",
                       help="Output image path")
    parser.add_argument("--conf-threshold", type=float, default=None,
                       help="Confidence threshold (default: from config yaml)")
    parser.add_argument("--iou-threshold", type=float, default=None,
                       help="IoU threshold for NMS (default: from config yaml)")
    parser.add_argument("--use-camera", action="store_true",
                       help="Use camera input instead of image file")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera device ID (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # 1) 本 example 的 config 目录；资源路径（test_image 等）相对 model_zoo/cv
        default_config = Path(__file__).parent.parent / "config" / "yolov8_pose.yaml"
        config_path = Path(args.config) if args.config else default_config
        project_root = Path(__file__).parent.parent.parent.parent  # model_zoo/cv

        model_name = config_path.stem
        # 2) 根据 yaml 创建模型
        print(f"创建 {model_name} 姿态估计器...")
        estimator = VisionServiceNative.create(
            str(config_path), model_path_override=args.model_path or ""
        )

        # 3) 从同一 yaml 读取推理参数（test_image 等）
        config = {}
        label_file = config_path
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Handle camera or image input
        if args.use_camera:
            print(f"使用摄像头 {args.camera_id}...")
            cap = cv2.VideoCapture(args.camera_id)
            if not cap.isOpened():
                raise ValueError(f"无法打开摄像头 {args.camera_id}")

            print("实时姿态估计中，按 'q' 退出，按 's' 保存当前帧...")
            frame_count = 0
            t_prev = time.perf_counter()
            fps = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break

                frame_count += 1

                # Run pose estimation
                status, results = estimator.infer_image(
                    frame,
                    conf=args.conf_threshold if args.conf_threshold is not None else -1.0,
                    iou=args.iou_threshold if args.iou_threshold is not None else -1.0,
                )
                if status != VisionServiceStatus.OK:
                    raise RuntimeError(estimator.last_error())

                if results:
                    # Draw results using C++ side drawing API
                    if estimator.supports_draw():
                        st, out = estimator.draw(frame)
                        result_frame = out if st == VisionServiceStatus.OK else frame.copy()
                    else:
                        result_frame = frame.copy()

                    # Print detection info for first few frames
                    if frame_count <= 5 or frame_count % 30 == 0:
                        print(f"帧 {frame_count}: 检测到 {len(results)} 个人")
                else:
                    result_frame = frame.copy()
                    if frame_count <= 5 or frame_count % 30 == 0:
                        print(f"帧 {frame_count}: 未检测到人")

                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLOv8 Pose Estimation", result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_path = f"camera_frame_{frame_count}.jpg"
                    cv2.imwrite(save_path, result_frame)
                    print(f"保存帧到: {save_path}")

                t_now = time.perf_counter()
                fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 1e-6 else 0.0
                t_prev = t_now

            cap.release()
            cv2.destroyAllWindows()
            print(f"已处理 {frame_count} 帧")
        else:
            # 输入图像：命令行优先，否则用 yaml 中的 test_image
            if args.image:
                image_path = resolve_path(args.image, project_root)
            else:
                image_path = config.get("test_image", "~/.cache/assets/image/013_pose.jpg")
                image_path = resolve_path(image_path, project_root)

            # Load image
            print(f"加载图像: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {image_path}")

            print(f"图像尺寸: {image.shape}")

            # Run pose estimation
            print("运行姿态估计...")
            status, results = estimator.infer_image(
                image,
                conf=args.conf_threshold if args.conf_threshold is not None else -1.0,
                iou=args.iou_threshold if args.iou_threshold is not None else -1.0,
            )
            if status != VisionServiceStatus.OK:
                raise RuntimeError(estimator.last_error())

            # Process results
            print(f"检测到 {len(results)} 个人:")

            for i, r in enumerate(results):
                x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
                confidence = r.score

                print(f"  人 {i+1}: 置信度 {confidence:.3f} at [{x1}, {y1}, {x2}, {y2}]")

                # Get keypoints (each has .x .y .visibility)
                keypoints = r.keypoints
                print(f"    关键点数量: {len(keypoints)}")

                # Count visible keypoints
                visible_count = sum(1 for kpt in keypoints if kpt.visibility > 0.5)
                print(f"    可见关键点: {visible_count}/{len(keypoints)}")

            # Draw results using C++ side drawing API
            if estimator.supports_draw():
                st, out = estimator.draw(image)
                result_image = out if st == VisionServiceStatus.OK else image
            else:
                result_image = image

            # Save result
            cv2.imwrite(args.output, result_image)
            print(f"结果已保存到: {args.output}")


    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
