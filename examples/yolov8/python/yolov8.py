#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv8 Detection Example using CV Model Factory

运行方式：通过 --config 指定 yaml 路径
→ ModelFactory.create_model(model_name, config_dir=config_dir) 创建模型
→ 从 yaml 读 test_image、label_file_path 等参数做推理。
"""

import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import yaml
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "src"))

from core.python.vision_model_factory import ModelFactory
from common.python.drawing import draw_detections
from common import load_labels


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Detection Example")
    parser.add_argument("--config", type=str, default=None,
                       help="Config yaml path (default: examples/yolov8/config/yolov8.yaml)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override model_path in yaml")
    parser.add_argument("--image", type=str,
                       help="Input image path (if not provided, uses config default)")
    parser.add_argument("--output", type=str, default="result.jpg",
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
        # 本 example 的 config 目录；yaml 中路径相对 model_zoo/cv
        default_config = Path(__file__).parent.parent / "config" / "yolov8.yaml"
        config_path = Path(args.config) if args.config else default_config
        config_dir = config_path.parent
        project_root = Path(__file__).parent.parent.parent.parent

        factory = ModelFactory()
        model_name = config_path.stem
        # 根据 config yaml 创建模型
        print(f"创建 {model_name} 检测器...")
        override_params = {}
        if args.conf_threshold is not None:
            override_params["conf_threshold"] = args.conf_threshold
        if args.iou_threshold is not None:
            override_params["iou_threshold"] = args.iou_threshold
        if args.model_path:
            override_params["model_path"] = args.model_path
        detector = factory.create_model(
            model_name,
            config_dir=config_dir,
            **override_params
        )

        # Load labels if available
        label_file = config_path

        labels = None
        image_path = None

        if label_file.exists():
            with open(label_file, 'r') as f:
                config = yaml.safe_load(f)

                # Load labels (路径相对 model_zoo/cv)
                label_file_path = config.get('label_file_path')
                if label_file_path:
                    if not Path(label_file_path).is_absolute():
                        label_file_path = project_root / label_file_path
                    try:
                        labels = load_labels(str(label_file_path))
                        print(f"加载标签文件: {label_file_path} ({len(labels)} 个标签)")
                    except Exception as e:
                        print(f"警告: 无法加载标签文件 {label_file_path}: {e}")
                        labels = None

                # Get test image if not provided
                if not args.image:
                    test_image = config.get('test_image')
                    if test_image:
                        if not Path(test_image).is_absolute():
                            image_path = project_root / test_image
                        else:
                            image_path = test_image
                        print(f"从 yaml 配置读取测试图像: {image_path}")

        # Handle camera or image input
        if args.use_camera:
            print(f"使用摄像头 {args.camera_id}...")
            cap = cv2.VideoCapture(args.camera_id)
            if not cap.isOpened():
                raise ValueError(f"无法打开摄像头 {args.camera_id}")

            print("实时检测中，按 'q' 退出，按 's' 保存当前帧...")
            frame_count = 0
            t_prev = time.perf_counter()
            fps = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break

                frame_count += 1

                # Run detection
                detections = detector.infer(frame)

                if detections:
                    # Draw results using unified drawing API
                    boxes = np.array([det['bbox'] for det in detections])
                    classes = np.array([det['class_id'] for det in detections])
                    scores = np.array([det['confidence'] for det in detections])
                    result_frame = draw_detections(frame, boxes, classes, scores, labels)

                    # Print detection info for first few frames
                    if frame_count <= 5 or frame_count % 30 == 0:
                        print(f"帧 {frame_count}: 检测到 {len(detections)} 个目标")
                else:
                    result_frame = frame

                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLOv8 Detection", result_frame)
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
            # Determine input image
            if args.image:
                image_path = args.image
            elif image_path is None:
                raise ValueError("No test image specified in config and --image not provided")

            # Load and process image
            print(f"加载图像: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")

            print("运行检测...")
            detections = detector.infer(image)

            if detections:
                print(f"检测到 {len(detections)} 个目标:")
                for i, det in enumerate(detections):
                    class_name = (
                        labels[det['class_id']]
                        if labels and det['class_id'] < len(labels)
                        else f"Class {det['class_id']}"
                    )
                    bbox = det['bbox']
                    print(f"  {class_name} (Class {det['class_id']}), Score: {det['confidence']:.6f}, "
                          f"Box: [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]")

                # Draw results using unified drawing API
                boxes = np.array([det['bbox'] for det in detections])
                classes = np.array([det['class_id'] for det in detections])
                scores = np.array([det['confidence'] for det in detections])
                result_image = draw_detections(image, boxes, classes, scores, labels)

                # Save result
                cv2.imwrite(args.output, result_image)
                print(f"结果图像已保存到: {args.output}")
            else:
                print("未检测到任何目标")
                cv2.imwrite(args.output, image)
                print(f"原始图像已保存到: {args.output}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
