#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv8 Instance Segmentation Example using CV Model Factory

This example demonstrates how to use YOLOv8 instance segmentation through the CV model factory.
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
from common.python.drawing import draw_segmentation
from common import load_labels


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Instance Segmentation Example")
    parser.add_argument("--config", type=str, default=None,
                       help="Config yaml path (default: examples/yolov8_seg/config/yolov8_seg.yaml)")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Override model_path in yaml")
    parser.add_argument("--image", type=str,
                       help="Input image path (if not provided, uses config default)")
    parser.add_argument("--output", type=str, default="yolov8_seg_result.jpg",
                       help="Output image path")
    parser.add_argument("--conf-threshold", type=float, default=None,
                       help="Confidence threshold (default: from config yaml)")
    parser.add_argument("--iou-threshold", type=float, default=None,
                       help="IoU threshold for NMS (default: from config yaml)")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Alpha blending factor for masks")
    parser.add_argument("--use-camera", action="store_true",
                       help="Use camera input instead of image file")
    parser.add_argument("--camera-id", type=int, default=0,
                       help="Camera device ID (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        # Config 位于 examples/yolov8_seg/config，资源路径相对 model_zoo/cv
        default_config = Path(__file__).parent.parent / "config" / "yolov8_seg.yaml"
        config_path = Path(args.config) if args.config else default_config
        config_dir = config_path.parent
        project_root = Path(__file__).parent.parent.parent.parent  # model_zoo/cv

        # Create model factory
        factory = ModelFactory()

        # Create YOLOv8 segmentation model
        model_name = config_path.stem
        print(f"创建 {model_name} 实例分割器...")
        override_params = {}
        if args.conf_threshold is not None:
            override_params["conf_threshold"] = args.conf_threshold
        if args.iou_threshold is not None:
            override_params["iou_threshold"] = args.iou_threshold
        if args.model_path is not None:
            override_params["model_path"] = args.model_path
        segmentor = factory.create_model(
            model_name,
            config_dir=config_dir,
            **override_params
        )

        # Get model config from local yaml
        config = {}
        label_file = config_path
        if label_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

        # Load labels if available
        labels = None
        label_file_path = config.get("label_file_path")
        if label_file_path:
            if not Path(label_file_path).is_absolute():
                label_file_path = project_root / label_file_path
            try:
                labels = load_labels(str(label_file_path))
                print(f"加载标签文件: {label_file_path} ({len(labels)} 个标签)")
            except Exception as e:
                print(f"警告: 无法加载标签文件 {label_file_path}: {e}")
                labels = None

        # Handle camera or image input
        if args.use_camera:
            print(f"使用摄像头 {args.camera_id}...")
            cap = cv2.VideoCapture(args.camera_id)
            if not cap.isOpened():
                raise ValueError(f"无法打开摄像头 {args.camera_id}")

            print("实时实例分割中，按 'q' 退出，按 's' 保存当前帧...")
            frame_count = 0
            t_prev = time.perf_counter()
            fps = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break

                frame_count += 1

                # Run segmentation
                detections, masks = segmentor.infer(frame)

                # Process results for drawing
                detections_for_draw = []
                for result in detections:
                    x1, y1, x2, y2 = map(int, result["bbox"])
                    confidence = result["confidence"]
                    class_id = result["class_id"]

                    detections_for_draw.append({
                        'bbox': [x1, y1, x2, y2],
                        'class_id': class_id,
                        'confidence': confidence
                    })

                if detections_for_draw:
                    # Draw results
                    result_frame = draw_segmentation(frame, detections_for_draw, masks, labels)

                    # Print detection info for first few frames
                    if frame_count <= 5 or frame_count % 30 == 0:
                        print(f"帧 {frame_count}: 检测到 {len(detections)} 个实例")
                else:
                    result_frame = frame

                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLOv8 Segmentation", result_frame)
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
            # Use provided image or default from config
            if args.image:
                image_path = args.image
            else:
                image_path = config.get("test_image", "test_data/images/bus.jpg")
                if not Path(image_path).is_absolute():
                    image_path = project_root / image_path

            # Load image
            print(f"加载图像: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"无法加载图像: {image_path}")

            print(f"图像尺寸: {image.shape}")

            # Run segmentation（infer 返回 (detections, masks) 元组，masks 为 [N,H,W]）
            print("运行实例分割...")
            detections, masks = segmentor.infer(image)

            # Process results
            print(f"检测到 {len(detections)} 个实例:")

            # Convert to format for drawing API
            detections_for_draw = []
            for i, result in enumerate(detections):
                x1, y1, x2, y2 = map(int, result["bbox"])
                confidence = result["confidence"]
                class_id = result["class_id"]

                if labels and class_id < len(labels):
                    class_name = labels[class_id]
                else:
                    class_name = f"Class_{class_id}"

                print(f"  {i+1}: {class_name} ({confidence:.3f}) at [{x1}, {y1}, {x2}, {y2}]")
                if masks is not None and i < len(masks):
                    print(f"    掩码尺寸: {masks[i].shape}")

                detections_for_draw.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_id': class_id,
                    'confidence': confidence
                })

            # Draw results（masks 为 [N,H,W] 或 None）
            result_image = draw_segmentation(image, detections_for_draw, masks, labels)

            # Save result
            cv2.imwrite(args.output, result_image)
            print(f"结果已保存到: {args.output}")


    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
