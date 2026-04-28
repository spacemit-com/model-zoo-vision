#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv5-Face 人脸检测示例（仅检测人脸框，不含 ArcFace 识别）

运行方式：通过 --config 指定 yaml 路径（与 yolov8.py 一致）。
"""

import sys
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from core.python.vision_model_factory import ModelFactory
from common.python.drawing import draw_detections


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5-Face 人脸检测示例（仅检测）")
    parser.add_argument("--config", type=str, default=None,
                        help="Config yaml 路径（默认: examples/yolov5-face/config/yolov5-face.yaml）")
    parser.add_argument("--image", type=str, help="输入图像路径（不指定则使用 config 默认）")
    parser.add_argument("--output", type=str, default="result_face.jpg", help="输出图像路径")
    parser.add_argument("--conf-threshold", type=float, default=None, help="置信度阈值（默认: 使用 config yaml）")
    parser.add_argument("--iou-threshold", type=float, default=None, help="NMS IoU 阈值（默认: 使用 config yaml）")
    parser.add_argument("--use-camera", action="store_true", help="使用摄像头")
    parser.add_argument("--camera-id", type=int, default=0, help="摄像头设备 ID")
    parser.add_argument("--model-path", type=str, default=None, help="覆盖 yaml 中的 model_path")
    return parser.parse_args()


def main():
    args = parse_args()
    default_config = Path(__file__).parent.parent / "config" / "yolov5-face.yaml"
    config_path = Path(args.config) if args.config else default_config
    config_dir = config_path.parent
    project_root = Path(__file__).parent.parent.parent.parent
    model_name = config_path.stem

    try:
        factory = ModelFactory()
        print(f"创建 {model_name} 检测器...")
        override_params = {}
        if args.model_path:
            p = Path(args.model_path).expanduser()
            override_params["model_path"] = str(
                p if p.is_absolute() else (project_root / p).resolve()
            )
        if args.conf_threshold is not None:
            override_params["conf_thres"] = args.conf_threshold
        if args.iou_threshold is not None:
            override_params["iou_thres"] = args.iou_threshold
        detector = factory.create_model(
            model_name,
            config_dir=config_dir,
            **override_params,
        )

        # 人脸检测只有一类
        labels = ["face"]

        if args.use_camera:
            cap = cv2.VideoCapture(args.camera_id)
            if not cap.isOpened():
                raise ValueError(f"无法打开摄像头 {args.camera_id}")
            print("实时人脸检测，按 'q' 退出，'s' 保存当前帧...")
            frame_count = 0
            t_prev = time.perf_counter()
            fps = 0.0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                detections, _ = detector.infer(frame)
                boxes = []
                classes = []
                scores = []
                for d in detections:
                    x1, y1, x2, y2 = map(int, d["bbox"])
                    boxes.append([x1, y1, x2, y2])
                    classes.append(0)
                    scores.append(d["confidence"])
                if boxes:
                    frame = draw_detections(
                        frame,
                        np.array(boxes),
                        np.array(classes),
                        np.array(scores),
                        labels,
                    )
                    if frame_count <= 5 or frame_count % 30 == 0:
                        print(f"帧 {frame_count}: 检测到 {len(detections)} 张人脸")
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLOv5-Face Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    path = f"camera_face_{frame_count}.jpg"
                    cv2.imwrite(path, frame)
                    print(f"已保存: {path}")
                t_now = time.perf_counter()
                fps = 1.0 / (t_now - t_prev) if (t_now - t_prev) > 1e-6 else 0.0
                t_prev = t_now
            cap.release()
            cv2.destroyAllWindows()
            return 0

        # 图像文件
        if args.image:
            image_path = resolve_path(args.image, project_root)
        else:
            config = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f) or {}
            image_path = config.get("test_image", "~/.cache/assets/image/006_test.jpg")
            image_path = resolve_path(image_path, project_root)
        if not image_path.exists():
            print(f"错误: 图像不存在 {image_path}")
            return 1

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return 1

        print(f"运行人脸检测: {image_path}")
        detections, _ = detector.infer(image)
        print(f"检测到 {len(detections)} 张人脸")

        boxes = []
        classes = []
        scores = []
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = map(int, d["bbox"])
            conf = d["confidence"]
            print(f"  人脸 {i+1}: 置信度 {conf:.3f} 框 [{x1},{y1},{x2},{y2}]")
            boxes.append([x1, y1, x2, y2])
            classes.append(0)
            scores.append(conf)

        result_image = draw_detections(
            image,
            np.array(boxes) if boxes else np.zeros((0, 4)),
            np.array(classes) if classes else np.array([], dtype=int),
            np.array(scores) if scores else np.array([]),
            labels,
        )
        cv2.imwrite(args.output, result_image)
        print(f"结果已保存: {args.output}")
        return 0

    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
