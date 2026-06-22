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
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus


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
    project_root = Path(__file__).parent.parent.parent.parent
    model_name = config_path.stem

    conf = args.conf_threshold if args.conf_threshold is not None else -1.0
    iou = args.iou_threshold if args.iou_threshold is not None else -1.0

    try:
        print(f"创建 {model_name} 检测器...")
        svc = VisionServiceNative.create(
            str(config_path),
            model_path_override=args.model_path or "",
        )

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
                status, detections = svc.infer_image(frame, conf=conf, iou=iou)
                if status != VisionServiceStatus.OK:
                    raise RuntimeError(svc.last_error())
                if detections:
                    if svc.supports_draw():
                        st, out = svc.draw(frame)
                        result_frame = out if st == VisionServiceStatus.OK else frame.copy()
                    else:
                        result_frame = frame.copy()
                    if frame_count <= 5 or frame_count % 30 == 0:
                        print(f"帧 {frame_count}: 检测到 {len(detections)} 张人脸")
                else:
                    result_frame = frame.copy()
                    if frame_count <= 5 or frame_count % 30 == 0:
                        print(f"帧 {frame_count}: 未检测到人脸")
                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLOv5-Face Detection", result_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    path = f"camera_face_{frame_count}.jpg"
                    cv2.imwrite(path, result_frame)
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
        status, detections = svc.infer_image(image, conf=conf, iou=iou)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(svc.last_error())
        print(f"检测到 {len(detections)} 张人脸")

        for i, r in enumerate(detections):
            x1, y1, x2, y2 = int(r.x1), int(r.y1), int(r.x2), int(r.y2)
            kp_info = ""
            if r.keypoints:
                kp_info = f" 关键点 {len(r.keypoints)} 个"
            print(f"  人脸 {i+1}: 置信度 {r.score:.3f} 框 [{x1},{y1},{x2},{y2}]{kp_info}")

        # 绘制（人脸框 + 关键点由 C++ 侧统一绘制）
        if svc.supports_draw():
            st, out = svc.draw(image)
            result_image = out if st == VisionServiceStatus.OK else image
        else:
            result_image = image
        cv2.imwrite(args.output, result_image)
        print(f"结果已保存: {args.output}")
        return 0

    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
