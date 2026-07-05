#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
YOLOE open-vocabulary instance segmentation example using spacemit_vision.

运行方式：--config 指定 yaml → VisionServiceNative.create(config) 创建模型。
文本词汇（= 类别）来自 yaml 的 default_params.prompts，或用 --prompts "a,b,c" 覆盖。
文本嵌入按 prompts 惰性缓存，重复同一词汇不重复调用 MobileCLIP。
"""

import argparse
import time
from pathlib import Path

import cv2

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOE Open-Vocabulary Segmentation Example")
    parser.add_argument("--config", type=str, default=None,
                        help="Config yaml path (default: examples/yoloe/config/yoloe.yaml)")
    parser.add_argument("--model-path", type=str, default=None, help="Override model_path in yaml")
    parser.add_argument("--image", type=str, help="Input image path (default: config test_image)")
    parser.add_argument("--output", type=str, default="yoloe_result.jpg", help="Output image path")
    parser.add_argument("--prompts", type=str, default="",
                        help='Comma-separated vocabulary override, e.g. "person,bus". '
                             "Empty -> use the config default prompts.")
    parser.add_argument("--conf-threshold", type=float, default=None,
                        help="Confidence threshold (default: from config yaml)")
    parser.add_argument("--iou-threshold", type=float, default=None,
                        help="IoU threshold for NMS (default: from config yaml)")
    parser.add_argument("--use-camera", action="store_true", help="Use camera input")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device ID (default: 0)")
    return parser.parse_args()


def main():
    args = parse_args()

    conf = args.conf_threshold if args.conf_threshold is not None else -1.0
    iou = args.iou_threshold if args.iou_threshold is not None else -1.0
    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]

    try:
        default_config = Path(__file__).parent.parent / "config" / "yoloe.yaml"
        config_path = Path(args.config) if args.config else default_config
        project_root = Path(__file__).parent.parent.parent.parent
        model_name = config_path.stem

        print(f"创建 {model_name} 分割器...")
        svc = VisionServiceNative.create(
            str(config_path),
            model_path_override=args.model_path or "",
        )

        def current_vocab():
            return prompts if prompts else svc.get_class_names()

        if args.use_camera:
            print(f"使用摄像头 {args.camera_id}...")
            cap = cv2.VideoCapture(args.camera_id)
            if not cap.isOpened():
                raise ValueError(f"无法打开摄像头 {args.camera_id}")

            print("实时分割中，按 'q' 退出，按 's' 保存当前帧...")
            frame_count = 0
            t_prev = time.perf_counter()
            fps = 0.0

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头帧")
                    break
                frame_count += 1

                status, results = svc.infer_image(frame, conf=conf, iou=iou, prompts=prompts)
                if status != VisionServiceStatus.OK:
                    raise RuntimeError(svc.last_error())

                if results and svc.supports_draw():
                    st, out = svc.draw(frame)
                    result_frame = out if st == VisionServiceStatus.OK else frame.copy()
                else:
                    result_frame = frame.copy()
                if frame_count <= 5 or frame_count % 30 == 0:
                    print(f"帧 {frame_count}: 检测到 {len(results)} 个实例")

                cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("YOLOE Segmentation", result_frame)
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
            if args.image:
                image_path = resolve_path(args.image, project_root)
            else:
                default_image = svc.get_default_image()
                if not default_image:
                    raise ValueError("未提供 --image，且 config 无 test_image")
                image_path = Path(default_image)

            print(f"加载图像: {image_path}")
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法加载图像: {image_path}")

            print(f"运行分割... 词汇: {current_vocab() or '(config 默认)'}")
            status, results = svc.infer_image(image, conf=conf, iou=iou, prompts=prompts)
            if status != VisionServiceStatus.OK:
                raise RuntimeError(svc.last_error())

            vocab = current_vocab()
            if results:
                print(f"检测到 {len(results)} 个实例:")
                for r in results:
                    name = vocab[r.label] if 0 <= r.label < len(vocab) else f"Class {r.label}"
                    print(f"  {name}, Score: {r.score:.6f}, "
                          f"Box: [{r.x1:.3f}, {r.y1:.3f}, {r.x2:.3f}, {r.y2:.3f}]")
                if svc.supports_draw():
                    st, out = svc.draw(image)
                    result_image = out if st == VisionServiceStatus.OK else image
                else:
                    result_image = image
                cv2.imwrite(args.output, result_image)
                print(f"结果图像已保存到: {args.output}")
            else:
                print("未检测到任何实例")
                cv2.imwrite(args.output, image)
                print(f"原始图像已保存到: {args.output}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
