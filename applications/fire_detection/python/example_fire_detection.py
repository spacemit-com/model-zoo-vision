# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fire Detection Example (火焰检测示例)

使用 YOLOv8 检测模型（yolov8_fire.q.onnx）进行火焰检测。
从 applications/fire_detection/config/fire_detection.yaml 读取应用配置，
模型路径由 detector_model_path 指定。
"""

import sys
from pathlib import Path

# 将 cv/src 加入路径（脚本在 applications/fire_detection/python/，parents[3]=cv）
_cv_src = Path(__file__).resolve().parents[3] / "src"
if str(_cv_src) not in sys.path:
    sys.path.insert(0, str(_cv_src))

import argparse  # noqa: E402
import yaml  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import create_model  # noqa: E402
from common import load_labels  # noqa: E402
from common import draw_detections  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fire Detection Example (火焰检测示例)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python example_fire_detection.py --image test.jpg
  python example_fire_detection.py --video test.mp4
  python example_fire_detection.py --use-camera
  python example_fire_detection.py   # 使用应用配置中的 test_image 或 test_video
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="应用配置 yaml 路径 (默认: applications/fire_detection/config/fire_detection.yaml)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="输入图片路径（与 --video/--use-camera 二选一）",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="输入视频路径",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="使用摄像头",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=None,
        help="摄像头设备 ID",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片/视频路径（图片模式默认 output_fire_detection.jpg）",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="置信度阈值",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="IoU 阈值，用于 NMS",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="推理线程数",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="检测模型 ONNX 路径，覆盖 yaml 中的 detector_model_path",
    )
    return parser.parse_args()


def _load_app_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    default_config = project_root / "applications" / "fire_detection" / "config" / "fire_detection.yaml"
    config_path = Path(args.config) if args.config else default_config
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    try:
        app_config = _load_app_config(config_path)
    except Exception as e:
        print(f"✗ 加载应用配置失败: {e}")
        return 1

    detector_config_path = app_config.get("detector_config_path", "")
    detector_model_path = app_config.get("detector_model_path", "")
    if not detector_config_path:
        print("✗ detector_config_path 未设置，请在 fire_detection.yaml 中指定")
        return 1
    if not Path(detector_config_path).is_absolute():
        detector_config_path = str((project_root / detector_config_path).resolve())
    detector_config_dir_abs = Path(detector_config_path).parent
    detector_model_name = Path(detector_config_path).stem
    if not detector_config_dir_abs.is_dir():
        print(f"✗ 检测器配置目录不存在: {detector_config_dir_abs}")
        return 1

    # 解析模型路径覆盖：命令行 --model-path 优先于 yaml 的 detector_model_path
    override_params = {}
    model_path_src = args.model_path if args.model_path else detector_model_path
    if model_path_src:
        p = Path(model_path_src).expanduser()
        override_params["model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
    if args.conf_threshold is not None:
        override_params["conf_threshold"] = args.conf_threshold
    if args.iou_threshold is not None:
        override_params["iou_threshold"] = args.iou_threshold
    if args.num_threads is not None:
        override_params["num_threads"] = args.num_threads

    # 标签文件
    label_path = app_config.get("label_file_path")
    labels = None
    if label_path:
        lp = Path(label_path)
        if not lp.is_absolute():
            lp = (project_root / label_path).resolve()
        if lp.exists():
            try:
                labels = load_labels(str(lp))
                print(f"✓ 加载标签: {lp} ({len(labels)} 个)")
            except Exception as e:
                print(f"警告: 无法加载标签 {lp}: {e}")

    # 输入源：优先 --image，其次 --video，其次 --use-camera，最后 yaml
    use_image = args.image is not None
    if not use_image and not args.video and not args.use_camera:
        test_image = app_config.get("test_image")
        test_video = app_config.get("test_video")
        if test_image:
            args.image = str(
                (project_root / test_image).resolve()
                if not Path(test_image).is_absolute() else test_image
            )
            use_image = True
        elif test_video:
            args.video = str(
                (project_root / test_video).resolve()
                if not Path(test_video).is_absolute() else test_video
            )

    if use_image:
        if not Path(args.image).exists():
            print(f"错误: 图片不存在: {args.image}")
            return 1
        if args.output is None:
            args.output = "output_fire_detection.jpg"
        print(f"应用配置: {config_path}")
        print(f"检测模型: {detector_config_path}")
        print(f"图片: {args.image}")
        print("=" * 60)
        print(f"\n从 {detector_config_path} 加载 YOLOv8 检测器...")
        detector = create_model(
            model_name=detector_model_name,
            config_dir=detector_config_dir_abs,
            **override_params,
        )
        print("✓ 检测器加载成功")
        image = cv2.imread(args.image)
        if image is None:
            print(f"错误: 无法读取图片 {args.image}")
            return 1
        detections = detector.infer(image)
        if detections:
            boxes = np.array([d["bbox"] for d in detections])
            classes = np.array([d["class_id"] for d in detections])
            scores = np.array([d["confidence"] for d in detections])
            result = draw_detections(image, boxes, classes, scores, labels)
            print(f"检测到 {len(detections)} 个目标")
        else:
            result = image
            print("未检测到目标")
        cv2.imwrite(args.output, result)
        print(f"结果已保存: {args.output}")
        return 0

    # 视频或摄像头
    if args.video and not Path(args.video).exists():
        args.video = str((project_root / args.video).resolve()) if not Path(args.video).is_absolute() else args.video
    if args.video and not Path(args.video).exists():
        print(f"错误: 视频不存在: {args.video}")
        return 1
    if args.use_camera:
        cap = cv2.VideoCapture(args.camera_id or app_config.get("camera_id", 0))
        source_desc = f"摄像头 {args.camera_id or 0}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = args.video
    if not cap.isOpened():
        print(f"✗ 无法打开: {source_desc}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(int(1000.0 / fps), 1)
    print(f"应用配置: {config_path}")
    print(f"检测模型: {detector_config_path}")
    print(f"输入: {source_desc}")
    print("按 'q' 退出")
    print("=" * 60)
    print(f"\n从 {detector_config_path} 加载 YOLOv8 检测器...")
    detector = create_model(
        model_name=detector_model_name,
        config_dir=detector_config_dir_abs,
        **override_params,
    )
    print("✓ 检测器加载成功")
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not args.use_camera:
                    print("视频结束")
                break
            detections = detector.infer(frame)
            if detections:
                boxes = np.array([d["bbox"] for d in detections])
                classes = np.array([d["class_id"] for d in detections])
                scores = np.array([d["confidence"] for d in detections])
                frame = draw_detections(frame, boxes, classes, scores, labels)
            cv2.imshow("Fire Detection", frame)
            if (cv2.waitKey(delay_ms) & 0xFF) == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print("✓ 完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
