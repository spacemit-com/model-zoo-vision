# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fire Detection Example (火焰检测示例)

使用 YOLOv8 检测模型（yolov8_fire.q.onnx）进行火焰检测。
从 applications/fire_detection/config/fire_detection.yaml 读取应用配置（引用 yolov8_fire.yaml）。
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
        help="应用配置 yaml (默认: applications/fire_detection/config/fire_detection.yaml)",
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
    return parser.parse_args()


def _load_app_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    default_app = project_root / "applications" / "fire_detection" / "config" / "fire_detection.yaml"
    app_path = Path(args.config) if args.config else default_app
    if not app_path.is_absolute():
        app_path = (project_root / app_path).resolve()
    try:
        app_config = _load_app_config(app_path)
    except Exception as e:
        print(f"✗ 加载应用配置失败: {e}")
        return 1
    detector_config_path = str(_resolve_path(
        str(app_path.parent / app_config["model"]), project_root))
    try:
        model_config = _load_app_config(Path(detector_config_path))
    except Exception as e:
        print(f"✗ 加载模型配置失败: {e}")
        return 1

    detector_model_path = model_config.get("model_path", "")
    detector_config_dir_abs = Path(detector_config_path).parent
    detector_model_name = Path(detector_config_path).stem
    if not detector_config_dir_abs.is_dir():
        print(f"✗ 检测器配置目录不存在: {detector_config_dir_abs}")
        return 1

    override_params = {}
    if detector_model_path:
        override_params["model_path"] = str(_resolve_path(detector_model_path, project_root))

    label_path = model_config.get("label_file_path")
    labels = None
    if label_path:
        lp = _resolve_path(label_path, project_root)
        if lp.exists():
            try:
                labels = load_labels(str(lp))
                print(f"✓ 加载标签: {lp} ({len(labels)} 个)")
            except Exception as e:
                print(f"警告: 无法加载标签 {lp}: {e}")

    # 输入源：优先 --image，其次 --video，其次 --use-camera，最后 yaml
    use_image = args.image is not None
    if not use_image and not args.video and not args.use_camera:
        test_image = model_config.get("test_image")
        test_video = model_config.get("test_video")
        if test_image:
            args.image = str(_resolve_path(test_image, project_root))
            use_image = True
        elif test_video:
            args.video = str(_resolve_path(test_video, project_root))
        else:
            print("错误: 未提供 --image / --video / --use-camera，且配置中无 test_image / test_video")
            return 1

    if use_image:
        if not Path(args.image).exists():
            print(f"错误: 图片不存在: {args.image}")
            return 1
        if args.output is None:
            args.output = "output_fire_detection.jpg"
        print(f"模型配置: {detector_config_path}")
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
        args.video = str(_resolve_path(args.video, project_root))
    if args.video and not Path(args.video).exists():
        print(f"错误: 视频不存在: {args.video}")
        return 1
    if args.use_camera:
        camera_id = args.camera_id if args.camera_id is not None else 0
        cap = cv2.VideoCapture(camera_id)
        source_desc = f"摄像头 {camera_id}"
    else:
        cap = cv2.VideoCapture(args.video)
        source_desc = args.video
    if not cap.isOpened():
        print(f"✗ 无法打开: {source_desc}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    delay_ms = max(int(1000.0 / fps), 1)
    print(f"模型配置: {detector_config_path}")
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
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if not args.use_camera:
                    print("视频结束")
                break
            frame_count += 1
            detections = detector.infer(frame)
            if detections:
                boxes = np.array([d["bbox"] for d in detections])
                classes = np.array([d["class_id"] for d in detections])
                scores = np.array([d["confidence"] for d in detections])
                display = draw_detections(frame, boxes, classes, scores, labels)
            else:
                display = frame.copy()
                if args.use_camera and (frame_count <= 5 or frame_count % 30 == 0):
                    print(f"帧 {frame_count}: 未检测到目标")
            cv2.imshow("Fire Detection", display)
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
