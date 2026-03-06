# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Emotion Recognition Example

This example demonstrates how to use the emotion recognition model to detect faces
and recognize emotions in images.
从 applications/emotion_detection/config/emotion_detection.yaml 读取应用配置，
模型由 emotion_config_path / face_detector_config_path 指向的 yaml 加载。
"""

import sys
from pathlib import Path

# 将 cv/src 加入路径（脚本在 applications/emotion_detection/python/，parents[3]=cv）
_cv_src = Path(__file__).resolve().parents[3] / "src"
if str(_cv_src) not in sys.path:
    sys.path.insert(0, str(_cv_src))

import argparse  # noqa: E402
import yaml  # noqa: E402
import cv2  # noqa: E402

from core import create_model  # noqa: E402
from common import draw_detections  # noqa: E402

# Emotion labels
EMOTION_LABELS = {
    0: "neutral",
    1: "happy",
    2: "sad",
    3: "angry",
    4: "fear",
    5: "disgust",
    6: "surprise"
}

def _load_app_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Emotion Recognition Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python example_emotion.py --config ../config/emotion_detection.yaml   # 在 python/ 目录下运行时
  python example_emotion.py --config applications/emotion_detection/config/emotion_detection.yaml --image test.jpg
  python example_emotion.py  # 使用默认配置（无需 --config）
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='应用配置 yaml 路径 (默认: applications/emotion_detection/config/emotion_detection.yaml)'
    )
    parser.add_argument(
        '--image',
        type=str,
        default=None,
        help='输入图片路径 (如果未提供，将从 yaml 配置中的 test_image 读取)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='输出图片路径 (默认: output_emotion.jpg)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=None,
        help='人脸检测置信度阈值 (覆盖 yaml 中的默认值)'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=None,
        help='IoU 阈值，用于 NMS (覆盖 yaml 中的默认值)'
    )
    parser.add_argument(
        '--num-threads',
        type=int,
        default=None,
        help='推理线程数 (覆盖 yaml 中的默认值)'
    )
    parser.add_argument(
        '--face-model-path',
        type=str,
        default=None,
        help='人脸检测模型 ONNX 路径，覆盖 yaml 中的 face_detector_path'
    )
    parser.add_argument(
        '--emotion-model-path',
        type=str,
        default=None,
        help='情绪模型 ONNX 路径，覆盖 yaml 中的 emotion_model_path'
    )
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]  # applications/emotion_detection/python/ -> cv
    default_config = project_root / "applications" / "emotion_detection" / "config" / "emotion_detection.yaml"
    config_path = Path(args.config) if args.config else default_config
    if not config_path.is_absolute():
        # Prefer cwd-relative path (e.g. ../config/emotion_detection.yaml when run from python/)
        cwd_resolved = (Path.cwd() / config_path).resolve()
        config_path = cwd_resolved if cwd_resolved.exists() else (project_root / config_path).resolve()
    try:
        app_config = _load_app_config(config_path)
    except Exception as e:
        print(f"✗ 加载应用配置失败: {e}")
        return

    # 如果没有提供图片路径，优先从应用配置的 test_image 读取
    if args.image is None:
        test_image_path = app_config.get("test_image")
        if test_image_path:
            if not Path(test_image_path).is_absolute():
                args.image = str((project_root / test_image_path).resolve())
            else:
                args.image = test_image_path
            print(f"从配置读取图片路径: {args.image}")
        else:
            print("错误: 未提供 --image，且应用配置中无 test_image")
            return

    # 检查图片文件是否存在
    if not Path(args.image).exists():
        print(f"错误: 图片文件不存在: {args.image}")
        return

    # 设置输出路径：命令行 > 应用配置 output_path > 默认
    if args.output is None:
        args.output = app_config.get("output_path") or "output_emotion.jpg"

    # 从应用配置读取模型 yaml 路径
    emotion_config_path = app_config.get("emotion_config_path", "")
    face_detector_config_path = app_config.get("face_detector_config_path", "")
    if not emotion_config_path:
        print("✗ emotion_config_path 未设置，请在 emotion_detection.yaml 中指定")
        return
    if not face_detector_config_path:
        print("✗ face_detector_config_path 未设置，请在 emotion_detection.yaml 中指定")
        return
    if not Path(emotion_config_path).is_absolute():
        emotion_config_path = str((project_root / emotion_config_path).resolve())
    if not Path(face_detector_config_path).is_absolute():
        face_detector_config_path = str((project_root / face_detector_config_path).resolve())
    emotion_config_dir_abs = Path(emotion_config_path).parent
    face_detector_config_dir_abs = Path(face_detector_config_path).parent
    emotion_model_name = Path(emotion_config_path).stem
    face_detector_model_name = Path(face_detector_config_path).stem

    print("=" * 60)
    print("Emotion Recognition Example")
    print("=" * 60)
    print(f"应用配置: {config_path}")
    print(f"情绪模型: {emotion_config_path}")
    print(f"人脸检测: {face_detector_config_path}")
    print(f"图片: {args.image}")
    if app_config.get("image_size"):
        print(f"Emotion 输入尺寸: {app_config['image_size']}，人脸检测: [640, 640] (固定)")
    print("=" * 60)

    # 覆盖参数：命令行 --face-model-path / --emotion-model-path 优先于 yaml
    face_override_params = {}
    face_path_src = args.face_model_path or app_config.get("face_detector_path", "")
    if face_path_src:
        p = Path(face_path_src).expanduser()
        face_override_params["model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
    if args.conf_threshold is not None:
        face_override_params["conf_thres"] = args.conf_threshold
    if args.iou_threshold is not None:
        face_override_params["iou_thres"] = args.iou_threshold
    if args.num_threads is not None:
        face_override_params["num_threads"] = args.num_threads

    if not face_detector_config_dir_abs.is_dir():
        print(f"✗ 人脸检测配置目录不存在: {face_detector_config_dir_abs}")
        return
    print(f"\n从 {face_detector_config_path} 加载人脸检测器...")
    try:
        face_detector = create_model(
            model_name=face_detector_model_name,
            config_dir=face_detector_config_dir_abs,
            **face_override_params,
        )
        print("✓ 人脸检测器加载成功!")
        print(f"  模型类: {type(face_detector).__name__}")
    except Exception as e:
        print(f"✗ 人脸检测器加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    emotion_override_params = {}
    emotion_path_src = args.emotion_model_path or app_config.get("emotion_model_path", "")
    if emotion_path_src:
        p = Path(emotion_path_src).expanduser()
        emotion_override_params["emotion_model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
    if args.num_threads is not None:
        emotion_override_params["num_threads"] = args.num_threads
    if not emotion_config_dir_abs.is_dir():
        print(f"✗ 情绪模型配置目录不存在: {emotion_config_dir_abs}")
        return
    print(f"\n从 {emotion_config_path} 加载 Emotion 模型...")
    try:
        recognizer = create_model(
            model_name=emotion_model_name,
            config_dir=emotion_config_dir_abs,
            **emotion_override_params,
        )
        print("✓ Emotion 模型加载成功!")
        print(f"  模型类: {type(recognizer).__name__}")
        if hasattr(recognizer, 'input_shape'):
            print(f"  输入尺寸: {recognizer.input_shape}")
    except Exception as e:
        print(f"✗ Emotion 模型加载失败: {e}")
        print(f"\n提示: 请确保 {emotion_config_path} "
              "存在并包含 class、emotion_model_path、default_params")
        import traceback
        traceback.print_exc()
        return

    # 加载图片
    print(f"\n加载图片: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"错误: 无法加载图片: {args.image}")
        return
    print(f"图片尺寸: {image.shape}")

    # 先做人脸检测，再做情绪分类（EmotionRecognizer.infer 只吃检测后的 face_images）
    print("\n运行人脸检测...")
    try:
        face_detections, face_images = face_detector.infer(image)
        print(f"检测到 {len(face_detections)} 个人脸")
    except Exception as e:
        print(f"✗ 人脸检测失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n运行情绪识别...")
    emotion_results = recognizer.infer(face_images)

    # 将 bbox/confidence 贴回输出，形成统一 detections
    detections = []
    for i, emo in enumerate(emotion_results):
        det = {
            "bbox": face_detections[i]["bbox"],
            "confidence": face_detections[i]["confidence"],
            "emotion": int(emo["emotion"]),
        }
        detections.append(det)

    # 显示结果
    if detections:
        print(f"\n检测到 {len(detections)} 个人脸及其情绪:")
        for i, detection in enumerate(detections, 1):
            emotion_id = int(detection['emotion'])  # core返回的是整数索引
            emotion_name = EMOTION_LABELS.get(emotion_id, f"emotion_{emotion_id}")
            confidence = detection['confidence']
            bbox = detection['bbox']
            print(f"  {i}. {emotion_name} (置信度: {confidence:.3f}) 位置: {bbox}")
    else:
        print("\n未检测到人脸")

    # 绘制结果
    print("\n绘制结果...")
    # draw_detections期望labels参数，它会将整数索引转换为名称
    result_image = draw_detections(
        image,
        [detection['bbox'] for detection in detections],
        [int(detection['emotion']) for detection in detections],  # 确保是整数索引
        [detection['confidence'] for detection in detections],
        labels=EMOTION_LABELS
    )

    # 保存输出
    cv2.imwrite(args.output, result_image)
    print(f"结果已保存到: {args.output}")
    print("\n✓ 完成!")


if __name__ == '__main__':
    main()
