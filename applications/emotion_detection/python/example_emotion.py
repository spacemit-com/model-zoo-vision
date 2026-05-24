# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Emotion Recognition Example

This example demonstrates how to use the emotion recognition model to detect faces
and recognize emotions in images.
从 config/emotion_detection.yaml 读取应用配置（引用 face_detector.yaml / emotion.yaml）。
默认测试图由 emotion.yaml 的 test_image 提供。
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


def _resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Emotion Recognition Example',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python example_emotion.py --image test.jpg
  python example_emotion.py  # 使用 emotion.yaml 中的 test_image
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='应用配置 yaml (默认: applications/emotion_detection/config/emotion_detection.yaml)'
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
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]  # applications/emotion_detection/python/ -> cv
    default_app = project_root / "applications" / "emotion_detection" / "config" / "emotion_detection.yaml"
    app_path = Path(args.config) if args.config else default_app
    if not app_path.is_absolute():
        app_path = (project_root / app_path).resolve()
    try:
        app_config = _load_app_config(app_path)
    except Exception as e:
        print(f"✗ 加载应用配置失败: {e}")
        return
    config_dir = app_path.parent
    face_yaml = _resolve_path(str(config_dir / app_config["face_model"]), project_root)
    emotion_yaml = _resolve_path(str(config_dir / app_config["emotion_model"]), project_root)

    try:
        face_model_config = _load_app_config(Path(face_yaml))
        emotion_model_config = _load_app_config(Path(emotion_yaml))
    except Exception as e:
        print(f"✗ 加载模型配置失败: {e}")
        return

    if args.image is None:
        test_image_path = emotion_model_config.get("test_image")
        if test_image_path:
            args.image = str(_resolve_path(str(test_image_path), project_root))
            print(f"从配置读取图片路径: {args.image}")
        else:
            print("错误: 未提供 --image，且 emotion.yaml 中无 test_image")
            return

    # 检查图片文件是否存在
    if not Path(args.image).exists():
        print(f"错误: 图片文件不存在: {args.image}")
        return

    # 设置输出路径：命令行 > 应用配置 output_path > 默认
    if args.output is None:
        args.output = "output_emotion.jpg"

    emotion_config_path = str(emotion_yaml)
    face_detector_config_path = str(face_yaml)
    emotion_config_dir_abs = Path(emotion_config_path).parent
    face_detector_config_dir_abs = Path(face_detector_config_path).parent
    emotion_model_name = Path(emotion_config_path).stem
    face_detector_model_name = Path(face_detector_config_path).stem

    print("=" * 60)
    print("Emotion Recognition Example")
    print("=" * 60)
    print(f"情绪模型配置: {emotion_config_path}")
    print(f"人脸模型配置: {face_detector_config_path}")
    print(f"图片: {args.image}")
    if emotion_model_config.get("image_size"):
        print(f"Emotion 输入尺寸: {emotion_model_config['image_size']}，人脸检测: [640, 640] (固定)")
    print("=" * 60)

    face_override_params = {}
    if face_model_config.get("model_path"):
        face_override_params["model_path"] = str(
            _resolve_path(str(face_model_config["model_path"]), project_root))

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
    if emotion_model_config.get("model_path"):
        emotion_override_params["emotion_model_path"] = str(
            _resolve_path(str(emotion_model_config["model_path"]), project_root))
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
