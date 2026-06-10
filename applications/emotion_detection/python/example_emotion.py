# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Emotion Recognition Example

image 模式: YOLOv5-Face 检测 + ResNet50 静态情绪分类（单帧）
camera 模式: YOLOv5-Face 检测 + ResNet50特征 + LSTM 动态情绪识别（10帧滑窗）

用法:
  python example_emotion.py --image test.jpg
  python example_emotion.py --use-camera
  python example_emotion.py --use-camera --camera-id 1
"""

import sys
from pathlib import Path

_cv_src = Path(__file__).resolve().parents[3] / "src"
if str(_cv_src) not in sys.path:
    sys.path.insert(0, str(_cv_src))

import argparse  # noqa: E402
from collections import deque  # noqa: E402
import yaml  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import create_model  # noqa: E402
from common import draw_detections, load_labels  # noqa: E402

_DEFAULT_EMOTION_LABELS = [
    "neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger",
]


def _load_emotion_labels(root: Path, cfg: dict) -> list:
    path = cfg.get("label_file_path")
    if path:
        try:
            return load_labels(str(_resolve(str(path), root)))
        except Exception as e:
            print(f"警告: 无法加载标签文件 {path}: {e}")
    return list(_DEFAULT_EMOTION_LABELS)


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Recognition Example")
    parser.add_argument("--config", default=None, help="应用配置 yaml")
    parser.add_argument("--image", default=None, help="输入图片路径（image 模式）")
    parser.add_argument("--output", default="output_emotion.jpg", help="输出图片路径")
    parser.add_argument("--use-camera", action="store_true", help="使用摄像头（camera 模式）")
    parser.add_argument("--camera-id", type=int, default=None, help="摄像头设备 ID")
    return parser.parse_args()


def _load_model(app_config: dict, key: str, config_dir: Path, root: Path):
    """按 app_config[key] 指向的子 yaml 加载单个模型。"""
    cfg_path = _resolve(str(config_dir / app_config[key]), root)
    cfg = _load_yaml(cfg_path)
    overrides = {}
    if cfg.get("model_path"):
        overrides["model_path"] = str(_resolve(str(cfg["model_path"]), root))
    return create_model(model_name=cfg_path.stem, config_dir=cfg_path.parent, **overrides)


def run_image(args, face_detector, emotion_model, root: Path, app_config: dict, config_dir: Path,
              emotion_labels: list):
    if args.image is None:
        cfg_path = _resolve(str(config_dir / app_config["emotion_model"]), root)
        cfg = _load_yaml(cfg_path)
        test_img = cfg.get("test_image")
        if not test_img:
            print("错误: 未提供 --image，且 emotion.yaml 中无 test_image")
            return
        args.image = str(_resolve(str(test_img), root))
        print(f"从配置读取图片: {args.image}")

    image = cv2.imread(args.image)
    if image is None:
        print(f"错误: 无法加载图片: {args.image}")
        return

    face_detections, face_images = face_detector.infer(image)
    print(f"检测到 {len(face_detections)} 个人脸")

    emotion_results = emotion_model.infer(face_images)

    detections = []
    for i, emo in enumerate(emotion_results):
        detections.append({
            "bbox": face_detections[i]["bbox"],
            "confidence": face_detections[i]["confidence"],
            "emotion": int(emo["emotion"]),
        })
        idx = int(emo["emotion"])
        label = emotion_labels[idx] if 0 <= idx < len(emotion_labels) else str(idx)
        print(f"  人脸 {i+1}: {label} (置信度 {face_detections[i]['confidence']:.3f})")

    result_image = draw_detections(
        image,
        [d["bbox"] for d in detections],
        [d["emotion"] for d in detections],
        [d["confidence"] for d in detections],
        labels=emotion_labels,
    )
    cv2.imwrite(args.output, result_image)
    print(f"结果已保存: {args.output}")


def run_camera(args, face_detector, backbone, emotion_lstm, camera_id: int):
    SEQ_LEN = 10
    feat_buffer = deque(maxlen=SEQ_LEN)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"✗ 无法打开摄像头 {camera_id}")
        return

    print(f"摄像头已打开 (id={camera_id})，按 'q' 退出，'s' 保存当前帧")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_idx += 1
        vis = frame.copy()

        try:
            detections, face_images = face_detector.infer(frame)
            # 取置信度最高的一张人脸
            best = max(
                ((d, fi) for d, fi in zip(detections, face_images) if fi.size > 0),
                key=lambda x: x[0]["confidence"],
                default=None,
            )
            if best is None:
                # 无人脸：清空滑窗，避免跨人脸/跨中断拼接出错误序列
                feat_buffer.clear()
            else:
                det, face_img = best
                # backbone 提取 512 维特征，应用层维护 10 帧滑窗
                feats = backbone.infer(face_img)[0]["features"]
                feat_buffer.append(feats)

                x1, y1, x2, y2 = map(int, det["bbox"])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if len(feat_buffer) < SEQ_LEN:
                    text = f"face {det['confidence']:.2f} (buffering {len(feat_buffer)}/{SEQ_LEN})"
                    color = (0, 255, 255)
                else:
                    seq = np.stack(list(feat_buffer), axis=0)  # (10, 512)
                    result = emotion_lstm.infer(seq)
                    label = result["emotion_label"]
                    conf = float(result["emotion_probs"].max())
                    text = f"{label} {conf:.2f} | face {det['confidence']:.2f}"
                    color = (0, 255, 0)
                cv2.putText(vis, text, (x1, max(y1 - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        except Exception as e:
            print(f"✗ 帧 {frame_idx} 推理失败: {e}")

        cv2.imshow("Dynamic Emotion Detection", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            out = f"emotion_camera_{frame_idx}.jpg"
            cv2.imwrite(out, vis)
            print(f"已保存: {out}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[3]
    default_app = root / "applications" / "emotion_detection" / "config" / "emotion_detection.yaml"
    app_path = Path(args.config) if args.config else default_app
    if not app_path.is_absolute():
        app_path = (root / app_path).resolve()

    app_config = _load_yaml(app_path)
    config_dir = app_path.parent
    camera_id = args.camera_id if args.camera_id is not None else int(app_config.get("camera_id", 0))

    print("=" * 60)
    print("Emotion Recognition Example")
    print("模式:", "camera (动态 LSTM)" if args.use_camera else "image (静态 ResNet50)")
    print("=" * 60)

    face_detector = _load_model(app_config, "face_model", config_dir, root)
    print(f"✓ 人脸检测器: {type(face_detector).__name__}")

    emotion_cfg_path = _resolve(str(config_dir / app_config["emotion_model"]), root)
    emotion_labels = _load_emotion_labels(root, _load_yaml(emotion_cfg_path))

    if args.use_camera:
        backbone = _load_model(app_config, "feature_model", config_dir, root)
        print(f"✓ 特征提取 backbone: {type(backbone).__name__}")
        emotion_lstm = _load_model(app_config, "lstm_model", config_dir, root)
        print(f"✓ 动态情绪 LSTM: {type(emotion_lstm).__name__}")
        run_camera(args, face_detector, backbone, emotion_lstm, camera_id)
    else:
        emotion_model = _load_model(app_config, "emotion_model", config_dir, root)
        print(f"✓ 静态情绪模型: {type(emotion_model).__name__}")
        run_image(args, face_detector, emotion_model, root, app_config, config_dir, emotion_labels)

    print("✓ 完成!")


if __name__ == "__main__":
    main()
