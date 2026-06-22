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

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus

_DEFAULT_EMOTION_LABELS = [
    "neutral", "happiness", "sadness", "surprise", "fear", "disgust", "anger",
]


def load_labels(path: str) -> list:
    """Read one class name per non-empty line."""
    names = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names


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


def _bbox_min_side(bbox) -> float:
    x1, y1, x2, y2 = bbox[:4]
    return min(abs(x2 - x1), abs(y2 - y1))


def parse_args():
    parser = argparse.ArgumentParser(description="Emotion Recognition Example")
    parser.add_argument("--config", default=None, help="应用配置 yaml")
    parser.add_argument("--image", default=None, help="输入图片路径（image 模式）")
    parser.add_argument("--output", default="output_emotion.jpg", help="输出图片路径")
    parser.add_argument("--use-camera", action="store_true", help="使用摄像头（camera 模式）")
    parser.add_argument("--camera-id", type=int, default=None, help="摄像头设备 ID")
    return parser.parse_args()


def _create_service(app_config: dict, key: str, config_dir: Path, root: Path):
    """按 app_config[key] 指向的子 yaml 创建一个 VisionServiceNative 实例。"""
    cfg_path = _resolve(str(config_dir / app_config[key]), root)
    cfg = _load_yaml(cfg_path)
    model_override = ""
    if cfg.get("model_path"):
        model_override = str(_resolve(str(cfg["model_path"]), root))
    svc = VisionServiceNative.create(str(cfg_path), model_path_override=model_override)
    return svc


def _draw_face_box(image, bbox, text, color):
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(image, text, (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def run_image(args, face_detector, emotion_model, root: Path, app_config: dict, config_dir: Path,
              emotion_labels: list, min_face_size: int = 0):
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

    # 人脸检测：返回带 bbox 的结果，由应用层裁剪人脸再做表情分类
    status, faces = face_detector.infer_image(image)
    if status != VisionServiceStatus.OK:
        print(f"✗ 人脸检测失败: {face_detector.last_error()}")
        return
    print(f"检测到 {len(faces)} 个人脸")

    h, w = image.shape[:2]
    for r in faces:
        bbox = (r.x1, r.y1, r.x2, r.y2)
        conf = float(r.score)
        min_side = _bbox_min_side(bbox)
        if min_side < min_face_size:
            # 过小人脸：仅画灰框，不做表情识别
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 1)
            continue

        x1 = max(0, int(r.x1))
        y1 = max(0, int(r.y1))
        x2 = min(w, int(r.x2))
        y2 = min(h, int(r.y2))
        if x2 <= x1 or y2 <= y1:
            continue
        face_crop = np.ascontiguousarray(image[y1:y2, x1:x2], dtype=np.uint8)

        # 表情分类：取 class_scores 的 argmax
        st_e, emo_results = emotion_model.infer_image(face_crop)
        if st_e != VisionServiceStatus.OK or not emo_results:
            _draw_face_box(image, bbox, f"face {conf:.2f}", (0, 255, 255))
            continue
        scores = list(emo_results[0].class_scores)
        if not scores:
            _draw_face_box(image, bbox, f"face {conf:.2f}", (0, 255, 255))
            continue
        idx = int(np.argmax(scores))
        label = emotion_labels[idx] if 0 <= idx < len(emotion_labels) else str(idx)
        print(f"  {label} (置信度 {conf:.3f})")
        _draw_face_box(image, bbox, f"{label} {conf:.2f}", (0, 255, 0))

    cv2.imwrite(args.output, image)
    print(f"结果已保存: {args.output}")


def run_camera(args, face_detector, backbone, emotion_lstm, emotion_labels, camera_id: int,
               min_face_size: int = 0):
    """Dynamic (LSTM) emotion recognition over a camera stream.

    Pipeline (orchestrated in Python against the wheel API):
      frame -> YOLOv5-Face detect -> crop best face
            -> ResNet50 backbone (feature_mode) 512-d embedding   [backbone.infer_embedding]
            -> 10-frame feature window
            -> LSTM over the flat (1,10,512) feature sequence      [emotion_lstm.infer_sequence]

    The backbone service is created from emotion_features.yaml (EmotionRecognizer with
    feature_mode=true, emits an Embedding result). The LSTM service is created from
    emotion_lstm.yaml; its expected_sequence_size is 10*512, and the C++ sequence path
    passes the flat float buffer straight to the model, so we feed the raveled window.
    """
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
        h, w = frame.shape[:2]

        try:
            status, faces = face_detector.infer_image(frame)
            if status != VisionServiceStatus.OK:
                print(f"✗ 帧 {frame_idx} 人脸检测失败: {face_detector.last_error()}")
                faces = []

            # 先按尺寸过滤，再在合格人脸里取置信度最高的一张：
            # 避免远处高分小脸抢占、导致漏识别近处大脸。
            eligible = [
                r for r in faces
                if _bbox_min_side((r.x1, r.y1, r.x2, r.y2)) >= min_face_size
            ]
            best = max(eligible, key=lambda r: float(r.score), default=None)

            if best is None:
                # 无合格人脸（无脸或全部过小）：清空滑窗，避免拼接出错误序列。
                feat_buffer.clear()
                for r in faces:
                    if _bbox_min_side((r.x1, r.y1, r.x2, r.y2)) < min_face_size:
                        bx = list(map(int, (r.x1, r.y1, r.x2, r.y2)))
                        cv2.rectangle(vis, (bx[0], bx[1]), (bx[2], bx[3]), (128, 128, 128), 1)
            else:
                conf = float(best.score)
                x1 = max(0, int(best.x1))
                y1 = max(0, int(best.y1))
                x2 = min(w, int(best.x2))
                y2 = min(h, int(best.y2))
                if x2 <= x1 or y2 <= y1:
                    feat_buffer.clear()
                    cv2.imshow("Dynamic Emotion Detection", vis)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                    continue
                face_crop = np.ascontiguousarray(frame[y1:y2, x1:x2], dtype=np.uint8)

                # backbone 提取 512 维特征，应用层维护 10 帧滑窗
                st_f, feats = backbone.infer_embedding(face_crop)
                if st_f != VisionServiceStatus.OK or not feats:
                    feat_buffer.clear()
                    text = f"face {conf:.2f} (feature failed)"
                    color = (0, 0, 255)
                else:
                    feat_buffer.append(np.asarray(feats, dtype=np.float32))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if len(feat_buffer) < SEQ_LEN:
                        text = f"face {conf:.2f} (buffering {len(feat_buffer)}/{SEQ_LEN})"
                        color = (0, 255, 255)
                    else:
                        seq = np.ascontiguousarray(
                            np.stack(list(feat_buffer), axis=0).ravel(), dtype=np.float32
                        )  # (10*512,) flat
                        st_l, scores = emotion_lstm.infer_sequence(seq, w, h)
                        if st_l != VisionServiceStatus.OK or not scores:
                            text = f"face {conf:.2f} (lstm failed)"
                            color = (0, 0, 255)
                        else:
                            scores = np.asarray(scores, dtype=np.float32)
                            idx = int(np.argmax(scores))
                            names = emotion_lstm.get_class_names() or emotion_labels
                            label = names[idx] if 0 <= idx < len(names) else str(idx)
                            text = f"{label} {float(scores[idx]):.2f} | face {conf:.2f}"
                            color = (0, 255, 0)
                    cv2.putText(vis, text, (x1, max(y1 - 8, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        except Exception as e:
            print(f"✗ 帧 {frame_idx} 推理失败: {e}")

        # 循环末尾统一处理显示与按键，保证任何分支下 q/s 都生效
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
    min_face_size = int(app_config.get("min_face_size", 0))

    print("=" * 60)
    print("Emotion Recognition Example")
    print("模式:", "camera (动态 LSTM)" if args.use_camera else "image (静态 ResNet50)")
    print("=" * 60)

    face_detector = _create_service(app_config, "face_model", config_dir, root)
    print("✓ 人脸检测器已创建")

    emotion_cfg_path = _resolve(str(config_dir / app_config["emotion_model"]), root)
    emotion_labels = _load_emotion_labels(root, _load_yaml(emotion_cfg_path))

    if args.use_camera:
        # 动态 LSTM 模式：backbone(特征) + LSTM(特征序列)，均通过 wheel 接口编排。
        backbone = _create_service(app_config, "feature_model", config_dir, root)
        print("✓ 特征提取 backbone 已创建")
        emotion_lstm = _create_service(app_config, "lstm_model", config_dir, root)
        print("✓ 动态情绪 LSTM 已创建")
        run_camera(args, face_detector, backbone, emotion_lstm, emotion_labels,
                   camera_id, min_face_size)
    else:
        emotion_model = _create_service(app_config, "emotion_model", config_dir, root)
        print("✓ 静态情绪模型已创建")
        run_image(args, face_detector, emotion_model, root, app_config, config_dir,
                  emotion_labels, min_face_size)

    print("✓ 完成!")


if __name__ == "__main__":
    main()
