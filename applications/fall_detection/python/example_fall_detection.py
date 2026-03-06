# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fall Detection Example (跌倒检测示例)

使用 YOLOv8-Pose 姿态估计 + STGCN 动作识别（多类别，含 Fall Down）进行跌倒检测。
仅通过 STGCN 判断动作/跌倒，无角度规则。
"""

import sys
from pathlib import Path
from collections import deque

# 将 cv/src 加入路径（脚本在 applications/fall_detection/python/，parents[3]=cv）
_cv_src = Path(__file__).resolve().parents[3] / "src"
if str(_cv_src) not in sys.path:
    sys.path.insert(0, str(_cv_src))

import argparse  # noqa: E402
import yaml  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import create_model  # noqa: E402
from common import draw_keypoints  # noqa: E402

# STGCN/TSSTG：30 帧、13 关键点（COCO 子集），用于 --use-stgcn 时
COCO17_TO_TSTSGO13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
STGCN_SEQUENCE_LENGTH = 30

# COCO Pose 关键点索引
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fall Detection Example (跌倒检测示例)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python example_fall_detection.py --use-camera
  python example_fall_detection.py --video test.mp4
  python example_fall_detection.py  # 使用应用配置中的 test_video
        """
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='应用配置 yaml 路径 (默认: applications/fall_detection/config/fall_detection.yaml)'
    )
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='输入视频路径 (如果未提供且不使用摄像头，将从应用配置的 test_video 读取)'
    )
    parser.add_argument(
        '--use-camera',
        action='store_true',
        help='使用摄像头作为输入'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=None,
        help='摄像头设备 ID (覆盖 yaml 中的默认值)'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=None,
        help='置信度阈值 (覆盖 yaml 中的默认值)'
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
        '--kp-threshold',
        type=float,
        default=None,
        help='关键点可见度阈值 (覆盖 yaml 中的默认值)'
    )
    parser.add_argument(
        '--pose-model',
        type=str,
        default=None,
        help='姿态模型 (YOLOv8-Pose) ONNX 路径，覆盖 yaml 中的 pose_model_path'
    )
    parser.add_argument(
        '--stgcn-model',
        type=str,
        default=None,
        help='STGCN/TSSTG ONNX 模型路径，覆盖 config 中 stgcn_action.yaml 的 model_path（--use-stgcn 时可选）'
    )
    parser.add_argument(
        '--stgcn-wait-frames',
        type=int,
        default=10,
        help='STGCN 每隔 N 帧推理一次（默认 10）'
    )
    parser.add_argument(
        '--smooth-window',
        type=int,
        default=None,
        help='预测类别平滑窗口：最近 N 次推理中超过一半为 Fall Down 才判跌倒（默认从 yaml 读取，若未配置则为 5）'
    )
    return parser.parse_args()

def normalize_keypoints_for_stgcn(keypoints, image_width, image_height, bbox=None):
    """将单帧关键点归一化供 STGCN 使用：优先按 bbox，否则按图像宽高。"""
    if bbox is not None and len(bbox) >= 4:
        x1, y1, x2, y2 = bbox[:4]
        bw, bh = float(x2 - x1), float(y2 - y1)
        if bw > 1.0 and bh > 1.0:
            return [((x - x1) / bw, (y - y1) / bh, float(vis)) for x, y, vis in keypoints]
    if image_width <= 1 or image_height <= 1:
        return [(float(x), float(y), float(vis)) for x, y, vis in keypoints]
    return [(x / image_width, y / image_height, float(vis)) for x, y, vis in keypoints]


def keypoint_buffer_to_tsstg_pts(keypoint_buffer, image_size):
    """将 keypoint_buffer（每帧 17 点 (x,y,vis)）转为 TSSTG 输入 pts (t, 13, 3)，像素坐标。"""
    w, h = image_size[0], image_size[1]
    arr = np.array(keypoint_buffer, dtype=np.float32)
    pts = arr[:, COCO17_TO_TSTSGO13, :].copy()
    if w > 0 and h > 0 and np.all(pts[:, :, :2] <= 1.0 + 1e-5) and np.all(pts[:, :, :2] >= -1e-5):
        pts[:, :, 0] *= w
        pts[:, :, 1] *= h
    return pts


def draw_fall_detection(image, detections, kp_threshold=0.3, action_results=None):
    """
    在图像上绘制动作/跌倒检测结果（仅 STGCN 多类别结果）。

    Args:
        image: 输入图像
        detections: 检测结果列表
        kp_threshold: 关键点可见度阈值
        action_results: 每人的动作结果 [{'action_name': str, 'is_fall': bool, 'fall_prob': float}, ...]；
                        未提供或对应索引无结果时显示 "—"

    Returns:
        绘制后的图像
    """
    result = image.copy()

    for i, det in enumerate(detections):
        bbox = det.get('bbox', det.get('box', []))
        keypoints = det.get('keypoints', [])

        if len(bbox) < 4 or len(keypoints) < 17:
            continue

        if action_results is not None and i < len(action_results):
            ar = action_results[i]
            action_name = ar.get('action_name', '—')
            is_fall = ar.get('is_fall', False)
            fall_prob = ar.get('fall_prob', 0.0)
        else:
            action_name = '—'
            is_fall = False
            fall_prob = 0.0

        box_color = (0, 0, 255) if is_fall else (0, 255, 0)
        kp_color = (0, 0, 255) if is_fall else (255, 0, 0)
        det_dict = {'box': bbox, 'keypoints': keypoints}
        result = draw_keypoints(
            result, [det_dict],
            box_color=box_color,
            kp_color=kp_color,
            confidence_threshold=kp_threshold,
        )

        x1, y1, x2, y2 = map(int, bbox)
        status_text = action_name if action_name else '—'
        text_color = (0, 0, 255) if is_fall else (0, 255, 0)
        (tw, th), bl = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(
            result,
            (x1, y1 - th - bl - 10),
            (x1 + tw + 10, y1),
            text_color,
            -1
        )
        cv2.putText(
            result, status_text, (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        if is_fall:
            sub_text = f"Fall Down ({fall_prob:.2f})"
            cv2.putText(
                result, sub_text, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
        elif action_name != '—' and fall_prob is not None:
            cv2.putText(
                result, f"P(fall)={fall_prob:.2f}", (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
            )

    return result

def _load_app_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def main():
    """Main function."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]  # applications/fall_detection/python/ -> cv
    default_config = project_root / "applications" / "fall_detection" / "config" / "fall_detection.yaml"
    config_path = Path(args.config) if args.config else default_config
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    try:
        app_config = _load_app_config(config_path)
    except Exception as e:
        print(f"✗ 加载应用配置失败: {e}")
        return
    app_config_dir = project_root / "applications" / "fall_detection" / "config"

    # 默认参数来自应用配置 yaml（命令行可覆盖）
    if args.kp_threshold is None:
        args.kp_threshold = float(app_config.get("kp_threshold", 0.3))
    if args.camera_id is None:
        args.camera_id = int(app_config.get("camera_id", 0))
    stgcn_model_name = str(app_config.get("stgcn_model_name", "stgcn_action"))
    stgcn_wait_frames = int(app_config.get("stgcn_wait_frames", args.stgcn_wait_frames))
    smooth_window = (
        int(args.smooth_window) if args.smooth_window is not None
        else int(app_config.get("stgcn_smooth_window", 5))
    )

    # 如果没有提供视频路径且不使用摄像头，优先从应用配置的 test_video 读取
    if args.video is None and not args.use_camera:
        test_video_path = app_config.get("test_video")
        if test_video_path:
            p = Path(str(test_video_path))
            args.video = str(p if p.is_absolute() else (project_root / p).resolve())
            print(f"从配置读取视频路径: {args.video}")
        else:
            print("错误: 未提供 --video 或 --use-camera，且应用配置中无 test_video")
            return

    # 检查视频文件是否存在（如果不使用摄像头）
    if not args.use_camera:
        if not Path(args.video).exists():
            print(f"错误: 视频文件不存在: {args.video}")
            return

    print("=" * 60)
    print("Fall Detection Example (跌倒检测示例)")
    print("=" * 60)
    print(f"应用配置: {config_path}")
    pose_config_path = app_config.get("pose_config_path", "")
    if not pose_config_path:
        print("✗ pose_config_path 未设置，请在 fall_detection.yaml 中指定")
        return
    if not Path(pose_config_path).is_absolute():
        pose_config_path = str((project_root / pose_config_path).resolve())
    print(f"姿态模型: {pose_config_path}")
    if args.use_camera:
        print(f"使用摄像头: {args.camera_id}")
    else:
        print(f"输入视频: {args.video}")
    print(f"关键点阈值: {args.kp_threshold}")
    print("动作/跌倒判断: STGCN 动作识别 (30 帧序列)")
    print("按 'q' 键退出")
    print("=" * 60)

    # 姿态模型：从 fall_detection.yaml 中的 pose_config_path 指向的 yaml 创建
    pose_model_name = Path(pose_config_path).stem
    pose_config_dir_abs = Path(pose_config_path).parent
    if not pose_config_dir_abs.is_dir():
        print(f"✗ 姿态配置目录不存在: {pose_config_dir_abs}")
        return
    override_params = {}
    pose_path_src = args.pose_model if args.pose_model else app_config.get("pose_model_path", "")
    if pose_path_src:
        p = Path(pose_path_src).expanduser()
        override_params["model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
    if args.conf_threshold is not None:
        override_params['conf_threshold'] = args.conf_threshold
    if args.iou_threshold is not None:
        override_params['iou_threshold'] = args.iou_threshold
    if args.num_threads is not None:
        override_params['num_threads'] = args.num_threads

    print(f"\n从 {pose_config_path} 加载姿态模型...")
    try:
        detector = create_model(
            model_name=pose_model_name,
            config_dir=pose_config_dir_abs,
            **override_params
        )
        print("✓ 姿态模型加载成功!")
        print(f"  模型类: {type(detector).__name__}")
        if hasattr(detector, 'input_shape'):
            print(f"  输入尺寸: {detector.input_shape}")
    except Exception as e:
        print(f"✗ 姿态模型加载失败: {e}")
        print(f"\n提示: 请确保 {pose_config_path} 存在并包含 "
              "class、model_path、default_params")
        import traceback
        traceback.print_exc()
        return

    # STGCN 动作识别（必选），30 帧序列，多类别；本应用 yaml 的 stgcn_model_path 为默认，--stgcn-model 可覆盖
    stgcn_override = {}
    raw_stgcn = app_config.get("stgcn_model_path", "")
    if raw_stgcn:
        p = Path(raw_stgcn).expanduser()
        stgcn_override["model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
    if args.stgcn_model:
        p = Path(args.stgcn_model).expanduser()
        stgcn_override["model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
    try:
        action_model = create_model(
            model_name=stgcn_model_name,
            config_dir=app_config_dir,
            **stgcn_override
        )
        keypoint_buffer = deque(maxlen=action_model.sequence_length)
        stgcn_infer_step = 0
        pred_class_hist = []   # 最近 smooth_window 次预测类别，用于平滑判跌倒
        last_probs = np.zeros(len(getattr(action_model, 'class_names', [])), dtype=np.float32)
        fall_down_class_index = action_model.fall_down_class_index
        last_stgcn_result = {'is_fall': False, 'action_name': '—', 'fall_prob': 0.0}
        class_names = getattr(action_model, 'class_names', None) or []
        class_str = ", ".join(class_names) if class_names else "(未知)"
        print(f"✓ STGCN 动作识别已加载（config: {stgcn_model_name}.yaml，"
              f"{action_model.sequence_length} 帧），每 {stgcn_wait_frames} 帧推理一次，"
              f"平滑窗口 {smooth_window}")
        print(f"  动作类别: {class_str}")
    except Exception as e:
        print(f"✗ STGCN 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 打开输入源（视频或摄像头）
    if args.use_camera:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"✗ 无法打开摄像头 {args.camera_id}")
            return
        source_desc = f"camera:{args.camera_id}"
    else:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"✗ 无法打开视频文件: {args.video}")
            return
        source_desc = f"video:{args.video}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay_ms = 1
    if fps and fps > 1e-3:
        delay_ms = max(int(1000.0 / fps), 1)
    print(f"\n输入源已打开: {source_desc} (fps={fps if fps else 'unknown'})")

    frame_idx = 0
    last_warn_frame = -9999
    warn_interval_frames = 30  # 控制台告警节流

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # 视频文件播放结束
                if not args.use_camera:
                    print("\n视频播放结束")
                    break
                # 摄像头短暂读不到，继续
                continue

            frame_idx += 1

            # 运行检测
            result = frame
            fall_count = 0
            try:
                detections = detector.infer(frame)
                if detections is not None and len(detections) > 0:
                    formatted_detections = list(detections)
                    h, w = frame.shape[:2]

                    # STGCN：对第一人维护 30 帧 buffer，满则推理并更新 last_stgcn_result
                    det0 = max(formatted_detections, key=lambda d: float(d.get('score', 0.0)))
                    kps = det0.get('keypoints', [])
                    box = det0.get('box', det0.get('bbox', []))
                    if len(kps) >= 17 and len(box) >= 4:
                        kps_norm = normalize_keypoints_for_stgcn(kps, w, h, bbox=box)
                        keypoint_buffer.append(kps_norm)
                        if len(keypoint_buffer) == action_model.sequence_length:
                            stgcn_infer_step += 1
                            if stgcn_infer_step % max(1, stgcn_wait_frames) == 0:
                                try:
                                    pts = keypoint_buffer_to_tsstg_pts(list(keypoint_buffer), (w, h))
                                    probs = action_model.predict(pts, (w, h))
                                    probs = np.asarray(probs)
                                    if probs.ndim >= 2:
                                        probs = probs[0]
                                    pred_class = int(np.argmax(probs))
                                    last_probs = np.asarray(probs, dtype=np.float32)
                                    pred_class_hist.append(pred_class)
                                    if len(pred_class_hist) > max(smooth_window, 1):
                                        pred_class_hist.pop(0)
                                    # 平滑：最近 smooth_window 次中超过一半为 Fall Down 才判跌倒
                                    is_fall_smooth = (
                                        (np.array(pred_class_hist) == fall_down_class_index).mean() > 0.5
                                        if pred_class_hist
                                        else (pred_class == fall_down_class_index)
                                    )
                                    fall_prob = (
                                        float(last_probs[fall_down_class_index])
                                        if last_probs.size > fall_down_class_index else 0.0
                                    )
                                    last_stgcn_result = {
                                        'is_fall': is_fall_smooth,
                                        'action_name': action_model.get_class_name(pred_class),
                                        'fall_prob': fall_prob,
                                    }
                                except Exception:
                                    pass
                    else:
                        # 与 test_stgcn 一致：关键点不足时不 popleft，只不 append，保证 30 帧为连续有效序列
                        pass

                    # 只画被跟踪的一个人（score 最高），与 test_stgcn 一致
                    single_detection = [det0]
                    single_action = (
                        last_stgcn_result if last_stgcn_result is not None
                        else {'action_name': '—', 'is_fall': False, 'fall_prob': 0.0}
                    )
                    result = draw_fall_detection(
                        frame, single_detection, args.kp_threshold, action_results=[single_action])

                    if single_action.get('is_fall', False):
                        fall_count = 1
                    else:
                        fall_count = 0

                    # 左上角：当前动作与跌倒计数
                    primary_action = last_stgcn_result.get('action_name', '—') if last_stgcn_result else '—'
                    primary_fall_prob = last_stgcn_result.get('fall_prob', 0.0) if last_stgcn_result else 0.0
                    info_line = f"Action: {primary_action}  P(fall): {primary_fall_prob:.2f}"
                    cv2.putText(result, info_line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                               0.65, (0, 0, 0), 2)
                    cv2.putText(result, info_line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                               0.65, (255, 255, 255), 1)
                    if fall_count > 0:
                        cv2.putText(
                            result, f"FALL COUNT: {fall_count}",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2
                        )
                        if frame_idx - last_warn_frame >= warn_interval_frames:
                            print(f"[警告] 检测到跌倒! 动作={primary_action}, "
                                  f"P(fall)={primary_fall_prob:.2f}, frame={frame_idx}")
                            last_warn_frame = frame_idx
                else:
                    # 无检测时 STGCN buffer 滑动丢弃一帧，画面仍显示上一帧动作信息
                    if keypoint_buffer:
                        keypoint_buffer.popleft()
                    primary_action = last_stgcn_result.get('action_name', '—') if last_stgcn_result else '—'
                    primary_fall_prob = last_stgcn_result.get('fall_prob', 0.0) if last_stgcn_result else 0.0
                    info_line = f"Action: {primary_action}  P(fall): {primary_fall_prob:.2f}"
                    cv2.putText(result, info_line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                               0.65, (0, 0, 0), 2)
                    cv2.putText(result, info_line, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                               0.65, (255, 255, 255), 1)
            except Exception as e:
                print(f"✗ 检测失败: {e}")
                result = frame

            cv2.imshow('Fall Detection', result)
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord('q'):
                print("\n用户停止")
                break
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ 完成!")

if __name__ == '__main__':
    main()
