# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Fall Detection Example (跌倒检测示例)

使用 YOLOv8-Pose 姿态估计 + STGCN 动作识别（多类别，含 Fall Down）进行跌倒检测。
仅通过 STGCN 判断动作/跌倒，无角度规则。
"""

from pathlib import Path
from collections import deque

import argparse
import yaml
import cv2
import numpy as np

from spacemit_vision import VisionServiceNative, VisionServiceStatus

# STGCN/TSSTG：30 帧、13 关键点（COCO 子集），用于 --use-stgcn 时
COCO17_TO_TSTSGO13 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4]
STGCN_SEQUENCE_LENGTH = 30

# COCO17 骨架连线（关键点索引对），用于手动绘制 skeleton
COCO17_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),       # arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # legs
    (5, 6), (11, 12), (5, 11), (6, 12),    # torso
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
]

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
        help='应用配置 yaml (默认: applications/fall_detection/config/fall_detection.yaml)'
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


def _draw_skeleton(image, bbox, keypoints, box_color, kp_color, kp_threshold):
    """手动用 cv2 绘制人体框 + 关键点 + 骨架连线。
    keypoints: list of (x, y, visibility)，像素坐标。"""
    x1, y1, x2, y2 = map(int, bbox[:4])
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

    pts = []
    for kp in keypoints:
        kx, ky, kv = float(kp[0]), float(kp[1]), float(kp[2])
        pts.append((int(kx), int(ky), kv))

    # 骨架连线
    for a, b in COCO17_SKELETON:
        if a < len(pts) and b < len(pts):
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va >= kp_threshold and vb >= kp_threshold:
                cv2.line(image, (xa, ya), (xb, yb), kp_color, 2)

    # 关键点圆点
    for (kx, ky, kv) in pts:
        if kv >= kp_threshold:
            cv2.circle(image, (kx, ky), 3, kp_color, -1)
    return image


def draw_fall_detection(image, detections, kp_threshold=0.3, action_results=None):
    """
    在图像上绘制动作/跌倒检测结果（仅 STGCN 多类别结果）。

    Args:
        image: 输入图像
        detections: 检测结果列表，每项为 {'box'/'bbox': [x1,y1,x2,y2],
                    'keypoints': [(x,y,vis), ...]}
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
        result = _draw_skeleton(
            result, bbox, keypoints,
            box_color=box_color,
            kp_color=kp_color,
            kp_threshold=kp_threshold,
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


def _resolve_path(path_str: str, project_root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def main():
    """Main function."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]  # applications/fall_detection/python/ -> cv
    default_app = project_root / "applications" / "fall_detection" / "config" / "fall_detection.yaml"
    app_path = Path(args.config) if args.config else default_app
    if not app_path.is_absolute():
        app_path = (project_root / app_path).resolve()
    try:
        app_config = _load_app_config(app_path)
    except Exception as e:
        print(f"✗ 加载应用配置失败: {e}")
        return
    app_config_dir = app_path.parent

    args.kp_threshold = float(app_config.get("kp_threshold", 0.3))
    if args.camera_id is None:
        args.camera_id = int(app_config.get("camera_id", 0))
    stgcn_wait_frames = int(app_config.get("stgcn_wait_frames", 10))
    smooth_window = int(app_config.get("stgcn_smooth_window", 5))

    if args.video is None and not args.use_camera:
        test_video_path = app_config.get("test_video")
        if test_video_path:
            args.video = str(_resolve_path(str(test_video_path), project_root))
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
    print(f"应用配置: {app_path}")
    pose_config_path = str(_resolve_path(
        str(app_config_dir / app_config["pose_model"]), project_root))
    print(f"姿态模型: {pose_config_path}")
    if args.use_camera:
        print(f"使用摄像头: {args.camera_id}")
    else:
        print(f"输入视频: {args.video}")
    print(f"关键点阈值: {args.kp_threshold}")
    print("动作/跌倒判断: STGCN 动作识别 (30 帧序列)")
    print("按 'q' 键退出")
    print("=" * 60)

    pose_config_dir_abs = Path(pose_config_path).parent
    if not pose_config_dir_abs.is_dir():
        print(f"✗ 姿态配置目录不存在: {pose_config_dir_abs}")
        return

    print(f"\n从 {pose_config_path} 加载姿态模型...")
    pose_svc = VisionServiceNative.create(str(pose_config_path))
    print("✓ 姿态模型加载成功!")

    stgcn_config_path = str(_resolve_path(
        str(app_config_dir / app_config["stgcn_model"]), project_root))
    stgcn_model_name = Path(stgcn_config_path).stem
    try:
        stgcn_svc = VisionServiceNative.create(str(stgcn_config_path))
        keypoint_buffer = deque(maxlen=STGCN_SEQUENCE_LENGTH)
        stgcn_infer_step = 0
        pred_class_hist = []   # 最近 smooth_window 次预测类别，用于平滑判跌倒
        class_names = list(stgcn_svc.get_class_names() or [])
        last_probs = np.zeros(len(class_names), dtype=np.float32)
        # fall_down_index is an application policy -> read from this app's own
        # config (fall_detection.yaml), aligned with the C++ app. Class names
        # remain a model property (stgcn_svc.get_class_names()).
        fall_down_class_index = int(app_config.get("fall_down_index", -1))
        if fall_down_class_index < 0:
            raise ValueError("fall_detection.yaml must define a valid fall_down_index")
        # Upper-bound check (mirror of the C++ app): a too-large index would
        # never fire a fall instead of crashing -- catch the misconfig early.
        if class_names and fall_down_class_index >= len(class_names):
            raise ValueError(
                f"fall_down_index ({fall_down_class_index}) is out of range for "
                f"the STGCN class count ({len(class_names)})")
        last_stgcn_result = {'is_fall': False, 'action_name': '—', 'fall_prob': 0.0}
        class_str = ", ".join(class_names) if class_names else "(未知)"
        print(f"✓ STGCN 动作识别已加载（config: {stgcn_model_name}.yaml，"
              f"{STGCN_SEQUENCE_LENGTH} 帧），每 {stgcn_wait_frames} 帧推理一次，"
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
                status, results = pose_svc.infer_image(frame)
                if status != VisionServiceStatus.OK:
                    print(f"✗ 推理失败: {pose_svc.last_error()}")
                    result = frame
                    cv2.imshow('Fall Detection', result)
                    if (cv2.waitKey(delay_ms) & 0xFF) == ord('q'):
                        break
                    continue
                # 将原生结果转换为本脚本使用的 dict 形式
                formatted_detections = []
                for r in results:
                    kp_list = [(float(kp.x), float(kp.y), float(kp.visibility))
                               for kp in r.keypoints]
                    formatted_detections.append({
                        'box': [float(r.x1), float(r.y1), float(r.x2), float(r.y2)],
                        'keypoints': kp_list,
                        'score': float(r.score),
                    })
                if formatted_detections:
                    h, w = frame.shape[:2]

                    # STGCN：对第一人维护 30 帧 buffer，满则推理并更新 last_stgcn_result
                    det0 = max(formatted_detections, key=lambda d: float(d.get('score', 0.0)))
                    kps = det0.get('keypoints', [])
                    box = det0.get('box', det0.get('bbox', []))
                    if len(kps) >= 17 and len(box) >= 4:
                        kps_norm = normalize_keypoints_for_stgcn(kps, w, h, bbox=box)
                        keypoint_buffer.append(kps_norm)
                        if len(keypoint_buffer) == STGCN_SEQUENCE_LENGTH:
                            stgcn_infer_step += 1
                            if stgcn_infer_step % max(1, stgcn_wait_frames) == 0:
                                try:
                                    pts = keypoint_buffer_to_tsstg_pts(list(keypoint_buffer), (w, h))
                                    pts_1d = np.ascontiguousarray(pts.ravel(), dtype=np.float32)
                                    st_seq, probs = stgcn_svc.infer_sequence(pts_1d, w, h)
                                    if st_seq != VisionServiceStatus.OK:
                                        raise RuntimeError(stgcn_svc.last_error())
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
                                    action_name = (
                                        class_names[pred_class]
                                        if 0 <= pred_class < len(class_names) else str(pred_class)
                                    )
                                    last_stgcn_result = {
                                        'is_fall': is_fall_smooth,
                                        'action_name': action_name,
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
                    if args.use_camera and (frame_idx <= 5 or frame_idx % 30 == 0):
                        print(f"帧 {frame_idx}: 未检测到人")
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
