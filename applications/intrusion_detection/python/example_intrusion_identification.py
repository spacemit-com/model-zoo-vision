# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Intrusion Identification Demo (区域闯入识别示例)

功能:
- 使用 ByteTrack(内置 YOLOv8 检测 + ByteTrack 跟踪) 在视频/摄像头中追踪目标
- 使用封闭多边形区域(红色标记/半透明填充)，默认使用图像中间的一块区域；也支持交互式重画
- 只要追踪到的“人”进入该区域，就认为闯入，并给跟踪框添加红色闯入信息
- 可选: --counting-mode 开启后，统计“进入区域”的次数(按 track_id 的 outside->inside 变化计数)

使用示例:
  python example_intrusion_identification.py --video test.mp4
  python example_intrusion_identification.py --use-camera --camera-id 0
  python example_intrusion_identification.py   # 使用 yaml 中的 test_video

ROI 说明:
- 默认使用“画面中心矩形 ROI”
- 也支持用 3 个输入点(顺时针)定义封闭区域：`--roi-points x1,y1 x2,y2 x3,y3`
  这 3 个点视为连续顶点，脚本会自动推算第 4 点，生成封闭四边形(平行四边形)区域
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 将 cv/src 加入路径（脚本在 applications/intrusion_detection/python/，parents[3]=cv）
_cv_src = Path(__file__).resolve().parents[3] / "src"
if str(_cv_src) not in sys.path:
    sys.path.insert(0, str(_cv_src))

import argparse  # noqa: E402
import yaml  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from core import create_model  # noqa: E402
from common import load_labels  # noqa: E402


Point = Tuple[int, int]


def _project_root() -> Path:
    """Return cv 根目录 (model_zoo/cv)。脚本在 applications/intrusion_detection/python/ -> parents[3]=cv"""
    return Path(__file__).resolve().parents[3]

def _load_app_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def _resolve_path(path_str: str, extra_bases: Optional[List[Path]] = None) -> Path:
    """
    Resolve a possibly-relative path.
    Strategy:
    - If absolute: return as-is (expanded)
    - Else: try cwd/path
    - Else: try each base/path (e.g. cv dir)
    - Finally: return expanded cwd/path (even if not exists)
    """
    p = Path(path_str).expanduser()
    if p.is_absolute():
        return p

    # 1) current working directory
    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return cand

    # 2) extra bases (e.g. cv dir)
    for base in extra_bases or []:
        cand2 = (base / p).resolve()
        if cand2.exists():
            return cand2

    return (Path.cwd() / p).resolve()


def _parse_point(text: str) -> Point:
    """Parse 'x,y' into (x, y)."""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Invalid point '{text}', expected format x,y")
    try:
        x = int(float(parts[0].strip()))
        y = int(float(parts[1].strip()))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid point '{text}', expected numbers like 100,200") from e
    return (x, y)


def _clamp_point(pt: Point, w: int, h: int) -> Point:
    x, y = pt
    x = max(0, min(int(x), max(0, w - 1)))
    y = max(0, min(int(y), max(0, h - 1)))
    return (x, y)


def _roi_from_three_points(p1: Point, p2: Point, p3: Point, w: int, h: int) -> List[Point]:
    """
    Build a closed quadrilateral ROI from 3 clockwise consecutive vertices.

    Given p1->p2->p3 are consecutive vertices (clockwise), infer p4:
      p4 = p1 + (p3 - p2)
    This forms a parallelogram (p1, p2, p3, p4).
    """
    x4 = p1[0] + (p3[0] - p2[0])
    y4 = p1[1] + (p3[1] - p2[1])
    pts = [
        _clamp_point(p1, w, h),
        _clamp_point(p2, w, h),
        _clamp_point(p3, w, h),
        _clamp_point((x4, y4), w, h),
    ]
    return _ensure_clockwise(pts)


def _ensure_clockwise(points: List[Point]) -> List[Point]:
    """
    Ensure polygon points are in clockwise order.
    """
    if len(points) < 3:
        return points
    pts = np.array(points, dtype=np.int32)
    # Shoelace formula (signed area)
    x = pts[:, 0].astype(np.float32)
    y = pts[:, 1].astype(np.float32)
    area2 = float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    # In image coordinates (y down), clockwise often yields positive/negative depending on convention;
    # We'll normalize by enforcing "clockwise" based on cv2 contour area sign behavior.
    # Empirically, OpenCV uses same math; we just pick a consistent rule:
    # If area2 > 0, reverse to make it clockwise.
    if area2 > 0:
        return list(reversed(points))
    return points


def _default_roi_polygon(frame_w: int, frame_h: int) -> List[Point]:
    """
    Default ROI polygon in the middle of the image.
    Use a centered rectangle occupying ~45% width and ~35% height.
    """
    if frame_w <= 0 or frame_h <= 0:
        return [(0, 0), (1, 0), (1, 1), (0, 1)]
    cx, cy = frame_w // 2, frame_h // 2
    half_w = int(frame_w * 0.45 / 2)
    half_h = int(frame_h * 0.35 / 2)
    x1, x2 = max(0, cx - half_w), min(frame_w - 1, cx + half_w)
    y1, y2 = max(0, cy - half_h), min(frame_h - 1, cy + half_h)
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return _ensure_clockwise(pts)


def _draw_roi_overlay(image: np.ndarray, roi_pts: List[Point], alpha: float = 0.25) -> np.ndarray:
    """Draw red filled ROI and border."""
    if len(roi_pts) < 3:
        return image
    overlay = image.copy()
    poly = np.array(roi_pts, dtype=np.int32).reshape((-1, 1, 2))

    # Fill (red) + border
    cv2.fillPoly(overlay, [poly], (0, 0, 255))
    cv2.polylines(overlay, [poly], isClosed=True, color=(0, 0, 255), thickness=2)

    # Blend
    out = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw vertices
    for idx, (x, y) in enumerate(roi_pts, start=1):
        cv2.circle(out, (x, y), 5, (0, 255, 255), -1)
        cv2.putText(out, str(idx), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out


def _point_in_roi(pt: Point, roi_pts: List[Point]) -> bool:
    if len(roi_pts) < 3:
        return False
    poly = np.array(roi_pts, dtype=np.int32).reshape((-1, 1, 2))
    # pointPolygonTest >= 0 means inside or on edge
    return cv2.pointPolygonTest(poly, pt, False) >= 0


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Intrusion Identification Demo (ByteTrack + ROI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="应用配置 yaml 路径 (默认: applications/intrusion_detection/config/intrusion_detection.yaml)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="输入视频文件路径 (如果未提供，将从 yaml 配置中的 test_video 读取)",
    )
    parser.add_argument(
        "--use-camera",
        action="store_true",
        help="使用摄像头作为输入",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="摄像头设备 ID (默认: 0)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=None,
        help="置信度阈值 (覆盖 yaml 中的默认值)",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=None,
        help="IoU 阈值，用于 NMS (覆盖 yaml 中的默认值)",
    )
    parser.add_argument(
        "--frame-rate",
        type=int,
        default=None,
        help="视频帧率 (覆盖 yaml 中的默认值)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=None,
        help="推理线程数 (覆盖 yaml 中的默认值)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="追踪模型 ONNX 路径，覆盖 yaml 中的 tracker_model_path",
    )
    parser.add_argument(
        "--roi-points",
        type=_parse_point,
        nargs=3,
        default=None,
        metavar=("x1,y1", "x2,y2", "x3,y3"),
        help=(
            "3 个顺时针输入点，用于定义封闭 ROI 区域(自动推算第 4 点形成四边形)。"
            "例如: --roi-points 100,200 300,200 320,400"
        ),
    )
    parser.add_argument(
        "--counting-mode",
        action="store_true",
        help="是否显示实时闯入人数(当前帧在 ROI 内的人数)；不设置则默认为 False",
    )
    return parser.parse_args()


def _load_labels_from_config(config: Optional[dict], base_dir: Path) -> Optional[List[str]]:
    if not config or "label_file_path" not in config:
        return None
    raw_label_path = config["label_file_path"]
    label_path = Path(raw_label_path)
    if not label_path.is_absolute():
        label_path = (base_dir / raw_label_path).resolve()
    if not label_path.exists():
        print(f"警告: 标签文件不存在: {label_path}")
        return None
    try:
        labels = load_labels(str(label_path))
        if labels:
            print(f"✓ 从 {label_path} 加载了 {len(labels)} 个标签")
        return labels
    except Exception as e:
        print(f"✗ 加载标签文件失败: {e}")
        return None


def _is_person(class_id: int, labels: Optional[List[str]]) -> bool:
    """
    Determine whether a class_id represents a person.
    - If labels are available: match label text containing "person"
    - Else: assume COCO-style class 0 == person
    """
    if labels is None:
        return class_id == 0
    if class_id < 0 or class_id >= len(labels):
        return False
    name = str(labels[class_id]).strip().lower()
    return "person" in name  # 兼容 person / persons / etc.


def main():
    args = _parse_args()
    project_root = _project_root()
    default_config = project_root / "applications" / "intrusion_detection" / "config" / "intrusion_detection.yaml"
    config_path = Path(args.config) if args.config else default_config
    if not config_path.is_absolute():
        config_path = (project_root / config_path).resolve()
    try:
        app_config = _load_app_config(config_path)
    except Exception as e:
        print(f"✗ 加载应用配置失败: {e}")
        return

    # 从应用配置读取追踪模型名与配置目录（与 fall_detection 一致）
    tracker_config_path = app_config.get("tracker_config_path", "")
    if not tracker_config_path:
        print("✗ tracker_config_path 未设置，请在 intrusion_detection.yaml 中指定")
        return
    if not Path(tracker_config_path).is_absolute():
        tracker_config_path = str((project_root / tracker_config_path).resolve())
    tracker_config_dir_abs = Path(tracker_config_path).parent
    tracker_model_name = Path(tracker_config_path).stem
    if not tracker_config_dir_abs.is_dir():
        print(f"✗ 追踪模型配置目录不存在: {tracker_config_dir_abs}")
        return

    # 标签文件从应用配置读取
    labels = _load_labels_from_config(app_config, project_root)

    # 若未提供视频且非摄像头，优先从应用配置的 test_video 读取
    if args.video is None and not args.use_camera:
        test_video_path = app_config.get("test_video")
        if test_video_path:
            if not Path(test_video_path).is_absolute():
                args.video = str((project_root / test_video_path).resolve())
            else:
                args.video = test_video_path
            print(f"从配置读取视频路径: {args.video}")
        else:
            print("错误: 未提供 --video 或 --use-camera，且应用配置中无 test_video")
            return
    elif args.video is not None and not Path(args.video).is_absolute():
        resolved = _resolve_path(args.video, extra_bases=[project_root])
        if str(resolved) != args.video:
            print(f"视频路径解析: {args.video} -> {resolved}")
        args.video = str(resolved)

    override_params: Dict = {}
    tracker_path_src = args.model_path or app_config.get("tracker_model_path", "")
    if tracker_path_src:
        p = Path(tracker_path_src).expanduser()
        override_params["model_path"] = str(p if p.is_absolute() else (project_root / p).resolve())
    if args.conf_threshold is not None:
        override_params["conf_threshold"] = args.conf_threshold
    if args.iou_threshold is not None:
        override_params["iou_threshold"] = args.iou_threshold
    if args.frame_rate is not None:
        override_params["frame_rate"] = args.frame_rate
    if args.num_threads is not None:
        override_params["num_threads"] = args.num_threads

    print(f"\n从 {tracker_config_path} 加载 ByteTrack 模型中...")
    tracker = create_model(model_name=tracker_model_name, config_dir=tracker_config_dir_abs, **override_params)
    print(f"✓ 模型加载成功: {type(tracker).__name__}")

    # Open input source
    if args.use_camera:
        cap = cv2.VideoCapture(args.camera_id)
        if not cap.isOpened():
            print(f"✗ 无法打开摄像头 {args.camera_id}")
            return
    else:
        vpath = _resolve_path(args.video, extra_bases=[project_root])
        args.video = str(vpath)
        if not vpath.exists():
            print(f"✗ 视频文件不存在: {vpath}")
            return
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"✗ 无法打开视频文件: {vpath}")
            print("  可能原因: 文件损坏/未完整拷贝，或编码格式 OpenCV/FFmpeg 不支持")
            return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.video else -1

    # Read first frame (for ROI size + writer init)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        print("✗ 无法读取第一帧")
        cap.release()
        return

    default_roi = _default_roi_polygon(first_frame.shape[1], first_frame.shape[0])
    if args.roi_points is not None:
        p1, p2, p3 = args.roi_points
        roi_pts = _roi_from_three_points(p1, p2, p3, first_frame.shape[1], first_frame.shape[0])
        print(f"使用命令行 ROI(3点推算四边形): {roi_pts}")
    else:
        roi_pts = default_roi
        print(f"使用默认中心 ROI: {roi_pts}")

    print("\n开始处理... 按 'q' 退出")
    frame_idx = 0

    try:
        frame = first_frame
        while True:
            if frame is None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if args.video:
                        print("\n视频结束")
                    else:
                        print("\n✗ 无法从摄像头读取帧")
                    break

            frame_idx += 1

            # Track
            results = tracker.track(frame)

            # Prepare visualization
            vis = frame.copy()
            vis = _draw_roi_overlay(vis, roi_pts, alpha=0.22)

            # Draw tracking boxes + intrusion (only person)
            inside_ids = set()
            for r in results:
                tlbr = r.get("tlbr", None)
                track_id = int(r.get("track_id", -1))
                score = float(r.get("score", 0.0))
                class_id = int(r.get("class_id", 0))
                if tlbr is None or len(tlbr) != 4:
                    continue

                # Only handle person; ignore other classes (no box, no counting)
                if not _is_person(class_id, labels):
                    continue

                x1, y1, x2, y2 = map(int, tlbr)
                x1 = max(0, min(x1, vis.shape[1] - 1))
                x2 = max(0, min(x2, vis.shape[1] - 1))
                y1 = max(0, min(y1, vis.shape[0] - 1))
                y2 = max(0, min(y2, vis.shape[0] - 1))

                # Use bottom-center as "foot point"
                foot = (int((x1 + x2) / 2), int(y2))

                intrusion = _point_in_roi(foot, roi_pts)
                if intrusion and track_id >= 0:
                    inside_ids.add(track_id)

                # Draw bbox
                if intrusion:
                    color = (0, 0, 255)  # red
                    thickness = 3
                else:
                    color = (0, 255, 0)  # green
                    thickness = 2
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(vis, foot, 4, color, -1)

                # Label
                if labels and 0 <= class_id < len(labels):
                    base_name = labels[class_id]
                else:
                    base_name = "obj"
                if intrusion:
                    text = f"{base_name} ID:{track_id} {score:.2f} INTRUSION"
                else:
                    text = f"{base_name} ID:{track_id} {score:.2f}"

                (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx1, ty1 = x1, max(0, y1 - th - bl - 6)
                tx2, ty2 = x1 + tw + 6, y1
                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, -1)
                cv2.putText(vis, text, (x1 + 3, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # HUD
            cv2.putText(vis, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            if args.video and total_frames > 0:
                cv2.putText(
                    vis,
                    f"Progress: {frame_idx}/{total_frames}",
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            if args.counting_mode:
                cv2.putText(
                    vis,
                    f"Intrusions (live): {len(inside_ids)}",
                    (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Intrusion Identification (ByteTrack)", vis)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n用户退出")
                break

            frame = None

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ 完成")


if __name__ == "__main__":
    main()

