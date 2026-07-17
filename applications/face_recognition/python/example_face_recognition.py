# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
Face recognition application (buffalo_l pipeline).

Usage:
  python example_face_recognition.py analyze /path/to/image.jpg
  python example_face_recognition.py register alice /path/to/face.jpg
  python example_face_recognition.py recognize /path/to/query.jpg
  python example_face_recognition.py camera 0 --enable-recognition
"""

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus

_DEFAULT_APP_CONFIG = "applications/face_recognition/config/face_recognition.yaml"

_ARCFACE_SRC5 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src").is_dir() and (candidate / "applications").is_dir():
            return candidate.resolve()
    return start.resolve()


def _resolve(path_str: str, root: Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _create_service(app_config: dict, key: str, config_dir: Path, root: Path):
    rel = app_config[key]
    cfg_path = _resolve(str(config_dir / rel), root)
    cfg = _load_yaml(cfg_path)
    model_override = ""
    if cfg.get("model_path"):
        model_override = str(_resolve(str(cfg["model_path"]), root))
    return VisionServiceNative.create(str(cfg_path), model_path_override=model_override)


def _estimate_similarity_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    src_var = float(np.sum(src_centered * src_centered))
    if src_var <= 1e-6:
        return None
    dst_var = float(np.sum(dst_centered * dst_centered))
    scale = np.sqrt(dst_var / src_var)
    src_angle = np.arctan2(src[1, 1] - src[0, 1], src[1, 0] - src[0, 0])
    dst_angle = np.arctan2(dst[1, 1] - dst[0, 1], dst[1, 0] - dst[0, 0])
    angle = dst_angle - src_angle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    a = scale * cos_a
    b = scale * sin_a
    c = -scale * sin_a
    d = scale * cos_a
    tx = dst_mean[0] - (a * src_mean[0] + b * src_mean[1])
    ty = dst_mean[1] - (c * src_mean[0] + d * src_mean[1])
    return np.array([[a, b, tx], [c, d, ty]], dtype=np.float32)


def _align_face_5pt(image: np.ndarray, keypoints, output_size: int = 112) -> np.ndarray:
    if len(keypoints) < 5:
        return np.array([])
    scale = output_size / 112.0
    src = np.array([[kp.x, kp.y] for kp in keypoints[:5]], dtype=np.float32)
    dst = (_ARCFACE_SRC5 * scale).astype(np.float32)
    affine = _estimate_similarity_affine(src, dst)
    if affine is None:
        return np.array([])
    return cv2.warpAffine(image, affine, (output_size, output_size), flags=cv2.INTER_LINEAR)


def _pose_keypoints(pose_result):
    if hasattr(pose_result, "keypoints"):
        return pose_result.keypoints
    return pose_result.get("keypoints", [])


def _pose_bbox(pose_result):
    if hasattr(pose_result, "bbox"):
        b = pose_result.bbox
        return b.x1, b.y1, b.x2, b.y2
    if all(hasattr(pose_result, k) for k in ("x1", "y1", "x2", "y2")):
        return pose_result.x1, pose_result.y1, pose_result.x2, pose_result.y2
    b = pose_result.get("bbox", [0, 0, 0, 0])
    return b[0], b[1], b[2], b[3]


def _pose_score(pose_result):
    return pose_result.score if hasattr(pose_result, "score") else pose_result.get("score", 0.0)


def _save_embedding(path: Path, embedding: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(embedding)))
        f.write(struct.pack(f"<{len(embedding)}f", *embedding))


def _load_embedding(path: Path):
    with open(path, "rb") as f:
        dim = struct.unpack("<Q", f.read(8))[0]
        return list(struct.unpack(f"<{dim}f", f.read(dim * 4)))


def _infer_poses(service, image):
    status, results = service.infer_image(image)
    if status != VisionServiceStatus.OK:
        return status, []
    poses = [r for r in results if hasattr(r, "keypoints") or isinstance(r, dict)]
    return (VisionServiceStatus.OK if poses else VisionServiceStatus.INFER_FAILED), poses


def _infer_embedding(service, image):
    status, embedding = service.infer_embedding(image)
    return status, embedding


def _infer_genderage(service, image):
    status, results = service.infer_image(image)
    if status != VisionServiceStatus.OK or not results:
        return status, None
    return status, results[0]


def _gender_age_text(ga):
    if ga is None:
        return ""
    scores = getattr(ga, "class_scores", None)
    if scores is None and isinstance(ga, dict):
        scores = ga.get("class_scores", [])
    scores = scores or []
    if len(scores) < 3:
        return ""
    label = getattr(ga, "label", None)
    if label is None and isinstance(ga, dict):
        label = ga.get("label", 0)
    label = 0 if label is None else int(label)
    gender = "M" if label == 1 else "F"
    return f"{gender} age:{int(scores[2])}"


def _draw_face(canvas, pose, label, box_color, extra_points=None):
    x1, y1, x2, y2 = map(int, _pose_bbox(pose))
    cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2)
    kpts = _pose_keypoints(pose)[:5]
    for kp in kpts:
        px = int(kp.x if hasattr(kp, "x") else kp["x"])
        py = int(kp.y if hasattr(kp, "y") else kp["y"])
        cv2.circle(canvas, (px, py), 3, (0, 0, 255), -1)
    if extra_points:
        for px, py in extra_points:
            cv2.circle(canvas, (int(px), int(py)), 3, (255, 255, 0), -1)
    cv2.putText(canvas, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)
    if extra_points:
        cv2.putText(canvas, f"lm106={len(extra_points)}", (x1, min(canvas.shape[0] - 5, y2 + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)


def _save_result_image(image, poses, labels, colors, extra_points_list, output_path):
    canvas = image.copy()
    for pose, label, color, extra in zip(poses, labels, colors, extra_points_list):
        _draw_face(canvas, pose, label, color, extra)
    cv2.imwrite(str(output_path), canvas)
    print(f"Saved result image: {output_path}")


def _build_landmark106_input(image, pose, input_size=192):
    x1, y1, x2, y2 = _pose_bbox(pose)
    w = x2 - x1
    h = y2 - y1
    if w <= 1 or h <= 1:
        return None, None
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    side = max(w, h) * 1.5
    if side <= 1e-6:
        return None, None
    s = float(input_size) / float(side)
    affine = np.array(
        [[s, 0.0, input_size * 0.5 - cx * s],
         [0.0, s, input_size * 0.5 - cy * s]],
        dtype=np.float32,
    )
    crop = cv2.warpAffine(
        image, affine, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    inv_affine = cv2.invertAffineTransform(affine)
    return crop, inv_affine


def _process_image_pipeline(
    image,
    scrfd,
    arcface,
    genderage,
    landmark106,
    *,
    enable_landmark106=False,
    enable_recognition=False,
    run_embedding=True,
    face_db_dir=None,
    threshold=0.3,
):
    status, poses = _infer_poses(scrfd, image)
    if status != VisionServiceStatus.OK:
        print(f"Detection failed: {scrfd.last_error()}")
        return status, [], [], [], []
    poses = sorted(poses, key=_pose_score, reverse=True)
    labels, colors, extras = [], [], []
    for fi, pose in enumerate(poses):
        aligned = _align_face_5pt(image, _pose_keypoints(pose))
        if aligned.size == 0:
            labels.append(f"#{fi} det:{_pose_score(pose):.2f} align:failed")
            colors.append((0, 165, 255))
            extras.append([])
            continue

        query = []
        has_embedding = False
        if run_embedding:
            st, query = _infer_embedding(arcface, aligned)
            has_embedding = (st == VisionServiceStatus.OK and len(query) > 0)
        st_ga, ga = _infer_genderage(genderage, aligned)
        ga_text = _gender_age_text(ga if st_ga == VisionServiceStatus.OK else None)
        if ga_text:
            scores = getattr(ga, "class_scores", None) or ga.get("class_scores", [])
            label = getattr(ga, "label", None)
            if label is None and isinstance(ga, dict):
                label = ga.get("label", 0)
            gender = "male" if int(label or 0) == 1 else "female"
            print(f"Face {fi} gender/age: {gender}, age={int(scores[2])}")

        extra_pts = []
        if enable_landmark106 and landmark106 is not None:
            lm_crop, inv_affine = _build_landmark106_input(image, pose, input_size=192)
            if lm_crop is not None:
                st_lm, lm_poses = _infer_poses(landmark106, lm_crop)
                if st_lm == VisionServiceStatus.OK and lm_poses:
                    lm_kps = _pose_keypoints(lm_poses[0])
                    print(f"Face {fi} landmark106 points: {len(lm_kps)}")
                    for kp in lm_kps:
                        x = kp.x if hasattr(kp, "x") else kp["x"]
                        y = kp.y if hasattr(kp, "y") else kp["y"]
                        mx = float(inv_affine[0, 0] * x + inv_affine[0, 1] * y + inv_affine[0, 2])
                        my = float(inv_affine[1, 0] * x + inv_affine[1, 1] * y + inv_affine[1, 2])
                        extra_pts.append((mx, my))

        label = f"#{fi} det:{_pose_score(pose):.2f}"
        if ga_text:
            label += f" {ga_text}"
        color = (0, 255, 0)

        if run_embedding and not has_embedding:
            label += " emb:failed"
            color = (0, 165, 255)
            labels.append(label)
            colors.append(color)
            extras.append(extra_pts)
            continue

        if not enable_recognition or not run_embedding:
            labels.append(label)
            colors.append(color)
            extras.append(extra_pts)
            continue

        best_name = ""
        best_score = -1.0
        print(f"\nSimilarity scores (face {fi}):")
        if face_db_dir is None or not face_db_dir.exists():
            print(f"  (empty face db: {face_db_dir})")
        else:
            for entry in sorted(face_db_dir.glob("*.bin")):
                stored = _load_embedding(entry)
                if len(stored) != len(query):
                    continue
                score = VisionServiceNative.embedding_similarity(query, stored)
                print(f"  {entry.stem}: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_name = entry.stem
        if best_score >= threshold:
            label += f" {best_name}:{best_score:.2f}"
            print(f"Best match (face {fi}): {best_name} (score: {best_score:.4f})")
        else:
            if best_score >= 0:
                label += f" Unknown:{best_score:.2f}"
            color = (0, 165, 255)
            print(f"Best match (face {fi}): Unknown person (best: {best_score:.4f})")

        labels.append(label)
        colors.append(color)
        extras.append(extra_pts)

    return VisionServiceStatus.OK, poses, labels, colors, extras


def parse_args():
    parser = argparse.ArgumentParser(description="Face recognition (buffalo_l)")
    parser.add_argument("mode", nargs="?", default="analyze")
    parser.add_argument("items", nargs="*")
    parser.add_argument("--config", default=None, help="app config yaml")
    parser.add_argument("--output", default=None, help="annotated result image path")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--no-save-image", action="store_true")
    parser.add_argument("--enable-recognition", action="store_true")
    parser.add_argument("--disable-recognition", action="store_true")
    args = parser.parse_args()
    valid = {"analyze", "register", "recognize", "camera"}
    if args.mode not in valid:
        args.items = [args.mode] + args.items
        args.mode = "analyze"
    return args


def main():
    args = parse_args()
    root = _find_project_root(Path(__file__).resolve())
    app_config_path = _resolve(args.config or _DEFAULT_APP_CONFIG, root)
    app_config = _load_yaml(app_config_path)
    config_dir = app_config_path.parent

    threshold = float(app_config.get("recognize_threshold", 0.3))
    face_db_dir = _resolve(app_config.get("face_db_dir", "~/.cache/face_db"), root)
    enable_landmark106 = bool(app_config.get("enable_landmark106", False))
    output_image = args.output or app_config.get("output_image", "output_face_recognition.jpg")
    output_image_path = Path(output_image).expanduser()

    scrfd = _create_service(app_config, "scrfd_model", config_dir, root)
    arcface = _create_service(app_config, "arcface_model", config_dir, root)
    genderage = _create_service(app_config, "genderage_model", config_dir, root)
    landmark106 = None
    if enable_landmark106:
        landmark106 = _create_service(app_config, "landmark106_model", config_dir, root)

    if args.save_image and args.no_save_image:
        print("Error: --save-image and --no-save-image cannot both be set")
        return 1
    save_default = (args.mode == "analyze")
    save_image = args.save_image if args.save_image or args.no_save_image else save_default
    if args.no_save_image:
        save_image = False
    enable_recognition = args.enable_recognition
    if args.mode == "recognize":
        enable_recognition = not args.disable_recognition
    elif args.disable_recognition:
        enable_recognition = False

    if args.mode == "register":
        if len(args.items) < 2:
            print("Usage: register <name> <image>")
            return 1
        name = args.items[0]
        image_path = _resolve(args.items[1], root)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: cannot load image: {image_path}")
            return 1

        status, poses, labels, colors, extras = _process_image_pipeline(
            image, scrfd, arcface, genderage, landmark106,
            enable_landmark106=enable_landmark106,
            enable_recognition=False,
            run_embedding=False,
            face_db_dir=face_db_dir,
            threshold=threshold,
        )
        if status != VisionServiceStatus.OK or not poses:
            return 1
        aligned = _align_face_5pt(image, _pose_keypoints(poses[0]))
        if aligned.size == 0:
            print("Alignment failed")
            return 1

        st, emb = _infer_embedding(arcface, aligned)
        if st != VisionServiceStatus.OK:
            print(f"Embedding failed: {arcface.last_error()}")
            return 1

        st_ga, ga = _infer_genderage(genderage, aligned)
        if st_ga == VisionServiceStatus.OK and ga is not None:
            scores = getattr(ga, "class_scores", None) or ga.get("class_scores", [])
            if len(scores) >= 3:
                label = getattr(ga, "label", None)
                if label is None and isinstance(ga, dict):
                    label = ga.get("label", 0)
                gender = "male" if int(label or 0) == 1 else "female"
                print(f"Gender/Age: {gender}, age={int(scores[2])}")

        out_path = face_db_dir / f"{name}.bin"
        _save_embedding(out_path, emb)
        print(f"Registered: {name} -> {out_path}")
        labels[0] += f" register:{name}"
        if save_image:
            _save_result_image(image, poses, labels, colors, extras, output_image_path)
        return 0

    if args.mode in ("analyze", "recognize"):
        image_arg = args.items[0] if args.items else app_config.get("test_image")
        if not image_arg:
            print("Usage: analyze <image> | recognize <image>")
            return 1
        image_path = _resolve(image_arg, root)
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: cannot load image: {image_path}")
            return 1

        status, poses, labels, colors, extras = _process_image_pipeline(
            image, scrfd, arcface, genderage, landmark106,
            enable_landmark106=enable_landmark106,
            enable_recognition=enable_recognition,
            run_embedding=True,
            face_db_dir=face_db_dir,
            threshold=threshold,
        )
        if status != VisionServiceStatus.OK:
            return 1
        if save_image:
            _save_result_image(image, poses, labels, colors, extras, output_image_path)
        return 0

    if args.mode == "camera":
        cam_idx = int(args.items[0]) if args.items else 0
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            print(f"Error: cannot open camera index: {cam_idx}")
            return 1
        print("Camera started. Press q or ESC to quit.")
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            status, poses, labels, colors, extras = _process_image_pipeline(
                frame, scrfd, arcface, genderage, landmark106,
                enable_landmark106=enable_landmark106,
                enable_recognition=enable_recognition,
                run_embedding=enable_recognition,
                face_db_dir=face_db_dir,
                threshold=threshold,
            )
            if status == VisionServiceStatus.OK:
                for pose, label, color, extra in zip(poses, labels, colors, extras):
                    _draw_face(frame, pose, label, color, extra)
            cv2.imshow("face_recognition_camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        cap.release()
        cv2.destroyAllWindows()
        return 0

    print("Unknown mode")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
