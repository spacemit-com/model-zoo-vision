#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import cv2
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="AdaFace face embedding example")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--image1", type=str, default=None)
    parser.add_argument("--image2", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    default_config = Path(__file__).parent.parent / "config" / "adaface.yaml"
    config_path = Path(args.config) if args.config else default_config
    project_root = Path(__file__).parent.parent.parent.parent

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    image1_path = args.image1 or config.get("test_image1")
    image2_path = args.image2 or config.get("test_image2")
    if not image1_path or not image2_path:
        raise ValueError("Provide --image1/--image2 or set test_image1/test_image2 in config")
    image1_path = resolve_path(image1_path, project_root)
    image2_path = resolve_path(image2_path, project_root)

    model_path_override = ""
    if args.model_path:
        p = Path(args.model_path).expanduser()
        model_path_override = str(p if p.is_absolute() else (project_root / p).resolve())

    svc = VisionServiceNative.create(str(config_path), model_path_override=model_path_override)

    img1 = cv2.imread(str(image1_path))
    img2 = cv2.imread(str(image2_path))
    if img1 is None or img2 is None:
        raise FileNotFoundError("Failed to load input images")

    st1, emb1 = svc.infer_embedding(img1)
    st2, emb2 = svc.infer_embedding(img2)
    if st1 != VisionServiceStatus.OK or st2 != VisionServiceStatus.OK:
        raise RuntimeError(svc.last_error())

    similarity = VisionServiceNative.embedding_similarity(emb1, emb2)
    print(f"Similarity: {similarity:.4f}")
    print("Same person" if similarity >= args.threshold else "Different person")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
