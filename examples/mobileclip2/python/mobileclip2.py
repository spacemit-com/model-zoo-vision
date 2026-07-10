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
    parser = argparse.ArgumentParser(description="MobileCLIP2 image-text embedding example")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument(
        "--text",
        type=str,
        default="a photo of a dog,a photo of a cat,a photo of a car",
    )
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    default_config = Path(__file__).parent.parent / "config" / "mobileclip2.yaml"
    config_path = Path(args.config) if args.config else default_config
    project_root = Path(__file__).parent.parent.parent.parent

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    image_path = args.image or config.get("test_image")
    if not image_path:
        raise ValueError("Provide --image or set test_image in config")
    image_path = resolve_path(image_path, project_root)

    model_path_override = ""
    if args.model_path:
        p = Path(args.model_path).expanduser()
        model_path_override = str(p if p.is_absolute() else (project_root / p).resolve())

    svc = VisionServiceNative.create(str(config_path), model_path_override=model_path_override)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    st, image_emb = svc.infer_embedding(image)
    if st != VisionServiceStatus.OK:
        raise RuntimeError(svc.last_error())

    labels = [t.strip() for t in args.text.split(",") if t.strip()]
    best_label = ""
    best_score = -1.0
    print(f"Image: {image_path}")
    for label in labels:
        st, text_emb = svc.encode_text(label)
        if st != VisionServiceStatus.OK:
            raise RuntimeError(svc.last_error())
        score = VisionServiceNative.embedding_similarity(image_emb, text_emb)
        print(f"  {label} : {score:.4f}")
        if score > best_score:
            best_score = score
            best_label = label
    print(f"Best match: {best_label} ({best_score:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
