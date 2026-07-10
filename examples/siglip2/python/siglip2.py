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


def load_labels_file(path):
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            labels.append(line)
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description="SigLIP2 image-text embedding example")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Scene labels file (default: scene_labels_path from config)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Comma-separated text prompts (overrides --labels)",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    default_config = Path(__file__).parent.parent / "config" / "siglip2.yaml"
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

    if args.text:
        labels = [t.strip() for t in args.text.split(",") if t.strip()]
    else:
        labels_path = args.labels or config.get("scene_labels_path")
        if labels_path:
            labels = load_labels_file(resolve_path(labels_path, project_root))
        else:
            labels = ["a photo of a dog", "a photo of a cat", "a photo of a car"]

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

    scored = []
    print(f"Image: {image_path}")
    for label in labels:
        st, text_emb = svc.encode_text(label)
        if st != VisionServiceStatus.OK:
            raise RuntimeError(svc.last_error())
        score = VisionServiceNative.embedding_similarity(image_emb, text_emb)
        scored.append((label, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {min(args.topk, len(scored))} matches:")
    for label, score in scored[: args.topk]:
        print(f"  {label} : {score:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
