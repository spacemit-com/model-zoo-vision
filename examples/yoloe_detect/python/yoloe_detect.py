#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0

"""YOLOE text, visual and fused detection using the shared C++ service."""

import argparse
from pathlib import Path
import sys

import cv2
import yaml

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def load_visuals(profile):
    with profile.open(encoding="utf-8") as source:
        data = yaml.safe_load(source)
    if not isinstance(data, list) or not data:
        raise ValueError("Visual prompt YAML must be a nonempty list")
    references = []
    for item in data:
        image_path = Path(item["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = profile.parent / image_path
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read reference: {image_path}")
        references.append({
            "class_name": item["class_name"], "image": image,
            "bboxes": item["bboxes"],
        })
    return references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1]/"config/yoloe_detect.yaml",
    )
    parser.add_argument("--mode", choices=["text", "visual", "fused"], default="text")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--prompts", help="Comma-separated text classes")
    parser.add_argument("--visual-prompts", type=Path)
    parser.add_argument("--conf-threshold", type=float)
    parser.add_argument("--output", default="yoloe_detect_result.jpg")
    args = parser.parse_args()
    service = None
    try:
        config = args.config.expanduser().resolve()
        texts = [name.strip() for name in args.prompts.split(",")] if args.prompts else None
        if texts and any(not name for name in texts):
            raise ValueError("Empty class name")
        if args.mode == "visual" and texts:
            raise ValueError("Visual mode uses class_name from reference YAML, not --prompts")
        if args.mode == "fused" and not texts:
            with config.open(encoding="utf-8") as source:
                texts = yaml.safe_load(source)["default_params"]["prompts"]
            if not texts:
                raise ValueError("Fused mode requires text classes")
        references = None
        if args.mode != "text":
            profile = (args.visual_prompts or config.parent.parent/"visual_prompts.yaml").expanduser().resolve()
            references = load_visuals(profile)
        service = VisionServiceNative.create(str(config), model_path_override=args.model_path)
        image_path = str(args.image.expanduser()) if args.image else service.get_default_image()
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read target image: {image_path}")
        status, results = service.infer_image(
            image, conf=args.conf_threshold, prompts=texts,
            visual_prompts=references,
        )
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        names = service.get_class_names()
        print(f"Mode: {args.mode}; detected {len(results)} objects")
        for result in results:
            print(f"{names[result.label]} {result.score:.4f} "
                  f"[{result.x1:.1f}, {result.y1:.1f}, {result.x2:.1f}, {result.y2:.1f}]")
        status, drawn = service.draw(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(service.last_error())
        if not cv2.imwrite(args.output, drawn):
            raise RuntimeError(f"Cannot save: {args.output}")
        print(f"Result image saved to: {args.output}")
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if service is not None:
            service.release()


if __name__ == "__main__":
    sys.exit(main())
