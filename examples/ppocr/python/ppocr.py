#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
PP-OCRv6 text detection + recognition example using spacemit_vision.

--config 指定 yaml → VisionServiceNative.create(config) 创建模型 → infer_image 出文本。
每个结果带 .text（识别文字）、.polygon（文本框四点）、.score（识别置信度）。
"""

import argparse
from pathlib import Path

import cv2

from spacemit_vision import VisionServiceNative, VisionServiceStatus


def resolve_path(path_value, project_root):
    p = Path(path_value).expanduser()
    return p if p.is_absolute() else (project_root / p).resolve()


def parse_args():
    parser = argparse.ArgumentParser(description="PP-OCRv6 OCR Example")
    parser.add_argument("--config", type=str, default=None,
                        help="Config yaml path (default: examples/ppocr/config/ppocr.yaml)")
    parser.add_argument("--model-path", type=str, default=None, help="Override det model_path in yaml")
    parser.add_argument("--image", type=str, help="Input image path (default: config test_image)")
    parser.add_argument("--output", type=str, default="ppocr_result.jpg", help="Output image path")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        default_config = Path(__file__).parent.parent / "config" / "ppocr.yaml"
        config_path = Path(args.config) if args.config else default_config
        project_root = Path(__file__).parent.parent.parent.parent
        model_name = config_path.stem

        print(f"创建 {model_name} OCR 模型...")
        svc = VisionServiceNative.create(
            str(config_path),
            model_path_override=args.model_path or "",
        )

        if args.image:
            image_path = resolve_path(args.image, project_root)
        else:
            default_image = svc.get_default_image()
            if not default_image:
                raise ValueError("未提供 --image，且 config 无 test_image")
            image_path = Path(default_image)

        print(f"加载图像: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法加载图像: {image_path}")

        print("运行 OCR...")
        status, results = svc.infer_image(image)
        if status != VisionServiceStatus.OK:
            raise RuntimeError(svc.last_error())

        if results:
            print(f"识别到 {len(results)} 行文字:")
            for r in results:
                text = getattr(r, "text", "")
                poly = getattr(r, "polygon", [])
                pts = [(int(p.x), int(p.y)) for p in poly]
                print(f'  "{text}"  score={r.score:.3f}  quad={pts}')
            if svc.supports_draw():
                st, out = svc.draw(image)
                result_image = out if st == VisionServiceStatus.OK else image
            else:
                result_image = image
            cv2.imwrite(args.output, result_image)
            print(f"结果图像已保存到: {args.output}")
        else:
            print("未识别到文字")
            cv2.imwrite(args.output, image)
            print(f"原始图像已保存到: {args.output}")

    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
