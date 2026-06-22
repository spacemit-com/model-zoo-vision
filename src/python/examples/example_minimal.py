#!/usr/bin/env python3
# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# 构建并安装 wheel 扩展:
#   cmake -S . -B build && cmake --build build -j        # 仓库根，编出扩展
#   cd src/python && ./build_wheel.sh                     # 打包 wheel
#   pip install --force-reinstall dist/spacemit_vision-*.whl
#
# 运行本示例（仓库根为当前目录，需先安装 wheel）:
#   python3 src/python/examples/example_minimal.py
#
# 按需修改 CONFIG / IMAGE 路径。

from __future__ import annotations

from pathlib import Path

# src/python/examples/example_minimal.py -> 仓库根为 parents[3]
ROOT = Path(__file__).resolve().parents[3]

CONFIG = ROOT / "examples/yolov8/config/yolov8.yaml"
IMAGE = ""  # 空则使用 yaml 里 test_image


def main() -> None:
    import cv2

    from spacemit_vision import VisionServiceNative, VisionServiceStatus

    if not CONFIG.is_file():
        raise SystemExit(f"Missing config: {CONFIG}")

    svc = VisionServiceNative.create(str(CONFIG))
    path = IMAGE or svc.get_default_image()
    if not path:
        raise SystemExit("Set IMAGE=... or add test_image in yaml")

    # 方式一：磁盘路径
    st, boxes = svc.infer_image(path)
    if st != VisionServiceStatus.OK:
        raise RuntimeError(svc.last_error())

    # 方式二：BGR uint8 numpy（与 OpenCV imread 一致）
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"cv2.imread failed: {path}")
    st2, boxes2 = svc.infer_image(bgr)
    assert st2 == VisionServiceStatus.OK
    assert len(boxes) == len(boxes2)

    print(f"detections: {len(boxes)}")
    for i, r in enumerate(boxes[:5]):
        print(f"  #{i} score={r.score:.3f} label={r.label} box=({r.x1:.0f},{r.y1:.0f},{r.x2:.0f},{r.y2:.0f})")


if __name__ == "__main__":
    main()
