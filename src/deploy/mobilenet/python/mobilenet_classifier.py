# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
MobileNetV1 Image Classifier Implementation

MobileNetV1 (TF-Slim export) outputs 1001 classes where index 0 is a
"background" class. Drop it from the raw logits so the remaining 1000
entries align with the standard ImageNet label file.
"""

import numpy as np

from deploy.resnet import ResNetClassifier


class MobileNetV1Classifier(ResNetClassifier):
    """MobileNetV1 classifier — strips background class (index 0) from raw logits."""

    def infer(self, image: np.ndarray) -> np.ndarray:
        """Return raw logits with background class removed."""
        raw = np.squeeze(super().infer(image))   # [1001]
        return raw[..., 1:] if raw.shape[-1] == 1001 else raw
