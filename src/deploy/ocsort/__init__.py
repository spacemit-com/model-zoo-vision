# Copyright 2026 SpacemiT (Hangzhou) Technology Co. Ltd.
#
# SPDX-License-Identifier: Apache-2.0

"""
OCSort deploy package.

Expose the BaseModel-style tracker for factory usage:
  class: deploy.ocsort.OCSortTracker
"""

from .python.ocsort_tracker import OCSortTracker

__all__ = ["OCSortTracker"]

