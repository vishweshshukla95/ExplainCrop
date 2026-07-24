"""
ExplainCrop — Weak severity labeling.

None of the public datasets we found include ground-truth severity labels
(confirmed during dataset research — this is a real annotation gap, not an
oversight). Two honest options:

  (a) hand-label a subset yourselves for a validation set, or
  (b) bootstrap weak severity labels from image heuristics and treat them
      as noisy supervision, validated later against (a).

This module implements (b): it estimates the affected-leaf-area ratio using
HSV color thresholding (yellowing/browning vs healthy green) as a proxy for
severity. This is standard in the agri-vision literature (cited in the
"lightweight ML pipeline" paper we found uses HSV segmentation the same
way) — we're not presenting this heuristic itself as a contribution, only
using it to generate training signal, and the paper should be explicit
about this being weak supervision.
"""

import cv2
import numpy as np

from config import SEVERITY_LEVELS


def estimate_affected_area_ratio(image_bgr: np.ndarray) -> float:
    """Returns fraction of leaf pixels showing yellowing/browning/necrosis."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # Healthy green leaf tissue.
    green_lower, green_upper = (35, 40, 40), (85, 255, 255)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Whole-leaf mask (green + yellow/brown) to exclude background.
    leaf_lower, leaf_upper = (10, 30, 20), (90, 255, 255)
    leaf_mask = cv2.inRange(hsv, leaf_lower, leaf_upper)

    leaf_pixels = int(np.sum(leaf_mask > 0))
    if leaf_pixels == 0:
        return 0.0

    healthy_pixels = int(np.sum(green_mask > 0))
    affected_pixels = leaf_pixels - healthy_pixels
    return max(0.0, affected_pixels / leaf_pixels)


def bucket_severity(affected_ratio: float) -> str:
    """Maps a continuous affected-area ratio to a discrete severity bucket.

    Thresholds are a starting point, not fixed truth — tune them against a
    small hand-labeled validation set before trusting them for the paper's
    severity results.
    """
    if affected_ratio < 0.15:
        return SEVERITY_LEVELS[0]   # low
    elif affected_ratio < 0.40:
        return SEVERITY_LEVELS[1]   # moderate
    else:
        return SEVERITY_LEVELS[2]   # high


def weak_label_severity(image_bgr: np.ndarray) -> tuple:
    """Returns (severity_bucket:str, severity_score_0to100:float)."""
    ratio = estimate_affected_area_ratio(image_bgr)
    return bucket_severity(ratio), round(ratio * 100, 1)
