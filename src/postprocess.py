"""Thresholding + small-component removal.

Measured on the photo-disjoint holdout: morphological closing / opening both
*lowered* Crack IoU, so only the connected-component filter is enabled.
"""

import cv2
import numpy as np


def binarize(prob, threshold=0.495, min_area=64):
    """prob: float array (H, W) in [0, 1] -> uint8 array (H, W) with values {0, 1}."""
    mask = (np.asarray(prob) >= float(threshold)).astype(np.uint8)
    if min_area and min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        kept = np.zeros_like(mask)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= int(min_area):
                kept[labels == i] = 1
        mask = kept
    return mask


def foreground_ratio(mask):
    m = np.asarray(mask)
    return float((m > 0).mean()) if m.size else 0.0
