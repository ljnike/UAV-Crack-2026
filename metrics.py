"""Segmentation metrics accumulated over a full validation pass."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class Confusion:
    """Pixel-level confusion counts for the crack (positive) class."""

    tp: float = 0.0
    fp: float = 0.0
    fn: float = 0.0
    tn: float = 0.0

    def update(self, pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> None:
        p = pred.astype(bool)
        t = target.astype(bool)
        if valid is not None:
            v = valid.astype(bool)
            p = p & v
            t = t & v
        self.tp += float(np.count_nonzero(p & t))
        self.fp += float(np.count_nonzero(p & ~t))
        self.fn += float(np.count_nonzero(~p & t))
        self.tn += float(np.count_nonzero(~p & ~t))

    def merge(self, other: "Confusion") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn

    # ---- derived metrics
    @property
    def iou_crack(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def iou_background(self) -> float:
        d = self.tn + self.fp + self.fn
        return self.tn / d if d else float("nan")

    @property
    def miou(self) -> float:
        return float(np.nanmean([self.iou_crack, self.iou_background]))

    @property
    def dice(self) -> float:
        d = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / d if d else float("nan")

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else float("nan")

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else float("nan")

    @property
    def accuracy(self) -> float:
        d = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / d if d else float("nan")

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}miou": self.miou,
            f"{prefix}iou_crack": self.iou_crack,
            f"{prefix}iou_background": self.iou_background,
            f"{prefix}f1": self.f1,
            f"{prefix}precision": self.precision,
            f"{prefix}recall": self.recall,
            f"{prefix}dice": self.dice,
            f"{prefix}accuracy": self.accuracy,
        }


def per_image_iou(pred: np.ndarray, target: np.ndarray, valid: np.ndarray | None = None) -> Tuple[float, float]:
    c = Confusion()
    c.update(pred, target, valid)
    return c.iou_crack, c.miou


def best_threshold(probs: Sequence[np.ndarray], targets: Sequence[np.ndarray],
                   valids: Sequence[np.ndarray] | None = None,
                   thresholds: np.ndarray | None = None) -> Tuple[float, List[Dict[str, float]]]:
    """Sweep thresholds and return the one with the best crack F1 plus the full sweep."""
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.05)
    rows: List[Dict[str, float]] = []
    for t in thresholds:
        c = Confusion()
        for i, (p, y) in enumerate(zip(probs, targets)):
            v = valids[i] if valids is not None else None
            c.update(p >= t, y, v)
        row = c.as_dict()
        row["threshold"] = float(t)
        rows.append(row)
    best = max(rows, key=lambda r: (r["f1"] if np.isfinite(r["f1"]) else -1))
    return float(best["threshold"]), rows
