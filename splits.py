"""Leak-free, *stratified* train/val split for the UAV-Crack tiles.

Two leakage sources matter here:

1. Tiles that come from the same DJI frame overlap visually, so the split must
   be done per source photo (never per tile).
2. The X4/X8/X16 subsets have different crack density (mean foreground fraction
   roughly 0.9 % / 0.9 % / 1.2 %), so a plain random photo split can produce a
   holdout that is systematically easier or harder than the hidden test set.
   We therefore stratify by subset and match the crack-pixel mass of the
   holdout to the requested fraction, greedily, per subset.

Validation tiles are also *duplicate-free*: any tile whose source photo appears
in more than one subset is dropped from the holdout only if the whole photo is
in the holdout, which the photo-level split already guarantees.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

import config as C


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Optional[Path]
    photo: str
    split: str
    fg: float = 0.0          # crack pixel fraction of this tile

    @property
    def key(self) -> str:
        return self.image.name


def _mask_fraction(path: Path) -> float:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)
    return float((arr > 0).mean())


def build_index(with_fg: bool = True) -> List[Sample]:
    out: List[Sample] = []
    for split in C.SPLITS:
        img_dir = C.TRAIN_IMG_DIR / split
        gt_dir = C.TRAIN_GT_DIR / split
        for img in sorted(img_dir.glob("*.jpg")):
            mask = gt_dir / (img.stem + ".png")
            if not mask.exists():
                raise FileNotFoundError(f"mask missing for {img}")
            fg = _mask_fraction(mask) if with_fg else 0.0
            out.append(Sample(img, mask, C.photo_id(img.name), split, fg))
    return out


def unlabeled_index() -> List[Sample]:
    out: List[Sample] = []
    for img in sorted(C.VAL_IMG_DIR.glob("*.jpg")):
        out.append(Sample(img, None, C.photo_id(img.name), "val", 0.0))
    return out


def _photo_table(samples: Sequence[Sample]) -> Dict[str, Dict]:
    table: Dict[str, Dict] = {}
    for s in samples:
        e = table.setdefault(s.photo, {"n": 0, "fg_sum": 0.0, "groups": set(), "samples": []})
        e["n"] += 1
        e["fg_sum"] += s.fg
        e["groups"].add(s.split)
        e["samples"].append(s)
    return table


def test_overlapping_photos() -> set:
    """Source photos that appear in BOTH the labelled set and the 300 test tiles.

    294/300 test tiles come from a photo that also has labelled tiles, i.e. the
    official split is tile-level, not scene-level.  Any realistic estimate of the
    hidden score has to reproduce that overlap.
    """
    return {s.photo for s in build_index(with_fg=False)} & {s.photo for s in unlabeled_index()}


def load_split_v2(val_fraction: float = C.VAL_PHOTO_FRACTION, seed: int = 42,
                  with_fg: bool = True) -> Tuple[List[Sample], List[Sample], Dict]:
    """Leaderboard-matched holdout: the SAME photos supply tiles to both sides.

    This mirrors the official split, where 294/300 test tiles come from a photo
    that also has labelled training tiles.  Half of each selected photo's tiles
    go to the holdout, the rest stay in training, so every holdout tile has
    same-scene supervision available - exactly like the real test set.

    Optimistic by construction: never use it to choose an architecture, only to
    calibrate the decision threshold and to estimate the v1->leaderboard offset.
    """
    samples = build_index(with_fg=with_fg)
    photos = _photo_table(samples)
    shared = test_overlapping_photos()
    rng = random.Random(seed)

    val_photos: List[str] = []
    by_group: Dict[str, List[str]] = defaultdict(list)
    for p, e in photos.items():
        if p in shared:
            by_group[sorted(e["groups"])[0]].append(p)

    for group in sorted(by_group):
        names = sorted(by_group[group])
        rng.shuffle(names)
        n_target = max(1, int(round(len(names) * val_fraction)))
        chosen, remaining = [], list(names)
        total_fg = sum(photos[p]["fg_sum"] for p in names)
        target_fg = total_fg * val_fraction
        for _ in range(n_target):
            cur = sum(photos[p]["fg_sum"] for p in chosen)
            best_p, best_cost = None, None
            for p in remaining:
                cost = abs((cur + photos[p]["fg_sum"]) - target_fg)
                if best_cost is None or cost < best_cost:
                    best_cost, best_p = cost, p
            chosen.append(best_p)
            remaining.remove(best_p)
        val_photos.extend(chosen)

    # Holdout photos: half of their tiles become the holdout, the other half stay
    # in training, so each holdout tile has a same-scene training tile (like the
    # real test set).  All other photos contribute every tile to training.
    vset = set(val_photos)
    train: List[Sample] = []
    val: List[Sample] = []
    for p, e in photos.items():
        tiles = sorted(e["samples"], key=lambda s: s.image.name)
        if p in vset and len(tiles) > 1:
            rng.shuffle(tiles)
            n_val = max(1, len(tiles) // 2)
            val.extend(tiles[:n_val])
            train.extend(tiles[n_val:])
        elif p in vset:
            val.extend(tiles)
        else:
            train.extend(tiles)

    assert len(val) > 0 and len(train) > 0
    same_photo = sum(1 for s in val if any(t.photo == s.photo for t in train))
    meta = {
        "kind": "v2_photo_overlap", "n_photos": len(photos),
        "n_train": len(train), "n_val": len(val),
        "val_photos": sorted(vset),
        "overlap_photos": len(vset),
        "val_tiles_with_same_photo_in_train": same_photo,
        "val_tiles_with_same_photo_frac": same_photo / max(1, len(val)),
        "val_fg_mean": float(np.mean([s.fg for s in val])) if val else 0.0,
        "train_fg_mean": float(np.mean([s.fg for s in train])) if train else 0.0,
    }
    meta["fingerprint"] = hashlib.md5(
        json.dumps(sorted(s.image.name for s in val), sort_keys=True).encode()).hexdigest()[:12]
    return train, val, meta


def load_split(val_fraction: float = C.VAL_PHOTO_FRACTION, seed: int = C.SPLIT_SEED,
               cache_dir: Optional[Path] = None, with_fg: bool = True,
               kind: str = "v1") -> Tuple[List[Sample], List[Sample], Dict]:
    """Return (train, val, meta).

    kind='v1' (default) - photo-disjoint and subset-stratified: the honest
    generalisation signal, used for choosing models.
    kind='v2'           - photo-overlapping, matching the hidden test set's
    structure: used to calibrate thresholds and the v1->leaderboard offset.
    """
    if kind == "v2":
        return load_split_v2(val_fraction, seed, with_fg)
    samples = build_index(with_fg=with_fg)
    photos = _photo_table(samples)
    by_group: Dict[str, List[str]] = defaultdict(list)
    for p, e in photos.items():
        group = sorted(e["groups"])[0]
        by_group[group].append(p)

    rng = random.Random(seed)
    val_photos: List[str] = []
    for group in sorted(by_group):
        names = sorted(by_group[group])
        rng.shuffle(names)
        n_target = max(1, int(round(len(names) * val_fraction)))
        # 2-opt style greedy: repeatedly add the photo that keeps the running
        # crack-mass fraction closest to the target.
        chosen: List[str] = []
        remaining = list(names)
        total_fg = sum(photos[p]["fg_sum"] for p in names)
        target_fg = total_fg * val_fraction
        for _ in range(n_target):
            best_p, best_cost = None, None
            cur = sum(photos[p]["fg_sum"] for p in chosen)
            for p in remaining:
                cost = abs((cur + photos[p]["fg_sum"]) - target_fg)
                if best_cost is None or cost < best_cost:
                    best_cost, best_p = cost, p
            chosen.append(best_p)
            remaining.remove(best_p)
        val_photos.extend(chosen)

    vset = set(val_photos)
    train = [s for s in samples if s.photo not in vset]
    val = [s for s in samples if s.photo in vset]
    meta = {
        "n_photos": len(photos),
        "val_photos": sorted(vset),
        "train_photos": sorted(set(photos) - vset),
        "n_train": len(train), "n_val": len(val),
        "val_fg": [round(s.fg, 6) for s in val],
        "train_fg": [round(s.fg, 6) for s in train],
        "val_fg_mean": float(np.mean([s.fg for s in val])) if val else 0.0,
        "train_fg_mean": float(np.mean([s.fg for s in train])) if train else 0.0,
        "val_groups": {g: sum(1 for p in val_photos if sorted(photos[p]["groups"])[0] == g)
                       for g in sorted(by_group)},
    }
    meta["fingerprint"] = hashlib.md5(
        json.dumps(sorted(s.image.name for s in val), sort_keys=True).encode()).hexdigest()[:12]
    return train, val, meta
