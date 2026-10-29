"""Shared configuration for the UAV-Crack U-Net segmentation pipeline."""
from __future__ import annotations

import os
import re
from pathlib import Path

# ---------------------------------------------------------------- paths
# Override the dataset location on another machine without editing this file:
#   PowerShell:  $env:UAV_CRACK_DATA = "D:\data\UAV-Crack-dataset"
#   bash:        export UAV_CRACK_DATA=/data/UAV-Crack-dataset
_DEFAULT_DATA_ROOT = r"C:\Users\1\Desktop\新建文件夹 (2)\UAV-Crack-dataset(1)\UAV-Crack-dataset"
DATA_ROOT = Path(os.environ.get("UAV_CRACK_DATA", _DEFAULT_DATA_ROOT))
TRAIN_IMG_DIR = DATA_ROOT / "leftImg8bit" / "train"
TRAIN_GT_DIR = DATA_ROOT / "gtFine" / "train"
VAL_IMG_DIR = DATA_ROOT / "leftImg8bit" / "val"
YOLO_PRED_DIR = Path(os.environ.get(
    "UAV_CRACK_REFERENCE", r"C:\Users\1\Desktop\新建文件夹 (2)\result\result"))

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "runs"

SPLITS = ("UAV-CrackX4", "UAV-CrackX8", "UAV-CrackX16")

# ---------------------------------------------------------------- data
# Native tile size of every split (X4/X8/X16 are all 672x378 RGB + 0/1 mask).
# Training pads to PAD_* so the U-Net's /32 path divides evenly; validation and
# inference use the native size so outputs line up with the raw masks.
IMG_SIZE = (378, 672)          # (H, W) network input for training (native)
PAD_H, PAD_W = 384, 672        # (H, W) padded training size (divisible by 32)
TRAIN_SIZE = (384, 672)        # what trainz.py feeds the network while training
VAL_SIZE = (378, 672)          # what trainz.py / predictz.py use at test time

# Photos are identified by the DJI capture id, e.g.
#   937956_DJI_20231015154707_0002_Z.JPG_1_z.jpg  ->  937956_DJI_20231015154707_0002_Z.JPG
PHOTO_RE = re.compile(r"^(?P<photo>.+?\.JPG)_(?P<tile>\d+)_z\.(?P<ext>jpg|png)$", re.IGNORECASE)


def photo_id(filename: str) -> str:
    m = PHOTO_RE.match(os.path.basename(filename))
    return m.group("photo") if m else os.path.basename(filename)


# ---------------------------------------------------------------- split
VAL_PHOTO_FRACTION = 0.2
SPLIT_SEED = 42

# ---------------------------------------------------------------- model
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"    # falls back to random init if download fails
IN_CHANNELS = 3

# ---------------------------------------------------------------- train
BATCH_SIZE = 8
EPOCHS = 60
LR = 3e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2
NUM_WORKERS = 0                 # Windows / sandbox friendly
SEED = 42
AMP = True
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5
PATIENCE = 15

# ---------------------------------------------------------------- eval
THRESHOLD = 0.5
TTA = True
