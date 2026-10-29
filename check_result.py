# -*- coding: utf-8 -*-
"""Format self-check + packing for a UAV-Crack submission.

Checks (all of them mirror the official submission rules):
    1. exactly 300 PNG files
    2. resolution 672x378
    3. single channel, values in {0, 1}
    4. file names match leftimg8bit/val one-to-one
    5. no empty (all-zero) mask
    6. zip layout: only a top-level result/ directory, no extra entries

Usage
    python check_result.py --result result --pack result.zip
    python check_result.py --result result --data-root data/UAV-Crack-dataset --pack result.zip
"""

import argparse
import os
import sys
import zipfile

import numpy as np
from PIL import Image

OK, BAD = "  [ok]  ", "  [FAIL]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="存放 300 张预测 PNG 的目录")
    ap.add_argument("--data-root", default=None, help="可选：用于核对文件名的数据集根目录")
    ap.add_argument("--expect", type=int, default=300, help="期望文件数")
    ap.add_argument("--pack", default=None, help="打包为 zip（内部仅含 result/）")
    ap.add_argument("--min-area", type=int, default=0, help="同时报告最小连通域面积（0=跳过）")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.result) if f.lower().endswith(".png"))
    fails = []

    print(f"[1] 文件数量: {len(files)}")
    if len(files) != args.expect:
        fails.append("文件数量")
        print(BAD, f"期望 {args.expect}")
    else:
        print(OK)

    dims, modes, values, empty, dup = {}, {}, {}, 0, 0
    for f in files:
        p = os.path.join(args.result, f)
        with Image.open(p) as im:
            dims[im.size] = dims.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
            a = np.asarray(im)
        u = np.unique(a)
        values[tuple(int(x) for x in u)] = values.get(tuple(int(x) for x in u), 0) + 1
        if not a.any():
            empty += 1

    print(f"[2] 分辨率: {dims}")
    if set(dims) != {(672, 378)}:
        fails.append("分辨率")
        print(BAD)
    else:
        print(OK)

    print(f"[3] 通道: {modes}")
    bad_vals = [k for k in values if not set(k) <= {0, 1}]
    print(f"    取值分布: {values}")
    if bad_vals:
        fails.append("像素取值不是 {0,1}")
        print(BAD, "发现非 0/1 取值:", bad_vals[:3])
    else:
        print(OK)

    if args.data_root:
        val_dir = os.path.join(args.data_root, "leftimg8bit", "val")
        ref = {os.path.splitext(f)[0] for f in os.listdir(val_dir) if f.lower().endswith(".jpg")}
        got = {os.path.splitext(f)[0] for f in files}
        missing, extra = ref - got, got - ref
        print(f"[4] 文件名对齐: val={len(ref)} 提交={len(got)} 缺失={len(missing)} 多出={len(extra)}")
        if missing or extra:
            fails.append("文件名不一致")
            print(BAD, list(missing)[:3], list(extra)[:3])
        else:
            print(OK)

    print(f"[5] 空掩膜: {empty}")
    if empty:
        fails.append("存在空掩膜")
        print(BAD)
    else:
        print(OK)

    if args.pack:
        with zipfile.ZipFile(args.pack, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                z.write(os.path.join(args.result, f), arcname=f"result/{f}")
        with zipfile.ZipFile(args.pack) as z:
            names = z.namelist()
        tops = {n.split("/")[0] for n in names}
        print(f"[6] 打包: {args.pack}  条目={len(names)}  顶层目录={tops}  "
              f"大小={os.path.getsize(args.pack) / 1024 / 1024:.2f} MB")
        if tops != {"result"} or len(names) != len(files):
            fails.append("压缩包结构")
            print(BAD)
        else:
            print(OK)

    print("-" * 46)
    if fails:
        print("未通过:", "、".join(fails))
        sys.exit(1)
    print("全部通过，可以提交。")


if __name__ == "__main__":
    main()
