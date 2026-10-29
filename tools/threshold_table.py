import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from select_model import gt_lookup, load_gt, load_run  # noqa


def lg(p):
    p = np.clip(p.astype(np.float32), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def post(masks, ms=64):
    out = np.empty_like(masks)
    for i, m in enumerate(masks):
        lab, n = ndi.label(m)
        if n:
            sz = ndi.sum(m, lab, range(1, n + 1))
            keep = np.zeros(n + 1, bool)
            keep[1:] = sz >= ms
            m = keep[lab]
        out[i] = m
    return out


def main():
    rows = {}
    for kind in ("v1", "v2"):
        n0, pA = load_run("ST_A", kind)
        y = load_gt(n0, gt_lookup())

        def ms(tag, base):
            arrs = [base]
            for s in ("s085", "s115"):
                d = np.load(ROOT / "runs" / tag / f"val_probs_{kind}_{s}.npz")
                nn = [str(x) for x in d["names"]]
                arrs.append(d["probs"].astype(np.float32)[[nn.index(x) for x in n0]])
            return sum(arrs) / 3

        nE, pE = load_run("E_swins_prec", kind)
        nL, pL = load_run("L4_unetd_100", kind)
        z = 0.10 * lg(ms("ST_A", pA)) + 0.65 * lg(ms("E_swins_prec", pE)) + 0.25 * lg(pL)
        p = 1 / (1 + np.exp(-z))
        for th in (0.41, 0.43, 0.45, 0.47, 0.50):
            mask = post(p >= th)
            tp = np.count_nonzero(mask & y)
            fp = np.count_nonzero(mask & ~y)
            fn = np.count_nonzero(~mask & y)
            rows.setdefault(th, {})[kind] = (tp / (tp + fp + fn),
                                             tp / max(tp + fp, 1), tp / max(tp + fn, 1))
    print(f"{'thr':>5s} | {'v1 IoU':>7s} {'v1 P':>6s} {'v1 R':>6s} | {'v2 IoU':>7s} {'v2 P':>6s} {'v2 R':>6s}")
    for th, d in rows.items():
        a, b = d["v1"], d["v2"]
        print(f"{th:5.2f} | {a[0]:7.4f} {a[1]:6.3f} {a[2]:6.3f} | {b[0]:7.4f} {b[1]:6.3f} {b[2]:6.3f}")


if __name__ == "__main__":
    main()
