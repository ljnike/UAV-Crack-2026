"""Fusion utilities: probability -> logit, weighted average, AE group synthesis."""

import numpy as np

EPS = 1e-6


def to_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float32), EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def logit_average(probs, weights):
    """probs: list of (H, W) probability maps; weights: same length, summing to 1."""
    if len(probs) != len(weights):
        raise ValueError("probs 与 weights 数量不一致")
    acc = np.zeros_like(np.asarray(probs[0], dtype=np.float32))
    for p, w in zip(probs, weights):
        acc += float(w) * to_logit(p)
    return 1.0 / (1.0 + np.exp(-acc))


def ae_group(p_st, p_e, share_st=0.10, share_e=0.65):
    """AE_ms = logit(share_st * ST_A + share_e * E) / (share_st + share_e)."""
    s = share_st + share_e
    mixed = (share_st * np.asarray(p_st, np.float32) + share_e * np.asarray(p_e, np.float32)) / s
    return 1.0 / (1.0 + np.exp(-to_logit(mixed) / s))


def fuse(probs_by_id, cfg):
    """The exact v4 recipe: AE group (0.35) + L4 (0.10) + H (0.25) + H2 (0.30)."""
    ae = cfg["ae_synthesis"]
    p_ae = ae_group(probs_by_id["ST_A"], probs_by_id["E"],
                    ae["shares"]["ST_A"], ae["shares"]["E"])
    parts = [p_ae, probs_by_id["L4"], probs_by_id["H"], probs_by_id["H2"]]
    weights = [ae["group_weight"]] + [cfg["w"][k] for k in ("L4", "H", "H2")]
    return logit_average(parts, weights)
