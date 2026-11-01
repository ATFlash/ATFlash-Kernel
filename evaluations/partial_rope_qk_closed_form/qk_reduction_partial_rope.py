# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""Closed-form share of QK^T inner-product terms the window removes in
partial-RoPE models (prefill over the full causal triangle, k = 2).

Terms are counted per dimension pair over every attention layer of the model.
Pairs without rotation are never removed, and layers the window does not touch
count at their own size. No weights are needed: the per-pair widths come from
atflash.window_table and the layouts below from each model's config.

    python evaluations/partial_rope_qk_closed_form/qk_reduction_partial_rope.py [out.csv]
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from atflash import window_table as WT  # noqa: E402

K_RET = 2.0
LENGTHS = (8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576)


def band_terms(n: int, w: float) -> float:
    """Causal (query, key) terms within key distance w over a length-n prefill.

    Same closed form as the reduction curves in the top-level README; w >= n
    keeps the whole triangle n(n+1)/2.
    """
    m = min(math.floor(w), n)
    return m * (m + 1) / 2 + (n - m) * w


def rotary_kept(n: int, rotary_dim: int, rope_base: float) -> float:
    """Kept terms summed over the rotary pairs of one head."""
    widths = WT.wavelengths_sorted(rotary_dim, rope_base, K_RET).double().tolist()
    return sum(band_terms(n, w) for w in widths)


def triangle(n: int) -> float:
    """All causal (query, key) terms of a length-n prefill."""
    return n * (n + 1) / 2


def minimax_m27(n: int) -> float:
    """62 full-attention layers; head_dim 128 = 32 rotary pairs (rotary_dim 64,
    base 5e6) + 32 pairs without rotation. All layers alike, so they cancel."""
    kept = 32 * triangle(n) + rotary_kept(n, 64, 5e6)
    return 1.0 - kept / (64 * triangle(n))


def mimo_v26(n: int) -> float:
    """70 layers: 10 full-attention layers (head_dim 192 = 32 rotary pairs, base
    1e7, + 64 pairs without rotation) carry the window; 60 sliding-window layers
    (window 128) are left as they are and count at their own size."""
    sliding = band_terms(n, 128.0)
    full_kept = 64 * triangle(n) + rotary_kept(n, 64, 1e7)
    total = 10 * 96 * triangle(n) + 60 * 96 * sliding
    kept = 10 * full_kept + 60 * 96 * sliding
    return 1.0 - kept / total


def mimo_v26_full_layers(n: int) -> float:
    """The 10 full-attention layers of MiMo-V2.6 alone (denominator = causal
    terms x 96 pairs), the convention of evaluations/mimo_v26_longbench_v2/."""
    full_kept = 64 * triangle(n) + rotary_kept(n, 64, 1e7)
    return 1.0 - full_kept / (96 * triangle(n))


def smollm3_3b(n: int) -> float:
    """36 layers; head_dim 128 = 64 rotary pairs (base 5e6); every fourth layer
    (9 of 36) applies no RoPE and is never pruned."""
    kept = 27 * rotary_kept(n, 128, 5e6) + 9 * 64 * triangle(n)
    return 1.0 - kept / (36 * 64 * triangle(n))


MODELS = (("minimax_m27", minimax_m27), ("mimo_v26", mimo_v26),
          ("mimo_v26_full_layers", mimo_v26_full_layers), ("smollm3_3b", smollm3_3b))


def main() -> None:
    rows = [[n] + [f(n) for _, f in MODELS] for n in LENGTHS]
    print("%9s" % "N" + "".join("%22s" % name for name, _ in MODELS))
    for r in rows:
        print("%9d" % r[0] + "".join("%21.2f%%" % (100 * v) for v in r[1:]))
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["N"] + [name for name, _ in MODELS])
            w.writerows([[r[0]] + [repr(v) for v in r[1:]] for r in rows])


if __name__ == "__main__":
    main()
