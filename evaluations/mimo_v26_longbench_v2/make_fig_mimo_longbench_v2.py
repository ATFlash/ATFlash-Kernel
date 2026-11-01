# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""Draw the MiMo-V2.6 LongBench-v2 figure from results.csv and results_paired.csv.

(a) accuracy of full, k2 and the stock kernel per band; (b) k2 − full in points with the 95%
interval. Values are read from the two CSV files as they are (accuracy to 4 decimals, difference
and interval to 2 decimals); nothing is recomputed. Style and colours follow the other figures
(DejaVu Sans 8 pt, MiMo orange #ED7D31, full #444444, stock #A6A6A6, arrows on the metric axes).

    python make_fig_mimo_longbench_v2.py                      # five bands -> ../../figs/mimo_longbench_v2_512k.{png,pdf}
    python make_fig_mimo_longbench_v2.py --bands 4 --out DIR  # the 16K-128K bands only
"""
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BANDS = [(16384, "16K"), (32768, "32K"), (65536, "64K"), (131072, "128K"), (524288, "512K")]
FULL, STOCK, INK, INK2, MIMO = "#444444", "#A6A6A6", "#222222", "#555555", "#ED7D31"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "cm", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "axes.edgecolor": "#333333", "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 0.7,
    "ytick.major.width": 0.7, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 1.3,
    "lines.markersize": 4.5, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.unicode_minus": True, "axes.titlepad": 3,
})
MS = 4.5
BETTER = {"higher": "→ higher is better", "zero": "closer to 0 is better"}


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(nb, out_dir):
    bands = BANDS[:nb]
    res = {(int(r["band"]), r["condition"]): r for r in read("results.csv")}
    pair = {(int(r["band"]), r["condition"], r["reference"]): r for r in read("results_paired.csv")}
    acc = {c: [float(res[(b, c)]["accuracy"]) for b, _ in bands] for c in ("full", "k2", "stock")}
    n = [int(res[(b, "full")]["items"]) for b, _ in bands]
    d = [float(pair[(b, "k2", "full")]["difference_points"]) for b, _ in bands]
    lo = [float(pair[(b, "k2", "full")]["ci95_low_points"]) for b, _ in bands]
    hi = [float(pair[(b, "k2", "full")]["ci95_high_points"]) for b, _ in bands]

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.35), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.25})
    w = 0.27
    for j, (c, col) in enumerate((("full", FULL), ("k2", MIMO), ("stock", STOCK))):
        ax[0].bar(np.arange(nb) + (j - 1) * w, acc[c], w, color=col, edgecolor="white", linewidth=0.6, zorder=3)
    ax[0].set_xticks(range(nb))
    ax[0].set_xticklabels(["%s\nn = %d" % (lab, k) for (_, lab), k in zip(bands, n)], fontsize=7)
    top = max(0.6, math.ceil(max(max(v) for v in acc.values()) * 10) / 10)
    ax[0].set_ylim(0, top)
    ax[0].set_ylabel("accuracy\n" + BETTER["higher"])
    ax[1].errorbar(np.arange(nb), d, yerr=[np.subtract(d, lo), np.subtract(hi, d)], fmt="o",
                   color=MIMO, ms=MS, lw=1.0, capsize=2.5, zorder=3)
    ax[1].axhline(0, color=INK2, lw=0.7, zorder=2)
    ax[1].set_xticks(range(nb))
    ax[1].set_xticklabels([lab for _, lab in bands])
    ax[1].set_xlim(-0.5, nb - 0.5)
    ax[1].set_ylim(-6, 6)
    ax[1].set_ylabel("k2 − full [points]\n" + BETTER["zero"])
    ax[0].set_title("(a)", loc="left")
    ax[1].set_title("(b)", loc="left")
    h = [Patch(fc=FULL), Patch(fc=MIMO), Patch(fc=STOCK), Line2D([], [], color=MIMO, marker="o", ms=MS, lw=1.0)]
    fig.legend(h, ["full", "k2 (ours)", "stock kernel (unmodified vLLM)", "k2 − full, 95% CI"],
               loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=4, handlelength=2.0, columnspacing=1.4,
               handletextpad=0.5, borderaxespad=0.0)
    stem = "mimo_longbench_v2_512k" if nb == 5 else "mimo_longbench_v2"
    os.makedirs(out_dir, exist_ok=True)
    for ext, meta in (("pdf", {"CreationDate": None}), ("png", {})):
        fig.savefig(os.path.join(out_dir, "%s.%s" % (stem, ext)), bbox_inches="tight", pad_inches=0.03,
                    facecolor="white", metadata=meta)
    plt.close(fig)
    print("wrote", os.path.join(out_dir, stem + ".{png,pdf}"))


if __name__ == "__main__":
    nb, out = 5, os.path.join(HERE, "..", "..", "figs")
    args = sys.argv[1:]
    if "--bands" in args:
        nb = int(args[args.index("--bands") + 1])
    if "--out" in args:
        out = args[args.index("--out") + 1]
    main(nb, out)
