"""MTP compatibility of the RoPE window on MiMo-V2.6-Pro-RL (teacher forcing).

Reads mtp_compat_raw.json (per-document values, next to this script), averages over the 6 documents,
writes mtp_compat.csv and fig_mtp_compat.png. Colours follow the repository figure palette:
MiMo window = model orange, stock FA4 path = grey, full attention = dark grey.

Usage: python3 make_fig_mtp_compat.py
"""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = json.load(open(os.path.join(HERE, "mtp_compat_raw.json")))
BANDS = [str(n) for n in RAW["band_end_tokens"]]
NPOS = RAW["positions_per_band"]
# (key, label, colour, marker, filled)
# Full attention is shown by its repeated run, which reproduces the reference exactly (agreement 1.0).
CONDS = [
    ("full_attention_rerun", "full attention", "#444444", "o", True),
    ("stock_fa4_path", "stock FA4 path", "#A6A6A6", "D", False),
    ("window_k1", "window $k$=1", "#ED7D31", "o", True),
    ("window_k2", "window $k$=2", "#ED7D31", "s", True),
    ("window_k3", "window $k$=3", "#ED7D31", "^", True),
    ("window_k4", "window $k$=4", "#ED7D31", "v", True),
]
COLS = ["main_top1_agree", "mtp_top1_agree", "mtp_target_agree", "main_kl_mean", "mtp_kl_mean",
        "main_top1_eq_next", "mtp_top1_eq_next2"]


def mean(xs):
    return sum(xs) / len(xs)


rows = []
for N in BANDS:
    base = mean([d["bands"][N]["full_attention"]["mtp_target_agree"] for d in RAW["docs"]])
    for key, *_ in CONDS:
        per = [d["bands"][N][key] for d in RAW["docs"]]
        r = {"band_end_tokens": int(N), "condition": key, "n_docs": len(per), "n_positions": len(per) * NPOS}
        for c in COLS:
            r[c] = mean([p[c] for p in per]) if c in per[0] else ""
        if "table2x2" in per[0]:
            t = {k: sum(p["table2x2"][k] for p in per) for k in ("TT", "TF", "FT", "FF")}
            assert sum(t.values()) == len(per) * NPOS
            # every document has the same number of positions, so the document mean equals the pooled rate
            assert abs((t["TT"] + t["TF"]) / (len(per) * NPOS) - r["main_top1_agree"]) < 1e-12
            assert abs((t["TT"] + t["FT"]) / (len(per) * NPOS) - r["mtp_top1_agree"]) < 1e-12
        r["mtp_target_agree_change_vs_full"] = r["mtp_target_agree"] - base
        rows.append(r)

with open(os.path.join(HERE, "mtp_compat.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["band_end_tokens", "condition", "n_docs", "n_positions"] + COLS
                       + ["mtp_target_agree_change_vs_full"], lineterminator="\n")
    w.writeheader()
    for r in rows:
        out = {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}
        out["condition"] = {"full_attention_rerun": "full_attention"}.get(out["condition"], out["condition"])
        w.writerow(out)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "mathtext.fontset": "cm", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222", "xtick.color": "#222222", "ytick.color": "#222222",
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300, "axes.unicode_minus": True,
})
PANELS = [("main_top1_agree", "main model top-1 agreement"), ("mtp_top1_agree", "MTP top-1 agreement"),
          ("mtp_target_agree", "MTP–target agreement")]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.75))
off = {key: (i - (len(CONDS) - 1) / 2) * 0.1 for i, (key, *_) in enumerate(CONDS)}
for j, (ax, (col, ylab)) in enumerate(zip(axs, PANELS)):
    for key, _, colr, mk, filled in CONDS:
        pts = [(b + off[key], r[col]) for b, N in enumerate(BANDS) for r in rows
               if r["condition"] == key and r["band_end_tokens"] == int(N) and r[col] != ""]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls="none", marker=mk, ms=4.5, mew=1.0, color=colr,
                mfc=colr if filled else "white")
    ax.set_xticks(range(len(BANDS)))
    ax.set_xticklabels([f"{int(N) // 1024}K" for N in BANDS])
    ax.set_xlim(-0.6, len(BANDS) - 0.4)
    ax.set_xlabel("band end position [tokens]")
    ax.set_ylabel(ylab)
    ax.text(-0.22, 1.02, "(%s)" % "abc"[j], transform=ax.transAxes, fontsize=8)
handles = [Line2D([], [], ls="none", marker=mk, ms=4.5, mew=1.0, color=c, mfc=c if fl else "white")
           for _, _, c, mk, fl in CONDS]
# Legends sit in the empty band between full attention (1.0) and the other conditions:
# the two references in panel (a), the four window settings in panel (b).
labels = [lb for _, lb, *_ in CONDS]
axs[0].legend(handles[:2], labels[:2], loc="center", bbox_to_anchor=(0.55, 0.62), handletextpad=0.3,
              labelspacing=0.35)
axs[1].legend(handles[2:], labels[2:], loc="center", bbox_to_anchor=(0.55, 0.62), handletextpad=0.3,
              labelspacing=0.35)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_mtp_compat.png"))
print("wrote mtp_compat.csv and fig_mtp_compat.png")
