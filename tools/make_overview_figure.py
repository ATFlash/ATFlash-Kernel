# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
"""Draw figs/overview_top1.png (and .pdf): top-1 agreement with the unpruned
model, every row of four evaluation studies, one dot per row.

Every value is read from the studies' results.csv files; nothing is computed
except the horizontal jitter. The arms are those of the per-study figures:
  model_scale/          the window (k = 2), every row
  window_assignment/    wavelength order, reversed, random (same kept count)
  partial_rope/         R, N, RN on SmolLM3-3B and Phi-4-mini
  minimax_m27_partial_rope/  R, N, RN, and the stock kernel (implementation floor)
Style and colours follow the curated figures of evaluations/ (DejaVu Sans 8 pt,
no top/right spines, light horizontal grid). The window (ours / R) is coloured
by model, with the model colours of the curated figures; the controls (reversed,
random, N, RN) are greys with one marker each, and the stock kernel is light grey
with an open marker. The y-axis runs from 0 to 1.

    python tools/make_overview_figure.py
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "evaluations")
OUT = os.path.join(ROOT, "figs", "overview_top1")

C = {"stock": "#A6A6A6", "ink": "#222222", "ink2": "#555555"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "axes.edgecolor": "#333333",
    "axes.labelcolor": C["ink"], "xtick.color": C["ink"], "ytick.color": C["ink"],
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300,
    "pdf.fonttype": 42, "axes.unicode_minus": False,
})
MS = 4.5
# Model colours of the curated figures (validated for colour-vision deficiency): one hue per family,
# light -> dark = model size. Qwen3 shares the Qwen2.5 purples and is drawn with open markers.
# model -> (colour, filled)
MODEL = {
    "Qwen/Qwen2.5-7B": ("#A57FD8", True), "Qwen/Qwen2.5-14B": ("#9970D2", True),
    "Qwen/Qwen2.5-32B": ("#8459C6", True),
    "Qwen/Qwen3-8B-Base": ("#A57FD8", False), "Qwen/Qwen3-14B-Base": ("#8459C6", False),
    "unsloth/Meta-Llama-3.1-8B": ("#0668E1", True), "unsloth/Meta-Llama-3.1-70B": ("#0050B8", True),
    "Qwen/Qwen2.5-0.5B": ("#E84F97", True), "unsloth/Llama-3.2-3B": ("#0081FB", True),
    "HuggingFaceTB/SmolLM3-3B": ("#5301F9", True), "microsoft/Phi-4-mini-instruct": ("#663300", True),
    "MiniMax-M2.7": ("#C62828", True),
}
# control role -> (colour, marker, filled)
ROLE = {"reversed": ("#5E5E5E", "s", True), "random": ("#9A9A9A", "o", False),
        "N": ("#7F7F7F", "v", True), "RN": ("#5E5E5E", "^", True), "stock": (C["stock"], "D", False)}
TICK = {"RN": "R+N"}


def top1(study: str, key: str, keep) -> list:
    """(model, top1_agreement) of every row of <study>/results.csv whose <key> passes keep."""
    with open(os.path.join(EVAL, study, "results.csv"), encoding="utf-8") as f:
        return [(r.get("model", "MiniMax-M2.7"), float(r["top1_agreement"]))
                for r in csv.DictReader(f) if keep(r[key])]


# (group label, [(tick label, role, rows)])
GROUPS = [
    ("Qwen2.5, Qwen3\nLlama-3.1\n7B–70B", [
        ("ours", "ours", top1("model_scale", "model", lambda v: True))]),
    ("Qwen2.5-0.5B\nLlama-3.2-3B", [
        ("ours", "ours", top1("window_assignment", "policy", lambda v: v == "ours")),
        ("rev.", "reversed", top1("window_assignment", "policy", lambda v: v == "rev")),
        ("rand.", "random", top1("window_assignment", "policy", lambda v: v.startswith("rand")))]),
    ("SmolLM3-3B\nPhi-4-mini", [
        (c, "ours" if c == "R" else c, top1("partial_rope", "condition", lambda v, c=c: v == c))
        for c in ("R", "N", "RN")]),
    ("MiniMax-M2.7", [
        (c, "ours" if c == "R" else c, top1("minimax_m27_partial_rope", "condition", lambda v, c=c: v == c))
        for c in ("R", "N", "RN", "stock")]),
]


def main() -> None:
    rs = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x, ticks, labels, centers, seps = 0.0, [], [], [], []
    for gi, (glabel, arms) in enumerate(GROUPS):
        start = x
        for tick, role, rows in arms:
            xs = x + rs.uniform(-0.18, 0.18, len(rows))      # same jitter draws as before
            for xx, (model, val) in zip(xs, rows):
                if role == "ours":
                    col, filled = MODEL[model]
                    mk = "o"
                else:
                    col, mk, filled = ROLE[role]
                ax.plot(xx, val, mk, color=col, ms=MS, mfc=col if filled else "white", mew=1.0,
                        ls="none", alpha=0.9, zorder=3)
            ticks.append(x)
            labels.append(TICK.get(tick, tick))
            x += 1.0
        centers.append((start + x - 1.0) / 2)
        if gi < len(GROUPS) - 1:
            seps.append(x - 0.25)
            x += 0.5
    for s in seps:
        ax.axvline(s, color="#BBBBBB", lw=0.6, zorder=1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, x - 0.4)
    sec = ax.secondary_xaxis(-0.12)
    sec.set_xticks(centers)
    sec.set_xticklabels([g for g, _ in GROUPS])
    sec.tick_params(length=0)
    sec.spines["bottom"].set_visible(False)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("top-1 agreement with the unpruned model\n→ higher is better")   # rotated: → points up

    def dot(model):
        col, filled = MODEL[model]
        return Line2D([], [], color=col, marker="o", ls="none", ms=MS, mfc=col if filled else "white", mew=1.0)

    def ctl(role):
        col, mk, filled = ROLE[role]
        return Line2D([], [], color=col, marker=mk, ls="none", ms=MS, mfc=col if filled else "white", mew=1.0)

    h_models = [(dot("Qwen/Qwen2.5-7B"), dot("Qwen/Qwen2.5-14B"), dot("Qwen/Qwen2.5-32B")),
                (dot("Qwen/Qwen3-8B-Base"), dot("Qwen/Qwen3-14B-Base")),
                (dot("unsloth/Meta-Llama-3.1-8B"), dot("unsloth/Meta-Llama-3.1-70B")),
                dot("Qwen/Qwen2.5-0.5B"), dot("unsloth/Llama-3.2-3B"),
                dot("HuggingFaceTB/SmolLM3-3B"), dot("microsoft/Phi-4-mini-instruct"), dot("MiniMax-M2.7")]
    l_models = ["Qwen2.5 7B → 14B → 32B", "Qwen3 8B → 14B", "Llama-3.1 8B → 70B", "Qwen2.5-0.5B",
                "Llama-3.2-3B", "SmolLM3-3B", "Phi-4-mini", "MiniMax-M2.7"]
    order = [0, 4, 1, 5, 2, 6, 3, 7]        # legend fills by column: rows read in group order
    h_models, l_models = [h_models[i] for i in order], [l_models[i] for i in order]
    h_ctl = [ctl("reversed"), ctl("random"), ctl("N"), ctl("RN"), ctl("stock")]
    l_ctl = ["reversed", "random", "N", "R+N", "stock kernel (unmodified vLLM)"]
    leg_m = fig.legend(h_models, l_models, loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=4,
                       handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)}, handlelength=2.6,
                       handletextpad=0.4, columnspacing=1.2, borderaxespad=0.0)
    fig.legend(h_ctl, l_ctl, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=5, handletextpad=0.4,
               columnspacing=1.4, borderaxespad=0.0)
    fig.add_artist(leg_m)
    for ext, meta in (("png", {"Software": None}), ("pdf", {"CreationDate": None, "Producer": None})):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", pad_inches=0.03, facecolor="white",
                    metadata=meta)
    plt.close(fig)
    for glabel, arms in GROUPS:
        for tick, role, rows in arms:
            vals = [v for _, v in rows]
            print(f"{glabel.replace(chr(10), ' '):34} {tick:6} n={len(vals):2} "
                  f"top-1 {min(vals):.4f}-{max(vals):.4f}")


if __name__ == "__main__":
    main()
