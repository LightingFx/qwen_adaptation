#!/usr/bin/env python3
"""Reproduce numerical manuscript figures from matched predictions and analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "results/matched_block"
OUT = ROOT / "figures"
WIDTH = 390 / 72

RELEASES = ["qwen1.5", "qwen2", "qwen2.5", "qwen3"]
RLABEL = {"qwen1.5": "Qwen1.5", "qwen2": "Qwen2", "qwen2.5": "Qwen2.5", "qwen3": "Qwen3"}
XLABELS = ["1.5", "2", "2.5", "3"]
TASKS = [("tlue", "Ti-MMLU", "acc"), ("gsm", "GSM8K-bo", "acc"), ("bele", "Belebele", "acc"),
         ("zh2bo", "Chinese to Tibetan", "bleu"), ("bo2zh", "Tibetan to Chinese", "bleu")]
ARW = r"$\rightarrow$"
SHORT = {"tlue": "Ti-MMLU", "gsm": "GSM8K-bo", "bele": "Belebele", "zh2bo": f"zh{ARW}bo", "bo2zh": f"bo{ARW}zh"}
CFGS = ["orig", "va2", "va6", "cpt"]
RUN = {"orig": "{m}-instruct-original", "va2": "{m}-instruct-va-sft-reference",
       "va6": "{m}-instruct-va-sft-6epoch-seed42", "cpt": "{m}-instruct-cpt-sft-reference"}
CLABEL = {"orig": "Released, no adaptation", "va2": "VA-SFT, 2 epochs", "va6": "VA-SFT, 6 epochs", "cpt": "CPT+SFT"}

INK, INK2, INK3, GRID, SURF = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df", "#f6f5f2"
GRAY, VA2, VA6, CPT = "#9a9993", "#5598e7", "#184f95", "#eb6834"
CCOLOR = {"orig": GRAY, "va2": VA2, "va6": VA6, "cpt": CPT}
STYLE = {
    "orig": dict(ls=(0, (1, 1.5)), marker="D", mfc=GRAY, lw=1.0, ms=3.0),
    "va2": dict(ls=(0, (4, 2)), marker="o", mfc="white", lw=1.4, ms=4.2),
    "va6": dict(ls="-", marker="o", mfc=VA6, lw=1.4, ms=4.2),
    "cpt": dict(ls="-", marker="s", mfc=CPT, lw=1.4, ms=4.2),
}
DIVERGING = LinearSegmentedColormap.from_list(
    "gain_loss", ["#a8322f", "#e89a95", "#f0efec", "#9ec5f4", "#184f95"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#f0efec", "#b7d3f6", "#5598e7", "#184f95"])

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False,
    "mathtext.fontset": "dejavusans", "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
})


def score(cfg: str, m: str, task: str) -> float:
    t = json.load(open(BLOCK / RUN[cfg].format(m=m) / "summary.json"))["tasks"]
    if task in ("zh2bo", "bo2zh"):
        return t["mt"][task]["spbleu"]
    return 100 * t[task]["accuracy"]


def zh_rate(cfg: str, m: str) -> float:
    return 100 * json.load(open(BLOCK / RUN[cfg].format(m=m) / "summary.json"))["tasks"]["mt"]["bo2zh"]["chinese_output_rate"]


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold", color=INK)


def save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


# --------------------------------------------------------------------------- overview








# --------------------------------------------------------------------------- trajectories
def fig_trajectories():
    fig = plt.figure(figsize=(WIDTH, 3.55))
    gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.35, left=0.085, right=0.985, bottom=0.1, top=0.84)
    axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
            fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6])]
    order = [TASKS[1], TASKS[2], TASKS[0], TASKS[3], TASKS[4]]
    for k, (ax, (task, title, unit)) in enumerate(zip(axes, order)):
        for c in CFGS:
            st = STYLE[c]
            ys = [score(c, m, task) for m in RELEASES]
            ax.plot(range(4), ys, color=CCOLOR[c], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=st["ms"],
                    mfc=st["mfc"], mec=CCOLOR[c], mew=1.0, zorder=2 if c == "orig" else 3)
        ax.set_title(f"({'abcde'[k]}) {title}", loc="left", color=INK, pad=4)
        ax.yaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(range(4), XLABELS)
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Qwen release", color=INK2)
        if k in (0, 3):
            ax.set_ylabel("Accuracy (%)" if unit == "acc" else "spBLEU")
    handles = [Line2D([], [], color=CCOLOR[c], ls=STYLE[c]["ls"], lw=STYLE[c]["lw"], marker=STYLE[c]["marker"],
                      ms=STYLE[c]["ms"], mfc=STYLE[c]["mfc"], mec=CCOLOR[c], mew=1.0, label=CLABEL[c]) for c in CFGS]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0),
               handlelength=2.4, columnspacing=1.2)
    save(fig, "fig_trajectories")


# --------------------------------------------------------------------------- upgrade-return heatmap
def fig_return_heatmap():
    pairs = [("qwen1.5", "qwen2"), ("qwen2", "qwen2.5"), ("qwen2.5", "qwen3"), ("qwen1.5", "qwen3")]
    plabels = [f"1.5 {ARW} 2", f"2 {ARW} 2.5", f"2.5 {ARW} 3", f"1.5 {ARW} 3"]
    tasks = [t for t, _, _ in TASKS]
    data = {c: np.array([[score(c, b, t) - score(c, a, t) for t in tasks] for a, b in pairs]) for c in CFGS}
    scale = {t: max(abs(data[c][:, j]).max() for c in CFGS) for j, t in enumerate(tasks)}
    fig, axs = plt.subplots(2, 2, figsize=(WIDTH, 3.9), sharey=True, sharex=True)
    fig.subplots_adjust(left=0.1, right=0.88, bottom=0.15, top=0.93, wspace=0.06, hspace=0.3)
    axes = axs.ravel()
    for k, (ax, c) in enumerate(zip(axes, CFGS)):
        M = data[c]
        norm_M = np.column_stack([M[:, j] / scale[t] for j, t in enumerate(tasks)])
        ax.imshow(norm_M, cmap=DIVERGING, norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:+.1f}".replace("-", "\u2212"), ha="center", va="center", fontsize=7.6,
                        color="white" if abs(norm_M[i, j]) > 0.55 else INK)
        ax.set_xticks(range(len(tasks)), [SHORT[t] for t in tasks], fontsize=7.4, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(len(pairs)), plabels)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.axhline(2.5, color="white", lw=2.0)
        ax.set_xticks(np.arange(-0.5, len(tasks)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(pairs)), minor=True)
        ax.grid(which="minor", color="white", lw=0.8)
        ax.tick_params(which="minor", length=0)
        title = {"orig": "Released", "va2": "VA-SFT, 2 ep", "va6": "VA-SFT, 6 ep", "cpt": "CPT+SFT"}[c]
        ax.set_title(f"({'abcd'[k]}) {title}", fontsize=7.5, color=INK, pad=5)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), len(tasks), 0.12, color=CCOLOR[c], clip_on=False,
                                   transform=ax.transData, zorder=5))
    cax = fig.add_axes([0.9, 0.35, 0.015, 0.3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(0, -1, 1), cmap=DIVERGING), cax=cax, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["loss", "0", "gain"], fontsize=7.2)
    cb.outline.set_visible(False)
    save(fig, "fig_return_heatmap")


# --------------------------------------------------------------------------- route-gap dumbbells
def load_contrasts() -> dict:
    rows = {}
    with open(ROOT / "results/ipm/e14_contrasts.csv", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            rows[(r["family"], r["task"])] = (float(r["estimate"]), float(r["ci_low"]), float(r["ci_high"]))
    return rows


def fig_route_gaps():
    c = load_contrasts()
    fig, axes = plt.subplots(1, 5, figsize=(WIDTH, 2.15), sharey=True)
    fig.subplots_adjust(left=0.1, right=0.99, bottom=0.2, top=0.8, wspace=0.18)
    ys = np.arange(4)[::-1]
    for k, (ax, (task, title, unit)) in enumerate(zip(axes, TASKS)):
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        for y, m in zip(ys, RELEASES):
            e2, l2, h2 = c[(f"route_gap_{m}_2ep", task)]
            e6, l6, h6 = c[(f"route_gap_{m}_6ep", task)]
            ax.plot([e2, e6], [y, y], color="#c9c7c0", lw=2.2, zorder=2, solid_capstyle="round")
            ax.errorbar(e2, y + 0.13, xerr=[[e2 - l2], [h2 - e2]], fmt="o", ms=4.0, color=VA2, mfc="white",
                        mec=VA2, mew=1.0, elinewidth=0.9, capsize=0, zorder=3)
            ax.errorbar(e6, y - 0.13, xerr=[[e6 - l6], [h6 - e6]], fmt="o", ms=4.0, color=VA6, mfc=VA6,
                        mec=VA6, mew=1.0, elinewidth=0.9, capsize=0, zorder=4)
        ax.set_title(f"({'abcde'[k]}) " + (SHORT[task] if task in ("zh2bo", "bo2zh") else title), loc="left", color=INK, pad=4)
        ax.set_yticks(ys, [RLABEL[m] for m in RELEASES])
        ax.set_ylim(-0.6, 3.6)
        ax.xaxis.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("pp" if unit == "acc" else "spBLEU", color=INK2)
        ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], ls="none", marker="o", ms=4.0, color=VA2, mfc="white", mew=1.0,
                      label="Against VA-SFT, 2 epochs"),
               Line2D([], [], ls="none", marker="o", ms=4.0, color=VA6, mfc=VA6, mew=1.0,
                      label="Against VA-SFT, 6 epochs")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0), columnspacing=2.0)
    save(fig, "fig_route_gaps")


# --------------------------------------------------------------------------- reverse-direction diagnostics
def fig_reverse():
    summary = json.load(open(ROOT / "results/ipm/bo2zh_manual_check_summary.json", encoding="utf-8"))["systems"]
    fig = plt.figure(figsize=(WIDTH, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.05, 1.5], left=0.085, right=0.985, bottom=0.2, top=0.72, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    rate = np.array([[zh_rate(c, m) for c in CFGS] for m in RELEASES])
    bleu = np.array([[score(c, m, "bo2zh") for c in CFGS] for m in RELEASES])
    ax.imshow(rate, cmap=SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{bleu[i, j]:.1f}", ha="center", va="center", fontsize=7.6,
                    color="white" if rate[i, j] > 60 else INK)
    ax.set_xticks(range(4), ["None", "VA\n2 ep", "VA\n6 ep", "CPT\n+SFT"], fontsize=7.2)
    ax.set_yticks(range(4), [RLABEL[m] for m in RELEASES])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 4), minor=True)
    ax.set_yticks(np.arange(-0.5, 4), minor=True)
    ax.grid(which="minor", color="white", lw=1.0)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"(a) bo{ARW}zh spBLEU (numbers) and\nChinese-output rate (%, shading)", loc="left", fontsize=7.5, color=INK, pad=4)
    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=SEQUENTIAL, norm=plt.Normalize(0, 100)), cax=cax, ticks=[0, 50, 100])
    cb.ax.tick_params(labelsize=7.2, length=2)
    cb.outline.set_visible(False)

    fig.subplots_adjust(wspace=0.12)
    bx = fig.add_axes([0.64, 0.2, 0.345, 0.52])
    systems = [("qwen2.5_cpt_sft", "Qwen2.5\nCPT+SFT"), ("qwen3_va_sft_6epoch", "Qwen3\nVA-SFT 6 ep"),
               ("qwen3_cpt_sft", "Qwen3\nCPT+SFT")]
    segments = [("adequate", "Adequate", "#184f95", "white"), ("partial", "Partly correct", "#5598e7", INK),
                ("wrong", "Incorrect or empty", "#1baf7a", INK), ("tibetan", "Untranslated", "#eb6834", INK)]
    for y, (key, label) in enumerate(systems):
        s = summary[key]
        q, lang = s["quality"], s["output_language"]
        parts = {"adequate": q.get("2", 0), "partial": q.get("1", 0),
                 "wrong": q.get("0", 0) + lang.get("其他或空", 0), "tibetan": lang.get("藏文", 0)}
        left = 0
        for seg, _, color, txt in segments:
            w = parts[seg]
            if w:
                bx.barh(y, w, left=left, height=0.62, color=color, edgecolor="white", linewidth=1.0)
                if w >= 8:
                    bx.text(left + w / 2, y, f"{w}", ha="center", va="center", color=txt, fontsize=7.4)
                left += w
    bx.set_yticks(range(3), [lab for _, lab in systems])
    bx.tick_params(axis="y", length=0, labelcolor=INK)
    bx.set_xlim(0, 100)
    bx.set_xlabel("Share of 100 sampled outputs (%)", color=INK2)
    bx.xaxis.grid(True, color=GRID, lw=0.5)
    bx.set_axisbelow(True)
    bx.spines["left"].set_visible(False)
    bx.set_title("(b) Manual assessment", loc="left", fontsize=7.5, color=INK, pad=24)
    handles = [Patch(facecolor=col, edgecolor="none", label=lab) for _, lab, col, _ in segments]
    bx.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False,
              handlelength=1.0, columnspacing=1.0, fontsize=7.0, borderaxespad=0.2)
    save(fig, "fig_reverse")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_trajectories()
    fig_return_heatmap()
    fig_route_gaps()
    fig_reverse()
    print("figures written")
