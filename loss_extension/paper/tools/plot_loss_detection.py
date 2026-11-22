"""Rebuild the manuscript's primary frame-loss figure from the shipped estimates."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PAPER = Path(__file__).resolve().parents[1]
ARMS = {"none": "No loss", "random": "Uniform loss", "selective": "Selective loss"}
METHODS = {
    "snapshot_events": ("Event summaries", "#17658a", "#124f6c", ""),
    "events_thinned": ("Thinned history", "#f2c79e", "#87542c", "//"),
    "fresh_compact": ("Fresh compact", "#e4dff2", "#61567e", ""),
}


def read_estimates():
    estimates = {}
    with (PAPER / "data" / "primary_rates.csv").open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = row["loss_mode"], row["method"]
            if key[0] not in ARMS:
                continue
            if key in estimates:
                raise ValueError(f"Duplicate primary cell {key}")
            lo, mean, hi = (float(row[k]) for k in
                            ("detection_lo95_pct", "fault_detection_pct", "detection_hi95_pct"))
            if not 0 <= lo <= mean <= hi <= 100:
                raise ValueError(f"Invalid interval for {key}")
            estimates[key] = lo, mean, hi
    if estimates.keys() != {(arm, method) for arm in ARMS for method in METHODS}:
        raise ValueError("Expected one primary estimate per loss condition and method")
    return estimates


def main():
    estimates = read_estimates()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                         "pdf.fonttype": 42, "hatch.linewidth": 0.5})
    fig, ax = plt.subplots(figsize=(122 / 25.4, 49 / 25.4))
    fig.subplots_adjust(left=0.078, right=0.99, top=0.81, bottom=0.24)
    fig.text(0.078, 0.94, "Failed OFF commands detected within 6 h (%)",
             va="center", fontsize=9)
    width = 0.24
    for group, arm in enumerate(ARMS):
        for slot, (method, (_, face, edge, hatch)) in enumerate(METHODS.items()):
            lo, mean, hi = estimates[arm, method]
            x = group + (slot - 1) * 0.28
            ax.bar(x, mean, width, facecolor=face, edgecolor=edge, linewidth=0.7,
                   hatch=hatch, zorder=2)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none",
                        color="#202020", elinewidth=0.8, capsize=2, capthick=0.8, zorder=3)
            ax.annotate(f"{mean:.2f}", (x, hi), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.8, color="#171717")
    ax.set_xlim(-0.49, len(ARMS) - 0.51)
    ax.set_ylim(0, 75)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xticks(range(len(ARMS)), ARMS.values())
    ax.tick_params(axis="both", length=0, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#d5d8da", linewidth=0.55)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#73777a")
    handles = [Patch(facecolor=face, edgecolor=edge, hatch=hatch, linewidth=0.7, label=label)
               for label, face, edge, hatch in METHODS.values()]
    fig.legend(handles=handles, loc="lower center", ncol=len(METHODS), frameon=False,
               bbox_to_anchor=(0.5, 0.01), borderaxespad=0, handlelength=1.25,
               handletextpad=0.5, columnspacing=1.6, fontsize=8.3)
    fig.savefig(PAPER / "figures" / "loss_detection.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(PAPER / "figures" / "loss_detection.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
