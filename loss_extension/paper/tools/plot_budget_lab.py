"""Redraw the contact-budget comparison and the measured TCLab temperature trace."""

import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data" / "budget_lab"
WIDTH = 122 / 25.4
INK = "#222629"
BLUE = "#17658a"
ORANGE = "#a5581d"
GRID = "#dce0e1"
PROFILES = {
    "clear_current": ("Frequent current", "o", BLUE),
    "slow_sparse_current": ("Sparse current", "s", ORANGE),
    "thermal_only": ("Temperature only", "^", "#51466f"),
}
SCHEDULES = {"kazstsat_astana_10deg": "KazSTSAT", "synthetic_90min": "Periodic"}
LAB_FILES = ("predictions.csv", "results.json", "audit.json")


def csv_rows(name):
    with (DATA / name).open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def save(fig, name):
    out = PAPER / "figures"
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def budget_figure():
    rows = csv_rows("matched_budget.csv")
    rates = sorted({float(row["rate_bps"]) for row in rows})
    keys = {(s, p, r) for s in SCHEDULES for p in PROFILES for r in rates}
    by_key = {(r["schedule_id"], r["profile"], float(r["rate_bps"])): r for r in rows}
    if (by_key.keys() != keys or len(keys) != len(rows) or rates != [4, 8, 32]
            or any(r["method"] != "snapshot_events" or int(r["n_episodes"]) != 400 for r in rows)):
        raise ValueError("Expected the registered matched-budget grid of 400 episodes per profile.")

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 66 / 25.4), sharey=True)
    fig.subplots_adjust(left=.205, right=.982, top=.865, bottom=.23, wspace=.17)
    group_stride = len(rates) + .95
    row_y = [group * group_stride + i for group in range(len(SCHEDULES)) for i in range(len(rates))]
    metric_specs = [
        ("coverage_loss", 100, "Fewer timely decisions", "Percentage points"),
        ("delay_gap_s", 1 / 60, "Extra mean delay", "Minutes"),
    ]
    for ax, (metric, scale, title, unit) in zip(axes, metric_specs):
        ax.axvline(0, color="#7b8488", linewidth=.8)
        ax.axhline((row_y[2] + row_y[3]) / 2, color=GRID, linewidth=.6)
        for group, schedule in enumerate(SCHEDULES):
            for i, rate in enumerate(rates):
                y = group * group_stride + i
                for profile_index, (profile, (_, marker, color)) in enumerate(PROFILES.items()):
                    r = by_key[schedule, profile, rate]
                    mean, lo, hi = [float(r[f"{metric}_{suffix}"]) * scale
                                    for suffix in ("mean", "ci_low", "ci_high")]
                    if not np.isfinite([lo, mean, hi]).all() or not lo <= mean <= hi:
                        raise ValueError(f"Invalid paired interval for {schedule}/{profile}/{rate}/{metric}")
                    ax.errorbar(mean, y + (profile_index - 1) * .245,
                                xerr=[[mean - lo], [hi - mean]], fmt=marker, markersize=3.7,
                                color=color, markeredgewidth=.7, elinewidth=.9, capsize=1.7,
                                clip_on=False)
        ax.set_title(title, loc="left", pad=4, fontsize=9.1, weight="bold")
        ax.set_xlabel(unit, labelpad=2.5)
        ax.margins(x=.07)
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([tick for tick in ax.get_xticks() if xmin <= tick <= xmax])
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.grid(axis="x", color=GRID, linewidth=.55)
        ax.set_axisbelow(True)
    axes[0].set_ylim(row_y[-1] + .57, -.58)
    axes[0].set_yticks(row_y, [f"{rate:g}" for _ in SCHEDULES for rate in rates])
    for group, name in enumerate(SCHEDULES.values()):
        axes[0].annotate(name, xy=(0, group * group_stride + 1),
                         xycoords=axes[0].get_yaxis_transform(), xytext=(-19, 0),
                         textcoords="offset points", ha="right", va="center",
                         weight="bold", fontsize=8.1)
    axes[0].text(-.035, 1.038, "bit/s", ha="right", va="bottom", transform=axes[0].transAxes)
    legend = [Line2D([], [], color=c, marker=m, ls="none", markersize=4.7, label=name)
              for name, m, c in PROFILES.values()]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .002), ncol=3,
               handlelength=.8, columnspacing=1.15, handletextpad=.4, frameon=False)
    save(fig, "budget_boundary")


def lab_figure():
    result = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    audit = json.loads((DATA / "audit.json").read_text(encoding="utf-8"))
    if result["status"] != "completed" or audit["status"] != "passed":
        raise ValueError("Expected a completed, audited laboratory run.")
    rows = csv_rows("predictions.csv")
    time, measured, simple, lagged, power = [
        np.asarray([float(r[column]) for r in rows])
        for column in ("time_s", "measured_t1_c", "first_order_c", "sensor_lag_c", "heater1_percent")
    ]
    split = min(float(r["time_s"]) for r in rows if r["split"] == "held_out")
    logged = result["commanded_off_observation"]
    off = logged["start_s"]
    elapsed = time - off
    off_index = int(np.flatnonzero(elapsed >= 0)[0])
    peak_index = off_index + int(np.argmax(measured[off_index:]))
    rise = measured[peak_index] - measured[off_index]
    if not (power[off_index - 1] > 0 and np.all(power[off_index:] == 0)
            and np.isclose(elapsed[peak_index], logged["observed_peak_after_off_s"])
            and np.isclose(rise, logged["observed_rise_after_off_c"])):
        raise ValueError("The plotted OFF or peak disagrees with the scored hardware record.")

    fig = plt.figure(figsize=(WIDTH, 64 / 25.4))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.45, 1), height_ratios=(3.1, .9),
                           left=.105, right=.98, bottom=.27, top=.88, wspace=.51, hspace=.12)
    full = fig.add_subplot(grid[0, 0])
    command = fig.add_subplot(grid[1, 0], sharex=full)
    zoom = fig.add_subplot(grid[:, 1])

    for ax in (full, command):
        ax.axvspan(split, time[-1], facecolor="#e5eff3", zorder=0)
        ax.set_xlim(time[0], time[-1])
        ax.set_xticks([0, 200, 400, 600])
    first_line, = full.plot(time, simple, color=ORANGE, ls="--", lw=1.05, label="First-order model")
    lag_line, = full.plot(time, lagged, color=BLUE, lw=1.7, label="Two-state model")
    data_line, = full.plot(time, measured, color=INK, lw=.85, label="Measured")
    full.tick_params(axis="x", bottom=False, labelbottom=False)
    full.set_title("Full record", loc="left", weight="bold")
    full.set_ylabel("Temp. (°C)", labelpad=2)
    full.text((split + time[-1]) / 2, .10, "Test", transform=full.get_xaxis_transform(),
              color="#2b5369", ha="center")
    command.step(time, power, where="post", color=INK, lw=.9)
    command.set_ylim(-8, 108)
    command.set_yticks([0, 100])
    command.set_ylabel("Power (%)", labelpad=1, fontsize=7.6)
    command.set_xlabel("Elapsed time (s)", labelpad=2)

    # The measured trace alone shows the brief rise; model fit is displayed at left.
    zoom.plot(elapsed, measured, color=INK, lw=1.4)
    zoom.axvline(0, color="#7b8488", ls=":", lw=.8)
    zoom.plot(0, measured[off_index], marker="o", ms=4.3, mfc="white", mec=INK, mew=1)
    zoom.plot(elapsed[peak_index], measured[peak_index], marker="o", ms=4.3, color=INK)
    zoom.set_xlim(-4, 30)
    visible = (elapsed >= -4) & (elapsed <= 30)
    zoom.set_ylim(min(measured[visible]) - .22, max(measured[visible]) + .88)
    zoom.set_xticks([0, 10, 20, 30])
    zoom.annotate(f"+{rise:.2f} °C\nat {elapsed[peak_index]:.1f} s",
                  (elapsed[peak_index], measured[peak_index]), xytext=(7, 6),
                  textcoords="offset points", ha="left", va="bottom", fontsize=8.1)
    zoom.set_title("After OFF", loc="left", weight="bold")
    zoom.set_xlabel("Time from OFF (s)", labelpad=2)
    zoom.set_ylabel("Temp. (°C)", labelpad=1)
    for ax in (full, command, zoom):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=.5, color=GRID)
        ax.set_axisbelow(True)
    fig.legend(handles=[data_line, first_line, lag_line], loc="lower center",
               bbox_to_anchor=(.5, .017), ncol=3, frameon=False,
               handlelength=1.8, handletextpad=.4, columnspacing=1.2)
    save(fig, "lab_physics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-csv", type=Path, help="Refresh the archived matched_budget.csv")
    parser.add_argument("--lab-dir", type=Path, help="Refresh the archived predictions, score and audit")
    args = parser.parse_args()
    if args.budget_csv or args.lab_dir:
        DATA.mkdir(exist_ok=True)
    if args.budget_csv:
        shutil.copyfile(args.budget_csv, DATA / "matched_budget.csv")
    if args.lab_dir:
        for name in LAB_FILES:
            shutil.copyfile(args.lab_dir / name, DATA / name)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.edgecolor": "#9ca3a5", "axes.labelcolor": INK, "text.color": INK,
        "axes.linewidth": .6, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.size": 2.7, "ytick.major.size": 2.7, "pdf.fonttype": 42,
    })
    budget_figure()
    lab_figure()


if __name__ == "__main__":
    main()
