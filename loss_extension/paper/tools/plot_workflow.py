"""Reconstruct the worked contact from its archived trace and draw Figure 1."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
DATA = PAPER / "data" / "worked_trace.json"


def extract(trace):
    record = json.loads(trace.read_text(encoding="utf-8"))
    contact = record["contacts"][0]
    frame, = contact["frames"]
    first, event = frame["context_epochs"]
    targets = {int(t["target_epoch"]): t for t in record["targets"]}
    target = targets[event]
    samples = {m["time_s"]: m for m in record["public_measurements"]}
    next_on = next(c["time_s"] for c in record["public_events"]
                   if c["heater_on"] and c["time_s"] > target["start_s"])
    selected = [t for t, epoch in zip(frame["measurement_times_s"], frame["measurement_epochs"])
                if epoch == event]
    assert len(selected) == 5 and targets[first]["deadline_s"] < contact["end_s"]
    assert target["first_correct_time_s"] == contact["decision"]["time_s"]
    return dict(episode=record["run"]["episode_id"], rate_bps=record["run"]["rate_bps"],
                off_s=target["start_s"], next_on_s=next_on, selected_s=selected,
                acquired_s=[t for t, m in samples.items() if m["command_epoch"] == event],
                selected_current_s=[t for t in selected if samples[t]["current_a"] is not None],
                contact_s=[contact["start_s"], contact["end_s"]],
                tx_s=[frame["start_s"], frame["end_s"]], diagnosis_s=target["first_correct_time_s"],
                budget=contact["budget_bytes"], observations=len(frame["measurement_times_s"]),
                data=frame["data_bytes"], context=frame["context_bytes"], ack=frame["ack_bytes"],
                total=frame["wire_bytes"])


def plot_trace(d):
    """Show the selection and delivery of one OFF on their natural time scales."""
    def minutes_after_off(seconds):
        return (seconds - d["off_s"]) / 60

    color = "#00728B"
    ink = "#26343C"
    gray = "#69777D"
    current = set(d["selected_current_s"])
    selected = [t for t in d["selected_s"] if t not in current]
    next_on = minutes_after_off(d["next_on_s"])
    start, end = d["contact_s"]
    tx_start, tx_end = (t - start for t in d["tx_s"])
    decision = d["diagnosis_s"] - start

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "text.color": ink,
                         "xtick.color": ink, "ytick.color": ink})
    fig = plt.figure(figsize=(122 / 25.4, 56 / 25.4))
    # The labels sit outside the data rectangle. Both panels have enough
    # horizontal space to separate nearby events at the final print size.
    samples = fig.add_axes([.253, .62, .70, .21])
    samples.set_xlim(-1.8, next_on + 1.8)
    samples.set_ylim(-.45, 2.45)
    for level in (0, 1, 2):
        samples.hlines(level, 0, next_on, color="#D5DBDD", linewidth=.55, zorder=0)

    acquired = [minutes_after_off(t) for t in d["acquired_s"]]
    samples.vlines(acquired, 1.78, 2.22, linewidth=.55, color=gray)
    for values, level, marker in ((selected, 1, "o"), (sorted(current), 0, "^")):
        times = [minutes_after_off(t) for t in values]
        samples.scatter(times, [level] * len(times), s=16, marker=marker,
                        color=color, clip_on=False, zorder=3)
    samples.set_yticks([2, 1, 0], ["All samples", "Kept · temp. only", "Kept · with current"])
    samples.set_xticks([0, 30, 60, next_on], ["0 (OFF)", "30", "60", f"{next_on:g} (ON)"])
    samples.tick_params(axis="y", length=0, pad=4)
    samples.tick_params(axis="x", length=2, pad=2, width=.6)
    samples.spines[["top", "left", "right"]].set_visible(False)
    samples.spines["bottom"].set_color(gray)
    samples.spines["bottom"].set_linewidth(.55)
    fig.text(.015, .94, "On board", fontweight="bold", va="center")
    fig.text(.175, .94, f"{len(d['selected_s'])} of {len(acquired)} samples kept for this OFF",
             va="center")
    samples.set_xlabel("Minutes after the OFF command", labelpad=2, color=ink)

    # The archived frame also carries data for an older OFF, as explained
    # in the worked example. Its complete frame-plus-ACK airtime is shown.
    link = fig.add_axes([.075, .055, .86, .22])
    link.set_xlim(-.015 * decision, 1.015 * decision)
    link.set_ylim(0, 1)
    link.axis("off")
    fig.text(.015, .335, "Next contact", fontweight="bold", va="center")
    fig.text(.215, .335, f"Link opens {minutes_after_off(start):.0f} min after OFF",
             va="center")
    lane = .42
    link.plot([0, end - start], [lane, lane], color="#B4BEC1", linewidth=3,
              solid_capstyle="butt")
    link.plot([tx_start, tx_end], [lane, lane], color=color, linewidth=4.5,
              solid_capstyle="butt")
    link.plot([decision], [lane], marker="D", markersize=4.3, color=ink)
    link.text(tx_start, .66, "Frame + ACK", va="bottom")
    link.text(decision, .66, "Ground confirms OFF", ha="right", va="bottom")
    for pos in (0, tx_end, decision):
        link.text(pos, .17, f"{pos:.0f} s", ha="center", va="top")

    output = PAPER / "figures" / "workflow_trace"
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, help="refresh the compact input from the full JSON trace")
    args = parser.parse_args()
    if args.trace:
        DATA.write_text(json.dumps(extract(args.trace), indent=2) + "\n", encoding="utf-8")
    d = json.loads(DATA.read_text(encoding="utf-8"))
    plot_trace(d)
    overhead = d["total"] - d["data"] - d["context"] - d["ack"]
    budget = f"{d['budget']:,}".replace(",", r"\,")
    offsets = ", ".join(f"{(t - d['off_s']) / 60:g}" for t in d["selected_s"])
    text = (rf"For the second OFF in Fig.~\ref{{fig:workflow}}, Eq.~\eqref{{eq:onboard-summary}} selects "
            rf"records at $({offsets})$~min after the command. "
            rf"One frame carries {d['observations']} records for two OFF commands. "
            rf"Observations ({d['data']} bytes), command context ({d['context']}), framing ({overhead}) and "
            rf"the ACK ({d['ack']}) use {d['total']} of {budget} available bytes at {d['rate_bps']}~bit/s. "
            "The second OFF is confirmed in time. The first has missed its deadline.")
    (PAPER/"worked_trace.tex").write_text(text+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
