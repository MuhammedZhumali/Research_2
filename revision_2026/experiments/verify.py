"""Independent checks on revision outputs, physical sensor input and replay hashes."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import run


def read(folder, name):
    with (folder / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sensor_check():
    ep = run.frozen.generate_episode(61000000, profile="thermal_only")
    lagged = run.sensor_episode(ep, run.SPEC["lag_s"])
    n = 40
    shorter = replace(ep, measurements=ep.measurements[:n], truth=ep.truth[:n])
    assert lagged.measurements[:n] == run.sensor_episode(shorter, run.SPEC["lag_s"]).measurements
    assert run.sensor_episode(ep, 0).measurements == run.frozen.quantized(ep).measurements
    # Independently integrate the same continuous linear forcing using short midpoint steps.
    state = ep.truth[0].temperature_c
    step = 0.25
    alpha = -math.expm1(-step / run.SPEC["lag_s"])
    residuals = []
    for i in range(1, n):
        previous = ep.truth[i-1]
        next_sample = ep.truth[i]
        steps = round((next_sample.time_s - previous.time_s) / step)
        for j in range(steps):
            temperature = previous.temperature_c + (next_sample.temperature_c-previous.temperature_c)*(j+0.5)/steps
            state += alpha * (temperature - state)
        original = ep.measurements[i]
        residuals.append(abs(lagged.measurements[i].temperature_c -
                             (state + original.temperature_c-next_sample.temperature_c)))
        assert (original.time_s, original.current_a, original.command_epoch) == (
            lagged.measurements[i].time_s, lagged.measurements[i].current_a, lagged.measurements[i].command_epoch)
    assert max(residuals) < 0.0001
    return max(residuals)


def verify(output, compare=None):
    rates = read(output, "aggregate.csv")
    counts = defaultdict(Counter)
    costs = defaultdict(list)
    starts = {}
    for row in read(output, "targets.csv"):
        key = (row["profile"],row["episode_id"],row["target_epoch"])
        truth = (row["truth"],row["start_s"],row["deadline_s"])
        assert starts.setdefault(key, truth) == truth
        correct = int(bool(row["first_correct_s"]))
        delay = float(row["first_correct_s"] or row["deadline_s"]) - float(row["start_s"])
        for profile in ("all", row["profile"]):
            values = counts[(profile,row["lag_s"],row["rate_bps"],row["method"])]
            values.update(targets=1,correct=correct,ever_wrong=int(row["ever_wrong"]),
                          no_decision=int(row["no_decision"]),delay_s=delay)
    for row in read(output, "runs.csv"):
        for profile in ("all",row["profile"]):
            if row["wire_bytes"]:
                costs[(profile,row["lag_s"],row["rate_bps"],row["method"])].append(int(row["wire_bytes"]))
    cells = {}
    for r in rates:
        key = tuple(r[name] for name in ("profile","lag_s","rate_bps","method"))
        cells[key] = r
        c = counts[key]
        for metric in ("targets","correct","ever_wrong","no_decision"):
            assert c[metric] == int(r[metric])
        assert abs(c["delay_s"] / c["targets"] / 3600 - float(r["capped_delay_h"])) < 1e-7
        if r["bytes_day"]:
            assert abs(sum(costs[key]) / len(costs[key]) - float(r["bytes_day"])) < 1e-7
    contrasts = read(output, "contrasts.csv")
    for r in contrasts:
        a = cells[r["profile"], r["lag_s"], r["rate_bps"], r["method"]]
        b = cells[r["profile"], r["reference_lag_s"], r["rate_bps"], r["reference"]]
        diff = float(a[r["metric"]]) - float(b[r["metric"]])
        assert abs(diff - float(r["difference"])) < 2e-7
        assert float(r["lo95"]) <= float(r["hi95"])
    files = ["targets.csv","runs.csv","aggregate.csv","contrasts.csv"]
    hashes = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files}
    assert hashes == json.loads((output / "audit.json").read_text())["sha256"]
    if compare is not None:
        for name in files:
            assert (output / name).read_bytes() == (compare / name).read_bytes(), name
    return dict(independent_target_counts=True, aggregate_cells=len(rates), contrasts=len(contrasts),
                unique_targets=len(starts), prefix_causality=True, sensor_integration_max_abs_c=sensor_check(),
                byte_identical_scientific_files=files if compare is not None else [], scientific_sha256=hashes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = json.dumps(verify(args.output, args.compare), indent=2)
    print(report)
    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")
