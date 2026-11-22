"""Describe six-hour diagnostic outcomes in the frozen 32 bit/s KazSTSAT cell."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
SCHEDULE = "kazstsat_astana_10deg"
REASONS = (
    "timely",
    "no_window_intersects_horizon",
    "window_but_no_timed_opportunity",
    "contact_reference_only",
    "insufficient_continuous_evidence",
)


def rows(path):
    with path.open(encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)


def selected(row):
    return (row["role"] == "primary" and row["method"] == "snapshot_events"
            and row["schedule_id"] == SCHEDULE and int(row["rate_bps"]) == 32)


def reason(row, windows, reserve):
    start, deadline = float(row["start_s"]), float(row["deadline_s"])
    evidence = row["oracle_continuous_s"]
    has_evidence = bool(evidence) and float(evidence) <= deadline
    reference = int(row["oracle_contact_resolved"])
    if int(row["resolved"]):
        if not reference or not has_evidence:
            raise ValueError("A resolved target lacks a timely reference in this baseline")
        return "timely"
    if reference:
        if not has_evidence:
            raise ValueError("Contact reference lacks continuous evidence")
        return "contact_reference_only"
    if not has_evidence:
        return "insufficient_continuous_evidence"
    if any(a >= float(evidence) and b + reserve <= deadline for a, b in windows):
        raise ValueError("Timing alone cannot account for this contact-reference miss")
    if any(a < deadline and b > start for a, b in windows):
        return "window_but_no_timed_opportunity"
    return "no_window_intersects_horizon"


def main(run_dir, output):
    config = json.loads((STUDY / "v2/configs/protocol-v2.json").read_text(encoding="utf-8"))
    v3 = json.loads((STUDY / "v3_validation/configs/protocol-confirmatory.json").read_text(encoding="utf-8"))
    duration = v3["duration_s"]
    windows = [(float(r["start_s"]), float(r["end_s"])) for r in rows(STUDY / "data/contacts/kazstsat.csv")]
    schedules = {}
    for row in rows(run_dir / "runs.csv"):
        if not selected(row):
            continue
        eid = row["episode_id"]
        offset = float(row["schedule_offset_s"])
        if eid in schedules:
            raise ValueError(f"Duplicate episode {eid}")
        schedules[eid] = [(max(0, a-offset), min(duration, b-offset)) for a, b in windows
                          if b > offset and a < offset + duration]
        if len(schedules[eid]) != int(row["n_contacts"]):
            raise ValueError(f"Window reconstruction failed for {eid}")
    expected = len(v3["profiles"]) * v3["n_per_profile"]
    if len(schedules) != expected:
        raise ValueError(f"Expected {expected} episodes, found {len(schedules)}")
    counts, mistakes, unknowns = Counter(), Counter(), Counter()
    targets = set()
    for row in rows(run_dir / "targets.csv"):
        if not selected(row):
            continue
        eid = row["episode_id"]
        key = (eid, row["target_epoch"])
        if key in targets:
            raise ValueError(f"Duplicate target {key}")
        targets.add(key)
        label = row["truth_label"]
        why = reason(row, schedules[eid], config["delivery"]["ground_diagnosis_reservation_s"])
        counts[label, why] += 1
        mistakes[label] += int(row["ever_wrong"])
        unknowns[label] += int(row["no_decision"])
    if len(targets) != expected * 3 or set(label for label, _ in counts) != {"fault", "nominal"}:
        raise ValueError("Unexpected OFF target population")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "outcomes.csv").open("w", encoding="utf-8", newline="") as dest:
        writer = csv.writer(dest, lineterminator="\n")
        writer.writerow(["truth", "targets", *REASONS, "ever_wrong", "no_decision"])
        for name, labels in [("fault", ["fault"]), ("nominal", ["nominal"]), ("all", ["fault", "nominal"])]:
            values = [sum(counts[label, r] for label in labels) for r in REASONS]
            writer.writerow([name, sum(values), *values, sum(mistakes[l] for l in labels),
                             sum(unknowns[l] for l in labels)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=STUDY / "v3_validation/outputs/confirmatory-main")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    main(args.runs, args.output)
