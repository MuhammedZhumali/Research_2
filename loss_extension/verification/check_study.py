"""Reconcile the matched contact opportunities, actual accounting and primary outcomes."""
from collections import defaultdict
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json

EXT = Path(__file__).resolve().parents[1]
PILOT = EXT.parent
METHODS = ("snapshot_events", "events_thinned", "fresh_compact")
ARMS = ("none", "random", "selective")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_rows(path):
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, "rt", newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def group(row):
    return row["method"], row["loss_mode"]


def run(data, output):
    freeze = json.loads((EXT/"verification/pretest-freeze.json").read_text())
    assert sha(EXT/"registration.json") == freeze["scope_sha256"]
    assert sha(EXT/"analysis/PLAN.md") == freeze["analysis_plan_sha256"]
    assert all(sha(PILOT/rel) == h for rel,h in freeze["source_hashes"].items())
    checks = {}
    for base in [PILOT, PILOT/"v2", PILOT/"v3_validation"]:
        manifest = base/"artifact-manifest.json"
        if manifest.exists():  # The compact reproduction bundle includes the scientific dependencies.
            files = json.loads(manifest.read_text(encoding="utf-8-sig"))["files"]
            assert all(sha(base/item["path"]) == item["sha256"] for item in files), base
            checks[base.name] = len(files)
    complete = json.loads((data/"completed.json").read_text())
    assert complete["episodes"] == 480 and complete["runs"] == 17280 and complete["targets"] == 51840
    run_row_count = 0
    run_keys = set()
    tally = defaultdict(lambda: defaultdict(int))
    keys = ("episode_id","schedule_id","rate_bps","method","loss_mode")
    for row in read_rows(data/"runs.csv"):
        run_row_count += 1
        key = tuple(row[k] for k in keys)
        assert key not in run_keys
        run_keys.add(key)
        if row["schedule_id"] == "kazstsat_astana_10deg" and row["rate_bps"] == "32":
            entry = tally[group(row)]
            for col in ("attempted_frames","erased_frames","erased_data_bytes","erased_marker_frames","wire_bytes","reserved_bytes","exposed_active_contact"):
                entry[col] += int(row[col])
    assert run_row_count == complete["runs"]
    gate_map = {}
    contact_count = 0
    for row in read_rows(data/"contacts.csv.gz"):
        contact_count += 1
        key = tuple(row[k] for k in ("episode_id","schedule_id","contact_number"))
        gate = int(row["opportunity"])
        assert gate_map.setdefault(key,gate) == gate
        erased, attempted = int(row["erased_frames"]), int(row["attempted_frames"])
        assert erased == int(bool(gate and attempted and row["loss_mode"] != "none"))
        assert int(row["reserved_bytes"]) - int(row["wire_bytes"]) == 56 * erased
        assert int(row["reserved_bytes"]) <= int(row["budget_bytes"])
        if row["loss_mode"] == "selective" and erased and int(row["marker_frames"]):
            assert int(row["erased_marker_frames"]) == 1
        if row["schedule_id"] == "kazstsat_astana_10deg" and row["rate_bps"] == "32" and erased:
            entry = tally[group(row)]
            entry["singleton_erasure_contacts"] += int(attempted == 1)
            entry["choice_of_marker_possible"] += int(0 < int(row["marker_frames"]) < attempted)
    assert contact_count == complete["contacts"]
    primary_targets = defaultdict(dict)
    truth_by_target = {}
    target_count = 0
    for row in read_rows(data/"targets.csv"):
        target_count += 1
        target_key = row["episode_id"],row["target_epoch"]
        truth = row["truth_label"]
        assert truth_by_target.setdefault(target_key,truth) == truth
        at = row["first_correct_time_s"]
        start, deadline = float(row["start_s"]), float(row["deadline_s"])
        assert deadline - start == 21600 and int(row["resolved"]) == int(bool(at))
        if at:
            assert start <= float(at) <= deadline
        if truth == "fault" and row["schedule_id"] == "kazstsat_astana_10deg" and row["rate_bps"] == "32":
            primary_targets[group(row)][target_key] = int(bool(at))
    assert target_count == complete["targets"] and len(truth_by_target) == 1440
    all_faults = {k for k,v in truth_by_target.items() if v == "fault"}
    assert all(set(cell) == all_faults for cell in primary_targets.values()) and all_faults
    outcomes = []
    for method in METHODS:
        for arm in ARMS:
            cell = primary_targets[method,arm]
            episodes = defaultdict(list)
            for (ep,_),value in cell.items():
                episodes[ep].append(value)
            assert len(episodes) == 240
            p = sum(sum(values)/len(values) for values in episodes.values())/len(episodes)
            outcomes.append(dict(method=method,loss_mode=arm,correct=sum(cell.values()),targets=len(cell),
                                 detection_rate=p,**tally[method,arm]))
    output.mkdir(exist_ok=True,parents=True)
    (output/"independent-check.json").write_text(json.dumps(dict(status="passed",
        original_manifest_files=checks, frozen_inputs_passed=True, run_rows=run_row_count,
        target_rows=target_count, contact_rows=contact_count, physical_targets=len(truth_by_target),
        unique_episode_schedule_contacts=len(gate_map),primary=outcomes),indent=2)+"\n")
    for row in outcomes:
        print(row["method"],row["loss_mode"],row["correct"],row["targets"],round(row["detection_rate"],5),
              row.get("singleton_erasure_contacts",0),row["erased_frames"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=EXT/"experiment/outputs/heldout")
    parser.add_argument("--output", type=Path, default=EXT/"verification")
    args = parser.parse_args()
    run(args.data,args.output)
