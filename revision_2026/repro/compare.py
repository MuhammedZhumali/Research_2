"""Compare a new run to published decisions, uncertainty and physical measurements."""
import argparse
import csv
from itertools import zip_longest
import json

from replay import LOSS, REV, ROOT, V2, V3


def records(path):
    with path.open(encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)


def compare_targets(reference, replay, keys):
    mismatches = {}
    largest_drift_s = 0
    largest_reference_drift_s = 0
    total = 0
    for original, current in zip_longest(records(reference), records(replay)):
        if original is None or current is None or any(original[k] != current[k] for k in keys):
            raise ValueError(f"Different target population or ordering in {reference.name}")
        total += 1
        for name in ("truth_label", "resolved", "no_decision", "ever_wrong", "final_label",
                     "oracle_contact_resolved", "oracle_continuous_s"):
            if name in original and original[name] != current[name]:
                mismatches[name] = mismatches.get(name, 0) + 1
        drift = abs(float(original["capped_delay_s"]) - float(current["capped_delay_s"]))
        largest_drift_s = max(largest_drift_s, drift)
        if original.get("oracle_contact_capped_delay_s") and current.get("oracle_contact_capped_delay_s"):
            drift = abs(float(original["oracle_contact_capped_delay_s"]) - float(current["oracle_contact_capped_delay_s"]))
            largest_reference_drift_s = max(largest_reference_drift_s, drift)
    return {"target_rows": total, "decision_mismatches": mismatches,
            "max_absolute_capped_delay_change_s": largest_drift_s,
            "max_absolute_contact_reference_capped_delay_change_s": largest_reference_drift_s}


def compare_rates(reference, replay, keys, numeric, categorical=()):
    def index(path):
        result = {}
        for row in records(path):
            key = tuple(row[k] for k in keys)
            if key in result:
                raise ValueError(f"Duplicate group {key} in {path.name}")
            result[key] = row
        return result
    one, two = index(reference), index(replay)
    if one.keys() != two.keys():
        raise ValueError(f"Different comparison populations in {reference.name}")
    diffs = {n: 0.0 for n in numeric}
    mismatches = 0
    for key, a in one.items():
        b = two[key]
        mismatches += int(any(a[k] != b[k] for k in categorical))
        for n in numeric:
            diffs[n] = max(diffs[n], abs(float(a[n]) - float(b[n])))
    return {"compared_rows": len(one), "categorical_mismatches": mismatches, "max_absolute_change": diffs}


def compare(tag):
    output = ROOT / "replay" / tag
    prefix = ["episode_id", "schedule_id", "rate_bps", "method"]
    result = {}
    for study, original, fresh, keys in [
        ("exploratory", V2 / "outputs/main", V2 / "outputs" / tag, ["condition", "stress"]),
        ("validation", V3 / "outputs/confirmatory-main", V3 / "outputs" / tag, ["role"]),
        ("frame_loss", LOSS / "experiment/outputs/heldout", LOSS / "experiment/outputs" / tag, ["loss_mode"]),
    ]:
        result[study] = compare_targets(original / "targets.csv", fresh / "targets.csv",
                                        prefix + keys + ["target_epoch", "start_s", "deadline_s"])
        result[study]["traffic"] = compare_rates(original / "runs.csv", fresh / "runs.csv", prefix + keys, ["wire_bytes"])
    result["v3_paired_intervals"] = compare_rates(
        V3 / "outputs/confirmatory-main/tables/primary_endpoints.csv", V3 / "outputs" / tag / "tables/primary_endpoints.csv",
        ["profile", "schedule_id", "rate_bps", "endpoint"],
        ["mean_difference", "ci_low", "ci_high"], ["ni_supported", "eq_supported"])
    def published_interval(row):
        return tuple(f"{float(row[field]):.1f}" if row["endpoint"] == "delay_gap_s"
                     else f"{100 * float(row[field]):.2f}" for field in ("mean_difference", "ci_low", "ci_high"))
    reference_intervals = list(records(V3 / "outputs/confirmatory-main/tables/primary_endpoints.csv"))
    rebuilt_intervals = list(records(V3 / "outputs" / tag / "tables/primary_endpoints.csv"))
    interval_pairs = [(published_interval(a), published_interval(b))
                      for a, b in zip(reference_intervals, rebuilt_intervals, strict=True)]
    result["v3_paired_intervals"]["rounded_mismatches"] = sum(
        tuple(map(float, a)) != tuple(map(float, b)) for a, b in interval_pairs)
    result["v3_paired_intervals"]["sign_of_rounded_zero_only"] = sum(
        a != b and tuple(map(float, a)) == tuple(map(float, b)) for a, b in interval_pairs)
    result["frame_loss_rates"] = compare_rates(
        LOSS / "analysis/results/primary_rates.csv", output / "loss-analysis/primary_rates.csv",
        ["method", "loss_mode"], ["fault_detection_pct", "daily_wire_bytes", "detection_lo95_pct", "detection_hi95_pct"])
    result["lab"] = compare_rates(
        V3 / "lab/main/prediction-metrics.csv", V3 / "lab" / tag / "prediction-metrics.csv",
        ["model", "split"], ["rmse_c", "mae_c", "max_abs_error_c"])
    result["attribution_counts_match"] = (list(records(output / "attribution/outcomes.csv")) ==
                                         list(records(REV / "attribution/outputs/outcomes.csv")))
    result["revision_scientific_tables_match"] = {
        name: (output / "revision" / name).read_bytes() == (REV / "experiments/outputs" / name).read_bytes()
        for name in ("runs.csv", "targets.csv", "aggregate.csv", "contrasts.csv")}
    article = LOSS / "paper"
    fragments = [*article.glob("*.tex"), *(article / "tables").glob("*.tex")]
    result["paper_tex_mismatches"] = [
        source.relative_to(article).as_posix() for source in sorted(fragments)
        if source.read_text(encoding="utf-8") !=
        (output / "paper" / source.relative_to(article)).read_text(encoding="utf-8")]
    result["all_checks_agree"] = (all(not result[s]["decision_mismatches"] and
        result[s]["traffic"]["max_absolute_change"]["wire_bytes"] == 0 for s in ("exploratory", "validation", "frame_loss"))
        and result["v3_paired_intervals"]["categorical_mismatches"] == 0
        and result["v3_paired_intervals"]["rounded_mismatches"] == 0
        and all(v < 1e-9 for v in result["frame_loss_rates"]["max_absolute_change"].values())
        and all(v < 1e-7 for v in result["lab"]["max_absolute_change"].values())
        and result["attribution_counts_match"] and all(result["revision_scientific_tables_match"].values())
        and not result["paper_tex_mismatches"])
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["all_checks_agree"]:
        raise ValueError("Replay differs on published scientific checks; inspect comparison.json including timing changes.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    compare(parser.parse_args().tag)
