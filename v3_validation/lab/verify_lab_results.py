"""Independent saved-hardware-result audit; never fits or changes the models."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message):
    check(math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-8), message)


def recursive_prediction(times, commands, initial, fit):
    """Independent exact state propagation; no step-response implementation import."""
    heater, sensor = initial, initial
    output = [initial]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        h = fit["tau_h_s"]
        s = fit["tau_sensor_s"]
        equilibrium = fit["ambient_c"] + fit["gain_full_scale_c"] * commands[index - 1]
        dh = math.exp(-dt / h)
        if s is None:
            heater = equilibrium + (heater - equilibrium) * dh
            output.append(heater)
        else:
            ds = math.exp(-dt / s)
            factor = (dt / h) * dh if abs(h - s) < 1e-9 else h * (dh - ds) / (h - s)
            sensor = equilibrium + (sensor - equilibrium) * ds + (heater - equilibrium) * factor
            heater = equilibrium + (heater - equilibrium) * dh
            output.append(sensor)
    return np.asarray(output)


def interval(initial, elapsed, on, cfg):
    heater = (cfg["heater_rise_min_c"], cfg["heater_rise_max_c"]) if on else (0.,)
    values = []
    for t0, ambient, tau, rise in itertools.product(
            (initial - cfg["temperature_noise_bound_c"], initial + cfg["temperature_noise_bound_c"]),
            (cfg["ambient_min_c"], cfg["ambient_max_c"]),
            (cfg["tau_min_s"], cfg["tau_max_s"]), heater):
        eq = ambient + rise
        values.append(eq + (t0 - eq) * math.exp(-elapsed / tau))
    return min(values), max(values)


def audit(run):
    source = ROOT / "sources/lab"
    manifest = load(source / "manifest.json")
    freeze, calibration, results = (load(run / name) for name in ("freeze.json", "calibration.json", "results.json"))
    check(results["status"] == "completed", "Validation run incomplete")
    for item in manifest["files"]:
        check(sha(source / item["path"]) == item["sha256"], "Source hash mismatch")
    check(freeze["protocol_sha256"] == sha(ROOT / "lab/protocol-lab.json"), "Protocol changed after freeze")
    check(freeze["source_manifest_sha256"] == sha(source / "manifest.json"), "Manifest changed")
    for relative, expected in freeze["implementation_sha256"].items():
        check(sha(ROOT.parent / relative) == expected, "Frozen implementation changed: " + relative)
    check(results["freeze_sha256"] == sha(run / "freeze.json"), "Result/freeze mismatch")
    check(results["calibration_sha256"] == sha(run / "calibration.json"), "Calibration artifact changed")
    raw_path = source / manifest["intended_record"]
    check(freeze["raw_sha256"] == results["raw_sha256"] == sha(raw_path), "Raw CSV changed")
    raw = np.genfromtxt(raw_path, delimiter=",", names=True)
    predicted = np.genfromtxt(run / "predictions.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    check(len(raw) == len(predicted) == 600, "Wrong prediction/raw record count")
    t, q, measured = raw["Time"], raw["H1"] / 100., raw["T1"]
    check(np.array_equal(t, predicted["time_s"]), "Prediction timestamps not raw timestamps")
    check(np.array_equal(measured, predicted["measured_t1_c"]), "Measurements altered")
    check(np.array_equal(raw["T2"], predicted["measured_t2_c"]), "Second sensor altered")
    check(np.array_equal(raw["H1"], predicted["heater1_percent"]) and np.all(raw["H2"] == 0.), "Command mismatch")
    cal, held = t < 400., t >= 400.
    check(np.sum(cal) == results["calibration_rows"] == 400 and np.sum(held) == results["held_out_rows"] == 200,
          "Held-out split changed")
    check(np.all(predicted["split"][cal] == "calibration") and np.all(predicted["split"][held] == "held_out"),
          "CSV split labels incorrect")
    off = [i for i in range(1, len(q)) if q[i - 1] > 0. and q[i] == 0.]
    held_off = [i for i in off if t[i] >= 400.]
    check(len(off) == 3 and len(held_off) == results["independent_held_out_off_transitions"] == 1, "Wrong OFF count")
    ninitial = next(i for i, command in enumerate(q) if command > 0.)
    initial = float(np.mean(measured[:ninitial]))
    close(initial, calibration["initial_temperature_c"], "Initial-state statistic mismatch")
    computed_metrics = {}
    recurrence_max_diff = {}
    for name, column in (("first_order", "first_order_c"), ("sensor_lag", "sensor_lag_c")):
        fit = calibration[name]
        recursive = recursive_prediction(t, q, initial, fit)
        difference = float(np.max(np.abs(recursive - predicted[column])))
        check(difference < 1e-8, "Free-run prediction/state continuation mismatch")
        recurrence_max_diff[name] = difference
        # Independently recover A/G from calibration only at the frozen poles.
        base = recursive_prediction(t[cal], q[cal], initial, {**fit, "ambient_c": 0., "gain_full_scale_c": 0.})
        ambient = recursive_prediction(t[cal], q[cal], 0., {**fit, "ambient_c": 1., "gain_full_scale_c": 0.})
        gain = recursive_prediction(t[cal], q[cal], 0., {**fit, "ambient_c": 0., "gain_full_scale_c": 1.})
        recovered = np.linalg.lstsq(np.column_stack((ambient, gain)), measured[cal] - base, rcond=None)[0]
        close(recovered[0], fit["ambient_c"], "A not obtained from calibration-only least squares")
        close(recovered[1], fit["gain_full_scale_c"], "G not obtained from calibration-only least squares")
        computed_metrics[name] = {}
        calibration_residual = measured[cal] - recursive[cal]
        width = float(np.max(np.abs(calibration_residual))) + .1
        close(np.sum(calibration_residual ** 2), fit["calibration_sse"], "Calibration SSE mismatch")
        for split, mask in (("calibration", cal), ("held_out", held)):
            residual = measured[mask] - recursive[mask]
            values = dict(n=int(np.sum(mask)), rmse_c=float(np.sqrt(np.mean(residual ** 2))),
                          mae_c=float(np.mean(np.abs(residual))), max_abs_error_c=float(np.max(np.abs(residual))),
                          empirical_envelope_half_width_c=width,
                          empirical_envelope_coverage=float(np.mean(np.abs(residual) <= width)))
            for key, value in values.items():
                close(value, results["metrics"][name][split][key], f"Metric mismatch: {name}/{split}/{key}")
            computed_metrics[name][split] = values
    with (run / "calibration-grid.csv").open(encoding="utf-8", newline="") as handle:
        grid = list(csv.DictReader(handle))
    for name in ("first_order", "sensor_lag"):
        candidates = [r for r in grid if r["model"] == name]
        best = min(candidates, key=lambda r: float(r["calibration_sse"]))
        close(best["calibration_sse"], calibration[name]["calibration_sse"], "Model selected by a nonminimal calibration SSE")
    accepted = [r for r in grid if r["model"] == "first_order" and
                float(r["calibration_sse"]) <= calibration["first_order"]["calibration_sse"] * 1.1]
    check(len(accepted) == calibration["lab_first_order_envelope"]["accepted_models"], "Changed envelope acceptance threshold")
    cfg = results["lab_detector_configuration"]
    check(cfg["min_pair_interval_s"] == 5. and cfg["max_pair_interval_s"] == 80., "Changed diagnostic temporal bounds")
    index = held_off[0]
    samples = [(float(tt), float(yy)) for tt, yy in zip(t[index:], measured[index:])]
    pairs, counts = [], dict(eligible_pairs=0, off_only=0, retained_power_only=0, both=0, neither=0)
    earliest_nominal, earliest_fault = None, None
    for i, (first_t, first_c) in enumerate(samples):
        for second_t, second_c in samples[i + 1:]:
            dt = second_t - first_t
            if not 5. <= dt <= 80.:
                continue
            possibilities = []
            for on in (False, True):
                low, high = interval(first_c, dt, on, cfg)
                possibilities.append(second_c - cfg["temperature_noise_bound_c"] <= high + 1e-5
                                     and second_c + cfg["temperature_noise_bound_c"] >= low - 1e-5)
            a, b = possibilities
            counts["eligible_pairs"] += 1
            counts["off_only" if a and not b else "retained_power_only" if b and not a else "both" if a else "neither"] += 1
            pairs.append((first_t, second_t, int(a), int(b)))
            if a and not b:
                earliest_nominal = second_t if earliest_nominal is None else min(earliest_nominal, second_t)
            if b and not a:
                earliest_fault = second_t if earliest_fault is None else min(earliest_fault, second_t)
    with (run / "heldout-pair-compatibility.csv").open(encoding="utf-8", newline="") as handle:
        saved_pairs = [(float(r["first_s"]), float(r["second_s"]), int(r["off_compatible"]),
                        int(r["retained_power_compatible"])) for r in csv.DictReader(handle)]
    check(pairs == saved_pairs, "Independent scalar pair compatibility differs")
    adapted = next(r for r in results["diagnostic_outcomes"] if r["configuration"] == "lab_adapted_first_order")
    for key, value in counts.items():
        check(adapted[key] == value, "Pair count mismatch")
    check(earliest_nominal is not None and earliest_fault is None, "Reported nominal outcome unsupported")
    close(earliest_nominal - t[index], adapted["first_support_after_off_s"], "Incorrect earliest certificate time")
    certificates = load(run / "certificates.json")["lab_adapted_first_order"]
    check({c["label"] for c in certificates} == {"nominal"}, "Certificate conflict omitted")
    close(min(c["as_of_s"] for c in certificates), earliest_nominal, "Certificate timestamp mismatch")
    for direct in results["diagnostic_outcomes"]:
        if direct["configuration"].startswith("unchanged_v2_"):
            check(direct["eligible_pairs"] == 0 and direct["final_label"] == "insufficient", "Incorrect direct-transfer report")
    inputs = [raw_path, run / "freeze.json", run / "calibration.json", run / "results.json",
              run / "predictions.csv", run / "heldout-pair-compatibility.csv", run / "calibration-grid.csv"]
    return dict(status="passed", created_utc=datetime.now(timezone.utc).isoformat(),
                scope="Independent recursive predictions, calibration-only A/G recovery, scalar pair checks, saved metrics and frozen hashes; no new fitting or held-out retuning",
                audit_script_sha256=sha(__file__),
                input_sha256={str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in inputs},
                raw_records=600, calibration_records=400, held_out_records=200, held_out_off_transitions=1,
                recursive_prediction_max_difference_c=recurrence_max_diff,
                metrics=computed_metrics, pair_counts=counts,
                first_nominal_compatible_after_off_s=float(earliest_nominal - t[index]),
                real_fault_labels_available=False, current_measurements_available=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="lab/main")
    parser.add_argument("--output", default="lab/main/audit.json")
    args = parser.parse_args()
    run, output = (ROOT / args.run).resolve(), (ROOT / args.output).resolve()
    check(run.is_relative_to(ROOT / "lab") and output.is_relative_to(ROOT / "lab"), "Audit paths outside lab")
    if output.exists():
        raise SystemExit("Audit output exists; choose a new --output for a replay")
    result = audit(run)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "records": result["raw_records"],
                      "pairs": result["pair_counts"]["eligible_pairs"], "output": str(output)}))


if __name__ == "__main__":
    main()
