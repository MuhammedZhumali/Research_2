"""Frozen calibration/held-out check on a published physical TCLab trace.

No simulated fault is added. This checks predictions and consistency with one
commanded OFF, not fault recall or independently certified hardware health.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import csv
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT.parent / "v2"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V2 / "src"))
from context_telemetry_v2.model import Measurement, CommandEvent
from context_telemetry_v2.diagnosis import DiagnosisConfig, diagnosis_config, diagnose, latest_outcomes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def survival(time_s, tau_h, tau_s=None):
    """Homogeneous response with both initial states equal; vectorized models."""
    t = np.asarray(time_s, dtype=float)[None, :]
    h = np.asarray(tau_h, dtype=float)[:, None]
    if tau_s is None:
        return np.exp(-np.maximum(t, 0.) / h)
    s = np.asarray(tau_s, dtype=float)[:, None]
    close = np.abs(h - s) < 1e-9 * np.maximum(h, s)
    denominator = np.where(close, 1., h - s)
    ordinary = (h * np.exp(-np.maximum(t, 0.) / h) - s * np.exp(-np.maximum(t, 0.) / s)) / denominator
    repeated = np.exp(-np.maximum(t, 0.) / h) * (1. + np.maximum(t, 0.) / h)
    return np.where(close, repeated, ordinary)


def model_basis(time_s, command_fraction, tau_h, tau_s=None):
    """Exact causal ZOH model basis. Later commands have zero earlier response."""
    t = np.asarray(time_s, dtype=float)
    q = np.asarray(command_fraction, dtype=float)
    if len(t) != len(q) or np.any(np.diff(t) <= 0):
        raise ValueError("Time/command arrays must align and increase")
    initial = survival(t - t[0], tau_h, tau_s)
    ambient = 1. - initial
    gain = np.zeros_like(initial)
    changes = np.r_[q[0], np.diff(q)]
    for index in np.flatnonzero(changes):
        elapsed = t - t[index]
        response = np.where(elapsed[None, :] >= 0., 1. - survival(elapsed, tau_h, tau_s), 0.)
        gain += changes[index] * response
    return initial, ambient, gain


def grid_fit(time_s, q, measured, initial_c, tau_h, tau_s, ambient_bounds, gain_bounds):
    """Independent A/G least squares at each grid point; fit only supplied rows."""
    records = []
    tau_h = np.asarray(tau_h)
    tau_s = None if tau_s is None else np.asarray(tau_s)
    for start in range(0, len(tau_h), 256):
        h = tau_h[start:start + 256]
        s = None if tau_s is None else tau_s[start:start + 256]
        ic, a, g = model_basis(time_s, q, h, s)
        response = np.asarray(measured)[None, :] - initial_c * ic
        aa, ag, gg = np.sum(a * a, axis=1), np.sum(a * g, axis=1), np.sum(g * g, axis=1)
        ay, gy = np.sum(a * response, axis=1), np.sum(g * response, axis=1)
        determinant = aa * gg - ag * ag
        for offset in range(len(h)):
            if determinant[offset] <= 1e-12:
                continue
            ambient = (ay[offset] * gg[offset] - ag[offset] * gy[offset]) / determinant[offset]
            gain = (aa[offset] * gy[offset] - ag[offset] * ay[offset]) / determinant[offset]
            if not (ambient_bounds[0] <= ambient <= ambient_bounds[1] and gain_bounds[0] <= gain <= gain_bounds[1]):
                continue
            prediction = initial_c * ic[offset] + ambient * a[offset] + gain * g[offset]
            residual = np.asarray(measured) - prediction
            records.append(dict(tau_h_s=float(h[offset]), tau_sensor_s=None if s is None else float(s[offset]),
                                ambient_c=float(ambient), gain_full_scale_c=float(gain),
                                calibration_sse=float(np.sum(residual ** 2)),
                                calibration_max_abs_error_c=float(np.max(np.abs(residual)))))
    if not records:
        raise RuntimeError("No grid model meets the registered physical bounds; no post-test expansion allowed")
    return sorted(records, key=lambda r: (r["calibration_sse"], r["tau_h_s"], r["tau_sensor_s"] or 0.))


def predict(time_s, q, initial_c, fit):
    sensor = None if fit["tau_sensor_s"] is None else [fit["tau_sensor_s"]]
    ic, ambient, gain = model_basis(time_s, q, [fit["tau_h_s"]], sensor)
    return (initial_c * ic + fit["ambient_c"] * ambient + fit["gain_full_scale_c"] * gain)[0]


def error_metrics(measured, prediction, envelope):
    residual = np.asarray(measured) - np.asarray(prediction)
    return dict(n=int(len(residual)), rmse_c=float(np.sqrt(np.mean(residual ** 2))),
                mae_c=float(np.mean(np.abs(residual))), max_abs_error_c=float(np.max(np.abs(residual))),
                empirical_envelope_half_width_c=float(envelope),
                empirical_envelope_coverage=float(np.mean(np.abs(residual) <= envelope)))


def registered_grid(description):
    return np.geomspace(description["minimum"], description["maximum"], description["points"])


def pair_audit(points, config):
    times = np.asarray([m.time_s for m in points])
    temperature = np.asarray([m.temperature_c for m in points])
    i, j = np.triu_indices(len(points), 1)
    elapsed = times[j] - times[i]
    use = (elapsed >= config.min_pair_interval_s) & (elapsed <= config.max_pair_interval_s)
    i, j, elapsed = i[use], j[use], elapsed[use]
    flags = []
    for on in (False, True):
        eq_lo = config.ambient_min_c + (config.heater_rise_min_c if on else 0.)
        eq_hi = config.ambient_max_c + (config.heater_rise_max_c if on else 0.)
        lo, hi = temperature[i] - config.temperature_noise_bound_c, temperature[i] + config.temperature_noise_bound_c
        d1, d2 = np.exp(-elapsed / config.tau_min_s), np.exp(-elapsed / config.tau_max_s)
        lower = np.minimum(eq_lo + (lo - eq_lo) * d1, eq_lo + (lo - eq_lo) * d2)
        upper = np.maximum(eq_hi + (hi - eq_hi) * d1, eq_hi + (hi - eq_hi) * d2)
        flags.append((temperature[j] - config.temperature_noise_bound_c <= upper + 1e-5)
                     & (temperature[j] + config.temperature_noise_bound_c >= lower - 1e-5))
    off, retained_on = flags
    details = [dict(first_s=float(times[a]), second_s=float(times[b]), elapsed_s=float(dt),
                    off_compatible=int(x), retained_power_compatible=int(y))
               for a, b, dt, x, y in zip(i, j, elapsed, off, retained_on)]
    count = len(i)
    return dict(eligible_pairs=count, off_only=int(np.sum(off & ~retained_on)),
                retained_power_only=int(np.sum(retained_on & ~off)), both=int(np.sum(off & retained_on)),
                neither=int(np.sum(~off & ~retained_on))), details


def diagnostic_record(name, points, event, config):
    certs = diagnose(points, (event,), config=config, decision_time_s=points[-1].time_s)
    labels = latest_outcomes(certs)
    first = certs[0] if certs else None
    counts, pairs = pair_audit(points, config)
    row = dict(configuration=name, off_start_s=event.time_s,
               observed_off_duration_s=points[-1].time_s - event.time_s,
               first_label=first.label if first else "insufficient",
               first_support_after_off_s=first.as_of_s - event.time_s if first else None,
               final_label=labels.get(event.epoch, "insufficient"),
               opposing_certificates=int(len({c.label for c in certs}) > 1),
               **counts)
    return row, [asdict(c) for c in certs], pairs


def run_validation(output):
    protocol_path = ROOT / "lab/protocol-lab.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    source = ROOT / "sources/lab"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        if sha(source / item["path"]) != item["sha256"]:
            raise AssertionError("Source snapshot changed: " + item["path"])
    raw_path = source / manifest["intended_record"]
    code_paths = [Path(__file__), ROOT / "tests/test_lab.py",
                  V2 / "src/context_telemetry_v2/model.py", V2 / "src/context_telemetry_v2/diagnosis.py"]
    freeze = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage="before any fitting",
                  protocol_sha256=sha(protocol_path), source_manifest_sha256=sha(source / "manifest.json"),
                  raw_sha256=sha(raw_path), source_commit=manifest["commit"],
                  implementation_sha256={str(path.relative_to(ROOT.parent)).replace("\\", "/"): sha(path) for path in code_paths},
                  numpy_version=np.__version__, python=sys.version)
    # The freeze is written before fitting and records the exact registered plan.
    write_json(output / "freeze.json", freeze)
    data = np.genfromtxt(raw_path, delimiter=",", names=True)
    time_s, q, measured = data["Time"], data["H1"] / 100., data["T1"]
    if len(data) != 600 or not np.all(data["H2"] == 0.) or np.any(np.diff(time_s) <= 0):
        raise AssertionError("Hardware record does not match the registered selection")
    calibration, held_out = time_s < 400., time_s >= 400.
    first_nonzero = int(np.flatnonzero(q > 0.)[0])
    initial = float(np.mean(measured[:first_nonzero]))
    bounds = (initial - 5., initial + 5.)
    model_plan = protocol["models"]
    first_fits = grid_fit(time_s[calibration], q[calibration], measured[calibration], initial,
                          registered_grid(model_plan["first_order_tau_grid_s"]), None,
                          bounds, model_plan["gain_bounds_C_at_100_percent"])
    heater, sensor = np.meshgrid(registered_grid(model_plan["sensor_lag_tau_h_grid_s"]),
                                registered_grid(model_plan["sensor_lag_tau_s_grid_s"]), indexing="ij")
    ordered = heater >= sensor
    lag_fits = grid_fit(time_s[calibration], q[calibration], measured[calibration], initial,
                        heater[ordered], sensor[ordered], bounds, model_plan["gain_bounds_C_at_100_percent"])
    accepted = [fit for fit in first_fits if fit["calibration_sse"] <= first_fits[0]["calibration_sse"] * 1.10]
    envelope = dict(ambient_min_c=min(f["ambient_c"] for f in accepted),
                    ambient_max_c=max(f["ambient_c"] for f in accepted),
                    tau_min_s=min(f["tau_h_s"] for f in accepted), tau_max_s=max(f["tau_h_s"] for f in accepted),
                    gain_min_c=min(f["gain_full_scale_c"] for f in accepted),
                    gain_max_c=max(f["gain_full_scale_c"] for f in accepted),
                    temperature_discrepancy_bound_c=max(f["calibration_max_abs_error_c"] for f in accepted) + .1,
                    accepted_models=len(accepted))
    calibration_artifact = dict(initial_temperature_c=initial, initial_rows=first_nonzero,
                                 calibration_rows=int(np.sum(calibration)), held_out_rows=int(np.sum(held_out)),
                                 first_order=first_fits[0], sensor_lag=lag_fits[0],
                                 lab_first_order_envelope=envelope,
                                 label="Calibration-only fit and empirical discrepancy, frozen before held-out scoring")
    write_json(output / "calibration.json", calibration_artifact)
    write_csv(output / "calibration-grid.csv", [dict(model=model, **fit)
              for model, fits in (("first_order", first_fits), ("sensor_lag", lag_fits)) for fit in fits])
    predictions, metrics = {}, {}
    for name, fit in (("first_order", first_fits[0]), ("sensor_lag", lag_fits[0])):
        predicted = predict(time_s, q, initial, fit)
        predictions[name] = predicted
        tolerance = fit["calibration_max_abs_error_c"] + .1
        metrics[name] = dict(calibration=error_metrics(measured[calibration], predicted[calibration], tolerance),
                             held_out=error_metrics(measured[held_out], predicted[held_out], tolerance))
    # One held-out OFF: command identity is public, independently labelled health is absent.
    off_indices = [i for i in range(1, len(q)) if q[i - 1] > 0 and q[i] == 0.]
    held_out_off = [i for i in off_indices if time_s[i] >= 400.]
    if len(held_out_off) != 1:
        raise AssertionError("Registered single held-out OFF not found")
    index = held_out_off[0]
    next_nonzero = next((j for j in range(index + 1, len(q)) if q[j] != 0.), len(q))
    selected = np.arange(index, next_nonzero)
    event = CommandEvent(float(time_s[index]), False, 3984321741)
    points = tuple(Measurement(float(time_s[i]), float(measured[i]), None, event.epoch) for i in selected)
    diagnostics, certificate_archive, all_pairs = [], {}, []
    for profile in ("clear_current", "slow_sparse_current", "thermal_only"):
        name = "unchanged_v2_" + profile
        row, certs, pairs = diagnostic_record(name, points, event, diagnosis_config(profile))
        diagnostics.append(row)
        certificate_archive[name] = certs
    lab_config = DiagnosisConfig(
        envelope["ambient_min_c"], envelope["ambient_max_c"], envelope["tau_min_s"], envelope["tau_max_s"],
        envelope["gain_min_c"] * float(q[index - 1]), envelope["gain_max_c"] * float(q[index - 1]),
        envelope["temperature_discrepancy_bound_c"], min_pair_interval_s=5., max_pair_interval_s=80.)
    row, certs, pairs = diagnostic_record("lab_adapted_first_order", points, event, lab_config)
    diagnostics.append(row)
    certificate_archive["lab_adapted_first_order"] = certs
    all_pairs.extend(pairs)
    raw = tuple(Measurement(m.time_s, m.temperature_c, None, -1) for m in points)
    no_context = diagnose(raw, (), config=lab_config, decision_time_s=raw[-1].time_s)
    peak_index = int(selected[np.argmax(measured[selected])])
    results = dict(
        status="completed", source_commit=manifest["commit"], raw_sha256=sha(raw_path),
        protocol_sha256=sha(protocol_path), freeze_sha256=sha(output / "freeze.json"),
        calibration_sha256=sha(output / "calibration.json"),
        calibration_rows=int(np.sum(calibration)), held_out_rows=int(np.sum(held_out)),
        independent_held_out_off_transitions=1,
        metrics=metrics, diagnostic_outcomes=diagnostics,
        lab_detector_configuration=asdict(lab_config),
        raw_without_context_certificates=len(no_context),
        commanded_off_observation=dict(start_s=event.time_s, last_observation_s=float(time_s[selected[-1]]),
            prior_power_percent=float(q[index - 1] * 100), current_measurements_available=False,
            independently_labelled_fault=False,
            observed_peak_after_off_s=float(time_s[peak_index] - event.time_s),
            observed_rise_after_off_c=float(measured[peak_index] - measured[index])),
        interpretation=[
            "Hardware temperatures and logged commands; no artificial faults were introduced.",
            "Held-out predictions are free-run continuations; no state or parameter reset at the split.",
            "First-order and sensor-lag models were each selected only by calibration SSE.",
            "Lab-adapted intervals are calibration discrepancy envelopes, not physical sensor specifications or confidence intervals.",
            "Nominal means compatible with commanded OFF in this experiment, not independently certified health.",
            "One held-out transition cannot validate fault rates, stuck-ON recall, current rules or KazSTSAT hardware.",
        ],
    )
    write_json(output / "results.json", results)
    write_json(output / "certificates.json", certificate_archive)
    write_csv(output / "diagnostic-outcomes.csv", diagnostics)
    write_csv(output / "heldout-pair-compatibility.csv", all_pairs)
    write_csv(output / "prediction-metrics.csv", [dict(model=name, split=split, **value)
              for name, splits in metrics.items() for split, value in splits.items()])
    write_csv(output / "predictions.csv", [dict(time_s=float(t), split="calibration" if calibration[i] else "held_out",
              heater1_percent=float(q[i] * 100), heater2_percent=float(data["H2"][i]), measured_t1_c=float(measured[i]),
              measured_t2_c=float(data["T2"][i]), first_order_c=float(predictions["first_order"][i]),
              sensor_lag_c=float(predictions["sensor_lag"][i])) for i, t in enumerate(time_s)])
    plot_results(output, time_s, q, measured, predictions, metrics, event.time_s)
    return results


def plot_results(output, time_s, q, measured, predictions, metrics, off_time):
    # Scientific plots use only the measured trace and computed model arrays.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 8.), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    axes[0].plot(time_s, measured, color="black", lw=1., label="Measured hardware sensor T1")
    colors = {"first_order": "#d95f02", "sensor_lag": "#0072b2"}
    for name, prediction in predictions.items():
        label = "First-order fit" if name == "first_order" else "Two-state sensor-lag fit"
        axes[0].plot(time_s, prediction, color=colors[name], lw=1.5, label=label)
        residual = measured - prediction
        axes[1].plot(time_s, residual, color=colors[name], lw=1., label=label)
        width = metrics[name]["calibration"]["empirical_envelope_half_width_c"]
        axes[1].axhline(width, color=colors[name], ls=":", alpha=.7)
        axes[1].axhline(-width, color=colors[name], ls=":", alpha=.7)
    axes[2].step(time_s, q * 100., where="post", color="#3b6d2b", label="Logged H1 command")
    for axis in axes:
        axis.axvspan(400., time_s[-1], color="#b3cde3", alpha=.23)
        axis.axvline(400., color="#555555", ls="--", lw=1)
        axis.axvline(off_time, color="#555555", ls=":", lw=1)
        axis.grid(alpha=.2)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel("Temperature (°C)")
    axes[1].set_ylabel("Measured − model (°C)")
    axes[2].set_ylabel("Heater 1 (%)")
    axes[2].set_xlabel("Logged elapsed time (s); blue region is held out")
    axes[0].set_title("TCLab hardware: calibration before 400 s; one held-out OFF")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / ("hardware-prediction." + extension), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="lab/main")
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not output.is_relative_to((ROOT / "lab").resolve()):
        raise SystemExit("Output must stay inside lab/")
    if output.exists():
        raise SystemExit("Output already exists; choose a separate --output for replay")
    output.mkdir(parents=True)
    result = run_validation(output)
    print(json.dumps(dict(status=result["status"], metrics=result["metrics"],
                         diagnostic_outcomes=result["diagnostic_outcomes"], output=str(output))))


if __name__ == "__main__":
    main()
