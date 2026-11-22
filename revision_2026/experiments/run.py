"""Replay the registered composition and sensor-lag series, without editing the frozen pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
V2 = PILOT / "v2"
sys.path.insert(0, str(V2 / "scripts"))
import run_v2 as frozen
from context_telemetry_v2 import policies, wire

SPEC = json.loads((HERE / "protocol.json").read_text())
SETTINGS = json.loads((V2 / "configs/protocol-v2.json").read_text())["delivery"]
ORBITS = frozen.load_orbits()


def fingerprint():
    paths = [HERE / name for name in ("protocol.json", "run.py", "analyze.py")]
    paths += [V2 / "scripts/run_v2.py", V2 / "configs/protocol-v2.json",
              PILOT / "data/contacts/kazstsat.csv", PILOT / "requirements-lock.txt"]
    paths.extend(sorted((V2 / "src/context_telemetry_v2").glob("*.py")))
    return {p.relative_to(PILOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def sensor_episode(ep, tau_s):
    if tau_s == 0:
        return frozen.quantized(ep)
    observations = []
    previous = ep.truth[0]
    state = previous.temperature_c
    for observed, physical in zip(ep.measurements, ep.truth, strict=True):
        dt = physical.time_s - previous.time_s
        if dt:
            response = -math.expm1(-dt / tau_s)
            slope = (physical.temperature_c - previous.temperature_c) / dt
            state += (previous.temperature_c - state) * response + slope * (dt - tau_s * response)
        noise = observed.temperature_c - physical.temperature_c
        observations.append(replace(observed, temperature_c=state + noise))
        previous = physical
    return frozen.quantized(replace(ep, measurements=tuple(observations)))


def shorter_buffer(public, method):
    if method == "full_summary":
        return public
    if method not in {"endpoints", "endpoints_current"}:
        raise ValueError(method)
    chosen = []
    for event in public.contexts:
        if event.heater_on:
            continue
        group = [m for m in public.measurements if m.command_epoch == event.epoch]
        if not group:
            continue
        records = [group[0], group[-1]]
        if method == "endpoints_current":
            current = next((m for m in group if m.current_a is not None), None)
            if current is not None:
                records.append(current)
        chosen.extend(sorted(set(records), key=lambda m: m.time_s))
    # Running the unchanged selector on this restricted group preserves its packing and
    # first/current/latest priority: the time-nearest searches cannot introduce new records.
    return policies.PublicBuffer(tuple(chosen), public.contexts, public.cutoff_s)


def simulate(ep, contacts, detector, rate, method):
    received, known_events = {}, {}
    measurements = ep.measurements
    times = [m.time_s for m in measurements]
    decisions = []
    sent = seq = 0
    timing = dict(planner_max_s=0., detector_max_s=0., planner_over=0, detector_over=0)
    plan_s = SETTINGS["planner_reservation_s"]
    ground_s = SETTINGS["ground_diagnosis_reservation_s"]
    for contact in contacts:
        begin, end = contact["start_s"], contact["end_s"]
        public = policies.PublicBuffer(measurements[:bisect_right(times, begin)],
                                      tuple(e for e in ep.commands if e.time_s <= begin), begin)
        if method == "contact_history":
            points, events = public.measurements, public.contexts
        else:
            budget = max(0, math.floor((end - begin - plan_s) * rate / 8))
            tick = time.perf_counter()
            view = shorter_buffer(public, method)
            frames = policies.plan("snapshot_events", view, received, known_events, budget, SETTINGS, seq)
            elapsed = time.perf_counter() - tick
            timing["planner_max_s"] = max(timing["planner_max_s"], elapsed)
            timing["planner_over"] += int(elapsed > plan_s)
            contact_bytes = 0
            for frame in frames:
                assert frame.sequence == seq
                assert all(m.time_s <= begin and m.time_s not in received for m in frame.measurements)
                assert all(e.time_s <= begin for e in frame.contexts)
                contact_bytes += frame.wire_bytes
                ack = wire.encode_ack(seq, begin + plan_s + contact_bytes * 8 / rate)
                assert len(frame.blob) + len(ack) == frame.wire_bytes
                received.update((m.time_s, m) for m in frame.measurements)
                known_events.update((e.epoch, e) for e in frame.contexts)
                seq += 1
            assert contact_bytes <= budget
            sent += contact_bytes
            points, events = tuple(received.values()), tuple(known_events.values())
        tick = time.perf_counter()
        cert = frozen.diagnose(points, events, config=detector, decision_time_s=end)
        elapsed = time.perf_counter() - tick
        timing["detector_max_s"] = max(timing["detector_max_s"], elapsed)
        timing["detector_over"] += int(elapsed > ground_s)
        decisions.append(dict(time_s=end + ground_s, certificates=[asdict(x) for x in cert]))
    return decisions, sent, seq, timing


def score(target, decisions):
    evidence = frozen.target_decisions(target, decisions)
    labels = [entry["label"] for _, entry in evidence]
    first = next((d["time_s"] for d, v in evidence if v["label"] == target.label), None)
    return dict(target_epoch=target.command_epoch, truth=target.label, start_s=target.start_s,
                deadline_s=target.deadline_s, first_correct_s=first, correct=int(first is not None),
                ever_wrong=int(any(x not in (target.label, "insufficient") for x in labels)),
                no_decision=int(all(x == "insufficient" for x in labels)),
                final_correct=int(bool(labels) and labels[-1] == target.label),
                conflict=int(any(v["conflicting"] for _, v in evidence)),
                capped_delay_s=(target.deadline_s if first is None else first)-target.start_s)


def continuous_rows(ep, detector):
    counts = []
    for t, data in zip(ep.targets, frozen.continuous_oracle(ep, detector), strict=True):
        first = data["first_correct_time_s"]
        counts.append(dict(target_epoch=t.command_epoch, truth=t.label, start_s=t.start_s, deadline_s=t.deadline_s,
                           first_correct_s=first, correct=int(first is not None), ever_wrong=data["ever_wrong"],
                           no_decision=int(data["first_label"] == "insufficient"),
                           final_correct=int(data["final_label"] == t.label), conflict=data["conflicting"],
                           capped_delay_s=(t.deadline_s if first is None else first)-t.start_s))
    return counts


def episode_run(spec):
    profile, index, seed = spec
    faulty = bool(index % 2)
    raw = frozen.generate_episode(seed, profile=profile, faulty=faulty)
    contacts, offset = frozen.contacts_for(SPEC["schedule"], seed, raw.duration_s, ORBITS)
    detector = frozen.diagnosis_config(profile)
    runs, targets, timings = [], [], []
    keys = dict(profile=profile, episode_id=f"{profile}-{index:03d}", seed=seed, faulty_episode=int(faulty))
    for lag in (0, SPEC["lag_s"]):
        ep = sensor_episode(raw, lag)
        rates = SPEC["composition_rates_bps"] if not lag else [SPEC["primary_rate_bps"]]
        for rate in rates:
            methods = list(SPEC["methods"])
            if rate == SPEC["primary_rate_bps"]:
                methods += ["contact_history", "continuous_history"]
            method_rows = {}
            for method in methods:
                key = dict(**keys, lag_s=lag, rate_bps=rate, method=method)
                sent = frames = None
                if method == "continuous_history":
                    scored = continuous_rows(ep, detector)
                else:
                    decisions, sent, frames, timing = simulate(ep, contacts, detector, rate, method)
                    scored = [score(t, decisions) for t in ep.targets]
                    timings.append(dict(**key, **timing))
                    if method == "contact_history":
                        sent = frames = None
                    if method == "full_summary" and index == 0 and lag == 0:
                        old = frozen.simulate(policies.PublicBuffer(ep.measurements, ep.commands, ep.duration_s),
                                              contacts, "snapshot_events", rate, detector, SETTINGS)
                        assert old["wire_bytes"] == sent and old["frames_sent"] == frames
                        assert [d["certificates"] for d in old["_decisions"]] == [d["certificates"] for d in decisions]
                        assert old["planner_overruns"] == old["diagnosis_overruns"] == 0
                        assert [score(t, old["_decisions"]) for t in ep.targets] == scored
                method_rows[method] = (scored, sent)
                runs.append(dict(**key, offset_s=offset, n_contacts=len(contacts), frames=frames, wire_bytes=sent))
                targets.extend(dict(**key, **row) for row in scored)
            if profile == "thermal_only":
                assert method_rows["endpoints"] == method_rows["endpoints_current"]
    return runs, targets, timings


def run(out, smoke=False, workers=1):
    frozen_record = json.loads((HERE / "freeze.json").read_text())
    if fingerprint() != frozen_record["sha256"]:
        raise ValueError("Study inputs have changed since the revision freeze.")
    count = 2 if smoke else SPEC["episodes_per_profile"]
    base = 61000000 if smoke else SPEC["seed_base"]
    specs = [(profile, i, base + pi * SPEC["seed_stride"] + i)
             for pi, profile in enumerate(SPEC["profiles"]) for i in range(count)]
    out.mkdir(parents=True, exist_ok=False)
    tables = {name: (out / f"{name}.csv").open("w", encoding="utf-8", newline="") for name in
              ("runs", "targets", "timings")}
    writers = {}
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, rows in enumerate(pool.map(episode_run, specs, chunksize=2), 1):
                for (name, stream), block in zip(tables.items(), rows, strict=True):
                    if name not in writers:
                        writers[name] = csv.DictWriter(stream, fieldnames=list(block[0]), lineterminator="\n")
                        writers[name].writeheader()
                    writers[name].writerows(block)
                if i % 60 == 0:
                    print(f"{i}/{len(specs)} episodes", flush=True)
    finally:
        for stream in tables.values():
            stream.close()
    (out / "completed.json").write_text(json.dumps(dict(episodes=len(specs), smoke=smoke,
           frozen_full_summary_parity_cells=9, freeze_sha256=hashlib.sha256((HERE/"freeze.json").read_bytes()).hexdigest()),
           indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze", action="store_true", help="Register a design once, before new episodes.")
    args = parser.parse_args()
    if args.freeze:
        with (HERE / "freeze.json").open("x", encoding="utf-8") as stream:
            json.dump(dict(frozen_utc=datetime.now(timezone.utc).isoformat(), sha256=fingerprint()), stream, indent=2)
            stream.write("\n")
    elif args.output:
        run(args.output, args.smoke, args.workers)
    else:
        parser.error("Supply --output (a new directory) or --freeze.")
