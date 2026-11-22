"""Aggregate revision targets and form matched, stratified episode bootstrap intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "protocol.json").read_text())
METHODS = SPEC["methods"]
PROFILES = SPEC["profiles"]
KEY = ["profile", "episode_id", "faulty_episode", "lag_s", "rate_bps", "method"]
CELL = ["lag_s", "rate_bps", "method"]
SUM = ["targets", "correct", "ever_wrong", "no_decision", "final_correct", "conflict",
       "fault_targets", "fault_detected", "nominal_targets", "nominal_confirmed",
       "false_alarm_targets", "missed_fault_wrong_nominal"]
METRICS = ["correct_pct", "ever_wrong_pct", "no_decision_pct", "capped_delay_h", "bytes_day"]


def aggregate(episodes):
    result = []
    for profile, subset in [("all", episodes), *episodes.groupby("profile")]:
        for (lag, rate, method), cell in subset.groupby(CELL, sort=False):
            row = {name: int(cell[name].sum()) for name in SUM}
            means = {metric: cell[metric].mean() for metric in METRICS}
            result.append(dict(profile=profile, lag_s=lag, rate_bps=rate, method=method,
                               episodes=len(cell), **row, unresolved=row["targets"] - row["correct"], **means))
    return pd.DataFrame(result)


def comparisons(cells):
    for lag, rate in sorted({(lag, rate) for lag, rate, _ in cells}):
        for left, right in [(METHODS[1], METHODS[0]), (METHODS[2], METHODS[1]), (METHODS[2], METHODS[0])]:
            yield "composition", (lag, rate, left), (lag, rate, right)
    for method in [*METHODS, "contact_history", "continuous_history"]:
        yield "lag", (SPEC["lag_s"], SPEC["primary_rate_bps"], method), (0, SPEC["primary_rate_bps"], method)


def paired(episodes):
    cells = {key: cell.set_index(["profile", "faulty_episode", "episode_id"]).sort_index()
             for key, cell in episodes.groupby(CELL)}
    result = []
    for profile in ["all", *PROFILES]:
        full_index = next(iter(cells.values())).index
        idx = full_index if profile == "all" else full_index[full_index.get_level_values("profile") == profile]
        groups = idx.to_frame(index=False).groupby(["profile", "faulty_episode"]).indices.values()
        rng = np.random.default_rng(SPEC["bootstrap_seed"])
        # Multinomial counts are resampling with replacement of whole episodes in each stratum.
        weights = np.zeros((SPEC["bootstrap_samples"], len(idx)), dtype=float)
        for group in groups:
            weights[:, group] = rng.multinomial(len(group), np.ones(len(group)) / len(group),
                                                 size=len(weights))
        weights /= len(idx)
        for comparison, (la, ra, a), (lb, rb, b) in comparisons(cells):
            diff = cells[la, ra, a].loc[idx, METRICS] - cells[lb, rb, b].loc[idx, METRICS]
            valid = diff.columns[diff.notna().all(axis=0)]
            values = diff[valid].to_numpy()
            lower, upper = np.quantile(weights @ values, [0.025, 0.975], axis=0)
            for j, metric in enumerate(valid):
                result.append(dict(profile=profile, comparison=comparison, rate_bps=ra, lag_s=la,
                                   reference_lag_s=lb, method=a, reference=b, metric=metric,
                                   difference=values[:, j].mean(), lo95=lower[j], hi95=upper[j],
                                   paired_episodes=len(idx)))
    return pd.DataFrame(result)


def analyze(output):
    complete = json.loads((output / "completed.json").read_text())
    targets = pd.read_csv(output / "targets.csv")
    runs = pd.read_csv(output / "runs.csv")
    assert len(runs) == complete["episodes"] * 16 and len(targets) == 3 * len(runs)
    assert not runs.duplicated(KEY).any() and not targets.duplicated(KEY + ["target_epoch"]).any()
    target_by_arm = targets.groupby(["profile", "episode_id", "target_epoch"])
    for field in ["start_s", "deadline_s", "truth"]:
        assert (target_by_arm[field].nunique() == 1).all()
    assert (target_by_arm.size() == 16).all()
    assert (targets["correct"] == targets.first_correct_s.notna().astype(int)).all()
    assert np.allclose(targets.capped_delay_s, targets.first_correct_s.fillna(targets.deadline_s)-targets.start_s)
    assert ((targets.capped_delay_s >= 0) & (targets.capped_delay_s <= 21600)).all()
    assert (targets.loc[targets.no_decision.astype(bool), "correct"] == 0).all()
    constrained = runs.method.isin(METHODS)
    assert runs.loc[constrained, "wire_bytes"].notna().all() and runs.loc[~constrained, "wire_bytes"].isna().all()
    faults = targets.truth.eq("fault")
    targets = targets.assign(targets=1, fault_targets=faults, nominal_targets=~faults,
                             fault_detected=targets.correct * faults, nominal_confirmed=targets.correct * ~faults,
                             false_alarm_targets=targets.ever_wrong * ~faults,
                             missed_fault_wrong_nominal=targets.ever_wrong * faults)
    grouped = targets.groupby(KEY, as_index=False)[[*SUM, "capped_delay_s"]].sum()
    ep = runs[KEY + ["wire_bytes"]].merge(grouped, on=KEY, validate="one_to_one")
    assert ep.targets.eq(3).all() and len(ep) == len(runs)
    ep["capped_delay_h"] = ep.capped_delay_s / (ep.targets * 3600)
    ep["bytes_day"] = ep.wire_bytes
    for field in ("correct", "ever_wrong", "no_decision"):
        ep[field + "_pct"] = ep[field] * 100 / ep.targets
    aggregate(ep).to_csv(output / "aggregate.csv", index=False, float_format="%.8f", lineterminator="\n")
    paired(ep).to_csv(output / "contrasts.csv", index=False, float_format="%.8f", lineterminator="\n")
    timing = pd.read_csv(output / "timings.csv")
    ts = {name: {"plan_max_s": float(frame.planner_max_s.max()), "plan_overruns": int(frame.planner_over.sum()),
                 "diagnose_max_s": float(frame.detector_max_s.max()), "diagnose_overruns": int(frame.detector_over.sum())}
          for name, frame in timing.groupby(np.where(timing.method.eq("contact_history"), "reference", "constrained"))}
    scientific = {f: hashlib.sha256((output/f).read_bytes()).hexdigest() for f in
                  ("runs.csv", "targets.csv", "aggregate.csv", "contrasts.csv")}
    audit = dict(episodes=complete["episodes"], runs=len(runs), target_rows=len(targets),
                 unique_targets=target_by_arm.ngroups, observed_timings=ts, sha256=scientific,
                 paired=True, lag_isolated=True, capped_delay_checked=True)
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    analyze(parser.parse_args().output)
