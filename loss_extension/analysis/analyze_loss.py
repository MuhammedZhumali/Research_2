"""Analyse matched episode-level erasure experiments without importing the simulator."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ("snapshot_events", "events_thinned", "fresh_compact")
ARMS = ("none", "random", "selective")
SCHEDULES = ("kazstsat_astana_10deg", "synthetic_90min")
RATES = (8, 32)
PROFILES = ("clear_current", "slow_sparse_current", "thermal_only")
DRAWS = 20_000
SEED = 8317402


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        yield from csv.DictReader(handle)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return (row["schedule_id"], int(row["rate_bps"]), row["loss_mode"],
            row["method"], row["profile"], row["episode_id"])


def num(row, name):
    return float(row[name])


def check_target(row):
    start, deadline = num(row, "start_s"), num(row, "deadline_s")
    resolved = int(row["resolved"])
    first_correct = row["first_correct_time_s"]
    assert math.isclose(deadline-start, 21600.0)
    assert bool(first_correct) == bool(resolved)
    if resolved:
        assert start <= float(first_correct) <= deadline
    stop = float(first_correct) if resolved else deadline
    assert math.isclose(num(row, "capped_delay_s"), stop-start, abs_tol=1e-6)
    assert int(row["no_decision"]) == (row["first_label"] == "insufficient")
    assert bool(row["first_time_s"]) == (not int(row["no_decision"]))
    assert not (int(row["no_decision"]) and resolved)
    assert int(row["no_decision"]) + int(row["ever_wrong"]) <= 1
    if row["first_label"] == row["truth_label"]:
        assert resolved and row["first_time_s"] == first_correct
    if row["first_label"] not in (row["truth_label"], "insufficient"):
        assert int(row["ever_wrong"])
    assert bool(row["wire_to_correct_bytes"]) == bool(resolved)
    assert bool(row["contacts_after_target_to_correct"]) == bool(resolved)
    if resolved:
        assert num(row,"wire_to_correct_bytes") >= 0
        assert num(row,"contacts_after_target_to_correct") >= 0
        assert math.isclose(num(row,"wire_to_correct_bytes")*8/int(row["rate_bps"]),
                            num(row,"airtime_to_correct_s"), abs_tol=2e-5)


def frame_counts(run):
    attempted = int(run["attempted_frames"])
    dropped = int(run["erased_frames"])
    wire = int(run["wire_bytes"])
    assert 0 <= dropped <= attempted
    assert wire == int(run["ack_bytes"]) + int(run["data_frame_bytes"])
    assert int(run["ack_bytes"]) == (attempted-dropped)*56
    assert int(run["reserved_bytes"]) == wire + dropped*56
    assert 0 <= int(run["erased_data_bytes"]) <= int(run["data_frame_bytes"])
    if run["loss_mode"] == "none":
        assert dropped == 0 and int(run["retry_observations"]) == 0


def episode_statistics(run, targets):
    faulty = [r for r in targets if r["truth_label"] == "fault"]
    healthy = [r for r in targets if r["truth_label"] == "nominal"]
    assert len(healthy) + len(faulty) == len(targets) == 3
    assert len({r["target_epoch"] for r in targets}) == 3
    correct = [r for r in faulty if int(r["resolved"])]
    n = len(faulty)
    return {
        "fault_targets": n, "fault_detected": len(correct),
        "detection": len(correct)/n if n else None,
        "never_definitive": mean(num(r,"no_decision") for r in faulty) if n else None,
        "incorrect_nominal": mean(num(r,"ever_wrong") for r in faulty) if n else None,
        "fault_delay_s": mean(num(r,"capped_delay_s") for r in faulty) if n else None,
        "healthy_false_alarm": max(num(r,"ever_wrong") for r in healthy) if not n else None,
        "episode_any_wrong": max(num(r,"ever_wrong") for r in targets),
        "fault_wire_cutoff_bytes": mean(num(r,"wire_to_cutoff_bytes") for r in faulty) if n else None,
        "fault_prefix_if_resolved": mean(num(r,"wire_to_correct_bytes") for r in correct) if correct else None,
        "fault_contacts_if_resolved": mean(num(r,"contacts_after_target_to_correct") for r in correct) if correct else None,
        "daily_bytes": num(run,"wire_bytes"), "daily_booked_bytes": num(run,"reserved_bytes"),
        "daily_occupied_s": num(run,"wire_bytes")*8/int(run["rate_bps"]),
        "daily_reserved_s": num(run,"reserved_bytes")*8/int(run["rate_bps"]),
        "attempted_frames": int(run["attempted_frames"]), "erased_frames": int(run["erased_frames"]),
        "retry_observations": int(run["retry_observations"]),
    }


def descriptive(rows):
    faulty = [r for r in rows if r["fault_targets"]]
    def column(col):
        values = [r[col] for r in rows if r[col] is not None]
        return mean(values) if values else None
    return {
        "episodes": len(rows), "faulty_episodes": len(faulty),
        "fault_targets": sum(r["fault_targets"] for r in rows),
        "correct_fault_targets": sum(r["fault_detected"] for r in rows),
        "fault_detection_pct": 100*column("detection"),
        "fault_no_definitive_pct": 100*column("never_definitive"),
        "fault_unresolved_pct": 100*(1-column("detection")),
        "fault_ever_incorrect_nominal_pct": 100*column("incorrect_nominal"),
        "healthy_episode_false_alarm_pct": 100*column("healthy_false_alarm"),
        "all_episode_any_wrong_pct": 100*column("episode_any_wrong"),
        "fault_capped_delay_s": column("fault_delay_s"),
        "daily_wire_bytes": column("daily_bytes"), "daily_reserved_bytes": column("daily_booked_bytes"),
        "daily_occupied_airtime_s": column("daily_occupied_s"), "daily_reserved_airtime_s": column("daily_reserved_s"),
        "daily_attempted_frames": column("attempted_frames"), "daily_erased_frames": column("erased_frames"),
        "daily_retry_observations": column("retry_observations"),
        "fault_prefix_bytes_through_detection_or_deadline": column("fault_wire_cutoff_bytes"),
        "fault_prefix_bytes_conditional_episode_mean": column("fault_prefix_if_resolved"),
        "fault_contacts_conditional_episode_mean": column("fault_contacts_if_resolved"),
        "fault_episodes_with_detection": sum(r["fault_prefix_if_resolved"] is not None for r in rows),
    }


def clean_comparison(base, impaired):
    baseline = {r["target_epoch"]:r for r in base if r["truth_label"] == "fault"}
    changed = {r["target_epoch"]:r for r in impaired if r["truth_label"] == "fault"}
    assert set(baseline) == set(changed)
    pairs = [(r,changed[k]) for k,r in baseline.items()]
    both = [(a,b) for a,b in pairs if int(a["resolved"]) and int(b["resolved"])]
    def diff(col):
        return mean(num(b,col)-num(a,col) for a,b in both) if both else None
    return {
        "lost":sum(int(a["resolved"]) and not int(b["resolved"]) for a,b in pairs),
        "gained":sum(int(b["resolved"]) and not int(a["resolved"]) for a,b in pairs),
        "both":len(both), "targets":len(pairs),
        "delay": diff("capped_delay_s"), "contacts":diff("contacts_after_target_to_correct"),
        "bytes":diff("wire_to_correct_bytes"),
    }


def analyse(source, destination, n_expected):
    assert json.loads((source/"completed.json").read_text())["completed"], "Incomplete dataset"
    destination.mkdir(parents=True,exist_ok=True)
    run_path,target_path=source/"runs.csv",source/"targets.csv"
    runs={}
    for row in read_rows(run_path):
        frame_counts(row)
        k=key(row)
        assert k not in runs
        runs[k]=row
    targets=defaultdict(list)
    truth={}
    for row in read_rows(target_path):
        k=key(row)
        assert k in runs
        check_target(row)
        target=(k[-2],k[-1],row["target_epoch"])
        expected=(row["truth_label"],row["start_s"],row["deadline_s"])
        assert truth.setdefault(target,expected) == expected
        targets[k].append(row)
    assert set(runs)==set(targets)
    episodes={p:sorted({k[-1] for k in runs if k[-2]==p}) for p in PROFILES}
    assert all(len(ids)==n_expected for ids in episodes.values())
    assert len(set(e for es in episodes.values() for e in es)) == n_expected*len(PROFILES)
    assert len(truth)==3*n_expected*len(PROFILES)
    assert len(runs)==3*n_expected*len(SCHEDULES)*len(RATES)*len(METHODS)*len(ARMS)

    summaries, by_profile, contrasts, recovery, episode_export = [],[],[],[],[]
    arm_differences=[]
    random_estimates={}
    rng=np.random.Generator(np.random.PCG64(SEED))
    n_fault=n_expected//2
    assert n_expected%2==0
    # Shared stratified bootstrap counts. Column order = sorted faulty episode IDs in each profile.
    weights = {p:rng.multinomial(n_fault,[1/n_fault]*n_fault,size=DRAWS)/n_fault for p in PROFILES}
    equal_tail=lambda a,alpha:np.quantile(a,[alpha/2,1-alpha/2],method="linear").tolist()
    for schedule,rate,arm in itertools.product(SCHEDULES,RATES,ARMS):
        distributions={}
        point_estimates={}
        fault_episode_ids={}
        for method in METHODS:
            identifiers=dict(schedule_id=schedule,rate_bps=rate,loss_mode=arm,method=method)
            all_metrics, matched_recovery, profile_points=[],[],[]
            distribution=np.zeros(DRAWS)
            for profile in PROFILES:
                metrics=[]
                fault_ids=[]
                for eid in episodes[profile]:
                    k=(schedule,rate,arm,method,profile,eid)
                    stats=episode_statistics(runs[k],targets[k])
                    metrics.append(stats)
                    if stats["fault_targets"]:
                        fault_ids.append(eid)
                    episode_export.append(dict(**identifiers,profile=profile,episode_id=eid,**stats))
                    if arm!="none":
                        clean=(schedule,rate,"none",method,profile,eid)
                        matched_recovery.append(clean_comparison(targets[clean],targets[k]))
                assert len(fault_ids)==n_fault
                assert fault_episode_ids.setdefault(profile,fault_ids)==fault_ids
                fault_values=np.array([r["detection"] for r in metrics if r["fault_targets"]])
                distribution+=100*(weights[profile]@fault_values)/len(PROFILES)
                profile_summary=descriptive(metrics)
                by_profile.append(dict(**identifiers,profile=profile,**profile_summary))
                profile_points.append(profile_summary["fault_detection_pct"])
                all_metrics.extend(metrics)
            value=mean(profile_points)
            distributions[method]=distribution
            point_estimates[method]=value
            pair_key=(schedule,rate,method)
            if arm=="random":
                random_estimates[pair_key]=(value,distribution)
            elif arm=="selective":
                random_point,random_distribution=random_estimates[pair_key]
                low,high=equal_tail(distribution-random_distribution,.05)
                arm_differences.append(dict(schedule_id=schedule,rate_bps=rate,method=method,
                    selective_minus_random_pp=value-random_point,pointwise_lo95_pp=low,pointwise_hi95_pp=high,
                    interpretation="exploratory paired secondary estimate"))
            summary=descriptive(all_metrics)
            assert math.isclose(summary["fault_detection_pct"],value,abs_tol=1e-9)
            low,high=equal_tail(distribution,.05)
            summaries.append(dict(**identifiers,**summary,detection_lo95_pct=low,detection_hi95_pct=high))
            if matched_recovery:
                eligible=[r for r in matched_recovery if r["both"]]
                def cond(metric):
                    return mean(r[metric] for r in eligible) if eligible else None
                recovery.append(dict(**identifiers,fault_targets=sum(r["targets"] for r in matched_recovery),
                    clean_detections_lost=sum(r["lost"] for r in matched_recovery),newly_detected=sum(r["gained"] for r in matched_recovery),
                    jointly_detected=sum(r["both"] for r in matched_recovery),episodes_with_joint_detection=len(eligible),
                    extra_delay_s_joint=cond("delay"),extra_contacts_joint=cond("contacts"),extra_wire_bytes_joint=cond("bytes")))
        primary=(schedule==SCHEDULES[0] and rate==32 and arm in ("random","selective"))
        for a,b in itertools.combinations(METHODS,2):
            difference=distributions[a]-distributions[b]
            # Round only tiny floating-point artifacts in the clone distributions for zero-variance diagnostics.
            low,high=equal_tail(difference,.05)
            adjusted=equal_tail(difference,.05/6) if primary else [None,None]
            pair=dict(schedule_id=schedule,rate_bps=rate,loss_mode=arm,candidate=a,comparator=b,
                      primary=int(primary),difference_pp=point_estimates[a]-point_estimates[b],
                      pointwise_lo95_pp=low,pointwise_hi95_pp=high,
                      family_adjusted_low_pp=adjusted[0],family_adjusted_high_pp=adjusted[1],
                      bootstrap_degenerate=int(np.ptp(difference)<1e-10))
            contrasts.append(pair)
    write_csv(destination/"episode_metrics.csv",episode_export)
    write_csv(destination/"all_cells.csv",summaries)
    write_csv(destination/"profile_cells.csv",by_profile)
    write_csv(destination/"paired_differences.csv",contrasts)
    write_csv(destination/"clean_recovery.csv",recovery)
    write_csv(destination/"exploratory_arm_differences.csv",arm_differences)
    primary_rows=[r for r in contrasts if r["primary"]]
    assert len(primary_rows)==6
    compact=[]
    for row in summaries:
        if row["schedule_id"]==SCHEDULES[0] and row["rate_bps"]==32:
            compact.append({k:row[k] for k in ("loss_mode","method","faulty_episodes","correct_fault_targets","fault_targets",
                           "fault_detection_pct","detection_lo95_pct","detection_hi95_pct","fault_capped_delay_s","daily_wire_bytes")})
    write_csv(destination/"primary_rates.csv",compact)
    write_csv(destination/"primary_contrasts.csv",primary_rows)
    qa=dict(run_rows=len(runs),target_rows=sum(map(len,targets.values())),distinct_targets=len(truth),
            episodes=sum(map(len,episodes.values())),source_hashes={p.name:file_hash(p) for p in (run_path,target_path)},
            full_paired_grid=True,checks=["frozen truth across repeated conditions","three distinct transitions per episode/run",
            "balanced faulty and healthy episodes per profile","complete cluster pairing","deadline and first correct accounting",
            "missingness, unknown and wrong report semantics","actual wire and ACK reservation identities"],
            draws=DRAWS,seed=SEED,primary_comparisons=len(primary_rows),
            bootstrap="stratified paired episode multinomial resampling, equal profile weights",numpy=np.__version__)
    (destination/"analysis_qa.json").write_text(json.dumps(qa,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(qa,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT.parent/"experiment/outputs/heldout")
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--episodes-per-profile",type=int,default=160)
    args=parser.parse_args()
    analyse(args.source,args.output,args.episodes_per_profile)
