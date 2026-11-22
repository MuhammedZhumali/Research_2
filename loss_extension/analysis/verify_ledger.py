"""Independent, streaming reconciliation of the completed erasure ledgers."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import groupby
from functools import lru_cache
import hashlib
import json
from math import isclose
from pathlib import Path

from analyze_loss import ROOT, file_hash, key, read_rows, write_csv


@lru_cache(maxsize=8192)
def rank(seed, schedule, contact, tag):
    material=f"R2-frame-loss-v1|{seed}|{schedule}|{contact-1}|{tag}".encode("ascii")
    return int(hashlib.sha256(material).hexdigest()[:16],16)


def run_audit(source,destination):
    completed=json.loads((source/"completed.json").read_text())
    assert completed["completed"]
    target_groups=iter(groupby(read_rows(source/"targets.csv"),key))
    contact_groups=iter(groupby(read_rows(source/"contacts.csv.gz"),key))
    frame_groups=iter(groupby(read_rows(source/"frames.csv.gz"),key))
    next_frames=next(frame_groups,None)
    aggregations=defaultdict(Counter)
    random_paths={}
    timing_counts=Counter()
    seen=Counter()
    for run in read_rows(source/"runs.csv"):
        k=key(run)
        schedule,rate,mode,method,profile,eid=k
        target_key,targets=next(target_groups)
        contact_key,contacts=next(contact_groups)
        assert target_key==contact_key==k
        targets=list(targets)
        contacts=list(contacts)
        frames=[]
        if next_frames and next_frames[0]==k:
            frames=list(next_frames[1])
            next_frames=next(frame_groups,None)
        within_contact=defaultdict(list)
        for frame in frames:
            within_contact[int(frame["contact_number"])].append(frame)
        sums=Counter()
        path=[]
        for c in contacts:
            j=int(c["contact_number"])
            batch=within_contact.pop(j,[])
            assert len(batch)==int(c["attempted_frames"])
            assert sum(int(f["reserved_bytes"]) for f in batch)<=int(c["budget_bytes"])
            exposed=3*rank(run["seed"],schedule,j,"opportunity") < 2**64
            assert exposed==bool(int(c["opportunity"]))
            marks=[int(f["marker"]) for f in batch]
            erased=[i for i,f in enumerate(batch) if int(f["erased"])]
            if mode!="none" and exposed and batch:
                eligible=[i for i,mark in enumerate(marks) if mark] if mode=="selective" and any(marks) else range(len(batch))
                choice=min(eligible,key=lambda i:(rank(run["seed"],schedule,j,f"rank_{i}"),i))
                assert erased==[choice]
            else:
                assert not erased
            for i,f in enumerate(batch):
                assert i==int(f["position"])
                start,data_end,end=[float(f[n]) for n in ("slot_start_s","data_end_s","slot_end_s")]
                assert start+1e-7 >= float(c["start_s"]) and end <= float(c["end_s"])+1e-7
                if i:
                    assert isclose(start,float(batch[i-1]["slot_end_s"]),abs_tol=1e-7)
                assert isclose((data_end-start)*rate/8,int(f["data_frame_bytes"]),abs_tol=1e-6)
                assert isclose((end-start)*rate/8,int(f["reserved_bytes"]),abs_tol=1e-6)
                assert int(f["ack_bytes"])==(0 if int(f["erased"]) else 56)
                assert int(f["data_frame_bytes"])+int(f["ack_bytes"])==int(f["wire_bytes"])
                assert int(f["data_frame_bytes"])+56==int(f["reserved_bytes"])
                assert int(f["retry_observations"])<=int(f["observations"])
                assert bool(int(f["retry_observations"]))==bool(int(f["retried_frame"]))
            for metric in ("wire_bytes","data_frame_bytes","ack_bytes","reserved_bytes","retry_observations","retried_frame"):
                total=sum(int(f[metric]) for f in batch)
                assert total==int(c[metric])
                sums[metric]+=total
            assert len(erased)==int(c["erased_frames"])
            sums["erased_frames"]+=len(erased)
            sums["attempted_frames"]+=len(batch)
            sums["erased_marker_frames"]+=sum(marks[i] for i in erased)
            sums["erased_data_bytes"]+=sum(int(batch[i]["data_frame_bytes"]) for i in erased)
            sums["marker_frames"]+=sum(marks)
            for field in ("planner_overrun","ground_overrun"):
                timing_counts[field]+=int(c[field])
            path.append((len(batch),tuple(erased),tuple(marks),
                tuple(int(f["data_frame_bytes"]) for f in batch),tuple(int(f["observations"]) for f in batch)))
        assert not within_contact
        assert all(sums[n]==int(run[n]) for n in sums)
        assert [int(f["sequence"]) for f in frames]==list(range(len(frames)))
        for target in targets:
            cutoff=float(target["cutoff_s"])
            # Integrate actual occupied channel segments; erased frames end at their data end.
            used=0.
            for f in frames:
                end=float(f["data_end_s"] if int(f["erased"]) else f["slot_end_s"])
                used+=max(0.,min(cutoff,end)-float(f["slot_start_s"]))*rate/8
            assert isclose(used,float(target["wire_to_cutoff_bytes"]),abs_tol=1e-5)
            if target["first_correct_time_s"]:
                n=sum(float(target["start_s"])<=float(c["end_s"])+max(float(c["ground_wall_s"]),0.1)<=cutoff for c in contacts)
                assert n==int(target["contacts_after_target_to_correct"])
        pair_key=(schedule,rate,method,profile,eid)
        if mode=="random":
            random_paths[pair_key]=path
        elif mode=="selective":
            identical=path==random_paths.pop(pair_key)
            for population in (("all","faulty") if int(run["faulty_episode"]) else ("all","healthy")):
                data=aggregations[(schedule,rate,method,population)]
                data["episode_pairs"]+=1
                data["identical"]+=int(identical)
        seen.update(runs=1,targets=len(targets),contacts=len(contacts),frames=len(frames))
    assert not random_paths
    assert next(target_groups,None) is None and next(contact_groups,None) is None and next_frames is None
    assert all(seen[name]==completed[name] for name in seen)
    paths=[]
    for (schedule,rate,method,population),value in sorted(aggregations.items()):
        paths.append(dict(schedule_id=schedule,rate_bps=rate,method=method,population=population,
            episode_pairs=value["episode_pairs"],identical_observed_frame_paths=value["identical"],
            identical_pct=100*value["identical"]/value["episode_pairs"]))
    write_csv(destination/"exploratory_frame_path_agreement.csv",paths)
    result=dict(status="passed",counts=dict(seen),timing_counts=dict(timing_counts),
        description="All raw frame/contact sums, actual versus reserved airtime, transmission-sequence continuity, "
        "action ranks, and replayed target wire prefixes checked independently of the engine.",
        hashes={f:file_hash(source/f) for f in ("runs.csv","targets.csv","contacts.csv.gz","frames.csv.gz")})
    (destination/"ledger_qa.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT.parent/"experiment/outputs/heldout")
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    run_audit(args.source,args.output)
