"""Run the matched synthetic frame-loss study. Run with the pilot environment."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PILOT = HERE.parents[1]
V2 = PILOT / "v2"
sys.path.insert(0,str(V2/"scripts"))
import run_v2 as frozen
import numpy as np
from loss_delivery import ContactDraw, simulate, evaluate

PROFILES = ("clear_current", "slow_sparse_current", "thermal_only")
METHODS = ("snapshot_events","events_thinned","fresh_compact")
MODES = ("none","random","selective")
SCHEDULES = ("kazstsat_astana_10deg","synthetic_90min")
RATES = (8,32)
SETTINGS = json.loads((V2/"configs/protocol-v2.json").read_text())["delivery"]
ORBITS = frozen.load_orbits()
COLUMNS = ("episode_id","profile","schedule_id","rate_bps","method","loss_mode")


def fingerprint():
    paths = [HERE/"run_experiment.py",HERE/"loss_delivery.py",V2/"configs/protocol-v2.json",
             V2/"scripts/run_v2.py",*sorted((V2/"src/context_telemetry_v2").glob("*.py")),
             PILOT/"data/contacts/kazstsat.csv"]
    return {p.relative_to(PILOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def assert_lossless_parity(public,contacts,config,method,rate,decisions,run):
    """Test development episodes against the separate frozen implementation."""
    old = frozen.simulate(public,contacts,method,rate,config,SETTINGS)
    for key in ("wire_bytes","ack_bytes","receiver_count","receiver_context_count"):
        assert old[key] == run[key], f"Lossless {key} differs"
    assert run["attempted_frames"] == old["frames_sent"]
    for new_d, old_d in zip(decisions,old["_decisions"],strict=True):
        assert new_d["certificates"] == old_d["certificates"]
        if old["diagnosis_overruns"] == 0 and run["ground_overrun"] == 0:
            assert new_d["time_s"] == old_d["time_s"]


def run_episode(spec):
    profile, index, seed, smoke = spec
    episode_id = f"{'smoke' if smoke else 'loss'}-{profile}-{index:03d}"
    faulty = bool(index % 2)
    ep = frozen.quantized(frozen.generate_episode(seed,profile=profile,condition="base",faulty=faulty))
    public = frozen.PublicBuffer(ep.measurements,ep.commands,ep.duration_s)
    config = frozen.diagnosis_config(profile)
    tables = {name:[] for name in ("runs","targets","contacts","frames")}
    for schedule in SCHEDULES:
        contacts, offset = frozen.contacts_for(schedule,seed,ep.duration_s,ORBITS)
        draws = [ContactDraw(seed,schedule,i) for i in range(len(contacts))]
        for rate in RATES:
            for method in METHODS:
                for mode in MODES:
                    key = dict(zip(COLUMNS,(episode_id,profile,schedule,rate,method,mode)))
                    decisions,contact_rows,frames,run = simulate(public,contacts,method,rate,config,
                                                                  SETTINGS,mode,draws)
                    targets = list(evaluate(ep,decisions,frames,rate,frozen.target_decisions))
                    if smoke and mode == "none":
                        assert_lossless_parity(public,contacts,config,method,rate,decisions,run)
                    tables["runs"].append({**key,"seed":seed,"index":index,"faulty_episode":int(faulty),
                                           "schedule_offset_s":offset,**run})
                    tables["targets"].extend({**key,**row} for row in targets)
                    tables["contacts"].extend({**key,**row} for row in contact_rows)
                    tables["frames"].extend({**key,**row} for row in frames)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--freeze",type=Path,help="Pretest file with source_hashes object")
    args=parser.parse_args()
    hashes = fingerprint()
    if not args.smoke:
        if args.freeze is None:
            parser.error("Held-out run requires --freeze")
        if json.loads(args.freeze.read_text())["source_hashes"] != hashes:
            raise ValueError("Scientific source differs from pretest freeze.")
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    n=2 if args.smoke else 160
    base=21000000 if args.smoke else 41000000
    specs=[(p,i,base+pi*100000+i,args.smoke) for pi,p in enumerate(PROFILES) for i in range(n)]
    meta=dict(started_utc=datetime.now(timezone.utc).isoformat(),smoke=args.smoke,workers=args.workers,
              numpy=np.__version__,python=sys.version,source_hashes=hashes,
              pretest_freeze_sha256=hashlib.sha256(args.freeze.read_bytes()).hexdigest() if args.freeze else None)
    (out/"registration.json").write_text(json.dumps(meta,indent=2)+"\n")
    with (out/"episodes.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(("profile","index","seed","faulty_episode"))
        w.writerows((p,i,seed,int(bool(i%2))) for p,i,seed,_ in specs)
    handles={name:((out/(name+".csv")).open("w",newline="",encoding="utf-8")
                   if name in ("runs","targets") else
                   gzip.open(out/(name+".csv.gz"),"wt",newline="",encoding="utf-8",compresslevel=1))
             for name in ("runs","targets","contacts","frames")}
    counts={name:0 for name in handles}
    writers={}
    try:
        with ProcessPoolExecutor(args.workers) as pool:
            # Limit queued tasks and their result memory while preserving ordered output.
            for offset in range(0,len(specs),args.workers*4):
                for tables in pool.map(run_episode,specs[offset:offset+args.workers*4]):
                    for name,rows in tables.items():
                        if name not in writers:
                            writers[name]=csv.DictWriter(handles[name],fieldnames=list(rows[0]))
                            writers[name].writeheader()
                        writers[name].writerows(rows)
                        counts[name]+=len(rows)
                print(f'{min(len(specs),offset+args.workers*4)}/{len(specs)} episodes; {time.perf_counter()-start:.1f}s',flush=True)
    finally:
        for stream in handles.values():
            stream.close()
    assert counts["runs"]==len(specs)*36 and counts["targets"]==counts["runs"]*3
    (out/"completed.json").write_text(json.dumps(dict(completed=True,**counts,
        episodes=len(specs),elapsed_s=time.perf_counter()-start,
        end_utc=datetime.now(timezone.utc).isoformat()),indent=2)+"\n")


if __name__ == "__main__":
    main()
