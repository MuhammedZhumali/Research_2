"""Thin orchestration of immutable v2 science on a new prespecified sample."""
from __future__ import annotations
import sys
sys.dont_write_bytecode=True
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT.parent
V2=PILOT/'v2'
sys.path.insert(0,str(V2/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
import run_v2 as frozen

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_protocol(path=None):
    path=Path(path) if path else ROOT/'configs/protocol-confirmatory.json'
    if not path.is_absolute():path=ROOT/path
    return path,json.loads(path.read_text(encoding='utf-8-sig'))

def verify_v2():
    freeze=json.loads((V2/'verification/freeze.json').read_text())
    for rel,digest in freeze['source_hashes'].items():
        if sha(PILOT/rel)!=digest:raise ValueError('Frozen v2 dependency changed: '+rel)
    if sha(V2/'configs/protocol-v2.json')!=freeze['protocol_sha256']:
        raise ValueError('Frozen v2 protocol changed')
    return freeze

def source_hashes():
    old=verify_v2()
    hashes=dict(old['source_hashes'])
    paths=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'scripts/run_confirmatory.py',
           ROOT/'scripts/analyze_confirmatory.py',ROOT/'requirements-lock.txt',
           V2/'verification/freeze.json',V2/'configs/protocol-v2.json']
    hashes.update({p.relative_to(PILOT).as_posix():sha(p) for p in paths})
    return hashes

def make_manifest(protocol,path):
    if path.exists():raise ValueError('Seed manifest is immutable; existing file refused')
    rng=random.Random(protocol['fault_assignment_seed'])
    rows=[]
    for pi,profile in enumerate(protocol['profiles']):
        for i in range(protocol['n_per_profile']):
            rows.append(dict(episode_id=f'confirmatory-{profile}-{i:04d}',profile=profile,index=i,
                seed=protocol['episode_seed_base']+pi*protocol['profile_seed_stride']+i,
                faulty=bool(rng.random()<.5),primary=True,
                secondary=i<protocol['secondary_n_per_profile'],boundary=i<protocol['boundary_n_per_profile']))
    assert len({r['seed'] for r in rows})==len(rows)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
    dump(path.with_suffix('.metadata.json'),dict(created_utc=datetime.now(timezone.utc).isoformat(),
        manifest_sha256=sha(path),n=len(rows),seed_formula='base + profile_index*stride + index',
        fault_assignment='Independent Bernoulli(0.5) draws from a separate fixed RNG stream; not index parity',
        by_profile={p:dict(n=sum(r['profile']==p for r in rows),faulty=sum(r['profile']==p and r['faulty'] for r in rows)) for p in protocol['profiles']},
        telemetry_generated=False,test_results_observed=False))
    return rows

def load_manifest(protocol):
    path=ROOT/protocol['seed_manifest_path']
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    if len(rows)!=protocol['n_per_profile']*len(protocol['profiles']):raise ValueError('Manifest count mismatch')
    return path,rows

def registered_cases(entry,protocol):
    for schedule in protocol['schedules']:
        for rate in protocol['primary_rates_bps']:
            yield schedule,rate,protocol['primary_method'],'primary'
            if entry['secondary']:
                for method in protocol['secondary_methods']:
                    yield schedule,rate,method,'secondary'
        if entry['boundary']:
            yield schedule,protocol['boundary_rate_bps'],protocol['primary_method'],'boundary'

def run(protocol_path,protocol,out,*,smoke=False):
    if out.exists():raise ValueError('Existing output refused')
    hashes=source_hashes();manifest_path,entries=load_manifest(protocol)
    if smoke:
        # Read-only v2 development seeds; no newly held-out telemetry is generated.
        entries=[dict(episode_id=f'smoke-{p}-{i:04d}',profile=p,index=i,seed=21000000+pi*10000+i,
                      faulty=bool(i%2),primary=True,secondary=True,boundary=True)
                 for pi,p in enumerate(protocol['profiles']) for i in range(2)]
    else:
        freeze=json.loads((ROOT/'verification/confirmatory-freeze.json').read_text())
        if freeze['protocol_sha256']!=sha(protocol_path) or freeze['source_hashes']!=hashes or freeze['seed_manifest_sha256']!=sha(manifest_path):
            raise ValueError('Confirmatory freeze mismatch')
        if not freeze['approved_before_new_telemetry']:raise ValueError('No pretest approval')
    out.mkdir(parents=True)
    v2_protocol=json.loads((V2/'configs/protocol-v2.json').read_text())
    dump(out/'registration.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol=protocol,
        protocol_sha256=sha(protocol_path),source_hashes=hashes,seed_manifest_sha256=sha(manifest_path),
        smoke=smoke,sampling='iid Bernoulli mixture' if not smoke else 'v2 development replay only',
        scientific_implementation='Unmodified frozen v2 model, detector, policies, binary framing and evaluator',
        interpreter=sys.version))
    for rel in hashes:
        dst=out/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PILOT/rel,dst)
    shutil.copyfile(manifest_path,out/'seed_manifest.jsonl')
    orbital=frozen.load_orbits();history=[]
    runs=frozen.Writer(out/'runs.csv');targets=frozen.Writer(out/'targets.csv')
    started=time.perf_counter();oracle_overruns=0;oracle_max=0.0
    with gzip.open(out/'episodes.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as stored:
        for number,entry in enumerate(entries,1):
            ep=frozen.quantized(frozen.generate_episode(entry['seed'],profile=entry['profile'],condition='base',faulty=entry['faulty']))
            config=frozen.diagnosis_config(entry['profile'])
            continuous=frozen.continuous_oracle(ep,config)
            history.extend(dict(episode_id=entry['episode_id'],profile=entry['profile'],index=entry['index'],**h) for h in continuous)
            stored.write(json.dumps(dict(manifest_entry=entry,episode=asdict(ep)))+'\n')
            public=frozen.PublicBuffer(ep.measurements,ep.commands,ep.duration_s)
            references={}
            for schedule in protocol['schedules']:
                contacts,offset=frozen.contacts_for(schedule,entry['seed'],ep.duration_s,orbital)
                reference=frozen.contact_oracle(ep,config,contacts,v2_protocol['delivery'])
                references[schedule]=(contacts,offset,reference)
                oracle_overruns+=sum(d['diagnosis_overrun'] for d in reference)
                oracle_max=max(oracle_max,max((d['diagnosis_wall_s'] for d in reference),default=0))
            for schedule,rate,method,role in registered_cases(entry,protocol):
                contacts,offset,reference=references[schedule]
                trace=entry['index']<2
                result=frozen.simulate(public,contacts,method,rate,config,v2_protocol['delivery'],trace=trace)
                base=dict(split='smoke' if smoke else 'confirmatory',episode_id=entry['episode_id'],
                          profile=entry['profile'],condition='base',schedule_id=schedule,rate_bps=rate,
                          stress='none',method=method,role=role,index=entry['index'],seed=entry['seed'],
                          faulty_episode=int(entry['faulty']))
                run_row={**base,'schedule_offset_s':offset,**{k:v for k,v in result.items() if not k.startswith('_')}}
                runs.add(run_row)
                target_rows=frozen.evaluate_targets(ep,result,continuous,reference,rate)
                for target in target_rows:targets.add({**base,**target})
                if trace:
                    name=f"{entry['episode_id']}__{schedule}__{rate}__{method}.json"
                    dump(out/'traces'/name,dict(run=run_row,targets=target_rows,oracle_continuous=continuous,
                        oracle_contact=reference,contacts=result['_trace'],wire_examples=result['_wire_examples'],
                        public_measurements=[asdict(m) for m in ep.measurements],public_events=[asdict(c) for c in ep.commands]))
            if number%100==0 or number==len(entries):
                runs.handle.flush();targets.handle.flush()
                print(json.dumps(dict(episodes=number,total_episodes=len(entries),run_rows=runs.count,
                                     target_rows=targets.count,elapsed_s=round(time.perf_counter()-started,2))),flush=True)
    runs.close();targets.close()
    dump(out/'full-history.json',history)
    summary=dict(completed=True,smoke=smoke,independent_episodes=len(entries),run_rows=runs.count,target_rows=targets.count,
        continuous_full_history_targets=len(history),continuous_resolved=sum(h['first_correct_time_s'] is not None for h in history),
        continuous_ever_wrong=sum(h['ever_wrong'] for h in history),
        reference_diagnosis_overruns=oracle_overruns,reference_diagnosis_max_wall_s=oracle_max,
        elapsed_s=time.perf_counter()-started)
    dump(out/'summary.json',summary);print(json.dumps(summary),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol',default='configs/protocol-confirmatory.json')
    ap.add_argument('--make-manifest',action='store_true')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--output')
    args=ap.parse_args();protocol_path,protocol=load_protocol(args.protocol)
    if args.make_manifest:
        path=ROOT/protocol['seed_manifest_path'];rows=make_manifest(protocol,path)
        print(json.dumps(dict(manifest=str(path),sha256=sha(path),episodes=len(rows))))
        return
    if not args.output:ap.error('--output is required')
    out=Path(args.output);out=(out if out.is_absolute() else ROOT/out).resolve()
    if not out.is_relative_to((ROOT/'outputs').resolve()):raise ValueError('Output must be under v3 outputs')
    run(protocol_path,protocol,out,smoke=args.smoke)

if __name__=='__main__':main()
