"""Frozen v2 experiment runner; public simulation is separated from evaluation."""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import random
import shutil
import struct
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT.parent
sys.path.insert(0,str(ROOT/'src'))
from context_telemetry_v2.model import generate_episode
from context_telemetry_v2.diagnosis import diagnose, diagnosis_config
from context_telemetry_v2.policies import PublicBuffer, METHODS
from context_telemetry_v2.delivery import simulate

KEYS=['split','episode_id','profile','condition','schedule_id','rate_bps','stress','method']

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_hashes():
    paths=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'scripts/run_v2.py',ROOT/'scripts/analyze_v2.py',
           PILOT/'data/contacts/kazstsat.csv',PILOT/'requirements-lock.txt',ROOT/'requirements-lock.txt']
    return {str(p.relative_to(PILOT)).replace('\\','/'):sha(p) for p in paths}

def load_orbits():
    with (PILOT/'data/contacts/kazstsat.csv').open(encoding='utf-8',newline='') as f:
        return list(csv.DictReader(f))

def quantized(ep):
    def f32(x): return struct.unpack('!f',struct.pack('!f',x))[0]
    return replace(ep,measurements=tuple(replace(m,temperature_c=f32(m.temperature_c),
                   current_a=None if m.current_a is None else f32(m.current_a)) for m in ep.measurements))

def contacts_for(schedule,seed,duration,orbital):
    rng=random.Random(seed+424242)
    if schedule=='synthetic_90min':
        first=rng.uniform(300,1800)
        return [dict(contact_id=i,start_s=first+i*5400,end_s=min(duration,first+i*5400+300))
                for i in range(17) if first+i*5400<duration],0.0
    if schedule!='kazstsat_astana_10deg':
        raise ValueError(schedule)
    offset=rng.uniform(0,6*86400)
    result=[]
    for c in orbital:
        start,end=float(c['start_s'])-offset,float(c['end_s'])-offset
        if end>0 and start<duration:
            result.append(dict(contact_id=c['contact_id'],start_s=max(0,start),end_s=min(duration,end)))
    return result,offset

def applicable(cert,target):
    if cert['label']=='insufficient':
        return False
    if cert['command_epoch'] is not None and cert['command_epoch']>=0:
        return cert['command_epoch']==target.command_epoch
    # Evaluator only: a raw low-current state observation provides evidence
    # within the actual OFF interval; no epoch/boundary is sent to diagnosis.
    return cert['label']=='nominal' and bool(cert['evidence_times_s']) and all(
        target.start_s<=t<target.end_s for t in cert['evidence_times_s'])

def continuous_oracle(ep,config):
    certs=[asdict(x) for x in diagnose(ep.measurements,ep.commands,config=config,decision_time_s=ep.duration_s)]
    rows=[]
    for target in ep.targets:
        found=sorted((c for c in certs if applicable(c,target) and c['as_of_s']<=target.deadline_s),
                     key=lambda c:(c['as_of_s'],c['label'],c['evidence_times_s']))
        timeline=[]
        for at in sorted({c['as_of_s'] for c in found}):
            labels={c['label'] for c in found if c['as_of_s']<=at}
            timeline.append(dict(as_of_s=at,label=next(iter(labels)) if len(labels)==1 else 'insufficient'))
        first=next((c for c in timeline if c['label']!='insufficient'),None)
        correct=next((c for c in timeline if c['label']==target.label),None)
        rows.append(dict(target_epoch=target.command_epoch,truth_label=target.label,start_s=target.start_s,
                         end_s=target.end_s,deadline_s=target.deadline_s,
                         first_label=first['label'] if first else 'insufficient',
                         first_time_s=first['as_of_s'] if first else None,
                         first_correct_time_s=correct['as_of_s'] if correct else None,
                         ever_wrong=int(any(c['label'] not in (target.label,'insufficient') for c in timeline)),
                         ever_wrong_certificate=int(any(c['label']!=target.label for c in found)),
                         final_label=timeline[-1]['label'] if timeline else 'insufficient',
                         conflicting=int(len({c['label'] for c in found})>1)))
    return rows

def contact_oracle(ep,config,contacts,settings):
    decisions=[]
    for i,c in enumerate(contacts):
        public=ep.public_prefix(c['start_s'])
        began=time.perf_counter()
        certs=diagnose(public.measurements,public.contexts,config=config,decision_time_s=c['end_s'])
        elapsed=time.perf_counter()-began
        at=c['end_s']+max(settings['ground_diagnosis_reservation_s'],elapsed)
        decisions.append(dict(time_s=at,contact_number=i+1,diagnosis_wall_s=elapsed,
                         diagnosis_overrun=int(elapsed>settings['ground_diagnosis_reservation_s']),
                         certificates=[asdict(x) for x in certs]))
    return decisions

def target_decisions(target,decisions):
    found=[]
    for d in decisions:
        if not target.start_s<=d['time_s']<=target.deadline_s:
            continue
        certs=[c for c in d['certificates'] if applicable(c,target)]
        labels={c['label'] for c in certs}
        found.append((d,dict(label=next(iter(labels)) if len(labels)==1 else 'insufficient',
                            conflicting=len(labels)>1,
                            wrong_certificate=any(c['label']!=target.label for c in certs))))
    return found

def prefix_cost(ledger,cutoff):
    # Complete acknowledged wire frames. A partly transmitted frame at a
    # censored deadline has no delivered bytes, but its occupied airtime is
    # accounted separately. Correct decisions occur after the whole contact.
    completed=sum(x['wire_bytes'] for x in ledger if x['end_s']<=cutoff)
    occupied=sum(max(0,min(x['end_s'],cutoff)-x['start_s']) for x in ledger)
    sent=sum(x['wire_bytes']*max(0,min(x['end_s'],cutoff)-x['start_s'])/(x['end_s']-x['start_s']) for x in ledger)
    return completed,occupied,sent

def evaluate_targets(ep,result,continuous,oracle,rate):
    rows=[]
    for target,baseline in zip(ep.targets,continuous):
        found=target_decisions(target,result['_decisions'])
        first=next((x for x in found if x[1]['label']!='insufficient'),None)
        correct=next((x for x in found if x[1]['label']==target.label),None)
        cref=target_decisions(target,oracle)
        co=next((x for x in cref if x[1]['label']==target.label),None)
        cutoff=correct[0]['time_s'] if correct else target.deadline_s
        acknowledged,air,wire=prefix_cost(result['_ledger'],cutoff)
        _,_,before=prefix_cost(result['_ledger'],target.start_s)
        actual=correct[0]['time_s'] if correct else None
        ct=co[0]['time_s'] if co else None
        continuous_time=baseline['first_correct_time_s']
        row=dict(target_epoch=target.command_epoch,truth_label=target.label,start_s=target.start_s,
                 end_s=target.end_s,deadline_s=target.deadline_s,
                 first_label=first[1]['label'] if first else 'insufficient',
                 first_time_s=first[0]['time_s'] if first else None,
                 first_correct_time_s=actual,first_correct=int(first is not None and first[1]['label']==target.label),
                 ever_wrong=int(any(c['label'] not in (target.label,'insufficient') for _,c in found)),
                 ever_wrong_certificate=int(any(c['wrong_certificate'] for _,c in found)),
                 conflicting=int(any(c['conflicting'] for _,c in found)),
                 final_label=found[-1][1]['label'] if found else 'insufficient',
                 final_correct=int(bool(found) and found[-1][1]['label']==target.label),
                 no_decision=int(first is None),resolved=int(correct is not None),
                 capped_delay_s=cutoff-target.start_s,
                 cutoff_s=cutoff,cutoff_kind='first_correct_contact_end' if correct else 'target_deadline',
                 wire_to_correct_bytes=wire if correct else None,airtime_to_correct_s=air if correct else None,
                 wire_to_cutoff_bytes=wire,airtime_to_cutoff_s=air,
                 acknowledged_wire_to_cutoff_bytes=acknowledged,
                 incremental_wire_to_cutoff_bytes=wire-before,
                 contacts_to_correct=correct[0]['contact_number'] if correct else None,
                 contacts_after_target_to_correct=sum(target.start_s<=d['time_s']<=actual for d in result['_decisions']) if correct else None,
                 oracle_continuous_s=continuous_time,oracle_contact_s=ct,
                 oracle_contact_resolved=int(co is not None),
                 oracle_contact_ever_wrong=int(any(c['label'] not in (target.label,'insufficient') for _,c in cref)),
                 oracle_contact_capped_delay_s=ct-target.start_s if co else target.deadline_s-target.start_s,
                 observation_wait_s=continuous_time-target.start_s if continuous_time is not None else None,
                 contact_wait_s=ct-continuous_time if ct is not None and continuous_time is not None else None,
                 delivery_wait_s=actual-ct if actual is not None and ct is not None else None)
        rows.append(row)
    return rows

def cases(protocol,condition):
    rates=protocol['rates_base_bps'] if condition=='base' else protocol['rates_robustness_bps']
    for schedule in protocol['schedules']:
        for rate in rates:
            yield schedule,rate,'none'
        if condition=='base':
            yield schedule,protocol['missed_contact_rate_bps'],'drop_second'

class Writer:
    def __init__(self,path):
        self.handle=path.open('w',encoding='utf-8',newline='')
        self.writer=None
        self.count=0
    def add(self,row):
        if self.writer is None:
            self.writer=csv.DictWriter(self.handle,fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count+=1
    def close(self):
        self.handle.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split',required=True,choices=['development','calibration','test'])
    parser.add_argument('--protocol',default='configs/protocol-v2.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--oracle-only',action='store_true')
    args=parser.parse_args()
    protocol_path=Path(args.protocol)
    if not protocol_path.is_absolute(): protocol_path=ROOT/protocol_path
    protocol=json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    out=Path(args.output)
    if not out.is_absolute(): out=ROOT/out
    out=out.resolve()
    if not out.is_relative_to(ROOT.resolve()): raise SystemExit('Output must be inside v2')
    if out.exists(): raise SystemExit('Existing output is immutable; choose a new output directory')
    if args.limit and args.split!='development': raise SystemExit('Only development may be limited')
    hashes=source_hashes()
    if args.split=='test':
        freeze=json.loads((ROOT/'verification/freeze.json').read_text(encoding='utf-8'))
        if freeze['protocol_sha256']!=sha(protocol_path) or freeze['source_hashes']!=hashes:
            raise SystemExit('Source/protocol does not match approved freeze')
        if not freeze['calibration_passed']: raise SystemExit('Calibration not approved')
    out.mkdir(parents=True)
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),split=args.split,
                      protocol=protocol,protocol_sha256=sha(protocol_path),source_hashes=hashes,
                      oracle_only=args.oracle_only,limited_development=args.limit,
                      python=sys.version,implementation='v2')
    dump(out/'registration.json',registration)
    for rel in hashes:
        destination=out/'source_snapshot'/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(PILOT/rel,destination)
    orbital=load_orbits()
    run_writer,target_writer=Writer(out/'runs.csv'),Writer(out/'targets.csv')
    history=[]
    count=args.limit or protocol['counts_per_profile'][args.split]
    started=time.perf_counter()
    episodes=0
    with gzip.open(out/'episodes.jsonl.gz','wt',encoding='utf-8') as stored:
        for profile_index,profile in enumerate(protocol['profiles']):
            conditions=protocol['conditions'] if profile=='slow_sparse_current' else ['base']
            for condition in conditions:
                for index in range(count):
                    seed=protocol['seed_bases'][args.split]+profile_index*10000+index
                    identifier=f'{args.split}-{profile}-{index:03d}'
                    ep=quantized(generate_episode(seed,profile=profile,condition=condition,faulty=bool(index%2),
                                        duration_s=protocol['duration_s']))
                    config=diagnosis_config(profile)
                    continuous=continuous_oracle(ep,config)
                    history.extend(dict(episode_id=identifier,profile=profile,condition=condition,**x) for x in continuous)
                    stored.write(json.dumps(dict(episode_id=identifier,episode=asdict(ep)))+'\n')
                    episodes+=1
                    if not args.oracle_only:
                        public=PublicBuffer(ep.measurements,ep.commands,ep.duration_s)
                        oracle_cache={}
                        for schedule,rate,stress in cases(protocol,condition):
                            contacts,offset=contacts_for(schedule,seed,ep.duration_s,orbital)
                            skipped=contacts[1]['contact_id'] if stress=='drop_second' and len(contacts)>1 else None
                            if stress=='drop_second': contacts=[c for j,c in enumerate(contacts) if j!=1]
                            key=(schedule,stress)
                            if key not in oracle_cache:
                                oracle_cache[key]=contact_oracle(ep,config,contacts,protocol['delivery'])
                            oracle=oracle_cache[key]
                            for method in protocol['methods']:
                                trace=index<2 and condition=='base' and rate in (2,8,32,2048) and stress=='none'
                                result=simulate(public,contacts,method,rate,config,protocol['delivery'],trace)
                                base=dict(split=args.split,episode_id=identifier,profile=profile,condition=condition,
                                          schedule_id=schedule,rate_bps=rate,stress=stress,method=method)
                                run_row={**base,'seed':seed,'schedule_offset_s':offset,'skipped_contact_id':skipped,
                                         **{k:v for k,v in result.items() if not k.startswith('_')}}
                                run_writer.add(run_row)
                                target_rows=evaluate_targets(ep,result,continuous,oracle,rate)
                                for row in target_rows: target_writer.add({**base,**row})
                                if trace:
                                    name=f'{identifier}__{condition}__{schedule}__{rate}__{method}.json'
                                    dump(out/'traces'/name,dict(run=run_row,targets=target_rows,oracle_continuous=continuous,
                                         oracle_contact=oracle,contacts=result['_trace'],wire_examples=result['_wire_examples'],
                                         public_measurements=[asdict(m) for m in ep.measurements],
                                         public_events=[asdict(c) for c in ep.commands]))
                    if (index+1)%10==0 or index==count-1:
                        print(json.dumps(dict(split=args.split,profile=profile,condition=condition,episodes=index+1,
                                         run_rows=run_writer.count,elapsed_s=round(time.perf_counter()-started,2))),flush=True)
    run_writer.close(); target_writer.close()
    dump(out/'full-history.json',history)
    gates={}
    for profile,condition in sorted({(r['profile'],r['condition']) for r in history}):
        subset=[r for r in history if r['profile']==profile and r['condition']==condition]
        resolved=sum(r['first_correct_time_s'] is not None for r in subset)/len(subset)
        errors=sum(r['ever_wrong'] for r in subset)/len(subset)
        gate=protocol['calibration_gate']
        first_errors=sum(r['first_label'] not in (r['truth_label'],'insufficient') for r in subset)/len(subset)
        gates[f'{profile}/{condition}']=dict(n_targets=len(subset),n_episodes=count,resolved_fraction=resolved,
                    first_wrong_fraction=first_errors,ever_wrong_fraction=errors,required_for_base_gate=condition=='base',
                    passed=resolved>=gate['base_full_history_correct_fraction_min'] and
                    first_errors<=gate['base_full_history_first_wrong_fraction_max'] and
                    errors<=gate['base_full_history_ever_wrong_fraction_max'])
    summary=dict(split=args.split,independent_base_episodes=count*len(protocol['profiles']),
                 condition_episode_realizations=episodes,run_rows=run_writer.count,target_rows=target_writer.count,
                 full_history_targets=len(history),gates=gates,
                 base_gate_passed=all(g['passed'] for g in gates.values() if g['required_for_base_gate']),
                 elapsed_s=time.perf_counter()-started,completed=True)
    dump(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
