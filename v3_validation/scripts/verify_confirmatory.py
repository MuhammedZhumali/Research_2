"""Independent registered-inventory, cost, decision and inference checks."""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_confirmatory import frozen,registered_cases,verify_v2
from confirmatory_stats import binomial_cdf

def read(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def same(a,b):
    if a in ('',None) or b in ('',None):return a in ('',None) and b in ('',None)
    try:return math.isclose(float(a),float(b),abs_tol=1e-6,rel_tol=1e-10)
    except (TypeError,ValueError):return a==b

def audit(out,replay=True):
    verify_v2()
    registration=json.loads((out/'registration.json').read_text());p=registration['protocol']
    summary=json.loads((out/'summary.json').read_text())
    assert summary['completed']
    entries=[json.loads(x) for x in (out/'seed_manifest.jsonl').read_text().splitlines()]
    if registration['smoke']:
        entries=[dict(episode_id=f'smoke-{profile}-{i:04d}',profile=profile,index=i,
                seed=21000000+pi*10000+i,faulty=bool(i%2),primary=True,secondary=True,boundary=True)
                for pi,profile in enumerate(p['profiles']) for i in range(2)]
    manifest={r['episode_id']:r for r in entries}
    expected={(e['episode_id'],s,str(rate),method,role) for e in entries for s,rate,method,role in registered_cases(e,p)}
    runs=read(out/'runs.csv');targets=read(out/'targets.csv')
    keys=['episode_id','schedule_id','rate_bps','method','role']
    observed={tuple(r[k] for k in keys):r for r in runs}
    assert len(runs)==len(observed)==len(expected) and set(observed)==expected
    counts=Counter(tuple(r[k] for k in keys) for r in targets)
    assert set(counts)==expected and set(counts.values())=={3}
    assert len(targets)==3*len(runs)==summary['target_rows']
    for rel,digest in registration['source_hashes'].items():
        assert hashlib.sha256((out/'source_snapshot'/rel).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((out/'seed_manifest.jsonl').read_bytes()).hexdigest()==registration['seed_manifest_sha256']
    for r in runs:
        e=manifest[r['episode_id']]
        assert int(r['seed'])==e['seed'] and bool(int(r['faulty_episode']))==e['faulty']
        assert int(r['index'])==e['index'] and r['profile']==e['profile']
        assert float(r['max_budget_excess_bytes'])==float(r['causality_violations'])==0
        assert same(float(r['wire_bytes'])-sum(float(r[k]) for k in ['data_bytes','context_bytes','ack_bytes']),56*int(r['frames_sent']))
        assert int(r['ack_bytes'])==56*int(r['frames_sent'])
        assert same(r['airtime_s'],float(r['wire_bytes'])*8/float(r['rate_bps']))
    for r in targets:
        base=observed[tuple(r[k] for k in keys)]
        resolved=bool(int(r['resolved']));start=float(r['start_s']);deadline=float(r['deadline_s'])
        assert deadline-start==21600
        end=float(r['first_correct_time_s']) if resolved else deadline
        assert start<=end<=deadline and same(r['capped_delay_s'],end-start)
        assert bool(r['wire_to_correct_bytes'])==resolved and bool(r['airtime_to_correct_s'])==resolved
        assert float(r['acknowledged_wire_to_cutoff_bytes'])<=float(r['wire_to_cutoff_bytes'])+1e-6<=float(base['wire_bytes'])+2e-6
        assert int(r['first_correct'])==int(r['first_label']==r['truth_label'])
        assert int(r['final_correct'])==int(r['final_label']==r['truth_label'])
    frames=0;examples=0;replays=[];cache={};orbital=frozen.load_orbits()
    v2p=json.loads((ROOT.parent/'v2/configs/protocol-v2.json').read_text())
    traces=sorted((out/'traces').glob('*.json'))
    for path in traces:
        trace=json.loads(path.read_text());old=trace['run'];ledger=[]
        for contact in trace['contacts']:
            assert sum(f['wire_bytes'] for f in contact['frames'])==contact['used_wire_bytes']<=contact['budget_bytes']
            for frame in contact['frames']:
                assert frame['start_s']>=contact['start_s'] and frame['end_s']<=contact['end_s']+1e-8
                assert all(t<=contact['start_s'] for t in frame['measurement_times_s']+frame['context_times_s'])
                frames+=1;ledger.append(frame)
        assert sum(f['wire_bytes'] for f in ledger)==old['wire_bytes']
        from context_telemetry_v2.wire import decode
        for example in trace['wire_examples']:
            decode(bytes.fromhex(example['data_frame_hex']));assert len(bytes.fromhex(example['ack_hex']))==56;examples+=1
        for target in trace['targets']:
            labels=[]
            for contact in trace['contacts']:
                at=contact['decision']['time_s']
                if not target['start_s']<=at<=target['deadline_s']:continue
                active=set()
                for cert in contact['decision']['certificates']:
                    epoch=cert['command_epoch']
                    matches=(epoch is not None and epoch>=0 and epoch==target['target_epoch']) or (
                        (epoch is None or epoch<0) and cert['label']=='nominal' and cert['evidence_times_s'] and
                        all(target['start_s']<=t<target['end_s'] for t in cert['evidence_times_s']))
                    if matches:active.add(cert['label'])
                labels.append((at,next(iter(active)) if len(active)==1 else 'insufficient'))
            first=next((d for d in labels if d[1]!='insufficient'),None)
            correct=next((d for d in labels if d[1]==target['truth_label']),None)
            assert target['first_label']==(first[1] if first else 'insufficient')
            assert same(target['first_correct_time_s'],correct[0] if correct else None)
            cutoff=target['cutoff_s']
            sent=sum(f['wire_bytes']*max(0,min(f['end_s'],cutoff)-f['start_s'])/(f['end_s']-f['start_s']) for f in ledger)
            assert same(sent,target['wire_to_cutoff_bytes'])
        if replay:
            e=manifest[old['episode_id']]
            if e['episode_id'] not in cache:
                ep=frozen.quantized(frozen.generate_episode(e['seed'],profile=e['profile'],condition='base',faulty=e['faulty']))
                cfg=frozen.diagnosis_config(e['profile']);cont=frozen.continuous_oracle(ep,cfg)
                cache[e['episode_id']]=(ep,cfg,cont)
            ep,cfg,cont=cache[e['episode_id']]
            contacts,_=frozen.contacts_for(old['schedule_id'],e['seed'],ep.duration_s,orbital)
            result=frozen.simulate(frozen.PublicBuffer(ep.measurements,ep.commands,ep.duration_s),contacts,old['method'],old['rate_bps'],cfg,v2p['delivery'])
            for key,value in result.items():
                if not key.startswith('_') and key not in ['planning_wall_s','diagnosis_wall_s']:
                    assert same(value,old[key]),(path.name,key)
            reference=frozen.contact_oracle(ep,cfg,contacts,v2p['delivery'])
            tt=frozen.evaluate_targets(ep,result,cont,reference,old['rate_bps'])
            for new,previous in zip(tt,trace['targets']):
                for key in new:assert same(new[key],previous[key]),(path.name,key)
            replays.append(dict(trace=path.name,scientific_match=True))
    endpoints=read(out/'tables/primary_endpoints.csv');inference=read(out/'tables/primary_cells.csv')
    assert len(endpoints)==36 and len(inference)==12
    delta=.05/36
    for r in endpoints:
        n=int(r['n']);m=float(r['mean_difference']);lo=float(r['ci_low']);hi=float(r['ci_high'])
        if not registration['smoke']:assert n==2000
        assert same(r['endpoint_alpha'],delta)
        if r['endpoint']!='excess_any_wrong':
            width=2 if r['endpoint']=='coverage_loss' else 43200
            variance=float(r['unbiased_variance'])
            radius=math.sqrt(2*variance*math.log(4/delta)/n)+7*width*math.log(4/delta)/(3*(n-1))
            assert same(lo,max(-width/2,m-radius)) and same(hi,min(width/2,m+radius))
            assert radius>0
        else:
            assert sum(int(r[k]) for k in ['n10','n01','n11','n00'])==n
            assert same(m,(int(r['n10'])-int(r['n01']))/n)
            assert same(lo,float(r['p10_low'])-float(r['p01_high']))
            assert same(hi,float(r['p10_high'])-float(r['p01_low']))
            for category in ['10','01']:
                k=int(r['n'+category]);lower=float(r['p'+category+'_low']);upper=float(r['p'+category+'_high'])
                if k==0:assert same(upper,1-(delta/4)**(1/n)) and lower==0
                elif k<n:
                    assert abs(binomial_cdf(k,n,upper)-delta/4)<1e-9
                    assert abs(binomial_cdf(k-1,n,lower)-(1-delta/4))<1e-9
        margin=float(r['margin'])
        assert int(r['ni_supported'])==int(hi<margin)
        assert int(r['eq_supported'])==int(lo>-margin and hi<margin)
    report=dict(passed=True,smoke=registration['smoke'],complete_registered_inventory=True,
        independent_episodes=len(entries),run_rows=len(runs),target_rows=len(targets),traces_checked=len(traces),
        frames_checked=frames,binary_examples_checked=examples,replayed_runs=len(replays),primary_intervals_checked=36,
        v2_source_hashes_unchanged=True,
        scope='All registered CSV rows and seed assignments, stored causal traces, fixed trace replay subset, independent interval formulas; no wall-clock reproducibility claim.')
    directory=out/'verification';directory.mkdir(exist_ok=True)
    (directory/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (directory/'replay.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['trace','scientific_match']);w.writeheader();w.writerows(replays)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',default='outputs/confirmatory-main');ap.add_argument('--no-replay',action='store_true')
    args=ap.parse_args();out=Path(args.run_dir);out=out if out.is_absolute() else ROOT/out
    print(json.dumps(audit(out,not args.no_replay),indent=2))
