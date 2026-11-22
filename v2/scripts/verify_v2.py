"""Independent accounting/invariant audit plus deterministic scientific replay.

The audit recomputes decisions/costs from stored traces rather than trusting
runner summary flags. CPU timings are measured, never expected to reproduce.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from context_telemetry_v2.model import generate_episode
from context_telemetry_v2.diagnosis import diagnosis_config
from context_telemetry_v2.policies import PublicBuffer
from context_telemetry_v2.delivery import simulate
from context_telemetry_v2.wire import decode, ACK_BYTES
from run_v2 import contacts_for, load_orbits, continuous_oracle, contact_oracle, evaluate_targets, quantized

def read(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))

def same(a,b):
    if a in ('',None) or b in ('',None): return a in ('',None) and b in ('',None)
    try: return math.isclose(float(a),float(b),abs_tol=1e-6,rel_tol=1e-10)
    except (ValueError,TypeError): return a==b

def audit(out,replay=True):
    summary=json.loads((out/'summary.json').read_text())
    if not summary['completed']: raise AssertionError('Run incomplete')
    registration=json.loads((out/'registration.json').read_text())
    protocol=registration['protocol']
    runs=read(out/'runs.csv'); targets=read(out/'targets.csv')
    assert len(runs)==summary['run_rows'] and len(targets)==summary['target_rows']==3*len(runs)
    keys=['split','episode_id','profile','condition','schedule_id','rate_bps','stress','method']
    runmap={tuple(r[k] for k in keys):r for r in runs}
    assert len(runmap)==len(runs)
    targetmap={tuple(r[k] for k in keys)+(r['target_epoch'],):r for r in targets}
    assert len(targetmap)==len(targets)
    count=registration.get('limited_development') or protocol['counts_per_profile'][registration['split']]
    expected_keys=set()
    for profile in protocol['profiles']:
        conditions=protocol['conditions'] if profile==protocol['robustness_profile'] else ['base']
        for condition in conditions:
            rates=protocol['rates_base_bps'] if condition=='base' else protocol['rates_robustness_bps']
            for schedule in protocol['schedules']:
                settings=[(r,'none') for r in rates]
                if condition=='base': settings.append((protocol['missed_contact_rate_bps'],'drop_second'))
                for rate,stress in settings:
                    for method in protocol['methods']:
                        for index in range(count):
                            identifier=f"{registration['split']}-{profile}-{index:03d}"
                            expected_keys.add((registration['split'],identifier,profile,condition,schedule,str(rate),stress,method))
    assert set(runmap)==expected_keys,'Missing or unregistered run cells'
    targets_per_run=Counter(tuple(r[k] for k in keys) for r in targets)
    assert set(targets_per_run)==expected_keys and set(targets_per_run.values())=={3}
    for key,r in runmap.items():
        index=int(r['episode_id'].split('-')[-1])
        expected_seed=protocol['seed_bases'][registration['split']]+protocol['profiles'].index(r['profile'])*10000+index
        assert int(r['seed'])==expected_seed
        if r['stress']=='drop_second':
            ref=list(key);ref[6]='none'
            assert int(r['n_contacts'])==int(runmap[tuple(ref)]['n_contacts'])-1
    for path,expected in registration['source_hashes'].items():
        saved=out/'source_snapshot'/path
        assert hashlib.sha256(saved.read_bytes()).hexdigest()==expected
    for r in runs:
        assert float(r['max_budget_excess_bytes'])==0 and float(r['causality_violations'])==0
        assert float(r['wire_bytes'])>=sum(float(r[k]) for k in ['data_bytes','context_bytes','ack_bytes'])
        assert same(float(r['wire_bytes'])-sum(float(r[k]) for k in ['data_bytes','context_bytes','ack_bytes']),
                    56*int(r['frames_sent']))
        assert same(r['airtime_s'],float(r['wire_bytes'])*8/float(r['rate_bps']))
        assert int(r['ack_bytes'])==int(r['frames_sent'])*ACK_BYTES
    for r in targets:
        run=runmap[tuple(r[k] for k in keys)]
        start,deadline=float(r['start_s']),float(r['deadline_s'])
        assert deadline-start==protocol['target_followup_s']
        resolved=int(r['resolved'])
        correct=float(r['first_correct_time_s']) if resolved else deadline
        assert start<=correct<=deadline and same(r['capped_delay_s'],correct-start)
        assert bool(r['wire_to_correct_bytes'])==bool(resolved)
        assert bool(r['airtime_to_correct_s'])==bool(resolved)
        assert int(r['no_decision'])==int(r['first_label']=='insufficient')
        assert int(r['first_correct'])==int(r['first_label']==r['truth_label'])
        assert int(r['final_correct'])==int(r['final_label']==r['truth_label'])
        assert float(r['wire_to_cutoff_bytes'])<=float(run['wire_bytes'])+1e-6
        assert float(r['acknowledged_wire_to_cutoff_bytes'])<=float(r['wire_to_cutoff_bytes'])+1e-6
        assert float(r['airtime_to_cutoff_s'])<=float(run['airtime_s'])+1e-6
    frames_checked=0;examples_checked=0;traces_checked=0;oracle_overruns=0
    trace_files=sorted((out/'traces').glob('*.json'))
    for path in trace_files:
        trace=json.loads(path.read_text())
        for reference in trace['oracle_contact']:
            oracle_overruns+=int(reference.get('diagnosis_overrun',0))
        ledger=[]
        for contact in trace['contacts']:
            used=sum(f['wire_bytes'] for f in contact['frames'])
            assert used==contact['used_wire_bytes']<=contact['budget_bytes']
            for frame in contact['frames']:
                assert frame['start_s']>=contact['start_s'] and frame['end_s']<=contact['end_s']+1e-8
                assert all(t<=contact['start_s'] for t in frame['measurement_times_s']+frame['context_times_s'])
                assert frame['ack_bytes']==ACK_BYTES
                frames_checked+=1;ledger.append(frame)
        assert sum(f['wire_bytes'] for f in ledger)==trace['run']['wire_bytes']
        for example in trace['wire_examples']:
            decode(bytes.fromhex(example['data_frame_hex']))
            assert len(bytes.fromhex(example['ack_hex']))==ACK_BYTES
            examples_checked+=1
        for target in trace['targets']:
            labels=[];conflicts=0
            for contact in trace['contacts']:
                decision=contact['decision'];at=decision['time_s']
                if not target['start_s']<=at<=target['deadline_s']: continue
                available=set()
                for cert in decision['certificates']:
                    epoch=cert['command_epoch']
                    matched=(epoch is not None and epoch>=0 and epoch==target['target_epoch']) or (
                        (epoch is None or epoch<0) and cert['label']=='nominal' and
                        cert['evidence_times_s'] and all(target['start_s']<=t<target['end_s'] for t in cert['evidence_times_s']))
                    if matched: available.add(cert['label'])
                label=next(iter(available)) if len(available)==1 else 'insufficient'
                labels.append((at,label));conflicts+=int(len(available)>1)
            first=next((x for x in labels if x[1]!='insufficient'),None)
            correct=next((x for x in labels if x[1]==target['truth_label']),None)
            assert target['first_label']==(first[1] if first else 'insufficient')
            assert same(target['first_correct_time_s'],correct[0] if correct else None)
            assert target['conflicting']==int(conflicts>0)
            assert target['ever_wrong']==int(any(label not in (target['truth_label'],'insufficient') for _,label in labels))
            cutoff=target['cutoff_s']
            transmitted=sum(f['wire_bytes']*max(0,min(f['end_s'],cutoff)-f['start_s'])/(f['end_s']-f['start_s']) for f in ledger)
            occupied=sum(max(0,min(f['end_s'],cutoff)-f['start_s']) for f in ledger)
            assert same(target['wire_to_cutoff_bytes'],transmitted)
            assert same(target['airtime_to_cutoff_s'],occupied)
        traces_checked+=1
    replay_rows=[]
    if replay:
        orbital=load_orbits()
        # Outcome-independent replay set: all base episode index000/001,
        # two schedules, two rates, every method; represented by existing traces.
        selected=[p for p in trace_files if json.loads(p.read_text())['run']['rate_bps'] in (8,32)]
        cache={}
        excluded={'planning_wall_s','diagnosis_wall_s'}
        for path in selected:
            trace=json.loads(path.read_text());old=trace['run']
            identity=(old['profile'],old['seed'],old['condition'])
            if identity not in cache:
                index=int(old['episode_id'].split('-')[-1])
                ep=quantized(generate_episode(old['seed'],profile=old['profile'],condition=old['condition'],faulty=bool(index%2)))
                config=diagnosis_config(old['profile']);cont=continuous_oracle(ep,config)
                cache[identity]=(ep,config,cont)
            ep,config,cont=cache[identity]
            contacts,_=contacts_for(old['schedule_id'],old['seed'],ep.duration_s,orbital)
            new=simulate(PublicBuffer(ep.measurements,ep.commands,ep.duration_s),contacts,old['method'],old['rate_bps'],config,protocol['delivery'])
            for key,value in new.items():
                if not key.startswith('_') and key not in excluded:
                    assert same(value,old[key]),(path.name,key,value,old[key])
            oracle=contact_oracle(ep,config,contacts,protocol['delivery'])
            new_targets=evaluate_targets(ep,new,cont,oracle,old['rate_bps'])
            for a,b in zip(new_targets,trace['targets']):
                for key in a: assert same(a[key],b[key]),(path.name,key,a[key],b[key])
            replay_rows.append(dict(trace=path.name,scientific_match=True))
    report=dict(passed=True,run_rows=len(runs),target_rows=len(targets),traces_checked=traces_checked,
                complete_registered_run_inventory=True,
                frames_checked=frames_checked,binary_examples_checked=examples_checked,replayed_runs=len(replay_rows),
                reference_diagnosis_overruns_in_saved_traces=oracle_overruns,
                excluded_replay_fields=['planning_wall_s','diagnosis_wall_s'],
                scope='Independent CSV and stored-frame/decision audit; fixed outcome-independent replay subset, not every run rerun.')
    (out/'verification').mkdir(exist_ok=True)
    (out/'verification/audit.json').write_text(json.dumps(report,indent=2)+'\n')
    with (out/'verification/replay.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['trace','scientific_match']);writer.writeheader();writer.writerows(replay_rows)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',default='outputs/main');ap.add_argument('--no-replay',action='store_true')
    args=ap.parse_args();out=Path(args.run_dir);out=out if out.is_absolute() else ROOT/out
    print(json.dumps(audit(out,not args.no_replay),indent=2))
