"""Episode-cluster analysis of fixed v2 outputs; does not run simulations."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE_SEED=20261113
CELL=['split','profile','condition','schedule_id','rate_bps','stress','method']
TARGET_METRICS={'resolved_fraction':'resolved','first_correct_fraction':'first_correct',
    'final_correct_fraction':'final_correct','ever_wrong_certificate_fraction':'ever_wrong_certificate',
    'ever_wrong_fraction':'ever_wrong','unknown_fraction':'no_decision','conflicting_fraction':'conflicting',
    'capped_delay_s':'capped_delay_s','wire_to_cutoff_bytes':'wire_to_cutoff_bytes',
    'airtime_to_cutoff_s':'airtime_to_cutoff_s','oracle_resolved_fraction':'oracle_contact_resolved',
    'oracle_ever_wrong_fraction':'oracle_contact_ever_wrong','oracle_capped_delay_s':'oracle_contact_capped_delay_s'}
CONDITIONAL={'wire_to_correct_bytes':'wire_to_correct_bytes','airtime_to_correct_s':'airtime_to_correct_s',
    'contacts_to_correct':'contacts_to_correct','contacts_after_target_to_correct':'contacts_after_target_to_correct'}
RUN_METRICS=['wire_bytes','data_bytes','context_bytes','ack_bytes','airtime_s','planning_wall_s','diagnosis_wall_s','frames_sent','receiver_count','receiver_context_count']

def load_csv(path):
    with path.open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    nonnumeric=set(CELL+['episode_id','truth_label','first_label','final_label','cutoff_kind','skipped_contact_id'])-{'rate_bps'}
    for row in rows:
        for k,v in row.items():
            if k not in nonnumeric: row[k]=float(v) if v else math.nan
    return rows

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:('' if isinstance(v,float) and not math.isfinite(v) else v) for k,v in row.items()})

def mean(values):
    x=[v for v in values if math.isfinite(v)]
    return float(np.mean(x)) if x else math.nan

def ci(values,seed,resamples=1000):
    arr=np.array(values,float)
    arr=arr[np.isfinite(arr)]
    if not len(arr): return math.nan,math.nan,math.nan
    rng=np.random.default_rng(seed)
    means=arr[rng.integers(0,len(arr),size=(resamples,len(arr)))].mean(axis=1)
    return float(arr.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps((BASE_SEED,parts),sort_keys=True).encode()).digest()[:8],'little')

def wilson(k,n):
    if not n: return math.nan,math.nan
    p=k/n; z=NormalDist().inv_cdf(.975); den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)

def analyze(out):
    global BASE_SEED
    registration=json.loads((out/'registration.json').read_text(encoding='utf-8'))
    protocol=registration['protocol']; resamples=protocol['statistics']['bootstrap_resamples']
    BASE_SEED=protocol['statistics']['bootstrap_seed']
    if not json.loads((out/'summary.json').read_text())['completed']: raise ValueError('Incomplete run')
    runs=load_csv(out/'runs.csv'); targets=load_csv(out/'targets.csv')
    run_lookup={tuple(r[k] for k in CELL)+(r['episode_id'],):r for r in runs}
    if len(run_lookup)!=len(runs): raise ValueError('Duplicate run')
    grouped=defaultdict(list)
    for r in targets: grouped[tuple(r[k] for k in CELL)+(r['episode_id'],)].append(r)
    episodes=[]
    for key,rows in grouped.items():
        if len(rows)!=3 or len({r['target_epoch'] for r in rows})!=3: raise ValueError('Missing/repeated target')
        run=run_lookup[key]
        row=dict(zip(CELL+['episode_id'],key))
        row['n_targets']=len(rows)
        row['resolved_n_targets']=sum(r['resolved'] for r in rows)
        for name,field in TARGET_METRICS.items(): row[name]=mean(r[field] for r in rows)
        row['first_wrong_fraction']=mean(int(r['first_label'] not in (r['truth_label'],'insufficient')) for r in rows)
        row['unresolved_fraction']=mean(1-r['resolved'] for r in rows)
        for name,field in CONDITIONAL.items(): row[name+'_conditional']=mean(r[field] for r in rows)
        joint=[r for r in rows if all(math.isfinite(r[f]) for f in ['observation_wait_s','contact_wait_s','delivery_wait_s'])]
        row['decomposition_n']=len(joint)
        for name in ['observation_wait_s','contact_wait_s','delivery_wait_s']:
            row[name+'_joint']=mean(r[name] for r in joint)
        for name in RUN_METRICS: row[name]=run[name]
        row['total_compute_wall_s']=run['planning_wall_s']+run['diagnosis_wall_s']
        row['any_wrong']=int(any(r['ever_wrong'] for r in rows))
        row['oracle_any_wrong']=int(any(r['oracle_contact_ever_wrong'] for r in rows))
        row['coverage_loss']=row['oracle_resolved_fraction']-row['resolved_fraction']
        row['excess_wrong']=row['ever_wrong_fraction']-row['oracle_ever_wrong_fraction']
        row['excess_any_wrong']=row['any_wrong']-row['oracle_any_wrong']
        row['delay_gap_s']=row['capped_delay_s']-row['oracle_capped_delay_s']
        row['false_alarm_fraction']=mean(r['ever_wrong'] for r in rows if r['truth_label']=='nominal')
        row['unresolved_fault_fraction']=mean(1-r['resolved'] for r in rows if r['truth_label']=='fault')
        episodes.append(row)
    cells=defaultdict(list)
    for r in episodes: cells[tuple(r[k] for k in CELL)].append(r)
    metrics=[k for k in episodes[0] if k not in CELL+['episode_id','n_targets','decomposition_n','resolved_n_targets']]
    aggregate=[]
    for key,rows in sorted(cells.items()):
        result=dict(zip(CELL,key)); result.update(n_episodes=len(rows),n_targets=sum(r['n_targets'] for r in rows),
                decomposition_n_targets=sum(r['decomposition_n'] for r in rows),
                resolved_n_targets=sum(r['resolved_n_targets'] for r in rows),
                resolved_n_episodes=sum(r['resolved_n_targets']>0 for r in rows))
        for metric in metrics:
            avg,low,high=ci([r[metric] for r in rows],stable_seed(key,metric),resamples)
            result[metric+'_mean']=avg;result[metric+'_ci_low']=low;result[metric+'_ci_high']=high
        lower,upper=wilson(sum(r['any_wrong'] for r in rows),len(rows))
        olower,oupper=wilson(sum(r['oracle_any_wrong'] for r in rows),len(rows))
        result.update(any_wrong_wilson_low=lower,any_wrong_wilson_high=upper,
                      oracle_any_wrong_wilson_low=olower,oracle_any_wrong_wilson_high=oupper,
                      excess_any_wrong_conservative_low=lower-oupper,
                      excess_any_wrong_conservative_high=upper-olower)
        margin=protocol['statistics']['sufficiency_margins']
        result['empirical_sufficient']=int(result['coverage_loss_mean']<=margin['correct_coverage_loss'] and
            result['excess_any_wrong_mean']<=margin['excess_ever_wrong'] and result['delay_gap_s_mean']<=margin['capped_delay_gap_s'])
        supported=(result['coverage_loss_ci_high']<=margin['correct_coverage_loss'] and
            result['excess_any_wrong_conservative_high']<=margin['excess_ever_wrong'] and result['delay_gap_s_ci_high']<=margin['capped_delay_gap_s'])
        inadequate=(result['coverage_loss_ci_low']>margin['correct_coverage_loss'] or
            result['excess_any_wrong_conservative_low']>margin['excess_ever_wrong'] or result['delay_gap_s_ci_low']>margin['capped_delay_gap_s'])
        result['interval_sufficiency']='supported' if supported else 'insufficient' if inadequate else 'inconclusive'
        aggregate.append(result)
    pair_specs=[('fresh_compact','fresh_raw'),('fifo_compact','fifo_raw'),('fresh_compact','fifo_compact'),
                ('snapshot_events','fresh_compact'),('fixed_groups','fresh_compact'),('greedy_groups','fixed_groups'),
                ('events_thinned','fresh_compact'),('snapshot_events','events_thinned')]
    paired=[]
    prefixes=sorted({key[:-1] for key in cells})
    pair_metrics=['resolved_fraction','ever_wrong_fraction','unknown_fraction','capped_delay_s','wire_bytes',
                  'total_compute_wall_s','wire_to_correct_bytes_conditional','airtime_to_correct_s_conditional']
    for prefix in prefixes:
        for candidate,reference in pair_specs:
            a={r['episode_id']:r for r in cells[prefix+(candidate,)]}
            b={r['episode_id']:r for r in cells[prefix+(reference,)]}
            if set(a)!=set(b): raise ValueError('Unpaired episode sets')
            for metric in pair_metrics:
                ids=sorted(a)
                diff=[];joint_targets=0
                for i in ids:
                    if metric.endswith('_conditional'):
                        field=metric.removesuffix('_conditional')
                        left={r['target_epoch']:r for r in grouped[prefix+(candidate,i)]}
                        right={r['target_epoch']:r for r in grouped[prefix+(reference,i)]}
                        common=[(left[e],right[e]) for e in left if left[e]['resolved'] and right[e]['resolved']]
                        joint_targets+=len(common)
                        diff.append(mean(x[field]-y[field] for x,y in common))
                    else:
                        diff.append(a[i][metric]-b[i][metric])
                avg,low,high=ci(diff,stable_seed(prefix,candidate,reference,metric),resamples)
                paired.append({**dict(zip(CELL[:-1],prefix)),'candidate':candidate,'reference':reference,
                    'metric':metric,'n_episode_pairs':sum(math.isfinite(x) for x in diff),
                    'mean_difference':avg,'ci_low':low,'ci_high':high,
                    'conditional':int(metric.endswith('_conditional')),
                    'joint_resolved_targets':joint_targets if metric.endswith('_conditional') else ''})
    # Same-episode sensitivity and contact-loss differences, never independent
    # samples of a new spacecraft or a new base episode.
    sensitivity=[]
    episode_lookup={tuple(r[k] for k in CELL)+(r['episode_id'],):r for r in episodes}
    for key,rows in sorted(cells.items()):
        if key[2]=='base' and key[5]=='none': continue
        reference_prefix=list(key);reference_prefix[2]='base';reference_prefix[5]='none';reference_prefix=tuple(reference_prefix)
        for metric in ['resolved_fraction','ever_wrong_fraction','unknown_fraction','capped_delay_s','wire_bytes']:
            diff=[]
            for r in rows:
                ref=episode_lookup.get(reference_prefix+(r['episode_id'],))
                if ref is not None: diff.append(r[metric]-ref[metric])
            avg,low,high=ci(diff,stable_seed(key,metric,'sensitivity'),resamples)
            sensitivity.append({**dict(zip(CELL,key)),'reference_condition':'base','reference_stress':'none',
                                'metric':metric,'n_episode_pairs':len(diff),'mean_difference':avg,'ci_low':low,'ci_high':high})
    expected=len(protocol['profiles'])*protocol['counts_per_profile'][registration['split']]*len(protocol['schedules'])*(len(protocol['rates_base_bps'])+1)*len(protocol['methods'])
    expected+=(len(protocol['conditions'])-1)*protocol['counts_per_profile'][registration['split']]*len(protocol['schedules'])*len(protocol['rates_robustness_bps'])*len(protocol['methods'])
    checks=dict(run_rows=len(runs),target_rows=len(targets),episode_cell_rows=len(episodes),cells=len(aggregate),
        expected_run_rows=expected,complete=len(runs)==expected,
        target_rows_wrong=sum(r['ever_wrong'] for r in targets),target_rows_unresolved=sum(1-r['resolved'] for r in targets),
        budget_excess_runs=sum(r['max_budget_excess_bytes']>0 for r in runs),
        causality_violations=sum(r['causality_violations'] for r in runs),
        planner_overrun_runs=sum(r['planner_overruns']>0 for r in runs),diagnosis_overrun_runs=sum(r['diagnosis_overruns']>0 for r in runs),
        note='Rows reuse base episodes across all conditions. CIs bootstrap episode averages, not individual OFF transitions. Conditional costs omit unresolved values only in explicitly conditional columns; all-target capped delay and coverage are always retained.')
    tables=out/'tables'
    for name,rows in [('aggregate',aggregate),('paired',paired),('sensitivity',sensitivity),('episode_metrics',episodes)]:
        write_csv(tables/(name+'.csv'),rows)
    (tables/'analysis_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    manifest=dict(source_hashes={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ['runs.csv','targets.csv','registration.json']},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),statistics=protocol['statistics'])
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return checks

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',default='outputs/main')
    a=ap.parse_args();out=Path(a.run_dir);out=out if out.is_absolute() else ROOT/out
    print(json.dumps(analyze(out),indent=2))
