"""Prespecified finite-sample primary inference; secondary episode bootstrap."""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from confirmatory_stats import empirical_bernstein,clopper_pearson,paired_binary_difference,infer_status

CELL=['profile','schedule_id','rate_bps','method','role']

def load_csv(path):
    with path.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    for row in rows:
        for key,value in row.items():
            if value=='':row[key]=math.nan
            else:
                try:row[key]=float(value)
                except ValueError:pass
    return rows

def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:'' if isinstance(v,float) and not math.isfinite(v) else v for k,v in row.items()})

def mean(values):
    values=[x for x in values if math.isfinite(x)]
    return float(np.mean(values)) if values else math.nan

def boot(values,*,seed,n_resamples):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    if not len(x):return math.nan,math.nan
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),size=(n_resamples,len(x)))].mean(axis=1)
    return tuple(float(y) for y in np.quantile(means,[.025,.975]))

def stable_seed(base,*parts):
    return int.from_bytes(hashlib.sha256(json.dumps([base,*parts],sort_keys=True).encode()).digest()[:8],'little')

def episode_rows(runs,targets):
    rmap={tuple(r[k] for k in CELL)+ (r['episode_id'],):r for r in runs}
    if len(rmap)!=len(runs):raise ValueError('Duplicate run')
    grouped=defaultdict(list)
    for r in targets:grouped[tuple(r[k] for k in CELL)+(r['episode_id'],)].append(r)
    result=[]
    for key,rr in grouped.items():
        if len(rr)!=3 or len({r['target_epoch'] for r in rr})!=3:raise ValueError('Incomplete transition cluster')
        source=rmap[key]
        row={k:source[k] for k in CELL+['episode_id','index','seed','faulty_episode']}
        row.update(n_targets=3,resolved_n_targets=sum(r['resolved'] for r in rr),
            resolved_fraction=mean(r['resolved'] for r in rr),
            first_correct_fraction=mean(r['first_correct'] for r in rr),
            first_wrong_fraction=mean(int(r['first_label'] not in (r['truth_label'],'insufficient')) for r in rr),
            ever_wrong_fraction=mean(r['ever_wrong'] for r in rr),
            any_wrong=int(any(r['ever_wrong'] for r in rr)),oracle_any_wrong=int(any(r['oracle_contact_ever_wrong'] for r in rr)),
            unresolved_fraction=mean(1-r['resolved'] for r in rr),unknown_fraction=mean(r['no_decision'] for r in rr),
            final_correct_fraction=mean(r['final_correct'] for r in rr),conflicting_fraction=mean(r['conflicting'] for r in rr),
            capped_delay_s=mean(r['capped_delay_s'] for r in rr),
            oracle_resolved_fraction=mean(r['oracle_contact_resolved'] for r in rr),
            oracle_capped_delay_s=mean(r['oracle_contact_capped_delay_s'] for r in rr),
            wire_to_correct_bytes_conditional=mean(r['wire_to_correct_bytes'] for r in rr),
            airtime_to_correct_s_conditional=mean(r['airtime_to_correct_s'] for r in rr),
            wire_to_cutoff_bytes=mean(r['wire_to_cutoff_bytes'] for r in rr),
            airtime_to_cutoff_s=mean(r['airtime_to_cutoff_s'] for r in rr))
        row['coverage_loss']=row['oracle_resolved_fraction']-row['resolved_fraction']
        row['delay_gap_s']=row['capped_delay_s']-row['oracle_capped_delay_s']
        row['excess_any_wrong']=row['any_wrong']-row['oracle_any_wrong']
        for field in ['wire_bytes','data_bytes','context_bytes','ack_bytes','airtime_s','planning_wall_s','diagnosis_wall_s','frames_sent','n_contacts']:
            row[field]=source[field]
        row['total_compute_wall_s']=row['planning_wall_s']+row['diagnosis_wall_s']
        result.append(row)
    if len(result)!=len(runs):raise ValueError('Run missing targets')
    return result,grouped

def analyze(out):
    registration=json.loads((out/'registration.json').read_text());protocol=registration['protocol']
    summary=json.loads((out/'summary.json').read_text())
    if not summary['completed']:raise ValueError('Incomplete results are not analyzed')
    runs,targets=load_csv(out/'runs.csv'),load_csv(out/'targets.csv')
    episodes,grouped_targets=episode_rows(runs,targets)
    cells=defaultdict(list)
    for r in episodes:cells[tuple(r[k] for k in CELL)].append(r)
    stats=protocol['statistics'];delta=stats['family_alpha']/(stats['primary_cells']*stats['endpoints'])
    endpoint_rows=[];cell_inference=[];aggregate=[]
    metrics=[k for k in episodes[0] if k not in CELL+['episode_id','index','seed','faulty_episode','n_targets','resolved_n_targets']]
    ci_metrics=['resolved_fraction','capped_delay_s','wire_bytes','wire_to_correct_bytes_conditional','airtime_to_correct_s_conditional','total_compute_wall_s']
    for key,rows in sorted(cells.items()):
        cell=dict(zip(CELL,key));n=len(rows)
        agg={**cell,'n_episodes':n,'n_targets':3*n,'faulty_episodes':sum(r['faulty_episode'] for r in rows),
             'resolved_n_targets':sum(r['resolved_n_targets'] for r in rows),
             'resolved_n_episodes':sum(r['resolved_n_targets']>0 for r in rows)}
        for metric in metrics:agg[metric+'_mean']=mean(r[metric] for r in rows)
        for metric in ci_metrics:
            low,high=boot([r[metric] for r in rows],seed=stable_seed(stats['bootstrap_seed'],key,metric),n_resamples=stats['bootstrap_resamples'])
            agg[metric+'_ci_low']=low;agg[metric+'_ci_high']=high
        k=sum(r['any_wrong'] for r in rows);kr=sum(r['oracle_any_wrong'] for r in rows)
        lo,hi=clopper_pearson(int(k),n);lr,ur=clopper_pearson(int(kr),n)
        agg.update(absolute_any_wrong_n=k,absolute_any_wrong_cp95_low=lo,absolute_any_wrong_cp95_high=hi,
                   reference_any_wrong_n=kr,reference_any_wrong_cp95_low=lr,reference_any_wrong_cp95_high=ur)
        aggregate.append(agg)
        if cell['role']!='primary':continue
        if not registration['smoke'] and n!=protocol['n_per_profile']:raise ValueError('Incomplete primary sample')
        current=[]
        for endpoint,low,high in [('coverage_loss',-1,1),('delay_gap_s',-21600,21600)]:
            interval=empirical_bernstein([r[endpoint] for r in rows],lower_bound=low,upper_bound=high,alpha=delta)
            current.append({**cell,'endpoint':endpoint,**interval,'interval_method':'empirical_Bernstein_two_sided',
                            'margin':stats['margins'][endpoint],'endpoint_alpha':delta,'family_alpha':stats['family_alpha']})
        interval=paired_binary_difference([r['any_wrong'] for r in rows],[r['oracle_any_wrong'] for r in rows],alpha=delta)
        current.append({**cell,'endpoint':'excess_any_wrong',**interval,'interval_method':'paired_discordance_Clopper_Pearson_union',
                        'margin':stats['margins']['excess_any_wrong'],'endpoint_alpha':delta,'family_alpha':stats['family_alpha']})
        for r in current:
            r['ni_supported']=int(r['ci_high']<r['margin'])
            r['eq_supported']=int(r['ci_low']>-r['margin'] and r['ci_high']<r['margin'])
        endpoint_rows.extend(current)
        cell_inference.append({**cell,'n':n,**infer_status(current),
            'candidate_any_wrong_n':k,'reference_any_wrong_n':kr,
            'candidate_any_wrong_cp95_low':lo,'candidate_any_wrong_cp95_high':hi,
            'reference_any_wrong_cp95_low':lr,'reference_any_wrong_cp95_high':ur,
            'empirical_ni':int(all(r['mean_difference']<r['margin'] for r in current)),
            'empirical_eq':int(all(abs(r['mean_difference'])<r['margin'] for r in current)),
            'inference_scope':'SMOKE_ONLY_NO_CONFIRMATORY_CLAIM' if registration['smoke'] else 'prespecified_simultaneous_family'})
    if len(cell_inference)!=stats['primary_cells']:raise ValueError('Missing primary cells')
    secondary=[]
    for key,rows in sorted(cells.items()):
        if key[-1]!='secondary':continue
        profile,schedule,rate,method,_=key
        references={r['episode_id']:r for r in cells[(profile,schedule,rate,protocol['primary_method'],'primary')]
            if r['index']<protocol['secondary_n_per_profile']}
        if set(references)!={r['episode_id'] for r in rows}:raise ValueError('Secondary comparator not paired')
        for metric in ['resolved_fraction','capped_delay_s','any_wrong','wire_bytes','total_compute_wall_s']:
            difference=[r[metric]-references[r['episode_id']][metric] for r in rows]
            low,high=boot(difference,seed=stable_seed(stats['bootstrap_seed'],key,metric,'paired'),n_resamples=stats['bootstrap_resamples'])
            secondary.append(dict(profile=profile,schedule_id=schedule,rate_bps=rate,candidate=method,
                reference=protocol['primary_method'],metric=metric,n=len(rows),mean_difference=mean(difference),
                candidate_mean=mean(r[metric] for r in rows),
                reference_mean=mean(references[r['episode_id']][metric] for r in rows),
                ci_low=low,ci_high=high,inference_scope='prespecified_secondary_pointwise_descriptive'))
    tables=out/'tables'
    for name,rows in [('aggregate',aggregate),('episode_metrics',episodes),('primary_endpoints',endpoint_rows),
                      ('primary_cells',cell_inference),('paired_secondary',secondary)]:write_csv(tables/(name+'.csv'),rows)
    checks=dict(completed=True,smoke=registration['smoke'],run_rows=len(runs),target_rows=len(targets),
        cells=len(aggregate),primary_cells=len(cell_inference),primary_endpoint_intervals=len(endpoint_rows),
        ni_supported_cells=sum(r['ni_status']=='supported' for r in cell_inference),
        eq_supported_cells=sum(r['eq_status']=='supported' for r in cell_inference),
        zero_variance_intervals_with_zero_width=sum(r.get('unbiased_variance')==0 and r['ci_high']==r['ci_low'] for r in endpoint_rows),
        alpha_allocation_sum=delta*len(endpoint_rows),
        errors_rows=sum(r['ever_wrong'] for r in targets),unresolved_rows=sum(1-r['resolved'] for r in targets),
        independent_unit='episode within profile; targets and repeated conditions are not independent replicates')
    dump(tables/'analysis_checks.json',checks)
    dump(out/'analysis_manifest.json',dict(source_hashes={x:sha(out/x) for x in ['runs.csv','targets.csv','registration.json']},
        analysis_script_sha256=sha(Path(__file__)),statistics_script_sha256=sha(ROOT/'src/confirmatory_stats.py'),
        primary_family=stats['primary_family'],analysis_contract=stats))
    return checks

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')

def design_precision(protocol):
    source=ROOT.parent/'v2/outputs/main/tables/episode_metrics.csv'
    prior=load_csv(source);delta=protocol['statistics']['family_alpha']/36
    rows=[]
    for profile in protocol['profiles']:
        for schedule in protocol['schedules']:
            for rate in protocol['primary_rates_bps']:
                selected=[r for r in prior if r['profile']==profile and r['schedule_id']==schedule and r['rate_bps']==rate and
                          r['condition']=='base' and r['stress']=='none' and r['method']==protocol['primary_method']]
                values=np.array([r['delay_gap_s'] for r in selected]);n=protocol['n_per_profile'];variance=float(values.var(ddof=1))
                radius=math.sqrt(2*variance*math.log(4/delta)/n)+7*43200*math.log(4/delta)/(3*(n-1))
                rows.append(dict(profile=profile,schedule_id=schedule,rate_bps=rate,v2_n=len(values),
                    v2_mean_delay_gap_s=float(values.mean()),v2_sd_delay_gap_s=float(values.std(ddof=1)),
                    planned_n=n,planned_EB_radius_s=radius,planned_upper_s=float(values.mean())+radius,
                    interpretation='Planning from v2 only; not a power guarantee or a result from new episodes'))
    directory=ROOT/'outputs/confirmatory-design';directory.mkdir(parents=True,exist_ok=True)
    write_csv(directory/'planning_precision.csv',rows)
    dump(directory/'planning_precision.json',dict(v2_source_sha256=sha(source),primary_endpoints=36,
        delta=delta,planned_n=protocol['n_per_profile'],
        source='https://www.cs.mcgill.ca/~colt2009/papers/012.pdf',theorem='Maurer and Pontil 2009, Theorem 4, two tails',
        formula='sqrt(2*s2*log(4/delta)/n) + 7*R*log(4/delta)/(3*(n-1))',
        iid_design='Independent Bernoulli fault class assignment is declared before telemetry generation.',
        zero_error_discordance_bound=1-(delta/4)**(1/protocol['n_per_profile']),
        no_optional_stopping=True))
    return rows

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',default='outputs/confirmatory-main');ap.add_argument('--design-only',action='store_true')
    args=ap.parse_args()
    if args.design_only:
        protocol=json.loads((ROOT/'configs/protocol-confirmatory.json').read_text())
        print(json.dumps(design_precision(protocol),indent=2))
    else:
        out=Path(args.run_dir);out=out if out.is_absolute() else ROOT/out
        print(json.dumps(analyze(out),indent=2))
