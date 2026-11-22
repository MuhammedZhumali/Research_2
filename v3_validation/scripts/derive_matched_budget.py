"""Presentation-only analysis of the predeclared first-400 episode subset.

This does not change the frozen confirmatory family or its 2,000-episode sample.
All plotted 4/8/32 bit/s points use exactly the same 400 episodes per profile.
"""
import sys
sys.dont_write_bytecode=True
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from analyze_confirmatory import load_csv,write_csv,boot,stable_seed,mean,dump

ROOT=Path(__file__).resolve().parents[1]
METRICS=['resolved_fraction','capped_delay_s','wire_bytes','airtime_s',
         'coverage_loss','delay_gap_s','oracle_resolved_fraction','oracle_capped_delay_s',
         'wire_to_correct_bytes_conditional','airtime_to_correct_s_conditional']
PAIR_METRICS=['resolved_fraction','capped_delay_s','wire_bytes','airtime_s','coverage_loss','delay_gap_s']

def derive(out):
    registration=json.loads((out/'registration.json').read_text())
    protocol=registration['protocol'];n=protocol['boundary_n_per_profile'];stats=protocol['statistics']
    assert n==protocol['secondary_n_per_profile']
    assert json.loads((out/'summary.json').read_text())['completed'] and not registration['smoke']
    assert json.loads((out/'verification/audit.json').read_text())['passed']
    source=out/'tables/episode_metrics.csv'
    rows=[r for r in load_csv(source) if r['method']==protocol['primary_method'] and r['index']<n]
    groups=defaultdict(list)
    for r in rows:groups[(r['profile'],r['schedule_id'],r['rate_bps'])].append(r)
    rates=[protocol['boundary_rate_bps'],*protocol['primary_rates_bps']]
    expected={(p,s,float(rate)) for p in protocol['profiles'] for s in protocol['schedules'] for rate in rates}
    assert set(groups)==expected
    aggregate=[];paired=[]
    for key,rr in sorted(groups.items()):
        assert len(rr)==n and {r['index'] for r in rr}==set(range(n))
        result=dict(profile=key[0],schedule_id=key[1],rate_bps=key[2],method=protocol['primary_method'],
            n_episodes=n,n_targets=3*n,resolved_n_targets=sum(r['resolved_n_targets'] for r in rr),
            resolved_n_episodes=sum(r['resolved_n_targets']>0 for r in rr),
            inference_scope='prespecified_secondary_matched_subset_pointwise_descriptive')
        for metric in METRICS:
            values=[r[metric] for r in rr]
            low,high=boot(values,seed=stable_seed(stats['bootstrap_seed'],'matched_budget',key,metric),n_resamples=stats['bootstrap_resamples'])
            result[metric+'_mean']=mean(values)
            result[metric+'_ci_low']=low;result[metric+'_ci_high']=high
        aggregate.append(result)
    for profile in protocol['profiles']:
        for schedule in protocol['schedules']:
            lookup={rate:{r['episode_id']:r for r in groups[(profile,schedule,float(rate))]} for rate in rates}
            assert all(set(lookup[r])==set(lookup[rates[0]]) for r in rates)
            for candidate,reference in [(rates[0],rates[1]),(rates[0],rates[2]),(rates[1],rates[2])]:
                ids=sorted(lookup[candidate])
                for metric in PAIR_METRICS:
                    differences=[lookup[candidate][i][metric]-lookup[reference][i][metric] for i in ids]
                    low,high=boot(differences,seed=stable_seed(stats['bootstrap_seed'],'matched_budget_pairs',profile,schedule,candidate,reference,metric),n_resamples=stats['bootstrap_resamples'])
                    paired.append(dict(profile=profile,schedule_id=schedule,method=protocol['primary_method'],
                        candidate_rate_bps=candidate,reference_rate_bps=reference,metric=metric,n_episode_pairs=n,
                        candidate_mean=mean(lookup[candidate][i][metric] for i in ids),
                        reference_mean=mean(lookup[reference][i][metric] for i in ids),mean_difference=mean(differences),
                        ci_low=low,ci_high=high,inference_scope='prespecified_secondary_matched_subset_pointwise_descriptive'))
    write_csv(out/'tables/matched_budget.csv',aggregate)
    write_csv(out/'tables/matched_budget_pairs.csv',paired)
    manifest=dict(classification='Post-test presentation export of a subset and budgets fixed before new telemetry generation',
        primary_inference_unchanged=True,primary_episodes_per_profile=protocol['n_per_profile'],
        matched_budget_episodes_per_profile=n,selection='episode index < boundary_n_per_profile for every plotted rate',
        cells=len(aggregate),paired_metric_rows=len(paired),
        inference='Ordinary pointwise episode-bootstrap intervals, not an extension of the 36-endpoint simultaneous family',
        conditional_costs='Within-episode resolved-target means, then mean among episodes with at least one resolution; resolved denominators retained. Paired cross-rate tests use unconditional daily costs only.',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    dump(out/'tables/matched_budget_manifest.json',manifest)
    return manifest

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run-dir',default='outputs/confirmatory-main')
    args=ap.parse_args();out=Path(args.run_dir);out=out if out.is_absolute() else ROOT/out
    print(json.dumps(derive(out),indent=2))
