"""Export completed v2 results as CSV, LaTeX tables and manuscript fragments.

No simulation or tuning is performed. Values come from recorded result files;
conditional means retain explicit eligible-episode/target denominators.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'fresh_raw':'Fresh raw', 'fifo_raw':'FIFO raw',
    'fresh_compact':'Fresh compact', 'fifo_compact':'FIFO compact',
    'snapshot_events':'Event summary', 'fixed_groups':'Fixed groups',
    'greedy_groups':'Greedy groups', 'events_thinned':'Thinned history',
}
PROFILE_LABELS={'clear_current':'Frequent current','slow_sparse_current':'Sparse current','thermal_only':'Temperature only'}
CONDITION_LABELS={'base':'Base','thermal_mismatch':'Thermal mismatch','temperature_gaussian':'Gaussian noise','current_bias':'Current bias'}

def read(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def number(value):
    try:
        return float(value)
    except (TypeError,ValueError):
        return math.nan

def avg(values):
    data=[number(x) for x in values]
    data=[x for x in data if math.isfinite(x)]
    return sum(data)/len(data) if data else math.nan

def fmt(value, digits=2, scale=1):
    value=number(value)
    return f'{value/scale:.{digits}f}' if math.isfinite(value) else '--'

def esc(text):
    return str(text).replace('\\',r'\textbackslash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')

def save_csv(path, rows):
    if not rows: raise ValueError(f'No rows for {path}')
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def tex_table(path,headers,rows,caption,label,align=None):
    align=align or 'l'+'r'*(len(headers)-1)
    text='\\begin{table}[htbp]\n\\centering\n\\small\n\\caption{'+caption+'}\n\\label{'+label+'}\n'
    text+='\\resizebox{\\textwidth}{!}{\\begin{tabular}{'+align+'}\n\\toprule\n'
    text+=' & '.join(headers)+' \\\\\n\\midrule\n'
    text+='\n'.join(' & '.join(str(x) for x in row)+' \\\\' for row in rows)
    text+='\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n'
    path.write_text(text,encoding='utf-8')

def parameter_tables(run_dir,out,paper,protocol):
    frozen=run_dir/'source_snapshot/v2/src/context_telemetry_v2/model.py'
    spec=importlib.util.spec_from_file_location('_frozen_model_for_documentation',frozen)
    model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model
    previous_bytecode=sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode=True
        spec.loader.exec_module(model)
    finally:
        sys.dont_write_bytecode=previous_bytecode
    rows=[]
    for name,c in model.PROFILES.items():
        rows.append({'profile':name,'tau_min_s':c['tau'][0],'tau_max_s':c['tau'][1],
                     'rise_min_c':c['rise'][0],'rise_max_c':c['rise'][1],
                     'ambient_amplitude_c':c['ambient_amplitude'],'temperature_noise_bound_c':c['noise'],
                     'current_interval_s':c['current_interval'],'evidence_status':'modeling assumption'})
    save_csv(out/'plant_parameters.csv',rows)
    tex_table(out/'plant_parameters.tex',['Profile','$\\tau$ (s)','$H$ (C)','Ambient amplitude (C)','Noise bound (C)','Current cadence (s)'],
              [[PROFILE_LABELS[r['profile']],fmt(r['tau_min_s'],0)+'--'+fmt(r['tau_max_s'],0),
                fmt(r['rise_min_c'],0)+'--'+fmt(r['rise_max_c'],0),fmt(r['ambient_amplitude_c'],1),
                fmt(r['temperature_noise_bound_c'],2),fmt(r['current_interval_s'],0)] for r in rows],
              'Declared baseline plant profiles read from the frozen model source. Every value is an analytical assumption, not measured spacecraft hardware. A dash means no current channel.',
              'tab:plant')
    orbit_path=ROOT.parent/'data/contacts/kazstsat.csv'
    if hashlib.sha256(orbit_path.read_bytes()).hexdigest()!=protocol['orbit']['sha256']:
        raise ValueError('Orbit CSV is not the frozen input')
    orbit=read(orbit_path);durations=[number(r['end_s'])-number(r['start_s']) for r in orbit]
    gaps=[number(b['start_s'])-number(a['end_s']) for a,b in zip(orbit,orbit[1:])]
    capacities=[]
    for rate in protocol['rates_base_bps']:
        budget=[math.floor(max(0,d-protocol['delivery']['planner_reservation_s'])*rate/8) for d in durations]
        capacities.append({'allocated_rate_bps':rate,'minimum_bytes':min(budget),'median_bytes':statistics.median(budget),
                           'maximum_bytes':max(budget),'budget_status':'calculated from assumed diagnostic allocation and geometric duration'})
    save_csv(out/'contact_capacity.csv',capacities)
    tex_table(out/'contact_capacity.tex',['Allocated rate (bit/s)','Minimum (bytes)','Median (bytes)','Maximum (bytes)'],
              [[r['allocated_rate_bps'],r['minimum_bytes'],fmt(r['median_bytes'],1),r['maximum_bytes']] for r in capacities],
              'Wire budget per complete frozen geometric contact after the common planning reservation. These diagnostic stream allocations are assumed; daily episode boundaries may truncate a contact.',
              'tab:capacity')
    station=protocol['orbit']['model_station']
    text=(f"The archived orbit identifies NORAD {protocol['orbit']['norad']} / COSPAR {protocol['orbit']['cospar']}, "
          f"with element epoch {esc(protocol['orbit']['element_epoch_utc'])}. "
          f"There are {len(orbit)} calculated windows in the saved seven-day interval. "
          f"Durations range from {fmt(min(durations),1)} to {fmt(max(durations),1)} s (median {fmt(statistics.median(durations),1)} s); "
          f"end-to-start gaps range from {fmt(min(gaps),2,3600)} to {fmt(max(gaps),2,3600)} h. "
          f"The model station uses latitude {station['latitude_deg']} degrees N, longitude {station['longitude_deg']} degrees E, "
          f"height {station['height_m']} m and an elevation mask of {station['minimum_elevation_deg']} degrees. "
          "Station parameters are assumptions; element identity and epoch are archived public data, and the window durations are calculated results. "
          "Table~\\ref{tab:capacity} gives the capacity implied by each assumed diagnostic allocation.\n"
          "\\input{../tables/contact_capacity.tex}\n")
    (paper/'orbit-generated.tex').write_text(text,encoding='utf-8')
    settings=protocol['delivery']
    policy_text=(f"The fixed and greedy fragments cover {settings['group_pre_s']} s before and {settings['group_post_s']} s after an observed OFF. "
                 "They become eligible only after the post-event portion is observed, which introduces a real collection delay. "
                 "The enlarged development-stage fragment includes one full declared sparse-current sampling period; it does not guarantee a current observation within a short OFF dwell. "
                 "The event-summary policy selects the first sample, first available current sample, an observation near the first sample plus the summary span, "
                 "one near the latest sample minus that span, and the latest sample for each observed OFF interval. "
                 f"The span is {settings['summary_span_s']} s; duplicates are removed and the summary carries context in the same packet. "
                 "It has no raw-history fallback. Updates at later contacts may accumulate more than five distinct samples per interval. "
                 f"Thinned history keeps a temperature sample every {settings['thinning_interval_s']} s and every sparse current sample. "
                 f"All methods allow at most {settings['max_records_per_frame']} observations per frame.\n")
    (paper/'policies-generated.tex').write_text(policy_text,encoding='utf-8')

def run(run_dir,out,paper):
    summary=json.loads((run_dir/'summary.json').read_text())
    if not summary.get('completed'): raise ValueError('Results are incomplete')
    checks=json.loads((run_dir/'tables/analysis_checks.json').read_text())
    if not checks.get('complete'): raise ValueError('Analysis is incomplete')
    registration=json.loads((run_dir/'registration.json').read_text())
    if registration.get('split')!='test':
        raise ValueError('Manuscript exports require a completed held-out test; development is not publication evidence')
    p=registration['protocol']
    aggregate=read(run_dir/'tables/aggregate.csv')
    episodes=read(run_dir/'tables/episode_metrics.csv')
    paired=read(run_dir/'tables/paired.csv')
    targets=read(run_dir/'targets.csv')
    out.mkdir(parents=True,exist_ok=True);paper.mkdir(parents=True,exist_ok=True)
    parameter_tables(run_dir,out,paper,p)
    # Full scientific tables remain available without editorial selection.
    for name in ('aggregate','paired','sensitivity','episode_metrics'):
        (out/(name+'.csv')).write_bytes((run_dir/'tables'/f'{name}.csv').read_bytes())
    chosen=[r for r in episodes if r['condition']=='base' and r['stress']=='none'
            and r['schedule_id']=='kazstsat_astana_10deg' and number(r['rate_bps'])==32]
    mixed=[]
    for method in p['methods']:
        rows=[r for r in chosen if r['method']==method]
        if len(rows)!=summary['independent_base_episodes']: raise ValueError('Primary mixture missing episodes')
        metrics=['resolved_fraction','ever_wrong_fraction','unknown_fraction','capped_delay_s','wire_bytes',
                 'wire_to_correct_bytes_conditional','airtime_to_correct_s_conditional','total_compute_wall_s',
                 'oracle_resolved_fraction','oracle_capped_delay_s']
        row={'method':method,'n_episodes':len(rows),'n_targets':sum(int(number(r['n_targets'])) for r in rows)}
        row.update({m+'_mean':avg(r[m] for r in rows) for m in metrics})
        row['resolved_targets']=sum(number(r['resolved_fraction'])*number(r['n_targets']) for r in rows)
        row['conditional_cost_eligible_episodes']=sum(math.isfinite(number(r['wire_to_correct_bytes_conditional'])) for r in rows)
        mixed.append(row)
    save_csv(out/'kazstsat_32_mixture.csv',mixed)
    tex_table(out/'kazstsat_32_mixture.tex',
              ['Method','Resolved (\\%)','Ever wrong (\\%)','Capped delay (h)','Prefix (KiB)','$n_\\mathrm{cost}$','Daily (KiB)','CPU (ms)'],
              [[LABELS[r['method']],fmt(r['resolved_fraction_mean']*100,1),fmt(r['ever_wrong_fraction_mean']*100,1),
                fmt(r['capped_delay_s_mean'],2,3600),fmt(r['wire_to_correct_bytes_conditional_mean'],2,1024),r['conditional_cost_eligible_episodes'],
                fmt(r['wire_bytes_mean'],2,1024),fmt(r['total_compute_wall_s_mean']*1000,2)] for r in mixed],
              'Base KazSTSAT-window scenario at 32 bit/s. Quality, delay and daily cost use an equally weighted mixture of the three synthetic profiles. Delay includes unresolved targets at the six-hour cap. Prefix is mean bytes through the correct diagnostic contact, averaged first within resolved targets of an episode and then across the eligible episodes counted by $n_\\mathrm{cost}$; eligibility can change the profile mix. This is not an all-target efficiency score. CPU is measured desktop selection plus diagnosis time per daily run.',
              'tab:primary')
    primary_pairs=[r for r in paired if r['condition']=='base' and r['stress']=='none' and
                   r['schedule_id']=='kazstsat_astana_10deg' and number(r['rate_bps'])==32 and
                   r['candidate']=='fresh_compact' and r['reference']=='fresh_raw' and
                   r['metric'] in ('resolved_fraction','capped_delay_s')]
    save_csv(out/'primary_pairs.csv',primary_pairs)
    tex_table(out/'primary_pairs.tex',['Profile','Endpoint','Difference','95\\% interval','Episode pairs'],
              [[PROFILE_LABELS[r['profile']],('Resolution (pp)' if r['metric']=='resolved_fraction' else 'Capped delay (h)'),
                fmt(r['mean_difference'],2,.01 if r['metric']=='resolved_fraction' else 3600),
                '['+fmt(r['ci_low'],2,.01 if r['metric']=='resolved_fraction' else 3600)+', '+fmt(r['ci_high'],2,.01 if r['metric']=='resolved_fraction' else 3600)+']',r['n_episode_pairs']] for r in primary_pairs],
              'Fresh compact minus fresh raw at 32 bit/s under the frozen KazSTSAT windows. Bootstrap intervals resample paired episodes, after averaging their OFF outcomes. The raw comparison is a context-information ablation.',
              'tab:pairs',align='llrrr')
    class_rows=[]
    for profile in p['profiles']:
        for schedule in p['schedules']:
            for method in ('fresh_raw','fresh_compact'):
                candidates=[r for r in targets if r['profile']==profile and r['condition']=='base' and r['stress']=='none'
                            and r['schedule_id']==schedule and number(r['rate_bps'])==32 and r['method']==method]
                row={'profile':profile,'schedule_id':schedule,'method':method}
                for label in ('nominal','fault'):
                    subset=[r for r in candidates if r['truth_label']==label]
                    clusters=defaultdict(list)
                    for r in subset:clusters[r['episode_id']].append(number(r['resolved']))
                    row[label+'_n_targets']=len(subset);row[label+'_n_episodes']=len(clusters)
                    row[label+'_resolved_episode_mean']=avg(avg(v) for v in clusters.values())
                class_rows.append(row)
    save_csv(out/'class_stratified_resolution.csv',class_rows)
    tex_table(out/'class_stratified_resolution.tex',
              ['Profile','Schedule','Method','Nominal resolved (\\%)','Nominal $n/E$','Fault resolved (\\%)','Fault $n/E$'],
              [[PROFILE_LABELS[r['profile']], 'KazSTSAT' if r['schedule_id']=='kazstsat_astana_10deg' else 'Periodic',LABELS[r['method']],
                fmt(r['nominal_resolved_episode_mean']*100,1),f"{r['nominal_n_targets']}/{r['nominal_n_episodes']}",
                fmt(r['fault_resolved_episode_mean']*100,1),f"{r['fault_n_targets']}/{r['fault_n_episodes']}"] for r in class_rows],
              'Descriptive class stratification at 32 bit/s: $n$ is the number of OFF outcomes and $E$ the number of independent episodes containing that class. Resolution is averaged within eligible episodes and then across episodes. Nominal OFF includes successful transitions before a later failure, not only wholly healthy episodes. Class subsets overlap in episode membership; this table does not create additional independent observations.',
              'tab:classes',align='lllrrrr')
    suff=[r for r in aggregate if r['condition']=='base' and r['stress']=='none' and
          r['schedule_id']=='kazstsat_astana_10deg' and number(r['rate_bps'])==32]
    save_csv(out/'sufficiency_32.csv',suff)
    tex_table(out/'sufficiency_32.tex',['Profile','Method','Point screen','Interval verdict'],
              [[PROFILE_LABELS[r['profile']],LABELS[r['method']], 'pass' if int(number(r['empirical_sufficient'])) else 'fail',esc(r['interval_sufficiency'])] for r in suff],
              'Predeclared sufficiency against the contact full-history reference at 32 bit/s. Point screens and interval-supported verdicts are distinct. The error guard uses marginal Wilson bounds on episode-any-wrong risk, so zero observed errors do not establish a two-point safety margin.',
              'tab:sufficiency',align='llll')
    robust=[r for r in aggregate if r['profile']=='slow_sparse_current' and r['stress']=='none'
            and r['schedule_id']=='kazstsat_astana_10deg' and number(r['rate_bps'])==32]
    save_csv(out/'robustness_32.csv',robust)
    tex_table(out/'robustness_32.tex',['Condition','Method','Ever correct (\\%)','Final correct (\\%)','Ever wrong (\\%)','Conflict (\\%)','Unknown (\\%)','Delay (h)'],
              [[CONDITION_LABELS[r['condition']],LABELS[r['method']],fmt(number(r['resolved_fraction_mean'])*100,1),
                fmt(number(r['final_correct_fraction_mean'])*100,1),fmt(number(r['ever_wrong_fraction_mean'])*100,1),
                fmt(number(r['conflicting_fraction_mean'])*100,1),fmt(number(r['unknown_fraction_mean'])*100,1),fmt(r['capped_delay_s_mean'],2,3600)] for r in robust],
              'One-factor sensitivities, sparse-current profile, frozen KazSTSAT windows and 32 bit/s. The same underlying episodes are reused across conditions. Ever-wrong and resolved can both apply to a target, because later correction does not erase earlier error.',
              'tab:robustness',align='llrrrrrr')
    gates=[]
    history=json.loads((run_dir/'full-history.json').read_text())
    for key,g in summary['gates'].items():
        profile,condition=key.split('/')
        h=[r for r in history if r['profile']==profile and r['condition']==condition]
        gates.append({'profile':profile,'condition':condition,**g,
                      'final_correct_fraction':avg(r['final_label']==r['truth_label'] for r in h),
                      'conflicting_fraction':avg(r['conflicting'] for r in h)})
    save_csv(out/'full_history.csv',gates)
    tex_table(out/'full_history.tex',['Profile','Condition','Targets','Ever correct (\\%)','Final correct (\\%)','Ever wrong (\\%)','Conflict (\\%)'],
              [[PROFILE_LABELS[r['profile']],CONDITION_LABELS[r['condition']],r['n_targets'],fmt(number(r['resolved_fraction'])*100,1),
                fmt(number(r['final_correct_fraction'])*100,1),fmt(number(r['ever_wrong_fraction'])*100,1),fmt(number(r['conflicting_fraction'])*100,1)] for r in gates],
              'Continuous full-history diagnostic check on held-out episodes. Ever-correct resolution may precede a later conflict; it must not be read as stable final correctness. Sensitivities change one factor while preserving the detector. These are diagnostic-model results, not delivery-policy victories.',
              'tab:fullhistory',align='llrrrrr')
    # Record complete outcome counts without interpreting repeated runs as n.
    verdicts={v:sum(r['interval_sufficiency']==v for r in aggregate) for v in ('supported','insufficient','inconclusive')}
    metrics={'run_summary':summary,'analysis_checks':checks,'primary_mixture':mixed,
             'sufficiency_cells':verdicts,'empirical_sufficient_cells':sum(int(number(r['empirical_sufficient'])) for r in aggregate),
             'total_cells':len(aggregate),'note':'Cell counts are descriptive repeated conditions, not independent observations.'}
    (out/'manuscript_numbers.json').write_text(json.dumps(metrics,indent=2)+'\n',encoding='utf-8')
    by_method={r['method']:r for r in mixed}; raw=by_method['fresh_raw']; compact=by_method['fresh_compact']; summary_method=by_method['snapshot_events']
    def texnum(x,d=2,scale=1): return fmt(x,d,scale)
    abstract=(f"The completed held-out campaign comprises {summary['independent_base_episodes']} independent base episodes and "
              f"{summary['run_rows']:,} paired delivery runs. At 32 bit/s in the calculated KazSTSAT schedule, "
              f"a coherent event summary resolves {round(summary_method['resolved_targets'])} of {summary_method['n_targets']} OFF outcomes, "
              f"compared with {round(summary_method['oracle_resolved_fraction_mean']*summary_method['n_targets'])} for contact full history. "
              f"It uses a mean of {texnum(summary_method['wire_bytes_mean'],0)} daily wire bytes, versus {texnum(compact['wire_bytes_mean'],0)} for fresh compact delivery. "
              f"Fresh compact and raw delivery resolve {texnum(compact['resolved_fraction_mean']*100,1)}\\% and {texnum(raw['resolved_fraction_mean']*100,1)}\\%, "
              f"with six-hour-capped delays of {texnum(compact['capped_delay_s_mean'],2,3600)} and {texnum(raw['capped_delay_s_mean'],2,3600)} h. "
              "The finite sample does not certify the narrow error-equivalence margin. The contribution is an empirical boundary of simple delivery representations, "
              "not a new scheduling algorithm or an operational spacecraft reliability result.\n")
    (paper/'results-abstract.tex').write_text(abstract,encoding='utf-8')
    setup=(f"The registered test contains {summary['independent_base_episodes']} independent base episodes, "
           f"{p['counts_per_profile']['test']} per observation profile, with three OFF targets each. "
           "Half of the episodes are healthy; the others develop persistent stuck-ON at the second or third OFF. "
           f"The base allocations are {', '.join(str(x) for x in p['rates_base_bps'])} bit/s. "
           "Base episodes use the saved KazSTSAT schedule and a synthetic 90-minute schedule. "
           f"The three one-factor sensitivities reuse the {p['counts_per_profile']['test']} sparse-current episodes at "
           f"{' and '.join(str(x) for x in p['rates_robustness_bps'])} bit/s. "
           "A separate sensitivity removes the second scheduled contact by ordinal position, independently of fault time. "
           f"This produces {summary['run_rows']:,} delivery runs and {summary['target_rows']:,} scored target rows; "
           "these repeated rows are not independent samples. Development and calibration use disjoint seed ranges.\n")
    (paper/'setup-generated.tex').write_text(setup,encoding='utf-8')
    results=("\\subsection{Full-history and delivery checks}\n"
             "Table~\\ref{tab:fullhistory} reports the fixed detector before attributing losses to communication. "
             f"The complete result grid contains {len(aggregate)} cells. The analysis records "
             f"{checks['budget_excess_runs']} runs exceeding their byte budget and "
             f"{checks['causality_violations']} causal-prefix violations. "
             "Verification artifacts separately check serialization, receiver state and replay; zero counted violations alone is not proof of correctness.\n"
             "\\input{../tables/full_history.tex}\n"
             "\\subsection{Context and simple alternatives}\n"
             "Tables~\\ref{tab:primary} and \\ref{tab:pairs} give the registered primary comparison. "
             "Raw data retain healthy low-current cases, but lack the command evidence needed to identify a failed OFF from a high-current observation. "
             "The full budget curves show where an allocation becomes too small to deliver informative evidence.\n"
             "\\input{../tables/kazstsat_32_mixture.tex}\n\\input{../tables/primary_pairs.tex}\n"
             "The descriptive class stratification in Table~\\ref{tab:classes} retains cases where context is less necessary: "
             "a low-current observation can support a nominal state without receiving a command. Half of the episodes have a late failure, "
             "but the proportion of failed OFF outcomes is smaller because the early OFF succeeds.\n"
             "\\input{../tables/class_stratified_resolution.tex}\n"
             "\\subsection{Sufficiency and uncertainty}\n"
             f"Across the full registered grid, {metrics['empirical_sufficient_cells']} of {len(aggregate)} cells pass the point-estimate sufficiency screen. "
             f"The interval rule labels {verdicts['supported']} supported, {verdicts['insufficient']} insufficient and {verdicts['inconclusive']} inconclusive. "
             "These are repeated-condition counts, not independent success trials. The conservative error guard intentionally prevents a small zero-error sample from certifying a narrow risk margin.\n"
             "\\input{../tables/sufficiency_32.tex}\n"
             "\\subsection{Robustness and costs}\n"
             "Table~\\ref{tab:robustness} separates the one-factor changes. Conditional prefix costs in Table~\\ref{tab:primary} must be read with resolution: "
             "unresolved targets retain their capped delay and are not counted as cheap successful diagnoses. "
             "The generated figures show budget curves, model sensitivity, prefix cost, cost--delay trade-offs, latency components and post-test illustrative traces.\n"
             "\\input{../tables/robustness_32.tex}\n"
             "\\IfFileExists{../figures/budget_slow_sparse_current.pdf}{\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{../figures/budget_slow_sparse_current.pdf}\\caption{Sparse-current budget sensitivity. Resolution accompanies capped delay; error bars are episode-bootstrap intervals.}\\end{figure}}{}\n"
             "\\IfFileExists{../figures/robustness.pdf}{\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{../figures/robustness.pdf}\\caption{Isolated model and sensor changes. No detector retuning across conditions.}\\end{figure}}{}\n"
             "\\IfFileExists{../figures/cost_delay.pdf}{\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{../figures/cost_delay.pdf}\\caption{Daily traffic and all-target capped delay. Annotations give correct resolution; low unresolved traffic is not efficiency.}\\end{figure}}{}\n")
    (paper/'results-generated.tex').write_text(results,encoding='utf-8')
    summary_base=[r for r in aggregate if r['condition']=='base' and r['stress']=='none' and r['method']=='snapshot_events']
    sufficient_rates=[rate for rate in p['rates_base_bps'] if all(int(number(r['empirical_sufficient'])) for r in summary_base if number(r['rate_bps'])==rate)]
    first_sufficient=min(sufficient_rates)
    lower=max(rate for rate in p['rates_base_bps'] if rate<first_sufficient)
    upper_rows=[r for r in summary_base if number(r['rate_bps'])>=first_sufficient]
    low_syn=[r for r in summary_base if number(r['rate_bps'])==lower and r['schedule_id']=='synthetic_90min']
    low_kaz=[r for r in summary_base if number(r['rate_bps'])==lower and r['schedule_id']=='kazstsat_astana_10deg']
    point_passes=sum(int(number(r['empirical_sufficient'])) for r in upper_rows)
    gates_by={(g['profile'],g['condition']):g for g in gates}
    gaussian=gates_by[('slow_sparse_current','temperature_gaussian')];thermal=gates_by[('slow_sparse_current','thermal_mismatch')]
    wrong_targets=[r for r in targets if number(r['ever_wrong'])>0]
    wrong_episodes={r['episode_id'] for r in wrong_targets}
    daily_saving=100*(1-summary_method['wire_bytes_mean']/compact['wire_bytes_mean'])
    gap=summary_method['capped_delay_s_mean']-summary_method['oracle_capped_delay_s_mean']
    discussion=(f"The main practical result is the coherent event summary. In the balanced base KazSTSAT mixture at 32 bit/s, "
                f"it resolves {round(summary_method['resolved_targets'])} of {summary_method['n_targets']} OFF targets, "
                f"compared with {round(summary_method['oracle_resolved_fraction_mean']*summary_method['n_targets'])} for contact full history. "
                f"Its mean capped-delay excess is {fmt(gap,1)} s, while its daily wire traffic is {fmt(daily_saving,1)}\\% lower than fresh compact delivery. "
                "This is useful evidence for retaining a small event-specific history: repeatedly sending only the freshest observations can omit evidence for an earlier OFF. "
                "It is not a claim that the summary is a formal sufficient statistic or that its conditional prefix mean is a minimum communication cost.\n\n"
                f"The budget boundary is explicit. All {point_passes} of {len(upper_rows)} base summary cells at tested allocations "
                f"{', '.join(str(x) for x in sufficient_rates)} bit/s pass the point-estimate margins. "
                f"At {lower} bit/s, periodic-contact resolution remains "
                f"{fmt(min(number(r['resolved_fraction_mean']) for r in low_syn)*100,2)}--{fmt(max(number(r['resolved_fraction_mean']) for r in low_syn)*100,2)}\\%, "
                f"but excess capped delay is {fmt(min(number(r['delay_gap_s_mean']) for r in low_syn),0)}--{fmt(max(number(r['delay_gap_s_mean']) for r in low_syn),0)} s. "
                f"Under KazSTSAT windows the same allocation resolves only {fmt(min(number(r['resolved_fraction_mean']) for r in low_kaz)*100,2)}--{fmt(max(number(r['resolved_fraction_mean']) for r in low_kaz)*100,2)}\\%, "
                "so the loss is not solely a latency penalty. Cached samples can form a useful pair across contacts; failure to fit a whole pair in one frame is not an impossibility theorem. "
                f"The apparent threshold at {first_sufficient} bit/s is specific to this serializer, observation cadence and tested schedules.\n\n"
                "The original pilot's rate-invariant fresh-policy delays were conditional on its single irreversible transition, shared OFF deadline and available recent evidence. "
                "The new lower allocations and repeated transitions expose a different bottleneck. Absolute delay values must not be compared directly across versions: "
                "v2 scores three historical outcomes with a six-hour cutoff rather than one permanent day-level label.\n\n"
                f"Full-history sensitivity checks bound the interpretation. Gaussian noise yields an earlier correct certificate for every target, "
                f"but only {round(gaussian['final_correct_fraction']*gaussian['n_targets'])} of {gaussian['n_targets']} retain a correct final outcome "
                f"and {round(gaussian['conflicting_fraction']*gaussian['n_targets'])} develop conflicting evidence. "
                f"Thermal mismatch leaves {round(thermal['final_correct_fraction']*thermal['n_targets'])} of {thermal['n_targets']} finally correct "
                f"and includes {round(thermal['ever_wrong_fraction']*thermal['n_targets'])} ever-wrong outcomes. "
                f"Across delivery repetitions, the {len(wrong_targets)} recorded ever-wrong target rows come from {len(wrong_episodes)} independent base episodes. "
                "A summary that omits contradictory observations can appear better than the full-history reference under mismatch; that is not evidence of a more valid physical model.\n\n"
                f"Finally, none of the {len(aggregate)} cells satisfies the complete interval-supported sufficiency rule. "
                "The point-estimate results identify promising operating conditions, whereas the narrow error-risk margin requires more independent evidence. "
                "The study therefore supports practical hypotheses about simple representations, not a certified safety-equivalence claim.\n")
    (paper/'discussion-results.tex').write_text(discussion,encoding='utf-8')
    conclusion=("A small, coherent event history can preserve nearly all diagnostic opportunity available at intermittent contacts in the tested base model. "
                "Its benefit comes from retaining context and appropriately separated observations, not from a new multi-contact optimizer. "
                "The study also identifies failure boundaries: insufficient packet budgets, short or missed contact opportunities, omitted observations, "
                "and violated sensor or thermal assumptions. Healthy low-current cases show that extra context is not always necessary. "
                "The supported contribution is a reproducible, causally separated comparison with byte accounting, censored outcomes, paired episode statistics "
                "and explicit limits of simple summaries. The evidence does not establish universal sufficiency or operational KazSTSAT performance. "
                "The next validation step is to replace assumed diagnostic physics and noise with documented hardware or laboratory telemetry, "
                "while preserving the frozen comparison and its unresolved-outcome accounting.\n")
    (paper/'conclusion-results.tex').write_text(conclusion,encoding='utf-8')
    manifest={'run_directory':str(run_dir.resolve()),'input_sha256':{str(x.relative_to(run_dir)):hashlib.sha256(x.read_bytes()).hexdigest()
              for x in [run_dir/'runs.csv',run_dir/'targets.csv',run_dir/'registration.json',run_dir/'summary.json',run_dir/'full-history.json',
                        *[run_dir/'tables'/f'{n}.csv' for n in ('aggregate','paired','sensitivity','episode_metrics')]]},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'outputs':[str(x.relative_to(ROOT)) for x in sorted(out.glob('*')) if x.is_file()]}
    (out/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'tables':str(out),'paper_fragments':str(paper),'cells':len(aggregate),'primary_n':len(chosen)},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',default='outputs/main');ap.add_argument('--output-dir',default='tables');ap.add_argument('--paper-dir',default='paper')
    a=ap.parse_args()
    def path(s):
        p=Path(s);return p if p.is_absolute() else ROOT/p
    run(path(a.run_dir),path(a.output_dir),path(a.paper_dir))
