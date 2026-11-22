"""Data-only manuscript export for the completed validation stage.

Inference is read from the separately audited analysis, never recomputed here.
The optional --lab-only flag exports the completed hardware case independently.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {'clear_current': 'Frequent current', 'slow_sparse_current': 'Sparse current', 'thermal_only': 'Temperature only'}
SCHEDULES = {'synthetic_90min': 'Synthetic', 'kazstsat_astana_10deg': 'KazSTSAT windows'}
METHODS = {'snapshot_events': 'Event summary', 'fresh_compact': 'Fresh compact', 'events_thinned': 'Thinned history'}

def read(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def num(value):
    try: return float(value)
    except (ValueError, TypeError): return math.nan

def boolean(value):
    return value is True or str(value).lower() in ('true', '1')

def fmt(value, digits=2, scale=1):
    value = num(value)
    return f'{value / scale:.{digits}f}' if math.isfinite(value) else '--'

def average(values):
    values = [num(x) for x in values]
    values = [x for x in values if math.isfinite(x)]
    return sum(values)/len(values) if values else math.nan

def esc(value):
    return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

def csv_save(path, rows):
    if not rows: raise ValueError(f'Empty table: {path}')
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def copy_file(source,destination):
    if Path(source).resolve()!=Path(destination).resolve():
        shutil.copyfile(source,destination)

def tex_table(path, headers, rows, caption, label):
    lines = [r'\begin{table}[htbp]',r'\centering',r'\small',r'\caption{'+caption+'}',r'\label{'+label+'}',
             r'\resizebox{\textwidth}{!}{\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}',r'\toprule',
             ' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(str(x) for x in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')

def write(path, text):
    Path(path).write_text(text+'\n',encoding='utf-8')

def file_record(path):
    path=Path(path)
    return {'path':os.path.relpath(path,ROOT), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

def validate_manuscript(paper, tables, figures):
    """Static live-main closure and provenance check; explicitly not a TeX build."""
    paper,tables,figures=map(Path,(paper,tables,figures))
    pending=[paper/'main.tex'];seen=set();contents={};missing=[];image_paths=set()
    while pending:
        path=pending.pop().resolve()
        if path in seen:continue
        seen.add(path)
        if not path.exists():missing.append(str(path));continue
        text=path.read_text(encoding='utf-8');contents[path]=text
        for rel in re.findall(r'\\(?:input|resultfile)\{([^}]+)\}',text):
            if '#' in rel:continue
            item=paper/rel
            if not item.suffix:item=item.with_suffix('.tex')
            pending.append(item)
        for rel in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
            image_paths.add((paper/rel).resolve())
    all_text='\n'.join(contents.values());bib=paper/'references.bib';bibtext=bib.read_text(encoding='utf-8')
    bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,\s]+)',bibtext));cites=set()
    for group in re.findall(r'\\cite(?:p|t)?(?:\[[^\]]*\])?\{([^}]+)\}',all_text):cites.update(group.split(','))
    labels=re.findall(r'\\label\{([^}]+)\}',all_text);refs=set(re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',all_text))
    duplicates=sorted({x for x in labels if labels.count(x)>1})
    unbalanced=[]
    for path,text in contents.items():
        clean=re.sub(r'(?<!\\)%[^\n]*','',text);level=0
        for token in re.findall(r'(?<!\\)[{}]',clean):
            level+=1 if token=='{' else -1
            if level<0:break
        if level:unbalanced.append(str(path))
    hash_checks=[];hash_errors=[]
    for manifest_path in [tables/'export_manifest.json',figures/'figure_manifest.json']:
        data=load(manifest_path);records=[]
        def collect(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value:records.append(value)
                for nested in value.values():collect(nested)
            elif isinstance(value,list):
                for nested in value:collect(nested)
        collect(data)
        for record in records:
            source=ROOT/record['path'];actual=hashlib.sha256(source.read_bytes()).hexdigest() if source.exists() else None
            ok=actual==record['sha256'];hash_checks.append({'path':record['path'],'matched':ok})
            if not ok:hash_errors.append(record['path'])
    missing_images=sorted(str(p) for p in image_paths if not p.exists())
    archive_checks=[]
    for candidate in [*contents,*image_paths]:
        for archive,original in [(tables/'v2_archive',ROOT.parent/'v2/tables'),(figures/'v2_archive',ROOT.parent/'v2/figures')]:
            try:relative=candidate.relative_to(archive.resolve())
            except ValueError:continue
            source=original/relative
            same=(candidate.exists() and source.exists() and hashlib.sha256(candidate.read_bytes()).digest()==hashlib.sha256(source.read_bytes()).digest())
            archive_checks.append({'copy':str(candidate.relative_to(ROOT)),'source':str(source),'matched':same})
            if not same:hash_errors.append(str(candidate))
    issues={'missing_inputs':missing,'missing_figures':missing_images,'unknown_citations':sorted(cites-bibkeys),
            'duplicate_labels':duplicates,'unresolved_references':sorted(refs-set(labels)),
            'unbalanced_tex_files':unbalanced,'manifest_hash_errors':hash_errors}
    result={'passed':not any(issues.values()),'scope':'Live main.tex recursive closure and recorded export/plot provenance; no compiled-PDF claim',
            'compiled':False,'compile_block':'Windows Application Control: An Application Control policy has blocked this file.',
            'input_tex_files':len(contents),'citation_keys':len(cites),'bib_entries':len(bibkeys),
            'table_environments':len(re.findall(r'\\begin\{(?:table|longtable)\}',all_text)),
            'referenced_figures':len(image_paths),'issues':issues,'manifest_hash_checks':hash_checks,
            'inherited_v2_asset_checks':archive_checks,
            'live_inputs':[file_record(p) for p in sorted(contents)],'references_bib':file_record(bib)}
    (paper/'validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if not result['passed']:raise ValueError('Manuscript static/provenance validation failed: '+json.dumps(issues))
    return result

def verified_run(run):
    summary=load(run/'summary.json')
    audit=load(run/'verification/audit.json')
    registration=load(run/'registration.json')
    if boolean(registration.get('smoke')):
        raise ValueError('Smoke results cannot be included as confirmatory results')
    if not boolean(summary.get('completed')) or not boolean(audit.get('passed')):
        raise ValueError('Only a completed, independently audited confirmatory run can be exported')
    return summary, audit

def lab_export(lab, out, paper):
    result=load(lab/'results.json')
    if result.get('status') != 'completed': raise ValueError('Laboratory scoring is incomplete')
    if load(lab/'audit.json').get('status') != 'passed': raise ValueError('Laboratory audit has not passed')
    metrics=read(lab/'prediction-metrics.csv')
    outcomes=read(lab/'diagnostic-outcomes.csv')
    calibration=load(lab/'calibration.json')
    for name in ['prediction-metrics.csv','diagnostic-outcomes.csv']:
        copy_file(lab/name,out/('lab_'+name))
    labels={'first_order':'First order','sensor_lag':'Two states with lag'}
    parameters=[]
    for model in labels:
        p=calibration[model]
        parameters.append({'model':model,'ambient_c':p['ambient_c'],'gain_full_scale_c':p['gain_full_scale_c'],'tau_h_s':p['tau_h_s'],'tau_sensor_s':p['tau_sensor_s']})
    csv_save(out/'lab_parameters.csv',parameters)
    tex_table(out/'lab_parameters.tex',['Model','Fitted environment (C)','Gain at full command (C)','$\\tau_h$ (s)','$\\tau_s$ (s)'],
              [[labels[r['model']],fmt(r['ambient_c'],2),fmt(r['gain_full_scale_c'],2),fmt(r['tau_h_s'],2),fmt(r['tau_sensor_s'],2)] for r in parameters],
              'Calibration-only fitted coefficients. These are model parameters, not independently measured thermal constants; the two-state poles are not uniquely assigned to physical components.', 'tab:lab-parameters')
    tex_table(out/'lab_prediction.tex',['Model','Split','$n$ samples','RMSE (C)','MAE (C)','Max error (C)','Envelope coverage (\\%)'],
              [[labels[r['model']],esc(r['split']),r['n'],fmt(r['rmse_c'],3),fmt(r['mae_c'],3),fmt(r['max_abs_error_c'],3),fmt(r['empirical_envelope_coverage'],1,.01)] for r in metrics],
              'Measured laboratory temperature prediction. Fits use calibration observations only; the held-out continuation has no state reset. Envelope coverage describes a fixed calibration-residual band, not a confidence interval. Sample counts are correlated time points from one experiment.', 'tab:lab-prediction')
    tex_table(out/'lab_diagnosis.tex',['Configuration','Eligible pairs','First supported outcome','Delay after OFF (s)','Opposing certificates'],
              [[esc(r['configuration']),r['eligible_pairs'],r['first_label'],fmt(r['first_support_after_off_s'],2),r['opposing_certificates']] for r in outcomes],
              'One held-out commanded OFF, with no measured current or independently labelled failure. Pairs are correlated evidence, not independent diagnostic trials. The laboratory adaptation has separate calibration-derived parameters.', 'tab:lab-diagnosis')
    first=result['metrics']['first_order']['held_out']; lag=result['metrics']['sensor_lag']['held_out']
    adapted=next(r for r in result['diagnostic_outcomes'] if r['configuration']=='lab_adapted_first_order')
    observation=result['commanded_off_observation']
    strict=[r for r in result['diagnostic_outcomes'] if r['configuration'].startswith('unchanged_v2')]
    text=(f"The source contains {result['calibration_rows']} calibration and {result['held_out_rows']} held-out temperature observations. "
          f"Only {result['independent_held_out_off_transitions']} held-out OFF transition is available. "
          f"First-order held-out RMSE is {fmt(first['rmse_c'],3)} C, compared with {fmt(lag['rmse_c'],3)} C for the predeclared two-state lag comparator. "
          f"Coverage of the calibration-residual bands is {fmt(first['empirical_envelope_coverage'],1,.01)}\\% and {fmt(lag['empirical_envelope_coverage'],1,.01)}\\%, respectively. "
          "This is evidence against treating the simple first-order free-run model as quantitatively adequate for this record, despite fitting its parameters on calibration. "
          "It is not evidence that the frozen synthetic generator has been physically validated.\n"
          "\\input{../tables/lab_parameters.tex}\n\\input{../tables/lab_prediction.tex}\n"
          f"After the logged OFF, measured temperature continues to rise by {fmt(observation['observed_rise_after_off_c'],2)} C, "
          f"reaching its observed peak {fmt(observation['observed_peak_after_off_s'],2)} s later. "
          "This illustrates why an isolated rising temperature cannot establish failed actuation. The measured response is better represented here by the predeclared lag model; the data do not uniquely identify which physical part produces each time constant.\n"
          f"All {len(strict)} unchanged v2 configurations have {sum(r['eligible_pairs'] for r in strict)} eligible pairs and remain insufficient over the observed {fmt(strict[0]['observed_off_duration_s'],2)} s OFF record. "
          f"The separately calibrated first-order interval check first supports the nominal commanded outcome after {fmt(adapted['first_support_after_off_s'],2)} s, "
          f"with {adapted['opposing_certificates']} opposing certificates. "
          "This nominal consistency result uses a broad calibration discrepancy envelope and shorter pair spans. Each compatibility pair starts from an observed temperature, whereas the free-run prediction accumulates error from the initial state; their different outcomes are therefore consistent. Neither result validates fault recall or justifies changing the frozen synthetic comparison.\n"
          "\\input{../tables/lab_diagnosis.tex}\n"
          "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{../figures/lab_physics.pdf}"
          "\\caption{Recorded laboratory temperatures and calibration-only model predictions. The shaded held-out interval is never used for fitting. The OFF detail shows measured thermal lag.}\\label{fig:lab-physics}\\end{figure}")
    write(paper/'laboratory-results.tex',text)
    return {'lab_first_order_rmse_c':first['rmse_c'],'lab_lag_rmse_c':lag['rmse_c'],
            'lab_nominal_delay_s':adapted['first_support_after_off_s'],'lab_independent_off':result['independent_held_out_off_transitions']}

def primary_export(run, out, paper, protocol):
    summary,_audit=verified_run(run)
    history=load(run/'full-history.json')
    endpoints=read(run/'tables/primary_endpoints.csv')
    cells=read(run/'tables/primary_cells.csv')
    aggregate=read(run/'tables/aggregate.csv')
    if len(endpoints)!=protocol['statistics']['primary_cells']*protocol['statistics']['endpoints']:
        raise ValueError('Incomplete primary endpoint family')
    grouped={}
    for r in endpoints:
        key=(r['profile'],r['schedule_id'],num(r['rate_bps']))
        grouped.setdefault(key,{})[r['endpoint']]=r
    rows=[]
    for key, value in sorted(grouped.items()):
        if set(value)!={'coverage_loss','excess_any_wrong','delay_gap_s'}: raise ValueError('Missing registered endpoint')
        coverage,error,delay=[value[e] for e in ['coverage_loss','excess_any_wrong','delay_gap_s']]
        ni=all(boolean(r['ni_supported']) for r in value.values());eq=all(boolean(r['eq_supported']) for r in value.values())
        row={'profile':key[0],'schedule_id':key[1],'rate_bps':int(key[2]),'n':int(num(coverage['n'])),'ni_supported':ni,'eq_supported':eq}
        for endpoint,r in value.items():
            for measure in ['mean_difference','ci_low','ci_high','margin']:
                row[endpoint+'_'+measure]=num(r[measure])
        rows.append(row)
    csv_save(out/'primary_summary.csv',rows)
    def interval(row,endpoint,scale=1,digits=2):
        return fmt(row[endpoint+'_mean_difference'],digits,scale)+' ['+fmt(row[endpoint+'_ci_low'],digits,scale)+', '+fmt(row[endpoint+'_ci_high'],digits,scale)+']'
    tex_table(out/'primary_summary.tex',['Profile','Schedule','bit/s','$n$','Coverage loss (pp)','Error excess (pp)','Delay excess (s)','NI / EQ'],
              [[PROFILES[r['profile']],SCHEDULES[r['schedule_id']],r['rate_bps'],r['n'],interval(r,'coverage_loss',.01),interval(r,'excess_any_wrong',.01),interval(r,'delay_gap_s',1,1),('yes' if r['ni_supported'] else 'no')+' / '+('yes' if r['eq_supported'] else 'no')] for r in rows],
              'Event summary minus contact full-history comparison, with coverage oriented as reference minus summary. Brackets are simultaneous intervals over the entire registered endpoint family. NI and EQ require all three endpoints. Non-support is not automatically evidence of inferiority.', 'tab:validation-primary')
    for name in ['primary_endpoints.csv','primary_cells.csv','aggregate.csv','paired_secondary.csv']:
        copy_file(run/'tables'/name,out/name)
    secondary=read(run/'tables/paired_secondary.csv')
    secondary_groups={}
    for r in secondary:
        key=(r['profile'],r['schedule_id'],num(r['rate_bps']),r['candidate'])
        secondary_groups.setdefault(key,{})[r['metric']]=r
    secondary_table=[]
    for key,metrics in sorted(secondary_groups.items()):
        record=[PROFILES[key[0]],SCHEDULES[key[1]],int(key[2]),METHODS[key[3]],metrics['resolved_fraction']['n']]
        for metric,scale,digits in [('resolved_fraction',.01,2),('capped_delay_s',1,1),('wire_bytes',1,1)]:
            r=metrics[metric]
            record.append(fmt(r['mean_difference'],digits,scale)+' ['+fmt(r['ci_low'],digits,scale)+', '+fmt(r['ci_high'],digits,scale)+']')
        secondary_table.append(record)
    tex_table(out/'secondary_paired.tex',['Profile','Schedule','bit/s','Candidate','$n$','Resolution difference (pp)','Delay difference (s)','Daily bytes difference'],secondary_table,
              'Prespecified secondary paired comparisons: candidate minus summary, using exactly the same registered subset. Positive resolution favors the candidate; negative delay or cost favors it. Brackets are descriptive pointwise episode-bootstrap intervals, outside the confirmatory family.', 'tab:validation-secondary')
    matched_budget=read(run/'tables/matched_budget.csv')
    copy_file(run/'tables/matched_budget.csv',out/'matched_budget.csv')
    boundary=[r for r in matched_budget if num(r['rate_bps'])==protocol['boundary_rate_bps']]
    tex_table(out/'boundary_summary.tex',['Profile','Schedule','$n$','Summary resolution (\\%)','Reference resolution (\\%)','Mean delay excess (s)','Daily bytes'],
              [[PROFILES[r['profile']],SCHEDULES[r['schedule_id']],r['n_episodes'],fmt(r['resolved_fraction_mean'],2,.01),fmt(r['oracle_resolved_fraction_mean'],2,.01),fmt(r['delay_gap_s_mean'],1),fmt(r['wire_bytes_mean'],1)] for r in boundary],
              'Registered secondary low-allocation boundary. The reference receives unlimited causal history at the same contact starts; it is not a feasible transmission competitor at the limited allocation. These descriptive estimates are outside the primary simultaneous family.', 'tab:validation-boundary')
    csv_save(out/'boundary_summary.csv',boundary)
    n_ni=sum(r['ni_supported'] for r in rows);n_eq=sum(r['eq_supported'] for r in rows)
    worst_cov=max(r['coverage_loss_ci_high'] for r in rows);worst_error=max(r['excess_any_wrong_ci_high'] for r in rows);worst_delay=max(r['delay_gap_s_ci_high'] for r in rows)
    n_independent=protocol['n_per_profile']*len(protocol['profiles'])
    sample_counts=[]
    for profile in protocol['profiles']:
        r=next(r for r in aggregate if r['role']=='primary' and r['profile']==profile)
        sample_counts.append({'profile':profile,'n_episodes':int(num(r['n_episodes'])),'late_failure_episodes':int(num(r['faulty_episodes'])),'no_failure_episodes':int(num(r['n_episodes'])-num(r['faulty_episodes']))})
    csv_save(out/'sampling_counts.csv',sample_counts)
    tex_table(out/'sampling_counts.tex',['Profile','Independent episodes','Late-failure episodes','No-failure episodes'],
              [[PROFILES[r['profile']],r['n_episodes'],r['late_failure_episodes'],r['no_failure_episodes']] for r in sample_counts],
              'Realized independent sample from the declared Bernoulli one-half episode mixture. A late-failure episode contains nominal OFF outcomes before failure; these counts are not target-class counts.', 'tab:validation-sample')
    max_absolute_error_upper=max(num(r['candidate_any_wrong_cp95_high']) for r in cells)
    largest_candidate_error_count=max(num(r['candidate_any_wrong_n']) for r in cells)
    largest_reference_error_count=max(num(r['reference_any_wrong_n']) for r in cells)
    timing=load(run/'verification/timing-sensitivity/audit.json')
    if timing.get('status')!='passed' or not timing.get('registered_analysis_unchanged'):
        raise ValueError('Timing diagnostic sensitivity is incomplete or modifies the primary analysis')
    reservation=load(ROOT.parent/'v2/configs/protocol-v2.json')['delivery']['ground_diagnosis_reservation_s']
    full_history=[]
    for profile in protocol['profiles']:
        rr=[r for r in history if r['profile']==profile]
        full_history.append({'profile':profile,'n_episodes':len({r['episode_id'] for r in rr}),'n_targets':len(rr),
                             'ever_correct':sum(r['first_correct_time_s'] is not None for r in rr),
                             'final_correct':sum(r['final_label']==r['truth_label'] for r in rr),
                             'ever_wrong':sum(r['ever_wrong'] for r in rr),'conflicting':sum(r['conflicting'] for r in rr)})
    if sum(r['ever_correct'] for r in full_history)!=summary['continuous_resolved'] or len(history)!=summary['continuous_full_history_targets']:
        raise ValueError('Continuous full-history summary differs from individual records')
    csv_save(out/'continuous_full_history.csv',full_history)
    unresolved_history=[r for r in history if r['first_correct_time_s'] is None]
    rare_explanation=''
    if unresolved_history:
        dwells=[r['end_s']-r['start_s'] for r in unresolved_history]
        rare_explanation=(f"Post-test inspection of the {len(unresolved_history)} unresolved continuous-history outcomes found nominal OFF intervals lasting {fmt(min(dwells)/60,0)}--{fmt(max(dwells)/60,0)} min between sparse current observations. "
                          "No current reading was acquired inside those OFF intervals, and all eligible thermal pairs remained compatible with both hypotheses under the frozen bounds. This is an acquisition-cadence and conservative-detector limitation rather than a delivery failure. It does not establish fundamental indistinguishability for every possible estimator. The explanatory case inspection is archived separately and did not change the model or thresholds.\n")
    tex_table(out/'continuous_full_history.tex',['Profile','Independent episodes','OFF targets','Ever correct','Finally correct','Ever wrong','Conflicting'],
              [[PROFILES[r['profile']],r['n_episodes'],r['n_targets'],r['ever_correct'],r['final_correct'],r['ever_wrong'],r['conflicting']] for r in full_history],
              'Continuous full-history diagnostic reference on the new synthetic sample. Ever-correct and ever-wrong counts can overlap; final correctness and contradictory certificates are retained. Targets within an episode are correlated.', 'tab:validation-full-history')
    text=(f"The registered validation sample contains {n_independent:,} independent episodes, with {protocol['n_per_profile']:,} in each profile. "
          f"Before restricting delivery, continuous full history resolves {summary['continuous_resolved']:,}/{summary['continuous_full_history_targets']:,} OFF outcomes by their deadlines and issues {summary['continuous_ever_wrong']} ever-wrong outcomes. "
          f"At the final recorded point, {sum(r['final_correct'] for r in full_history):,} outcomes are correct and {sum(r['conflicting'] for r in full_history)} contain contradictory certificates. "
          f"All {len(rows)} primary event-summary cells and {len(endpoints)} endpoints are retained. "
          f"Non-inferiority is supported in {n_ni}/{len(rows)} cells, and two-sided equivalence in {n_eq}/{len(rows)}, using simultaneous familywise intervals. "
          f"The largest upper bounds across these cells are {fmt(worst_cov,2,.01)} percentage points for resolution loss, "
          f"{fmt(worst_error,2,.01)} points for excess episode-any-error probability, and {fmt(worst_delay,1)} s for capped-delay excess. "
          f"The largest candidate and contact-reference any-error counts in a primary cell are {fmt(largest_candidate_error_count,0)} and {fmt(largest_reference_error_count,0)}, respectively; the largest descriptive candidate absolute-risk 95\\% upper bound is {fmt(max_absolute_error_upper,3,.01)}\\%. "
          "These are bounds for the declared synthetic population at the selected allocations, not a universal minimum transmission rate or a spacecraft safety certificate. The bandwidth-unlimited contact reference is an evidence-availability benchmark, not an attainable constrained-rate transmission policy.\n"
          "\\input{../tables/sampling_counts.tex}\n\\input{../tables/continuous_full_history.tex}\n"+rare_explanation+
          "\\input{../tables/primary_summary.tex}\n"
          f"Measured computation produced {timing['candidate_planner_overruns']} candidate planning overruns and {timing['candidate_diagnosis_overruns']} candidate diagnostic overruns, "
          f"but {timing['reference_overrun_events']} full-history reference overruns, with a maximum of {fmt(timing['reference_max_wall_s'],6)} s against a {fmt(reservation,1)} s reservation. "
          "The registered primary analysis retains these charged times. A separately labelled post-test timing sensitivity reconstructed fixed-reservation outcomes from saved evidence and contact starts. "
          f"It checked {timing['unique_reference_targets']:,} distinct reference targets across {timing['checked_repeated_target_rows']:,} repeated rows, with {timing['resolution_changes']} resolution changes and {timing['error_changes']} error changes. "
          f"Only {timing['affected_first_correct_reference_targets']} first-correct reference target shifts, by at most {fmt(timing['observed_max_first_correct_delay_shift_s'],6)} s; "
          f"all {timing['fixed_reserve_delay_ni_cells']} delay non-inferiority and {timing['fixed_reserve_delay_eq_cells']} delay-equivalence classifications remain supported. "
          "This is a diagnostic sensitivity, not a second confirmatory test or a replacement of the registered results. The complete per-contact timing log was not retained outside the trace subset; total/max overruns and all outcome-relevant shifts are available. Desktop timing is not a flight-latency distribution.\n"
          "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{../figures/primary_intervals.pdf}"
          "\\caption{Registered summary comparisons. Each row is a paired independent-episode comparison; shaded bands are the prespecified symmetric engineering margins. Error bars use the simultaneous finite-sample procedure, not an all-zero bootstrap.}\\label{fig:primary-intervals}\\end{figure}\n"
          f"Secondary policies use the preselected {protocol['secondary_n_per_profile']} episodes per profile, paired against summaries from that exact subset. "
          "Their pointwise intervals are reported separately from the confirmatory declarations. The registered low-budget check is likewise descriptive and cannot expand the primary guarantee to another allocation.\n"
          "\\input{../tables/secondary_paired.tex}\n\\input{../tables/boundary_summary.tex}\n"
          "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{../figures/budget_boundary.pdf}"
          "\\caption{Matched budget comparison on the same preselected episode subset at every allocation. Solid lines show summaries; dashed lines show bandwidth-unlimited contact history. Marginal bootstrap intervals are descriptive; primary inference separately uses the larger registered sample.}\\label{fig:validation-boundary}\\end{figure}\n"
          "\\begin{figure}[htbp]\\centering\\includegraphics[width=\\textwidth]{../figures/secondary_tradeoff.pdf}"
          "\\caption{Secondary policies minus summary on exactly the same episode subset. Negative cost and delay differences favor the candidate; annotations show resolution differences in percentage points, so lower cost cannot conceal reduced resolution. Points are descriptive paired means.}\\label{fig:validation-secondary}\\end{figure}")
    write(paper/'validation-results.tex',text)
    write(paper/'validation-abstract.tex',f"An independent validation of {n_independent:,} synthetic episodes supports registered non-inferiority in {n_ni}/{len(rows)} summary cells and two-sided equivalence in {n_eq}/{len(rows)} cells under simultaneous finite-sample bounds. A separate public laboratory case tests physical-model transfer and preserves failures of direct transfer.")
    return {'independent_episodes':n_independent,'primary_cells':len(rows),'primary_endpoints':len(endpoints),
            'ni_supported_cells':n_ni,'eq_supported_cells':n_eq,'worst_coverage_upper':worst_cov,
            'worst_error_upper':worst_error,'worst_delay_upper_s':worst_delay,
            'max_absolute_error_cp95_upper':max_absolute_error_upper,'max_candidate_error_count':largest_candidate_error_count,
            'max_reference_error_count':largest_reference_error_count,
            'continuous_resolved':summary['continuous_resolved'],'continuous_targets':summary['continuous_full_history_targets'],
            'continuous_ever_wrong':summary['continuous_ever_wrong'],
            'continuous_final_correct':sum(r['final_correct'] for r in full_history),'continuous_conflicting':sum(r['conflicting'] for r in full_history),
            'reference_cpu_overruns':timing['reference_overrun_events'],'timing_affected_targets':timing['affected_first_correct_reference_targets'],
            'timing_max_first_correct_shift_s':timing['observed_max_first_correct_delay_shift_s']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,default=Path('outputs/confirmatory-main'));p.add_argument('--lab-dir',type=Path,default=Path('lab/main'));p.add_argument('--output-dir',type=Path,default=Path('tables'));p.add_argument('--paper-dir',type=Path,default=Path('paper'));p.add_argument('--lab-only',action='store_true');a=p.parse_args()
    a.run_dir=resolve(a.run_dir);a.lab_dir=resolve(a.lab_dir);out=resolve(a.output_dir);paper=resolve(a.paper_dir)
    out.mkdir(exist_ok=True,parents=True);paper.mkdir(exist_ok=True,parents=True)
    if out.resolve() != (ROOT/'tables').resolve():
        shutil.copytree(ROOT/'tables/v2_archive',out/'v2_archive',dirs_exist_ok=True)
    if paper.resolve() != (ROOT/'paper').resolve():
        for source in (ROOT/'paper').iterdir():
            if source.is_file() and (source.suffix in ('.tex','.bib','.md') or source.name=='build-status.json'):
                copy_file(source,paper/source.name)
    numbers=lab_export(a.lab_dir,out,paper)
    sources=[a.lab_dir/'results.json',a.lab_dir/'calibration.json',a.lab_dir/'prediction-metrics.csv',a.lab_dir/'diagnostic-outcomes.csv',a.lab_dir/'audit.json']
    if not a.lab_only:
        protocol=load(ROOT/'configs/protocol-confirmatory.json')
        numbers.update(primary_export(a.run_dir,out,paper,protocol))
        earlier=read(ROOT/'tables/v2_archive/kazstsat_32_mixture.csv')
        old_summary=next(r for r in earlier if r['method']=='snapshot_events')
        old_compact=next(r for r in earlier if r['method']=='fresh_compact')
        old_n=int(num(old_summary['n_targets']));old_resolved=int(num(old_summary['resolved_targets']))
        old_reference=round(num(old_summary['oracle_resolved_fraction_mean'])*old_n)
        write(paper/'integrated-abstract.tex',
              "Fresh telemetry can be insufficient to distinguish legitimate operation from failed actuation when command context or separated observations are missing. We compare simple delivery representations under fixed diagnostic rules, repeated ON/OFF transitions, late faults, and limited LEO contact opportunities. Serialized context, framing and acknowledgements share the budget; unresolved outcomes remain in quality and capped-delay statistics. "
              f"In the preserved v2 KazSTSAT-window scenario at 32 bit/s, event summaries resolve {old_resolved}/{old_n} OFF outcomes versus {old_reference}/{old_n} for bandwidth-unlimited contact history, "
              f"using {fmt(old_summary['wire_bytes_mean'],0)} daily bytes versus {fmt(old_compact['wire_bytes_mean'],0)} for fresh compact delivery. "
              f"A new independent sample of {numbers['independent_episodes']:,} synthetic episodes supports non-inferiority in {numbers['ni_supported_cells']}/{numbers['primary_cells']} registered summary cells and two-sided equivalence in {numbers['eq_supported_cells']}/{numbers['primary_cells']} at {' and '.join(str(r) for r in protocol['primary_rates_bps'])} bit/s, using simultaneous finite-sample bounds with predeclared engineering margins. "
              f"A separate public laboratory trace exposes a physical-model limitation: held-out temperature RMSE is {fmt(numbers['lab_first_order_rmse_c'],2)} C for the first-order model and {fmt(numbers['lab_lag_rmse_c'],2)} C for a predeclared lag comparator. "
              "The unchanged v2 detector has no eligible pairs in the short laboratory OFF interval; a separately calibrated nominal consistency check does not validate actual fault sensitivity. The contribution is a reproducible map of context-summary sufficiency and failure boundaries. All KazSTSAT diagnostic telemetry and link allocations are synthetic; geometric windows are calculated from a frozen public orbital snapshot.")
        sources += [ROOT/'configs/protocol-confirmatory.json',ROOT.parent/'v2/configs/protocol-v2.json',a.run_dir/'summary.json',a.run_dir/'full-history.json',a.run_dir/'tables/primary_endpoints.csv',a.run_dir/'tables/primary_cells.csv',a.run_dir/'tables/aggregate.csv',a.run_dir/'tables/paired_secondary.csv',a.run_dir/'verification/audit.json',a.run_dir/'verification/timing-sensitivity/audit.json']
        sources.append(ROOT/'tables/v2_archive/kazstsat_32_mixture.csv')
        sources += [a.run_dir/'tables/matched_budget.csv',a.run_dir/'tables/matched_budget_manifest.json',ROOT/'docs/rare-full-history-cases.md']
        write(paper/'validation-discussion.tex',
              'The larger independent sample addresses the precision limitation of the earlier exploratory study under unchanged synthetic behavior. Its inferential unit remains an episode; increased counts of frames, transitions and schedule replays do not create independent evidence. The separately recorded laboratory case addresses a different limitation: physical-model transfer. The first-order free-run prediction fails quantitatively on the held-out segment, while the predeclared two-state comparator tracks the measured response substantially better. A broad interval check can still agree with nominal OFF without validating the assumed physical model or its response to a real fault. Context, physical-model adequacy and inferential precision therefore require separate evidence.')
        interpretation=('This establishes the registered aggregate equivalence result throughout the primary synthetic grid.'
                        if numbers['eq_supported_cells']==numbers['primary_cells'] else
                        'The supported applicability region is restricted to the reported cells; the remaining claims are not established.'
                        if numbers['ni_supported_cells'] else
                        'The larger sample does not establish the registered sufficiency criterion, and the negative or inconclusive results remain part of the applicability boundary.')
        write(paper/'validation-conclusion.tex',
              f"The independent validation supports non-inferiority in {numbers['ni_supported_cells']}/{numbers['primary_cells']} registered summary cells and two-sided equivalence in {numbers['eq_supported_cells']}/{numbers['primary_cells']}. {interpretation} The laboratory case limits its physical interpretation: calibration-specific nominal consistency is feasible, but unchanged-model transfer and actual fault sensitivity are not established. The supported contribution is a reproducible set of evidence-delivery conditions and failure boundaries, not a new scheduling algorithm or demonstrated flight performance.")
    (out/'validation_numbers.json').write_text(json.dumps(numbers,indent=2)+'\n',encoding='utf-8')
    manifest={'lab_only':a.lab_only,'inputs':[file_record(f) for f in sources],'exporter':file_record(Path(__file__))}
    (out/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(numbers,indent=2))

if __name__=='__main__':main()
