"""Post-test timing sensitivity; never replaces the registered primary analysis.

Reconstruct the contact reference at a fixed 0.1 s processing reservation.
This shortcut is valid only after proving that complete history has no opposing
or wrong certificates: evidence availability is then monotone in a causal prefix.
The recorded main run and all its actual wall-clock charges remain unchanged.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'outputs/confirmatory-main'


def read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    global RUN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='outputs/confirmatory-main')
    args = parser.parse_args()
    supplied = Path(args.run_dir)
    RUN = supplied if supplied.is_absolute() else ROOT/supplied
    summary = json.loads((RUN/'summary.json').read_text())
    registration = json.loads((RUN/'registration.json').read_text())
    assert summary['completed'] and not registration['smoke']
    history = json.loads((RUN/'full-history.json').read_text())
    assert not any(r['conflicting'] or r['ever_wrong_certificate'] or r['ever_wrong'] for r in history), 'Prefix shortcut invalid under conflicting evidence'
    manifest = [json.loads(line) for line in (RUN/'seed_manifest.jsonl').read_text().splitlines()]
    entries = {r['episode_id']: r for r in manifest}
    by_target = {(r['episode_id'], r['target_epoch']): r for r in history}
    orbital = read(ROOT.parent/'data/contacts/kazstsat.csv')
    runs = read(RUN/'runs.csv')
    planner_overruns = sum(int(r['planner_overruns']) for r in runs)
    receiver_overruns = sum(int(r['diagnosis_overruns']) for r in runs)
    assert planner_overruns == receiver_overruns == 0, 'Candidate timing needs its own counterfactual'
    contacts = {}
    for entry in manifest:
        seed = entry['seed']
        first = random.Random(seed+424242).uniform(300, 1800)
        contacts[(entry['episode_id'], 'synthetic_90min')] = [
            (first+i*5400, min(86400., first+i*5400+300)) for i in range(17) if first+i*5400 < 86400]
        offset = random.Random(seed+424242).uniform(0, 6*86400)
        contacts[(entry['episode_id'], 'kazstsat_astana_10deg')] = [
            (max(0., float(c['start_s'])-offset), min(86400., float(c['end_s'])-offset))
            for c in orbital if float(c['end_s'])-offset > 0 and float(c['start_s'])-offset < 86400]
    extra_bound = max(0., summary['reference_diagnosis_max_wall_s']-.1)
    effects = {}
    episode_differences = defaultdict(list)
    checked = 0
    for r in read(RUN/'targets.csv'):
        episode, schedule, epoch = r['episode_id'], r['schedule_id'], int(r['target_epoch'])
        target = by_target[(episode, epoch)]
        evidence = target['first_correct_time_s']
        candidate_times = [end+.1 for start, end in contacts[(episode, schedule)]
                           if evidence is not None and evidence <= start and
                           target['start_s'] <= end+.1 <= target['deadline_s']]
        fixed = min(candidate_times) if candidate_times else None
        actual = float(r['oracle_contact_s']) if r['oracle_contact_s'] else None
        assert (fixed is not None) == bool(int(r['oracle_contact_resolved']))
        assert (actual is not None) == (fixed is not None), 'Deadline crossing changed resolution'
        assert int(r['oracle_contact_ever_wrong']) == 0
        fixed_delay = fixed-target['start_s'] if fixed is not None else 21600.
        actual_delay = float(r['oracle_contact_capped_delay_s'])
        difference = actual_delay-fixed_delay
        assert -1e-8 <= difference <= extra_bound+1e-8
        if actual is not None:
            assert abs(actual_delay-(actual-target['start_s'])) < 1e-7
        event_key = (episode, schedule, epoch)
        record = dict(episode_id=episode, profile=entries[episode]['profile'], schedule_id=schedule,
                      target_epoch=epoch, actual_reference_time_s=actual, fixed_reserve_time_s=fixed,
                      delay_shift_s=difference)
        if event_key in effects:
            assert effects[event_key] == record
        effects[event_key] = record
        if r['role'] == 'primary':
            cell = (r['profile'], schedule, int(r['rate_bps']))
            episode_differences[(cell, episode)].append(float(r['capped_delay_s'])-fixed_delay)
        checked += 1
    cells = defaultdict(list)
    for (cell, episode), values in episode_differences.items():
        assert len(values) == 3
        cells[cell].append(statistics.fmean(values))
    delta = .05/36
    intervals = []
    for cell, values in sorted(cells.items()):
        assert len(values) == 2000
        variance = statistics.variance(values)
        half = math.sqrt(2*variance*math.log(4/delta)/len(values))+7*43200*math.log(4/delta)/(3*(len(values)-1))
        mean = statistics.fmean(values)
        lo, hi = max(-21600, mean-half), min(21600, mean+half)
        intervals.append(dict(profile=cell[0], schedule_id=cell[1], rate_bps=cell[2], n=len(values),
                              fixed_reserve_mean_gap_s=mean, ci_low=lo, ci_high=hi,
                              ni_supported=hi<900, eq_supported=-900<lo and hi<900))
    changed = [r for r in effects.values() if r['delay_shift_s'] > 1e-8]
    output = RUN/'verification/timing-sensitivity'
    output.mkdir(parents=True, exist_ok=True)
    with (output/'fixed-reserve-delay.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(intervals[0]))
        writer.writeheader(); writer.writerows(intervals)
    result = dict(status='passed', classification='Post-test diagnostic sensitivity, not a second prespecified confirmatory test',
                  registered_analysis_unchanged=True, script_sha256=sha(Path(__file__)),
                  candidate_planner_overruns=planner_overruns, candidate_diagnosis_overruns=receiver_overruns,
                  reference_overrun_events=summary['reference_diagnosis_overruns'],
                  reference_max_wall_s=summary['reference_diagnosis_max_wall_s'], maximum_extra_reference_time_s=extra_bound,
                  checked_repeated_target_rows=checked, unique_reference_targets=len(effects),
                  resolution_changes=0, error_changes=0, affected_first_correct_reference_targets=len(changed),
                  observed_max_first_correct_delay_shift_s=max((r['delay_shift_s'] for r in changed), default=0.),
                  affected_targets=changed,
                  fixed_reserve_delay_ni_cells=sum(r['ni_supported'] for r in intervals),
                  fixed_reserve_delay_eq_cells=sum(r['eq_supported'] for r in intervals),
                  fixed_reserve_max_delay_ci_upper_s=max(r['ci_high'] for r in intervals),
                  logical_basis='All complete-history targets have no wrong or opposing certificate. Earliest correct evidence plus the registered contact-start snapshot reconstructs fixed-reserve decisions. All primary candidate runs had no computation overrun, so their behavior is already reservation-limited.',
                  logging_limit='All seven original reference overrun timestamps were not retained outside the fixed trace subset. Their total/max and every reference first-correct target outcome were retained. This audit identifies all outcome-relevant shifts but does not invent a full per-contact CPU log.',
                  inference_limit='The registered intervals use realized charged times. Host timing is not a flight latency distribution; this separately labelled deterministic-reserve sensitivity checks that tiny host overruns do not explain the sufficiency classification.',
                  inputs={str(p.relative_to(ROOT)): sha(p) for p in [RUN/'summary.json',RUN/'full-history.json',RUN/'seed_manifest.jsonl',RUN/'runs.csv',RUN/'targets.csv']})
    (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
