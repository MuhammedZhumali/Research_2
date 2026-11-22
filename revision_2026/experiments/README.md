# Rolling summary composition and sensor response

This lossless follow-up uses 360 fresh synthetic episodes with repeated heater commands and calculated KazSTSAT visibility intervals. The registered rules, sensor model, seeds, grid, statistics and scope are in [protocol.json](protocol.json). [freeze.json](freeze.json) records source and input hashes before the main runs. The prior studies motivated this follow-up. The selected rates and 120 s sensor time constant are modeling assumptions.

## Replay

Run from the pilot root with Python 3.12.14. The revision requires the sibling frozen `v2/src/context_telemetry_v2` package, `v2/scripts/run_v2.py`, `v2/configs/protocol-v2.json`, `data/contacts/kazstsat.csv` and the original `requirements-lock.txt` identified in the freeze. The orbit data and generator have the same provenance as the v2 telemetric study.

```powershell
python -m pip install -r revision_2026/experiments/requirements.txt
python -B revision_2026/experiments/run.py --workers 3 --output revision_2026/experiments/replay
python -B revision_2026/experiments/analyze.py revision_2026/experiments/replay
python -B revision_2026/experiments/verify.py revision_2026/experiments/outputs --compare revision_2026/experiments/replay
```

Select a new output directory on each run. The unchanged simulator requires NumPy; the analysis also uses pandas. The production simulator used NumPy 2.5.4 and the analysis used NumPy 2.3.5 and pandas 3.0.1. A second complete run used the single pinned environment above for both stages. The four scientific CSVs matched byte for byte. Desktop wall times are kept separately and vary. A six-episode development smoke run uses `--smoke`, a different seed range and a separate output directory.

## Design and checks

The unit of replication is one generated 24-hour episode. Three historical OFF commands share its trajectory and schedule. Each underlying episode is replayed in 16 study cells. The revision compares first/latest observations, first/latest plus the first available current observation, and the frozen full summary. Each OFF interval retains its own command context. Endpoints keep their measured current field whenever one exists. In a temperature-only profile the two shorter policies select identical records, a property checked in every run. The selector only receives a prefix available at the start of the contact. All three policies use the frozen v2 packer, wire encoder/decoder, ACK/cache and detector. A parity check against the original v2 simulation compares encoded traffic totals, frame counts, certificates and per-target decisions in nine no-lag cells.

The single sensor perturbation filters the simulated noiseless temperature and adds the original paired measurement noise after filtering. A separate test checks prefix causality and compares the analytic sensor response with numerical integration. In all arms selection reserves 0.1 s of contact capacity and ground diagnosis ends 0.1 s after contact end. The measured operation times and reservation overruns are available for each run in `timings.csv`. The two full-history benchmarks are diagnostic references without a bounded wire allocation.

## Saved results

- `outputs/runs.csv` records every cell and seed, episode schedule offset, number of contacts and constrained wire costs. Empty byte fields identify an unlimited diagnostic reference.
- `outputs/targets.csv` records the target truth and deadline, first-correct time, issued errors, conflicting certificates, final correctness, absence of a decision and capped delay. The series has 1,080 distinct targets and 17,280 rows after reuse across conditions.
- `outputs/aggregate.csv` gives target counts by true outcome, means of equally weighted episodes and byte cost. `profile=all` weights the three equally sized profiles equally. Error and correct counts refer to outcomes by the six-hour deadline. An episode can produce contradictory evidence under sensor mismatch.
- `outputs/contrasts.csv` gives method-minus-reference or lag-minus-zero paired differences, stratifying whole episodes by profile and health assignment. Differences for rates are in percentage points, capped delay in hours and traffic in bytes per day. Intervals are pointwise and descriptive. A zero-width empirical error interval signals no observed paired variation.
- `outputs/audit.json` and `outputs/independent-validation.json` cover matched population counts, metric recomputation, numerical sensor checks and replay hashes.

At 32 bit/s the full summary resolved 671 of 1,080 targets versus 546 with endpoints and a current sample, at 641.34 versus 523.77 mean wire bytes/day. The gain of 125 resolved outcomes was confined to the temperature-only profile. There, the two extra representative times jointly raised resolution from 97 to 222 of 360 targets. In both current-bearing profiles, the shorter current-bearing rule matched the observed full-summary resolution and capped delay at this allocation. The comparison does not isolate the two extra times individually.

With a 120 s sensor lag the full summary resolved 664 of 1,080 targets at the same wire cost, including 177 of 277 failed OFF commands. The contact-history reference resolved 607 and showed conflicting certificates in 64 targets. Continuous history produced a correct decision at some time for 1,013, a final correct outcome for 967, and an issued error for 65. More measurements can expose mutually incompatible evidence under a mismatched detector. A higher delivery-policy count in this setting cannot establish more accurate physical modeling.
