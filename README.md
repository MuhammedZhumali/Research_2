# Diagnostic telemetry experiment and reproduction guide

This research package contains the scientific source, recorded inputs and reference results for the conference manuscript in `loss_extension/paper/`. It covers a simulated actuator, calculated KazSTSAT visibility windows, one public TCLab hardware record, the loss experiment and the subsequent summary-composition and sensor-response comparison. All contact windows and diagnostic stream allocations belong to the simulation. The laboratory capture contains one held-out nominal OFF.

Save the original `R2_Frame_Loss_Research.zip` from the review PDF's attachments panel or use the separately supplied ZIP. Extract it to inspect the saved results and run the verification below. To reproduce the review PDF including its attachment, copy that original ZIP into `loss_extension/paper/supplement/R2_Frame_Loss_Research.zip` (create the directory if needed) before compiling the supplied manuscript sources. This keeps the kit finite and lets readers verify the exact attached file.

## Quick start on Windows

Extract the ZIP to a short directory, for example `C:\r2`. Run the following commands from the extracted root in PowerShell. The original scientific environment used CPython 3.12.14. The revision used the same Python version with a different NumPy version. Use two virtual environments to reproduce both dependency sets. Installation requires access to the package index. The actual study uses archived inputs offline.

```powershell
py -3.12 -m venv .venv
py -3.12 -m venv .venv-revision
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv-revision\Scripts\python.exe -m pip install -r revision_2026/experiments/requirements.txt
.\.venv\Scripts\python.exe revision_2026/repro/replay.py verify
.\.venv\Scripts\python.exe revision_2026/repro/replay.py all --tag review-01 --revision-python .\.venv-revision\Scripts\python.exe
```

On Linux or macOS use `python3.12 -m venv` twice and the corresponding `.venv/bin/python` and `.venv-revision/bin/python` paths. The full replay generates several hundred MB of temporary products. Allow at least 2 GB of free disk space. Registered v3 validation is the longest step.

## Outputs and review

The final scientific comparison is written to `replay/review-01/comparison.json`. The rebuilt paper, including new numerical tables and vector plots, is in `replay/review-01/paper/`. To compile it, select `main.tex` in Overleaf with pdfLaTeX, or use a local TeX Live installation:

```text
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Every simulation receives a new output tag and refuses to overwrite an earlier simulation. To resume after an interrupted downstream analysis, use `analyze --study <study> --tag <tag>` or `paper --tag <tag>` with the same scientific environment as above. The `simulate` and `analyze` stages also accept one study at a time, using the names below. Pass `--revision-python` when running the revision stage.

| Study | Computation | Fresh outputs |
|---|---|---|
| orbit | Propagate the frozen public TLE/OMM, check identifiers and numerical visibility, verify the 28 generated windows against the archived CSV | `replay/<tag>/orbit/` |
| v2 | 120 independent baseline episodes, repeated matched deliveries, single-factor stress tests, held-out audit, worked contact trace and tables | `v2/outputs/<tag>/` |
| v3 | 6,000 new episodes from the registered seed manifest, 36,000 deliveries, 36 simultaneous primary intervals, matched budget boundary and missed-outcome attribution | `v3_validation/outputs/<tag>/`, `replay/<tag>/attribution/` |
| lab | Refit both registered models to the 400 calibration observations, score the 200 held-out observations, and independently check predictions and diagnostic limits | `v3_validation/lab/<tag>/` |
| loss | Generate all 480 episodes and 17,280 deliveries, reanalyze the stratified paired comparisons, check each contact and frame ledger | `loss_extension/experiment/outputs/<tag>/`, `replay/<tag>/loss-analysis/` |
| revision | Regenerate the registered composition and sensor-response scenarios. Compare original and replay decisions, costs, aggregates and contrasts | `replay/<tag>/revision/` |

The v2/v3/loss simulators record local computation times. Algorithms charge overruns against the modeled contact budget where specified. Absolute times and execution-speed diagnostics can vary. The numerical comparison checks target populations, decisions, traffic, detection and attribution. All 36 primary interval estimates and limits must agree to 0.1 s or 0.01 percentage point after rounding, and the regenerated LaTeX tables and numerical text must match the published text. The comparison also reports maximum raw timing and interval changes. The revision records computation timings alongside its fixed-reservation outcomes. The archived reference CSVs provide the original observed results.

## Provenance and archive scope

The input manifests under `provenance/` list the original URLs, retrieval metadata and hashes. The active orbital manifest lists the three numerical element records. The active laboratory manifest lists the source data, acquisition context and license. The original manifests also identify ancillary PDFs, entire upstream repositories and mission-description pages. These ancillary records can be obtained using their cited sources. The scientific programs validate every packaged input listed in their active source manifests. The published research protocols and quantitative inputs retain their original bytes. The archive attributes the TCLab source to the APMonitor Arduino project (license at `v3_validation/sources/lab/LICENSE`). The proceedings class and bibliography style carry the publisher's license in `loss_extension/paper/template/README-llncs.md`.

The final manuscript includes the recorded reference values and figures. The replay builds a separate copy from regenerated observations and decisions and preserves the published copy for comparison. `SHA256SUMS.txt` checks file integrity against the package manifest. The repository attributes preserve bytes across checkouts for the frozen source and input hashes.
