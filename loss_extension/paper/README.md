# Diagnostic telemetry conference manuscript

Import the source ZIP into Overleaf, select `main.tex` as the main document and use pdfLaTeX. For a local build, run `pdflatex main`, `bibtex main`, then `pdflatex main` twice. The anonymous manuscript uses the Springer `llncs` class, A4 paper and a centered 122 × 193 mm text area. The bibliography is included in the 12 pages. The class license is in `template/README-llncs.md`.

The Overleaf project includes `supplement/R2_Frame_Loss_Research.zip`. The review PDF embeds that file as an attachment. In a PDF viewer with an attachments panel, save the ZIP, extract it and open its root `README.md` for the protocols, results and offline replay instructions. When rebuilding from the standalone research kit, place the original ZIP at `supplement/R2_Frame_Loss_Research.zip` beside the manuscript before compiling.

Figure 1 follows one OFF command from onboard sampling to ground confirmation. Figure 2 shows the gap between event summaries and complete contact history at each budget. Figure 3 and Table 8 compare fault detection under the three frame-loss conditions. Figure 4 compares two temperature predictions with the public nominal TCLab record. The independent lossless validation, subsequent composition/lag study and held-out loss experiment have separate samples and endpoints.

## Regenerate the manuscript material

From this directory, regenerate the tables and numerical text from the compact files in `data/` (Python standard library):

```
python tools/export_tables.py
python tools/export_revision.py
python tools/export_loss_text.py
```

Regenerate all four figures (Matplotlib 3.11.2):

```
python tools/plot_workflow.py
python tools/plot_budget_lab.py
python tools/plot_loss_detection.py
```

The manuscript uses the vector PDFs. Each plotting script also writes a PNG. The inputs in `data/budget_lab/` come from the v3 matched-budget results and the scored public APMonitor TCLab record. The latter includes measurements from `initial_models.csv` (commit `f36e65a70dd7122d1829883899e40e56bf6c4279`, license in `data/budget_lab/TCLab-LICENSE.txt`). To refresh these figure inputs from a simulation replay, pass `--budget-csv PATH --lab-dir DIRECTORY` to `plot_budget_lab.py`.

The separate research package documents how to regenerate the simulations, references and orbital windows. For Figure 1, use `--trace PATH` to refresh the compact input from `test-clear_current-000__base__kazstsat_astana_10deg__32__snapshot_events.json`. Run `python tools/export_revision.py --pilot PATH` against a complete research tree to refresh the compact inputs for the deadline, robustness and ablation results.

The [conference submission instructions](https://www.icecc.org/sub.html) and [proceedings template](https://www.icecc.org/files/template.doc) are the source for the review format. The class and bibliography style are from [CTAN](https://ctan.org/pkg/llncs).
