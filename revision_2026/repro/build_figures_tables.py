"""Connect the fresh simulation outputs to the actual manuscript's exporters."""
import argparse
import shutil
import sys

from replay import LOSS, ROOT, V2, V3, run


def build(tag):
    output = ROOT / "replay" / tag
    v2_run = V2 / "outputs" / tag
    v3_run = V3 / "outputs" / tag
    lab = V3 / "lab" / tag
    manuscript = output / "paper"
    if manuscript.exists():
        raise FileExistsError(f"Paper already rebuilt under tag {tag}. Choose a fresh tag.")
    shutil.copytree(LOSS / "paper", manuscript,
                    ignore=shutil.ignore_patterns("__pycache__", "*.aux", "*.log", "*.out", "*.bbl", "*.blg", "R2_*.pdf"))
    data = manuscript / "data"
    sys.path.insert(0, str(V3 / "scripts"))
    from export_validation_tables import load, primary_export
    v3_exports = v3_run / "paper-data"
    v3_exports.mkdir(exist_ok=True)
    primary_export(v3_run, v3_exports, v3_exports, load(V3 / "configs/protocol-confirmatory.json"))
    figure_names = ("budget_boundary", "lab_physics", "loss_detection", "workflow_trace")
    for name in figure_names:
        for extension in (".pdf", ".png"):
            (manuscript / "figures" / (name + extension)).unlink(missing_ok=True)
    for name in ("plant_parameters.csv", "kazstsat_32_mixture.csv"):
        shutil.copyfile(v2_run / "paper-tables" / name, data / name)
    shutil.copyfile(v3_exports / "primary_summary.csv", data / "primary_summary.csv")
    for name in ("primary_rates.csv", "primary_contrasts.csv", "all_cells.csv", "exploratory_frame_path_agreement.csv"):
        shutil.copyfile(output / "loss-analysis" / name, data / name)
    shutil.copyfile(output / "loss-verification/independent-check.json", data / "independent-check.json")
    # Keep the original compact files in the publication directory immutable.
    stage = output / "revision-inputs"
    for folder, source, names in [
        ("revision_2026/attribution/outputs", output / "attribution", ["outcomes.csv"]),
        ("v2/tables", v2_run / "paper-tables", ["full_history.csv", "robustness_32.csv"]),
        ("revision_2026/experiments/outputs", output / "revision", ["aggregate.csv", "contrasts.csv"]),
    ]:
        destination = stage / folder
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copyfile(source / name, destination / name)
    tools = manuscript / "tools"
    run(tools / "export_tables.py")
    run(tools / "export_loss_text.py")
    run(tools / "export_revision.py", "--pilot", stage)
    run(tools / "plot_budget_lab.py", "--budget-csv", v3_run / "tables/matched_budget.csv",
        "--lab-dir", lab)
    run(tools / "plot_loss_detection.py")
    trace = v2_run / "traces/test-clear_current-000__base__kazstsat_astana_10deg__32__snapshot_events.json"
    run(tools / "plot_workflow.py", "--trace", trace)
    for name in figure_names:
        for extension in (".pdf", ".png"):
            if not (manuscript / "figures" / (name + extension)).is_file():
                raise FileNotFoundError(name + extension)
    print(f"Rebuilt paper inputs and vector figures under replay/{tag}/paper.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    build(parser.parse_args().tag)
