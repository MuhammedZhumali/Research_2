"""Reproduce the scientific inputs and paper in a new, named output tree."""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V2 = ROOT / "v2"
V3 = ROOT / "v3_validation"
LOSS = ROOT / "loss_extension"
REV = ROOT / "revision_2026"


def run(script, *args, python=None):
    command = [python or sys.executable, "-u", "-B", "-X", "utf8", str(script), *map(str, args)]
    print("+", Path(script).name, *[str(a).replace(str(ROOT), ".") for a in args], flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def verify():
    manifest = ROOT / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise ValueError("Run verify inside the unpacked public research release.")
    for entry in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = entry.split("  ", 1)
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Changed or missing release file: {name}")
    print(f"Verified {len(manifest.read_text().splitlines())} package files.", flush=True)


def check_revision_environment(python):
    lock = REV / "experiments/requirements.txt"
    pins = dict(line.split("==") for line in lock.read_text().splitlines() if line.strip())
    query = "from importlib.metadata import version; import sys; print(' '.join(version(n) for n in sys.argv[1:]))"
    check = subprocess.run([python, "-B", "-c", query, *pins], capture_output=True, text=True)
    if check.returncode or check.stdout.split() != list(pins.values()):
        raise ValueError("The revision needs " + ", ".join(f"{k}=={v}" for k, v in pins.items())
                         + ". Install revision_2026/experiments/requirements.txt and pass its interpreter with --revision-python.")


def simulate(study, tag, workers, revision_python):
    output = ROOT / "replay" / tag
    if study == "orbit":
        run(ROOT / "scripts/build_contacts.py", "--output-dir", output / "orbit")
        if (output / "orbit/kazstsat.csv").read_bytes() != (ROOT / "data/contacts/kazstsat.csv").read_bytes():
            raise ValueError("Regenerated geometric windows differ from the frozen schedule.")
    elif study == "v2":
        run(V2 / "scripts/run_v2.py", "--split", "test", "--output", V2 / "outputs" / tag)
    elif study == "v3":
        run(V3 / "scripts/run_confirmatory.py", "--output", V3 / "outputs" / tag)
    elif study == "lab":
        run(V3 / "scripts/validate_lab.py", "--output", V3 / "lab" / tag)
    elif study == "loss":
        run(LOSS / "experiment/run_experiment.py", "--freeze", LOSS / "verification/pretest-freeze.json",
            "--output", LOSS / "experiment/outputs" / tag, "--workers", workers)
    elif study == "revision":
        run(REV / "experiments/run.py", "--output", output / "revision", "--workers", workers, python=revision_python)


def analyze(study, tag, revision_python):
    output = ROOT / "replay" / tag
    if study == "v2":
        run_dir = V2 / "outputs" / tag
        for script in ("analyze_v2", "verify_v2"):
            run(V2 / f"scripts/{script}.py", "--run-dir", run_dir)
        run(V2 / "scripts/export_tables_v2.py", "--run-dir", run_dir,
            "--output-dir", run_dir / "paper-tables", "--paper-dir", run_dir / "paper-fragments")
    elif study == "v3":
        run_dir = V3 / "outputs" / tag
        for script in ("analyze_confirmatory", "verify_confirmatory", "audit_timing_v3", "derive_matched_budget"):
            run(V3 / f"scripts/{script}.py", "--run-dir", run_dir)
        run(REV / "attribution/attribute.py", "--runs", run_dir, "--output", output / "attribution")
    elif study == "lab":
        lab = V3 / "lab" / tag
        run(V3 / "lab/verify_lab_results.py", "--run", lab, "--output", lab / "audit.json")
    elif study == "loss":
        run_dir = LOSS / "experiment/outputs" / tag
        for script in ("analyze_loss", "verify_ledger"):
            run(LOSS / f"analysis/{script}.py", "--source", run_dir, "--output", output / "loss-analysis")
        run(LOSS / "verification/check_study.py", "--data", run_dir, "--output", output / "loss-verification")
    elif study == "revision":
        run(REV / "experiments/analyze.py", output / "revision", python=revision_python)
        run(REV / "experiments/verify.py", output / "revision", "--compare", REV / "experiments/outputs",
            "--report", output / "revision/independent-validation.json", python=revision_python)


def paper(tag):
    run(HERE / "build_figures_tables.py", tag)
    run(HERE / "compare.py", tag)


def main():
    studies = ["orbit", "v2", "v3", "lab", "loss", "revision"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("verify", "simulate", "analyze", "paper", "all"))
    parser.add_argument("--study", choices=studies)
    parser.add_argument("--tag", default="review-01")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--revision-python", type=Path, help="Python from the smaller revision-specific locked environment.")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.tag) or args.tag in ("main", "heldout", "confirmatory-main", "confirmatory-design"):
        parser.error("Choose a new simple output tag such as review-01.")
    if args.workers < 1 or (args.study and args.stage not in ("simulate", "analyze")):
        parser.error("Use a positive worker count and select --study only with simulate or analyze.")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(ROOT / "replay" / args.tag / ".mpl-cache")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    if args.stage in ("all", "verify"):
        verify()
    revision_python = args.revision_python.resolve() if args.revision_python else sys.executable
    if args.stage in ("all", "simulate", "analyze") and args.study in (None, "revision"):
        check_revision_environment(revision_python)
    for study in ([args.study] if args.study else studies):
        if args.stage in ("all", "simulate"):
            simulate(study, args.tag, args.workers, revision_python)
        if args.stage in ("all", "analyze"):
            analyze(study, args.tag, revision_python)
    if args.stage in ("all", "paper"):
        paper(args.tag)


if __name__ == "__main__":
    main()
