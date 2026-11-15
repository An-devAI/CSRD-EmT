"""Command-line training and audited LOSO aggregation."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from . import training as run


def summarize(output):
    rows = []
    for fold in sorted(output.glob("fold_*")):
        if not (fold / "receipt.json").exists():
            continue
        run.verify_receipt(fold)
        row = json.loads((fold / "result.json").read_text())
        pred = np.load(fold / "target_predictions.npz", allow_pickle=False)
        for key, value in [
            ("ba", balanced_accuracy_score(pred["y"], pred["pred"])),
            ("macro_f1", f1_score(pred["y"], pred["pred"], average="macro")),
            ("accuracy", accuracy_score(pred["y"], pred["pred"])),
        ]:
            if abs(row[key] - value) > 1e-10:
                raise ValueError(f"Metric mismatch: {fold.name}/{key}")
        rows.append(row)
    targets = [r["target"] for r in rows]
    if len(set(targets)) != len(targets):
        raise ValueError("Duplicate target results")
    complete = set(targets) == set(range(1, 16))
    summary = dict(
        completed_targets=targets,
        complete=complete,
        mean_ba=float(np.mean([r["ba"] for r in rows])) if complete else None,
        mean_macro_f1=(
            float(np.mean([r["macro_f1"] for r in rows])) if complete else None
        ),
    )
    output.mkdir(parents=True, exist_ok=True)
    run.save(output / "summary.json", summary)
    pd.DataFrame(rows).to_csv(output / "fold_results.csv", index=False)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument(
        "--targets", default="all", help="all or comma-separated IDs 1..15"
    )
    train.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    train.add_argument("--resume", action="store_true")
    report = sub.add_parser("summarize")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.output)
        return
    targets = (
        list(range(1, 16))
        if args.targets == "all"
        else [int(v) for v in args.targets.split(",")]
    )
    if (
        not targets
        or len(set(targets)) != len(targets)
        or not set(targets) <= set(range(1, 16))
    ):
        parser.error("Targets must be distinct IDs in 1..15")
    if not args.data.exists():
        parser.error("Data path does not exist")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu explicitly")
    if not args.resume and any(
        (args.output / f"fold_{t:02}").exists() for t in targets
    ):
        parser.error("Existing folds found; choose a new output or use --resume")
    run.DEVICE = args.device
    expected = json.loads((Path(__file__).parent / "seed_hashes.json").read_text())
    for target in targets:
        run.run_fold(target, args.data, args.output, expected)
    summarize(args.output)


if __name__ == "__main__":
    main()
