"""Run the NILM workflow from raw binary waveform files.

Typical inference flow:
1. raw binary files -> minute features
2. minute features -> candidate events
3. candidate events -> model predictions

Optional bootstrap flow can also create rule-based pseudo labels and retrain the
baseline classifier when no external operation log is available.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end NILM pipeline from raw binary data.")
    parser.add_argument("--raw-data-dir", type=Path, default=Path(r"E:/华鹏波形数据"))
    parser.add_argument("--features", type=Path, default=Path("outputs/features/minute_features.csv"))
    parser.add_argument("--events", type=Path, default=Path("outputs/events/candidate_events.csv"))
    parser.add_argument("--labels", type=Path, default=Path("outputs/labels/auto_event_labels.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/event_classifier_rf.json"))
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions/event_predictions.csv"))
    parser.add_argument("--model-report-dir", type=Path, default=Path("outputs/model_reports/event_classifier"))
    parser.add_argument("--project-dir", type=Path, default=Path(r"E:/NILM_Project"))
    parser.add_argument("--top-n", type=int, default=300)
    parser.add_argument("--min-score", type=float, default=20.0)
    parser.add_argument("--max-fft-records", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Limit raw files for a quick smoke test")
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--skip-event-detection", action="store_true")
    parser.add_argument("--make-pseudo-labels", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true", help="Run prediction using an existing or newly trained model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_feature_extraction:
        command = [
            python,
            "scripts/extract_features.py",
            "--raw-data-dir",
            str(args.raw_data_dir),
            "--output",
            str(args.features),
            "--max-fft-records",
            str(args.max_fft_records),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        run_step(command, dry_run=args.dry_run)

    if not args.skip_event_detection:
        run_step(
            [
                python,
                "scripts/detect_events.py",
                "--features",
                str(args.features),
                "--output",
                str(args.events),
                "--top-n",
                str(args.top_n),
                "--min-score",
                str(args.min_score),
            ],
            dry_run=args.dry_run,
        )

    if args.make_pseudo_labels or args.train:
        run_step(
            [
                python,
                "scripts/auto_label_events.py",
                "--events",
                str(args.events),
                "--output",
                str(args.labels),
                "--project-dir",
                str(args.project_dir),
            ],
            dry_run=args.dry_run,
        )

    if args.train:
        run_step(
            [
                python,
                "scripts/train_event_classifier.py",
                "--events",
                str(args.events),
                "--labels",
                str(args.labels),
                "--output-dir",
                str(args.model_report_dir),
                "--model-output",
                str(args.model),
            ],
            dry_run=args.dry_run,
        )

    if args.predict or args.train:
        run_step(
            [
                python,
                "scripts/predict_event_classifier.py",
                "--events",
                str(args.events),
                "--model",
                str(args.model),
                "--output",
                str(args.predictions),
            ],
            dry_run=args.dry_run,
        )

    print("pipeline complete")


if __name__ == "__main__":
    main()
