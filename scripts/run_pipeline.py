"""Run the NILM workflow from raw binary waveform files.

Current flow:
1. raw binary files -> minute features
2. minute features -> event detection
3. candidate events -> phase-aligned steady waveform differences
4. submeter waveform events -> random forest with chronological holdout
5. main-meter waveform events -> startup/stop and submeter attribution
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
    parser.add_argument("--raw-data-dir", type=Path, default=Path("E:/\u534e\u9e4f\u6ce2\u5f62\u6570\u636e"))
    parser.add_argument("--features", type=Path, default=Path("outputs/features/minute_features.csv"))
    parser.add_argument("--events", type=Path, default=Path("outputs/events/candidate_events.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/steady_waveform_rf.json"))
    parser.add_argument("--model-report-dir", type=Path, default=Path("outputs/model_reports/steady_waveform_rf"))
    parser.add_argument("--project-dir", type=Path, default=Path(r"E:/NILM_Project"))
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--min-score", type=float, default=100.0)
    parser.add_argument("--target-min-score", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-fft-records", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Limit raw files for a quick smoke test")
    parser.add_argument("--skip-feature-extraction", action="store_true")
    parser.add_argument("--skip-event-detection", action="store_true")
    parser.add_argument("--train", action="store_true")
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
        command = [
                python,
                "scripts/detect_events.py",
                "--features",
                str(args.features),
                "--output",
                str(args.events),
                "--min-score",
                str(args.min_score),
            ]
        if args.top_n is not None:
            command.extend(["--top-n", str(args.top_n)])
        run_step(command, dry_run=args.dry_run)

    if args.train:
        run_step(
            [
                python,
                "scripts/train_steady_waveform_rf.py",
                "--raw-data-dir",
                str(args.raw_data_dir),
                "--features",
                str(args.features),
                "--train-min-score",
                str(args.min_score),
                "--target-min-score",
                str(args.target_min_score),
                "--train-ratio",
                str(args.train_ratio),
                "--report-dir",
                str(args.model_report_dir),
                "--model-output",
                str(args.model),
            ],
            dry_run=args.dry_run,
        )

    print("pipeline complete")


if __name__ == "__main__":
    main()
