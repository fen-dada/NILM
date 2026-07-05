# NILM Industrial Load Identification Project

This repository contains the code and project notes for industrial NILM
experiments. Raw waveform data and generated outputs are kept outside Git or
ignored by `.gitignore`.

## Folders

- `scripts/`: parsers, QC, feature extraction, event detection, pseudo-labeling,
  training, prediction, and pipeline scripts.
- `outputs/`: generated QC reports, features, event tables, labels, plots, and
  model reports. Generated content is ignored by Git.
- `models/`: trained model artifacts. Generated model files are ignored by Git.
- `notebooks/`: exploratory analysis notebooks.
- `docs/`: method notes, progress records, and experiment summaries.

## Current Data Format

The vendor waveform files are parsed as repeated records with no global file
header:

- record header: 18 bytes
  - `seq`: 4-byte unsigned int, little-endian
  - `data_len`: 4-byte unsigned int, little-endian
  - `timestamp`: 10 bytes
- content: `data_len` bytes
  - currently expected as 5376 bytes
  - `256 groups x 7 channels x 3 bytes`
  - channel order: `Ua, Ub, Uc, Ia, Ib, Ic, I0`
  - channel values are decoded as signed little-endian int24

## Current Workflow

The current end-to-end prototype is:

1. Parse raw binary waveform files.
2. Run QC checks.
3. Extract minute-level voltage/current/harmonic/power-like features.
4. Detect candidate load events from adjacent-minute feature changes.
5. Generate rule-based pseudo labels when no external operation log exists.
6. Train a first-pass event classifier.
7. Predict event type for candidate events.

Use `scripts/run_pipeline.py` as the high-level entry point for future
experiments from raw binary data.
