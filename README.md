# NILM Industrial Load Identification Project

This folder keeps project code and generated outputs separate from the raw waveform data.

## Folders

- `scripts/`: Python scripts for decoding waveform files, quality checks, plotting, and feature extraction.
- `outputs/qc/`: Quality-control tables and summaries.
- `outputs/sample_plots/`: Waveform plots used to verify decoding.
- `outputs/features/`: Extracted cycle-level or minute-level features for modeling.
- `notebooks/`: Exploratory analysis notebooks.
- `models/`: Trained model files and evaluation results.
- `docs/`: Project notes, method summaries, and experiment records.

## First Milestone

Decode the binary waveform files and generate a quality-control report:

- Parse each frame as 5394 bytes.
- Treat the first 18 bytes as frame header.
- Treat the remaining 5376 bytes as `896 samples x 3 channels x int16`.
- Plot sample waveforms from each measurement point.
- Scan all files for missing data, abnormal frame counts, clipping, and channel statistics.
