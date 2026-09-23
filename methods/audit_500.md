# 500-episode Pac-Man audit

**PASS** — 5507 checks; 0 errors; 1 warning.

Generated: 2026-09-18T16:28:29.106105+00:00

Run: `local extracted 500-run archive`

Status: **completed**. Episodes: **500**; decisions: **292152**; updates: **72789**.

Elapsed training seconds including periodic demos: 1150.267762866. Initial/final learning rates: 0.0001 / 5.120000000000001e-05.

## Official evaluation

| Seed | Untrained | Trained | Change |
|---|---:|---:|---:|
| 101 | 200 | 430 | +230 |
| 202 | 240 | 580 | +340 |
| 303 | 240 | 510 | +270 |
| 404 | 240 | 480 | +240 |
| 505 | 170 | 250 | +80 |
| **Mean** | **218.0** | **450.0** | **+232.0** |

Improved/tied/worse: 5/0/0. Time-limited games before/after: 0/0.

## Supplemental evaluation

| Seed | Untrained | Trained | Change |
|---|---:|---:|---:|
| 606 | 240 | 910 | +670 |
| 707 | 230 | 420 | +190 |
| 808 | 300 | 830 | +530 |
| 909 | 240 | 370 | +130 |
| 1010 | 250 | 1900 | +1650 |
| **Mean** | **252.0** | **886.0** | **+634.0** |

Improved/tied/worse: 5/0/0. Time-limited games before/after: 0/0.

## Learning-rate boundary checks

| Episode | Recorded rate |
|---|---:|
| 1 | 0.0001 |
| 125 | 0.0001 |
| 126 | 8e-05 |
| 250 | 8e-05 |
| 251 | 6.400000000000001e-05 |
| 375 | 6.400000000000001e-05 |
| 376 | 5.120000000000001e-05 |
| 500 | 5.120000000000001e-05 |

Every CSV row was checked against the schedule, not only these boundaries.

Notebook: 28 code cells, 59 saved outputs, 24 image outputs. Exact code match with prepared source: **True**.

**Export limitation:** all 28 execution counters are null. This opt-in audit uses matching prepared code and preserved outputs reconciled with all 500 CSV rows and progress counters. It does not reconstruct or verify individual cell execution counts.

## Findings

- **WARNING — colab_null_execution_counts:** All 28 execution counters are absent in this Colab export. No counters were reconstructed. Execution evidence is limited to preserved outputs, exact prepared-code matching, and artifact reconciliation; individual cell counters cannot be verified.

## Verification boundaries

- Input files were read only. No training or PyTorch/pickle deserialization was performed.
- Checkpoint validation establishes presence and nonempty files only. The executed notebook must perform the actual saved-model reload.
- GIF/PNG validation covers headers and dimensions, not full decoding, animation playback, or behavior interpretation.
- ZIP integrity, README evidence links, public GitHub rendering/access, and course submission must be checked separately.
- Saved execution counts and output checks do not independently rerun the notebook. Exact source comparison is limited to the supplied prepared notebook.
- Both five-seed evaluations use .05 exploration and a 3000-decision limit. Supplemental results are separate from the official classroom score.
- Training seed, episode budget, and learning-rate schedule changed together; this is not a causal test of one change. Fixed seeds do not force deterministic CUDA execution.
- Training elapsed time includes periodic demos and excludes setup, baseline/final evaluations, and archive download.
