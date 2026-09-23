# Verification of the completed 500-episode experiment

The raw Colab download and full results ZIP are preserved locally. The submission notebook changes only the final explanatory markdown cell; every code cell, saved output, and execution counter remains identical to the raw export. The earlier notebooks and original result files are also preserved.

## Evidence checked

- Full ZIP integrity passed: 53 files, including 22 nonempty playback checkpoints, 22 GIFs, and nine configuration/log/evaluation/dashboard files.
- The [numerical audit](audit_500.md) passed 5,507 checks, with zero errors and one documented export warning.
- All 28 code-cell sources match the [prepared source notebook](pacman_dqn_500_prepared_source.ipynb). This source copy is deliberately unexecuted; the submission entry point is the [completed notebook](../pacman_dqn.ipynb).
- All 500 printed training rows and periodic decision/update totals reconcile with `training.csv`; every applied learning rate matches the planned schedule.
- Both final evaluation tables match the separate official and supplemental JSON results. Saved outputs include the checkpoint reload, final evaluation, dashboard, and gameplay evidence.
- All 22 GIFs were decoded; their sampled frames and the dashboard were visually reviewed. The report distinguishes excerpt scores from complete-game scores.
- All 31 small evidence files for this run are copied unchanged into `results/run_500/`. The large checkpoints remain in the full local ZIP.

## Colab export limitation

The live completed Colab session displayed execution counts 1–28. After the saved notebook was reopened in Dia, its downloaded `.ipynb` retained 59 outputs in nine code cells, including 24 image outputs, but all 28 `execution_count` fields were null and no `executionInfo` metadata remained. No counters were reconstructed. The archive-download cell was later rerun to retrieve the ZIP; training and evaluation were not rerun.

The auditor normally rejects missing counters. The explicit `--allow-colab-null-counts` option was used for this documented export only. It requires matching prepared code, Colab output metadata, all 500 printed training rows, progress counters, 20 periodic images, both score tables, and matching run identifiers in the saved outputs. It leaves a warning because individual cell execution counters cannot be verified from the export. Completed output evidence is present; counters are unavailable.

## Source hashes

| Preserved source | SHA-256 |
|---|---|
| Raw `pacman_dqn_500_decay.ipynb` | `b32f63237ce06918cba2e35fbbd60d6faf8bbfa3c5aacfca63c785e68bb772fb` |
| Full `20260918_160133_643037.zip` (137,808,612 bytes) | `349b0800359f68b840426fb6d59f959af9c31b0a5f459352064ab4197b773043` |
| Prepared source notebook | `9c58b8c2d1783f5430cb9d0d2220704b00fe8686305c3e10d322d7eaaa841d78` |

The annotated submission notebook has a different file hash because its final markdown explanation was completed after inspecting the results. Its code and outputs are unchanged.

## Reproduce the artifact audit

Extract the full local ZIP and run the following from the repository root, replacing the run path with the extracted directory containing `config.json` and the checkpoints:

```sh
python3 tools/audit_500.py \
  --run500 /absolute/path/to/extracted/500-run \
  --notebook500 pacman_dqn.ipynb \
  --prepared methods/pacman_dqn_500_prepared_source.ipynb \
  --allow-colab-null-counts \
  --json-out /tmp/pacman-500-audit.json \
  --markdown-out /tmp/pacman-500-audit.md
```

This reads artifacts without training or deserializing PyTorch checkpoints. File presence and size checks do not independently validate checkpoint weights; the saved notebook contains the actual checkpoint reload and evaluation. Public-access and browser-rendering checks are recorded separately in [publication verification](publication_verification.md). bCourses submission is performed separately by the student.
