#!/usr/bin/env python3
"""Read-only, standard-library audit of the authorized 500-episode Pac-Man run.

python3 tools/audit_500.py --run500 EXTRACTED_DIR --notebook500 EXECUTED.ipynb \
  --prepared prepared/pacman_dqn_500_decay_UNEXECUTED.ipynb \
  --json-out audits/500.json --markdown-out audits/500.md

Exit 0 means no failed checks; 1 means validation errors; 2 means CLI misuse.
Checkpoint contents are never deserialized. Input ZIP integrity is a separate check.
"""
from __future__ import annotations

import argparse
import ast
import bisect
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


OFFICIAL = [101, 202, 303, 404, 505]
SUPPLEMENTAL = [606, 707, 808, 909, 1010]
PERIODIC = list(range(25, 501, 25))
FIELDS = ["episode", "score", "steps", "total_steps", "exploration", "mean_loss",
          "elapsed_seconds", "terminated", "truncated", "learning_rate"]
PACKAGES = ["torch", "gymnasium", "ale-py", "opencv-python-headless", "numpy", "matplotlib", "Pillow"]
FIXED = {
    "exploration": .20, "episodes_requested": 500, "learning_rate": .0001,
    "seed": 31415, "environment": "ALE/MsPacman-v5", "frame_skip": 4,
    "sticky_action_probability": .25, "noop_max": 30, "grayscale_size": 84,
    "stack_size": 4, "terminal_on_life_loss": False, "max_decisions_per_game": 3000,
    "replay_capacity": 5000, "warmup_decisions": 1000, "batch_size": 32,
    "train_every_decisions": 4, "target_sync_decisions": 1000, "gamma": .99,
    "training_reward_clipping": [-1, 1], "eval_exploration": .05,
    "eval_seeds": OFFICIAL, "supplemental_eval_seeds": SUPPLEMENTAL,
    "preview_seconds": 20, "preview_speed": 4, "preview_plays": 2,
    "preview_stride": 4, "preview_frame_ms": 67,
}
PARAMETERS = {"EXPLORATION": .20, "EPISODES": 500, "LEARNING_RATE": .0001,
              "SEED": 31415, "LR_DECAY_FACTOR": .8, "LR_DECAY_EVERY_EPISODES": 125,
              "EVAL_SEEDS": OFFICIAL, "SUPPLEMENTAL_EVAL_SEEDS": SUPPLEMENTAL,
              "EVAL_EXPLORATION": .05, "MAX_STEPS": 3000}


def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def integer(x):
    return isinstance(x, int) and not isinstance(x, bool)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def exact(a, b):
    return type(a) is type(b) and a == b


def rate(episode):
    return .0001 * .8 ** ((episode - 1) // 125)


def updates(decisions):
    return max(0, decisions // 4 - 249)


def joined(value):
    return "".join(value) if isinstance(value, list) else str(value or "")


def compare(before, after):
    differences = [a - b for b, a in zip(before["scores"], after["scores"])]
    delta = after["mean"] - before["mean"]
    return {"paired_deltas": differences, "mean_delta": delta,
            "relative_mean_change_percent": delta / before["mean"] * 100 if before["mean"] else None,
            "median_paired_delta": statistics.median(differences),
            "improved": sum(v > 0 for v in differences), "tied": sum(v == 0 for v in differences),
            "worse": sum(v < 0 for v in differences)}


class Auditor:
    def __init__(self):
        self.issues = []
        self.checked = 0
        self.training_rows = []

    def check(self, condition, check, detail, severity="error"):
        self.checked += 1
        if not condition:
            self.issues.append({"severity": severity, "check": check, "detail": detail})
        return bool(condition)

    def read_json(self, path, mapping=True):
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"),
                               parse_constant=lambda s: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value: {s}")))
            if mapping and not isinstance(value, dict):
                raise ValueError("Expected JSON object")
            return value
        except (OSError, ValueError, TypeError) as exc:
            self.check(False, "read_json", f"{path.name}: {exc}")
            return None

    def file(self, path, image_kind=None):
        try:
            if not self.check(path.is_file() and path.stat().st_size > 0, "required_file", f"Missing or empty: {path}"):
                return False
            if not image_kind:
                return True
            with path.open("rb") as handle:
                h = handle.read(24)
            if image_kind == "gif":
                valid = h[:6] in (b"GIF87a", b"GIF89a") and len(h) >= 10 and int.from_bytes(h[6:8], "little") > 0 and int.from_bytes(h[8:10], "little") > 0
            else:
                valid = h[:8] == b"\x89PNG\r\n\x1a\n" and len(h) >= 24 and h[12:16] == b"IHDR" and int.from_bytes(h[16:20], "big") > 0 and int.from_bytes(h[20:24], "big") > 0
            return self.check(valid, "image_header", f"Invalid {image_kind} header/dimensions: {path.name}")
        except OSError as exc:
            self.check(False, "read_file", f"{path.name}: {exc}")
            return False

    def config(self, data):
        if not isinstance(data, dict):
            return
        for key, expected in FIXED.items():
            self.check(exact(data.get(key), expected), "fixed_config", f"{key}: expected {expected!r}; found {data.get(key)!r}")
        for key in ("device", "python", "platform"):
            self.check(isinstance(data.get(key), str) and bool(data[key]), "runtime_metadata", f"Missing/invalid {key}")
        if data.get("device") != "cuda":
            self.check(False, "device_changed", f"Expected planned CUDA device; recorded {data.get('device')!r}", "warning")
        packages = data.get("packages")
        self.check(isinstance(packages, dict) and all(isinstance(packages.get(p), str) and packages[p] for p in PACKAGES), "package_metadata", "All seven package versions must be recorded")
        schedule = data.get("learning_rate_schedule")
        if not self.check(isinstance(schedule, dict), "lr_schedule", "Missing learning_rate_schedule object"):
            return
        for key, expected in (("kind", "episode_step_decay"), ("factor", .8), ("every_completed_episodes", 125)):
            self.check(exact(schedule.get(key), expected), "lr_schedule", f"Schedule {key} must be {expected!r}")
        blocks = schedule.get("rates_by_episode_block")
        if self.check(isinstance(blocks, list) and len(blocks) == 4, "lr_blocks", "Expected four learning-rate blocks"):
            for first, block in zip((1, 126, 251, 376), blocks):
                self.check(isinstance(block, dict) and exact(block.get("first_episode"), first)
                           and exact(block.get("last_episode"), first + 124)
                           and close(block.get("learning_rate"), rate(first)), "lr_blocks", f"Invalid schedule block beginning episode {first}: {block}")
        self.check(set(OFFICIAL).isdisjoint(SUPPLEMENTAL) and set(OFFICIAL + SUPPLEMENTAL).isdisjoint(range(31416, 31916)), "seed_disjointness", "Evaluation sets must be mutually disjoint and outside training reset seeds")

    def evaluation(self, data, seeds, label):
        if not self.check(isinstance(data, dict), "evaluation_schema", f"{label}: expected object"):
            return None
        valid = self.check(data.get("seeds") == seeds and all(integer(x) for x in data.get("seeds", [])), "evaluation_seeds", f"{label}: expected ordered seeds {seeds}")
        scores = data.get("scores")
        score_valid = isinstance(scores, list) and len(scores) == len(seeds) and all(finite(v) for v in scores)
        valid &= self.check(score_valid, "evaluation_scores", f"{label}: expected {len(seeds)} finite scores")
        if score_valid:
            valid &= self.check(close(data.get("mean"), statistics.mean(scores)), "evaluation_mean", f"{label}: mean does not equal arithmetic score mean")
        steps, caps = data.get("steps"), data.get("time_limited")
        step_valid = isinstance(steps, list) and len(steps) == len(seeds) and all(integer(s) and 1 <= s <= 3000 for s in steps)
        cap_valid = isinstance(caps, list) and len(caps) == len(seeds) and all(type(c) is bool for c in caps)
        valid &= self.check(step_valid, "evaluation_steps", f"{label}: invalid episode decision counts")
        valid &= self.check(cap_valid, "evaluation_caps", f"{label}: invalid time-limit flags")
        if step_valid and cap_valid:
            valid &= self.check(all(not c or s == 3000 for s, c in zip(steps, caps)), "evaluation_caps", f"{label}: time-limited game has fewer than 3000 decisions")
        if not valid:
            return None
        return {"seeds": seeds, "scores": scores, "mean": statistics.mean(scores), "median": statistics.median(scores),
                "steps": steps, "time_limited": caps, "time_limited_count": sum(caps)}

    def comparison(self, folder, supplemental=False):
        prefix, seeds = ("supplemental_", SUPPLEMENTAL) if supplemental else ("", OFFICIAL)
        baseline = self.read_json(folder / f"{prefix}baseline.json")
        comparison = self.read_json(folder / f"{prefix}comparison.json")
        before = self.evaluation(baseline, seeds, f"{prefix}baseline")
        after = None
        if comparison is not None:
            self.check(comparison.get("baseline_kind") == "untrained network", "baseline_kind", f"{prefix}comparison: baseline is not an untrained network")
            self.check(close(comparison.get("evaluation_exploration"), .05) and exact(comparison.get("max_decisions_per_game"), 3000), "evaluation_settings", f"{prefix}comparison: evaluation settings changed")
            if supplemental:
                self.check(comparison.get("evaluation_set") == "supplemental", "supplemental_label", "Supplemental comparison must be labeled separately")
            self.check(comparison.get("before") == baseline and baseline is not None, "baseline_consistency", f"{prefix}comparison.before differs from saved baseline")
            self.evaluation(comparison.get("before"), seeds, f"{prefix}comparison.before")
            after = self.evaluation(comparison.get("after"), seeds, f"{prefix}comparison.after")
        return {"before": before, "after": after, "change": compare(before, after) if before and after else None}

    def training(self, path, summary):
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.check(reader.fieldnames == FIELDS, "csv_columns", f"Expected columns {FIELDS}; found {reader.fieldnames}")
                raw_rows = list(reader)
        except (OSError, csv.Error) as exc:
            self.check(False, "read_csv", str(exc))
            return None
        self.check(len(raw_rows) == 500, "csv_row_count", f"Expected 500 completed episodes; found {len(raw_rows)}")
        cumulative, elapsed, scores, boundaries = 0, -1., [], {}
        for index, raw in enumerate(raw_rows, 1):
            try:
                row = {key: int(raw[key]) for key in ("episode", "steps", "total_steps")}
                row.update({key: float(raw[key]) for key in ("score", "exploration", "mean_loss", "elapsed_seconds", "learning_rate")})
                for key in ("terminated", "truncated"):
                    if raw[key] not in ("True", "False"):
                        raise ValueError(f"Invalid {key}")
                    row[key] = raw[key] == "True"
            except (KeyError, TypeError, ValueError) as exc:
                self.check(False, "csv_row", f"Row {index}: {exc}")
                continue
            self.check(row["episode"] == index, "csv_episode", f"Row {index} has episode {row['episode']}")
            self.check(1 <= row["steps"] <= 3000, "csv_steps", f"Episode {index}: steps outside [1,3000]")
            prior = cumulative
            cumulative += row["steps"]
            self.check(row["total_steps"] == cumulative, "csv_cumulative", f"Episode {index}: total_steps differs from sum {cumulative}")
            self.check(finite(row["score"]), "csv_score", f"Episode {index}: nonfinite score")
            self.check(close(row["exploration"], 1. if cumulative <= 1000 else .2), "csv_exploration", f"Episode {index}: incorrect final-action exploration")
            n_updates = updates(cumulative) - updates(prior)
            self.check(finite(row["mean_loss"]) and row["mean_loss"] >= 0 if n_updates else math.isnan(row["mean_loss"]), "csv_loss", f"Episode {index}: loss inconsistent with update count")
            self.check(finite(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0 and row["elapsed_seconds"] >= elapsed, "csv_elapsed", f"Episode {index}: elapsed time invalid/nonmonotonic")
            elapsed = row["elapsed_seconds"]
            self.check(row["terminated"] or row["truncated"], "csv_game_end", f"Episode {index}: neither terminated nor truncated")
            self.check(not row["truncated"] or row["steps"] == 3000, "csv_cap", f"Episode {index}: truncated below limit")
            self.check(close(row["learning_rate"], rate(index)), "csv_learning_rate", f"Episode {index}: expected {rate(index):.8g}; found {row['learning_rate']:.8g}")
            if index in (1, 125, 126, 250, 251, 375, 376, 500):
                boundaries[str(index)] = row["learning_rate"] if finite(row["learning_rate"]) else None
            if finite(row["score"]):
                scores.append(row["score"])
            self.training_rows.append(row)
        if summary is not None:
            self.check(exact(summary.get("completed_episodes"), len(raw_rows)), "summary_csv_episodes", "Summary episode count differs from CSV rows")
            self.check(exact(summary.get("total_decisions"), cumulative), "summary_csv_decisions", "Summary decisions differ from CSV sum")
            self.check(finite(summary.get("elapsed_seconds_including_periodic_demos")) and summary["elapsed_seconds_including_periodic_demos"] >= elapsed, "summary_csv_elapsed", "Summary elapsed time precedes CSV or is invalid")
        return {"rows": len(raw_rows), "total_completed_episode_decisions": cumulative,
                "learning_rate_boundaries": boundaries,
                "first_25_training_score_mean": statistics.mean(scores[:25]) if scores else None,
                "last_25_training_score_mean": statistics.mean(scores[-25:]) if scores else None}

    def summary(self, value):
        if value is None:
            return
        self.check(value.get("status") == "completed", "training_status", f"Run status is {value.get('status')!r}")
        self.check(exact(value.get("completed_episodes"), 500), "completed_budget", "Run did not complete exactly 500 episodes")
        decisions = value.get("total_decisions")
        if self.check(integer(decisions) and 500 <= decisions <= 1500000, "summary_decisions", "Invalid total_decisions"):
            self.check(exact(value.get("learning_updates"), updates(decisions)) and updates(decisions) > 0, "summary_updates", f"Expected {updates(decisions)} positive learning updates from {decisions} decisions")
        self.check(close(value.get("initial_learning_rate"), .0001), "summary_lr", "Initial learning rate differs from .0001")
        self.check(close(value.get("final_learning_rate"), rate(500)), "summary_lr", "Final learning rate differs from .0000512")
        self.check(finite(value.get("elapsed_seconds_including_periodic_demos")) and value["elapsed_seconds_including_periodic_demos"] > 0, "summary_elapsed", "Elapsed seconds must be positive and finite")

    def notebook(self, path, prepared_path, official, supplemental, allow_colab_null_counts=False):
        nb = self.read_json(path)
        if nb is None:
            return None
        cells = nb.get("cells")
        if not self.check(isinstance(cells, list), "notebook_cells", "Missing notebook cells list"):
            return None
        codes = [(i, c) for i, c in enumerate(cells) if isinstance(c, dict) and c.get("cell_type") == "code"]
        self.check(len(codes) == 28, "notebook_code_count", f"Expected all 28 code cells; found {len(codes)}")
        null_counts = [i for i, c in codes if c.get("execution_count") is None]
        colab_null_mode = allow_colab_null_counts and len(codes) == 28 and len(null_counts) == 28
        if colab_null_mode:
            self.check(False, "colab_null_execution_counts", "All 28 execution counters are absent in this Colab export. No counters were reconstructed. Execution evidence is limited to preserved outputs, exact prepared-code matching, and artifact reconciliation; individual cell counters cannot be verified.", "warning")
            self.check(isinstance(nb.get("metadata", {}).get("colab"), dict), "colab_export_metadata", "Missing Colab notebook metadata")
        assignments = {name: [] for name in PARAMETERS}
        required = {"baseline": [], "training": [], "saved_evaluation": [], "comparison": [], "archive": []}
        markers = {"baseline": "supplemental_baseline = evaluate(", "training": "for episode in range(1, EPISODES + 1)",
                   "saved_evaluation": "supplemental_after = evaluate(", "comparison": "supplemental_comparison = {",
                   "archive": "archive = shutil.make_archive("}
        outputs_count, images_count, code_texts = 0, 0, []
        for index, cell in codes:
            source = joined(cell.get("source"))
            code_texts.append(source)
            if not colab_null_mode:
                self.check(integer(cell.get("execution_count")) and cell["execution_count"] > 0, "notebook_execution", f"Code cell {index} lacks completed execution count")
            if not source.lstrip().startswith("%pip"):
                try:
                    tree = ast.parse(source)
                    for node in tree.body:
                        if isinstance(node, ast.Assign):
                            for target in node.targets:
                                if isinstance(target, ast.Name) and target.id in assignments:
                                    try:
                                        assignments[target.id].append(ast.literal_eval(node.value))
                                    except (ValueError, TypeError):
                                        assignments[target.id].append("<not a literal>")
                except SyntaxError as exc:
                    self.check(False, "notebook_syntax", f"Cell {index}: {exc}")
            output_list = cell.get("outputs")
            if not self.check(isinstance(output_list, list), "notebook_outputs", f"Cell {index}: malformed output list"):
                continue
            outputs_count += len(output_list)
            if colab_null_mode and output_list:
                metadata = cell.get("metadata", {})
                colab = metadata.get("colab", {})
                self.check(isinstance(metadata.get("outputId"), str) and bool(metadata["outputId"])
                           and isinstance(colab, dict) and isinstance(colab.get("base_uri"), str),
                           "colab_output_provenance", f"Output-producing cell {index} lacks Colab outputId/base_uri metadata")
            image_count = 0
            streams = []
            for output in output_list:
                if not self.check(isinstance(output, dict), "notebook_outputs", f"Cell {index}: malformed output item"):
                    continue
                self.check(output.get("output_type") != "error", "notebook_error", f"Cell {index}: {output.get('ename')}: {output.get('evalue')}")
                data = output.get("data") or {}
                image_count += int(isinstance(data, dict) and any(str(mime).startswith("image/") for mime in data))
                streams.append(joined(output.get("text")))
            images_count += image_count
            for name, marker in markers.items():
                if marker in source:
                    required[name].append({"index": index, "outputs": len(output_list), "images": image_count, "text": "\n".join(streams)})
        for name, expected in PARAMETERS.items():
            values = assignments[name]
            self.check(len(values) == 1 and exact(values[0], expected), "notebook_parameter", f"{name}: expected {expected!r}; found {values}")
        for name, matches in required.items():
            self.check(len(matches) == 1, "notebook_required_cell", f"Expected one {name} cell; found {len(matches)}")
            for match in matches:
                self.check(match["outputs"] > 0, "notebook_required_output", f"{name} cell has no saved outputs")
                if name == "baseline":
                    self.check(match["images"] >= 1, "notebook_baseline_image", "Baseline GIF output missing")
                if name == "comparison":
                    self.check(match["images"] >= 2, "notebook_final_images", "Final dashboard/GIF output missing")
                    self.tables(match["text"], official, supplemental)
        exact_code_match = None
        if prepared_path:
            prepared = self.read_json(prepared_path)
            if prepared is not None and isinstance(prepared.get("cells"), list):
                prepared_codes = [joined(c.get("source")) for c in prepared["cells"] if isinstance(c, dict) and c.get("cell_type") == "code"]
                exact_code_match = code_texts == prepared_codes
                changed = [i + 1 for i, (a, b) in enumerate(zip(prepared_codes, code_texts)) if a != b]
                self.check(exact_code_match, "prepared_code_match", f"Code source differs from prepared notebook; changed code ordinals {changed}; counts {len(prepared_codes)} / {len(code_texts)}")
            else:
                self.check(False, "prepared_schema", "Prepared notebook missing valid cells list")
        else:
            self.check(False, "prepared_not_provided", "Exact source comparison skipped; supply --prepared to verify all code cells", "warning")
        if colab_null_mode:
            self.check(exact_code_match is True, "colab_null_source_requirement", "Null-count mode requires exact matching of every code cell to the prepared notebook")
            self.reconcile_colab_training_output(required)
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "code_cells": len(codes), "saved_outputs": outputs_count, "saved_images": images_count,
                "parameter_assignments": assignments, "all_code_matches_prepared": exact_code_match,
                "prepared_path": str(prepared_path.resolve()) if prepared_path else None,
                "null_execution_count_cells": null_counts, "colab_null_count_mode_used": colab_null_mode,
                "execution_evidence": "preserved_outputs_and_artifact_reconciliation; individual counters unavailable" if colab_null_mode else "saved_execution_counts_and_outputs"}

    def reconcile_colab_training_output(self, required):
        rows = self.training_rows
        if not self.check(len(rows) == 500, "colab_null_artifact_requirement", "Null-count mode requires 500 parsed CSV rows from the retrieved run artifacts"):
            return
        matches = required.get("training", [])
        if not self.check(len(matches) == 1, "colab_null_training_output", "Missing unique training-output cell"):
            return
        recorded = matches[0]
        self.check(recorded["images"] == 20, "colab_null_training_images", f"Expected all 20 saved periodic gameplay images; found {recorded['images']}")
        pattern = (r"^Episode (\d+)/500 \| score (-?\d+(?:\.\d+)?) \| recent mean (-?\d+(?:\.\d+)?) "
                   r"\| exploration (\d+)% \| learning rate (\d+(?:\.\d+)?)\s*$")
        logged = re.findall(pattern, recorded["text"], re.MULTILINE)
        expected = []
        for index, row in enumerate(rows):
            recent = statistics.mean(r["score"] for r in rows[max(0, index - 24):index + 1])
            expected.append((str(row["episode"]), f"{row['score']:.0f}", f"{recent:.1f}",
                             f"{row['exploration'] * 100:.0f}", f"{row['learning_rate']:.8f}"))
        self.check(logged == expected, "colab_null_training_rows", "All 500 printed episode/score/recent-mean/exploration/rate rows must match the CSV in order")
        progress = re.findall(r"^\s+Episode (\d+)/500 \| ([\d,]+) decisions \| ([\d,]+) updates \|", recorded["text"], re.MULTILINE)
        cumulative = [r["total_steps"] for r in rows]
        expected_progress = [(str(bisect.bisect_left(cumulative, step) + 1), f"{step:,}", f"{updates(step):,}")
                             for step in range(500, cumulative[-1] + 1, 500)]
        self.check(progress == expected_progress, "colab_null_progress_counters", "Every 500-decision progress counter and learning-update count must reconcile with CSV cumulative steps")
        saved = re.findall(r"^Saved:\s*(\S+)\s*$", recorded["text"], re.MULTILINE)
        archives = required.get("archive", [])
        archived = re.findall(r"^Results folder:\s*(\S+)\s*$", archives[0]["text"], re.MULTILINE) if len(archives) == 1 else []
        self.check(len(saved) == len(archived) == 1 and Path(saved[0]).name == Path(archived[0]).name,
                   "colab_null_saved_run", "Training completion and archive output must identify the same saved run")

    def tables(self, text, official, supplemental):
        sections = text.split("SUPPLEMENTAL EVALUATION", 1)
        if not self.check(len(sections) == 2, "notebook_two_tables", "Final output must contain separate official and supplemental tables"):
            return
        for label, section, values, seeds in (("official", sections[0], official, list(range(1, 6))),
                                              ("supplemental", sections[1], supplemental, SUPPLEMENTAL)):
            before, after = values.get("before"), values.get("after")
            if not before or not after:
                continue
            rows = re.findall(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$", section, re.MULTILINE)
            expected = [(str(s), f"{b:.0f}", f"{a:.0f}") for s, b, a in zip(seeds, before["scores"], after["scores"])]
            self.check(rows == expected, "notebook_score_table", f"{label} notebook score rows differ from saved comparison")
            means = re.findall(r"^\s*Mean\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$", section, re.MULTILINE)
            self.check(means == [(f"{before['mean']:.1f}", f"{after['mean']:.1f}")], "notebook_score_means", f"{label} notebook mean row differs from saved comparison")


def render(report):
    errors = sum(x["severity"] == "error" for x in report["issues"])
    warnings = sum(x["severity"] == "warning" for x in report["issues"])
    lines = ["# 500-episode Pac-Man audit", "", f"**{'PASS' if report['passed'] else 'FAIL'}** — {report['checks_performed']} checks; {errors} errors; {warnings} warnings.", "",
             f"Generated: {report['generated_at_utc']}", "", f"Run: `{report['run_directory']}`", ""]
    s = report.get("training_summary") or {}
    if s:
        lines += [f"Status: **{s.get('status')}**. Episodes: **{s.get('completed_episodes')}**; decisions: **{s.get('total_decisions')}**; updates: **{s.get('learning_updates')}**.", "",
                  f"Elapsed training seconds including periodic demos: {s.get('elapsed_seconds_including_periodic_demos')}. Initial/final learning rates: {s.get('initial_learning_rate')} / {s.get('final_learning_rate')}.", ""]
    for label in ("official", "supplemental"):
        result = report[label]
        if result["before"] and result["after"]:
            before, after, change = result["before"], result["after"], result["change"]
            lines += [f"## {label.title()} evaluation", "", "| Seed | Untrained | Trained | Change |", "|---|---:|---:|---:|"]
            for seed, b, a in zip(before["seeds"], before["scores"], after["scores"]):
                lines.append(f"| {seed} | {b:g} | {a:g} | {a-b:+g} |")
            lines += [f"| **Mean** | **{before['mean']:.1f}** | **{after['mean']:.1f}** | **{change['mean_delta']:+.1f}** |", "",
                      f"Improved/tied/worse: {change['improved']}/{change['tied']}/{change['worse']}. Time-limited games before/after: {before['time_limited_count']}/{after['time_limited_count']}.", ""]
    if report.get("training_csv"):
        lines += ["## Learning-rate boundary checks", "", "| Episode | Recorded rate |", "|---|---:|"]
        lines += [f"| {episode} | {value} |" for episode, value in report["training_csv"]["learning_rate_boundaries"].items()]
        lines += ["", "Every CSV row was checked against the schedule, not only these boundaries.", ""]
    notebook = report.get("notebook")
    if notebook:
        lines += [f"Notebook: {notebook['code_cells']} code cells, {notebook['saved_outputs']} saved outputs, {notebook['saved_images']} image outputs. Exact code match with prepared source: **{notebook['all_code_matches_prepared']}**.", ""]
        if notebook.get("colab_null_count_mode_used"):
            lines += ["**Export limitation:** all 28 execution counters are null. This opt-in audit uses matching prepared code and preserved outputs reconciled with all 500 CSV rows and progress counters. It does not reconstruct or verify individual cell execution counts.", ""]
    lines += ["## Findings", ""]
    lines += [f"- **{x['severity'].upper()} — {x['check']}:** {x['detail']}" for x in report["issues"]] or ["- No failed checks."]
    lines += ["", "## Verification boundaries", ""]
    lines += [f"- {x}" for x in report["verification_boundaries"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run500", type=Path, required=True)
    parser.add_argument("--notebook500", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--allow-colab-null-counts", action="store_true", help="Explicitly permit an all-null Colab export only with exact source and complete output/artifact reconciliation; retain a warning")
    parser.add_argument("--json-out", "--json-output", type=Path)
    parser.add_argument("--markdown-out", "--markdown-output", type=Path)
    args = parser.parse_args(argv)
    if args.prepared is None:
        default = Path(__file__).resolve().parents[1] / "prepared/pacman_dqn_500_decay_UNEXECUTED.ipynb"
        if default.is_file():
            args.prepared = default
    outputs = [p.resolve() for p in (args.json_out, args.markdown_out) if p]
    protected = [p.resolve() for p in (args.notebook500, args.prepared) if p]
    if len(outputs) != len(set(outputs)) or any(p.is_relative_to(args.run500.resolve()) or p in protected for p in outputs):
        parser.error("Audit output files must be distinct and outside input directories/notebook paths")
    audit = Auditor()
    audit.check(args.run500.is_dir(), "run_directory", f"Not a directory: {args.run500}")
    config = audit.read_json(args.run500 / "config.json")
    audit.config(config)
    summary = audit.read_json(args.run500 / "training_summary.json")
    audit.summary(summary)
    training = audit.training(args.run500 / "training.csv", summary)
    official = audit.comparison(args.run500)
    supplemental = audit.comparison(args.run500, supplemental=True)
    audit.file(args.run500 / "training_dashboard.png", "png")
    for name in ["episode_0000.gif", "final_best.gif"] + [f"episode_{e:04d}.gif" for e in PERIODIC]:
        audit.file(args.run500 / "demos" / name, "gif")
    for name in ["untrained.pt", "trained.pt"] + [f"episode_{e:04d}.pt" for e in PERIODIC]:
        audit.file(args.run500 / name)
    demos = audit.read_json(args.run500 / "demo_scores.json", mapping=False)
    valid_demos = isinstance(demos, list) and all(isinstance(d, dict) for d in demos)
    if audit.check(valid_demos, "periodic_demo_schema", "Expected list of periodic demo results"):
        audit.check([d.get("episode") for d in demos] == PERIODIC, "periodic_demo_episodes", "Expected every 25th episode from 25 through 500")
        for demo in demos:
            audit.evaluation(demo, [101], f"periodic demo {demo.get('episode')}")
    notebook = audit.notebook(args.notebook500, args.prepared, official, supplemental, args.allow_colab_null_counts)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not any(x["severity"] == "error" for x in audit.issues),
        "checks_performed": audit.checked, "run_directory": str(args.run500.resolve()),
        "config": config, "training_summary": summary, "training_csv": training,
        "official": official, "supplemental": supplemental, "notebook": notebook,
        "expected_periodic_episodes": PERIODIC, "issues": audit.issues,
        "verification_boundaries": [
            "Input files were read only. No training or PyTorch/pickle deserialization was performed.",
            "Checkpoint validation establishes presence and nonempty files only. The executed notebook must perform the actual saved-model reload.",
            "GIF/PNG validation covers headers and dimensions, not full decoding, animation playback, or behavior interpretation.",
            "ZIP integrity, README evidence links, public GitHub rendering/access, and course submission must be checked separately.",
            "Saved execution counts and output checks do not independently rerun the notebook. Exact source comparison is limited to the supplied prepared notebook.",
            "Both five-seed evaluations use .05 exploration and a 3000-decision limit. Supplemental results are separate from the official classroom score.",
            "Training seed, episode budget, and learning-rate schedule changed together; this is not a causal test of one change. Fixed seeds do not force deterministic CUDA execution.",
            "Training elapsed time includes periodic demos and excludes setup, baseline/final evaluations, and archive download.",
        ],
    }
    markdown = render(report)
    for path, content in ((args.json_out, json.dumps(report, indent=2, allow_nan=False) + "\n"), (args.markdown_out, markdown)):
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
    sys.stdout.write(markdown)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
