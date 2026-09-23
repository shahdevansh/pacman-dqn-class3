#!/usr/bin/env python3
"""Audit the supplied classroom Pac-Man outputs using Python's standard library.

Usage:
  python3 tools/audit_results.py --run50 EXTRACTED_RUN_50 --notebook50 RUN_50.ipynb \
      --run200 EXTRACTED_RUN_200 --notebook200 RUN_200.ipynb \
      --json-out audit.json --markdown-out audit.md

Input folders and notebooks are read only. No training or deserialization of
PyTorch/pickle data occurs. Exit status is 1 for failed checks, 0 otherwise.
Notebook omission is a warning: such an audit cannot establish submission readiness.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path


SEEDS = [101, 202, 303, 404, 505]
CSV_FIELDS = ["episode", "score", "steps", "total_steps", "exploration", "mean_loss",
              "elapsed_seconds", "terminated", "truncated"]
FIXED_CONFIG = {
    "exploration": 0.20, "learning_rate": 0.0001, "seed": 42,
    "environment": "ALE/MsPacman-v5", "frame_skip": 4,
    "sticky_action_probability": 0.25, "noop_max": 30, "grayscale_size": 84,
    "stack_size": 4, "terminal_on_life_loss": False, "max_decisions_per_game": 3000,
    "replay_capacity": 5000, "warmup_decisions": 1000, "batch_size": 32,
    "train_every_decisions": 4, "target_sync_decisions": 1000, "gamma": 0.99,
    "training_reward_clipping": [-1, 1], "eval_exploration": 0.05,
    "eval_seeds": SEEDS, "preview_seconds": 20, "preview_speed": 4,
    "preview_plays": 2, "preview_stride": 4, "preview_frame_ms": 67,
}
ENVIRONMENT_FIELDS = {"device", "python", "platform", "packages"}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def whole_number(value):
    return isinstance(value, int) and not isinstance(value, bool)


def same_number(a, b, tolerance=1e-8):
    return finite_number(a) and finite_number(b) and math.isclose(a, b, rel_tol=0, abs_tol=tolerance)


def expected_updates(decisions):
    return max(0, decisions // 4 - 249)


def text_value(value):
    return "".join(value) if isinstance(value, list) else str(value or "")


def score_summary(scores):
    return {
        "scores": scores, "mean": statistics.mean(scores), "median": statistics.median(scores),
        "minimum": min(scores), "maximum": max(scores),
        "sample_standard_deviation": statistics.stdev(scores),
    }


def delta_summary(before, after):
    values = [b - a for a, b in zip(before, after)]
    baseline_mean = statistics.mean(before)
    difference = statistics.mean(after) - baseline_mean
    return {
        "paired_deltas": values, "mean_delta": difference,
        "median_paired_delta": statistics.median(values),
        "relative_mean_change_percent": 100 * difference / baseline_mean if baseline_mean else None,
        "improved_seeds": sum(x > 0 for x in values),
        "tied_seeds": sum(x == 0 for x in values),
        "worse_seeds": sum(x < 0 for x in values),
    }


class Audit:
    def __init__(self):
        self.issues = []

    def issue(self, severity, run, check, message):
        self.issues.append({"severity": severity, "run": run, "check": check, "message": message})

    def check(self, condition, run, check, message, severity="error"):
        if not condition:
            self.issue(severity, run, check, message)
        return bool(condition)

    def read_json(self, path, run):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"),
                              parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON value {x}")))
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            return data
        except (OSError, ValueError, TypeError) as exc:
            self.issue("error", run, "read_json", f"{path.name}: {exc}")
            return None

    def nonempty_file(self, path, run):
        try:
            okay = path.is_file() and path.stat().st_size > 0
        except OSError:
            okay = False
        self.check(okay, run, "required_file", f"Missing or empty file: {path}")
        return okay

    def image_file(self, path, run, kind):
        if not self.nonempty_file(path, run):
            return False
        try:
            with path.open("rb") as handle:
                header = handle.read(24)
            okay = (header[:6] in (b"GIF87a", b"GIF89a") and len(header) >= 10
                    and int.from_bytes(header[6:8], "little") > 0
                    and int.from_bytes(header[8:10], "little") > 0) if kind == "gif" else (
                        header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 24
                        and header[12:16] == b"IHDR"
                        and int.from_bytes(header[16:20], "big") > 0
                        and int.from_bytes(header[20:24], "big") > 0)
            return self.check(okay, run, "image_header", f"Invalid {kind.upper()} header/dimensions: {path.name}")
        except OSError as exc:
            self.issue("error", run, "image_header", f"{path.name}: {exc}")
            return False

    def evaluation(self, value, run, label):
        if not self.check(isinstance(value, dict), run, "evaluation", f"{label} is not an object"):
            return None
        okay = self.check(value.get("seeds") == SEEDS, run, "evaluation_seeds", f"{label}: expected seeds {SEEDS}")
        scores = value.get("scores")
        score_okay = isinstance(scores, list) and len(scores) == 5 and all(finite_number(x) for x in scores)
        okay &= self.check(score_okay, run, "evaluation_scores", f"{label}: expected five finite numeric scores")
        if score_okay:
            okay &= self.check(same_number(value.get("mean"), statistics.mean(scores)), run, "evaluation_mean",
                                f"{label}: recorded mean differs from arithmetic mean {statistics.mean(scores)}")
        steps, capped = value.get("steps"), value.get("time_limited")
        steps_okay = isinstance(steps, list) and len(steps) == 5 and all(whole_number(s) and 1 <= s <= 3000 for s in steps)
        caps_okay = isinstance(capped, list) and len(capped) == 5 and all(type(c) is bool for c in capped)
        okay &= self.check(steps_okay, run, "evaluation_steps", f"{label}: expected five decision counts in [1,3000]")
        okay &= self.check(caps_okay, run, "evaluation_caps", f"{label}: expected five boolean time-limit flags")
        if steps_okay and caps_okay:
            okay &= self.check(all(not c or s == 3000 for s, c in zip(steps, capped)), run, "evaluation_caps",
                                f"{label}: a time-limited game has fewer than 3,000 decisions")
        if not okay:
            return None
        return {**score_summary(scores), "seeds": SEEDS, "steps": steps,
                "time_limited": capped, "time_limited_count": sum(capped)}

    def training_csv(self, path, run, summary):
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                self.check(reader.fieldnames == CSV_FIELDS, run, "csv_columns",
                           f"Expected columns {CSV_FIELDS}; found {reader.fieldnames}")
                raw_rows = list(reader)
        except (OSError, csv.Error) as exc:
            self.issue("error", run, "training_csv", str(exc))
            return None
        rows, total, last_elapsed = [], 0, -1.0
        for index, raw in enumerate(raw_rows, 1):
            try:
                row = {key: int(raw[key]) for key in ("episode", "steps", "total_steps")}
                row.update({key: float(raw[key]) for key in ("score", "exploration", "mean_loss", "elapsed_seconds")})
                for key in ("terminated", "truncated"):
                    if raw[key] not in ("True", "False"):
                        raise ValueError(f"{key} must be True or False")
                    row[key] = raw[key] == "True"
            except (KeyError, TypeError, ValueError) as exc:
                self.issue("error", run, "csv_row", f"Data row {index}: {exc}")
                continue
            prefix = f"Episode row {index}"
            self.check(row["episode"] == index, run, "csv_episodes", f"{prefix}: episode is {row['episode']}")
            self.check(1 <= row["steps"] <= 3000, run, "csv_steps", f"{prefix}: steps outside [1,3000]")
            previous_total = total
            total += row["steps"]
            self.check(row["total_steps"] == total, run, "csv_cumulative_steps", f"{prefix}: total_steps differs from sum of steps {total}")
            self.check(finite_number(row["score"]), run, "csv_score", f"{prefix}: nonfinite score")
            expected_epsilon = 1.0 if total <= 1000 else 0.20
            self.check(same_number(row["exploration"], expected_epsilon), run, "csv_exploration",
                       f"{prefix}: final-action exploration should be {expected_epsilon}")
            episode_updates = expected_updates(total) - expected_updates(previous_total)
            loss_okay = (finite_number(row["mean_loss"]) and row["mean_loss"] >= 0) if episode_updates else math.isnan(row["mean_loss"])
            self.check(loss_okay, run, "csv_loss", f"{prefix}: loss inconsistent with {episode_updates} updates")
            self.check(finite_number(row["elapsed_seconds"]) and row["elapsed_seconds"] >= 0
                       and row["elapsed_seconds"] >= last_elapsed, run, "csv_elapsed", f"{prefix}: elapsed seconds invalid/nonmonotonic")
            last_elapsed = row["elapsed_seconds"]
            self.check(row["terminated"] or row["truncated"], run, "csv_episode_end", f"{prefix}: neither terminated nor truncated")
            self.check(not row["truncated"] or row["steps"] == 3000, run, "csv_time_limit",
                       f"{prefix}: truncated before 3,000 decisions")
            rows.append(row)
        if summary:
            self.check(len(raw_rows) == summary.get("completed_episodes"), run, "csv_row_count",
                       f"{len(raw_rows)} rows versus summary completed_episodes={summary.get('completed_episodes')}")
            summary_steps = summary.get("total_decisions")
            if summary.get("status") == "completed":
                self.check(summary_steps == total, run, "csv_summary_steps", f"CSV sum {total} differs from summary total {summary_steps}")
            elif whole_number(summary_steps):
                self.check(summary_steps >= total, run, "csv_summary_steps", "Summary decisions are smaller than completed-game CSV total")
                if summary_steps > total:
                    self.issue("warning", run, "partial_episode", f"Checkpoint includes {summary_steps - total} decisions from an incomplete episode absent from CSV")
            elapsed = summary.get("elapsed_seconds_including_periodic_demos")
            if finite_number(elapsed) and finite_number(last_elapsed):
                self.check(elapsed >= last_elapsed, run, "summary_elapsed", "Summary elapsed time precedes last CSV row")
        scores = [r["score"] for r in rows if finite_number(r["score"])]
        return {"row_count": len(raw_rows), "parsed_row_count": len(rows), "sum_completed_episode_decisions": total,
                "first_up_to_25_training_score_mean": statistics.mean(scores[:25]) if scores else None,
                "last_up_to_25_training_score_mean": statistics.mean(scores[-25:]) if scores else None}

    def notebook(self, path, run, budget, before, after):
        if path is None:
            self.issue("warning", run, "notebook_not_provided", "Executed notebook not audited; pass --notebook50/--notebook200")
            return None
        nb = self.read_json(path, run)
        if nb is None:
            return None
        cells = nb.get("cells")
        if not self.check(isinstance(cells, list), run, "notebook_cells", "Notebook cells missing or malformed"):
            return None
        codes = [c for c in cells if isinstance(c, dict) and c.get("cell_type") == "code" and text_value(c.get("source")).strip()]
        self.check(bool(codes), run, "notebook_cells", "No code cells in notebook")
        settings = {"EXPLORATION": [], "EPISODES": [], "LEARNING_RATE": []}
        unexecuted, error_outputs, output_count, rich_count = [], [], 0, 0
        marked = {"baseline": [], "training": [], "final_evaluation": [], "comparison": [], "archive": []}
        markers = {"baseline": 'baseline = evaluate(', "training": 'for episode in range(1, EPISODES + 1)',
                   "final_evaluation": 'after = evaluate(', "comparison": 'comparison = {', "archive": 'archive = shutil.make_archive('}
        for index, cell in enumerate(cells):
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue
            source = text_value(cell.get("source"))
            if source.strip() and not (whole_number(cell.get("execution_count")) and cell["execution_count"] >= 1):
                unexecuted.append(index)
            for name in settings:
                for match in re.finditer(rf"^{name}\s*=\s*(.+)$", source, re.MULTILINE):
                    try:
                        settings[name].append(ast.literal_eval(ast.parse(match.group(1), mode="eval")))
                    except (ValueError, SyntaxError):
                        settings[name].append("<not a literal>")
            outputs = cell.get("outputs", [])
            if not isinstance(outputs, list):
                self.issue("error", run, "notebook_outputs", f"Cell {index}: malformed outputs")
                continue
            output_count += len(outputs)
            for output in outputs:
                if not isinstance(output, dict):
                    self.issue("error", run, "notebook_outputs", f"Cell {index}: malformed output item")
                    continue
                if output.get("output_type") == "error":
                    error_outputs.append({"cell": index, "name": output.get("ename"), "value": output.get("evalue")})
                rich_count += int(any(mime.startswith("image/") for mime in output.get("data", {})))
            for name, marker in markers.items():
                if marker in source:
                    marked[name].append((index, outputs))
        self.check(not unexecuted, run, "notebook_execution", f"Code cells without completed execution counts: {unexecuted}")
        self.check(not error_outputs, run, "notebook_errors", f"Error outputs found: {error_outputs}")
        expected_settings = {"EXPLORATION": 0.20, "EPISODES": budget, "LEARNING_RATE": 0.0001}
        for name, expected in expected_settings.items():
            self.check(len(settings[name]) == 1 and same_number(settings[name][0], expected), run, "notebook_parameters",
                       f"Expected a single {name}={expected}; found {settings[name]}")
        comparison_text = ""
        for name, matches in marked.items():
            self.check(len(matches) == 1, run, "notebook_required_cells", f"Expected one {name} cell; found {len(matches)}")
            for index, outputs in matches:
                self.check(bool(outputs), run, "notebook_required_outputs", f"{name} cell {index} has no saved outputs")
                if name == "baseline":
                    baseline_images = sum(any(k.startswith("image/") for k in o.get("data", {})) for o in outputs if isinstance(o, dict))
                    self.check(baseline_images >= 1, run, "notebook_baseline_image", "Baseline cell has no saved gameplay image output")
                if name == "comparison":
                    comparison_text = "\n".join(text_value(o.get("text")) for o in outputs if isinstance(o, dict))
                    image_outputs = sum(any(k.startswith("image/") for k in o.get("data", {})) for o in outputs if isinstance(o, dict))
                    self.check(image_outputs >= 2, run, "notebook_final_images",
                               f"Final comparison contains {image_outputs} saved image outputs; expected dashboard and gameplay GIF")
        if before and after and comparison_text:
            numeric_rows = re.findall(r"^\s*([1-5])\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$", comparison_text, re.MULTILINE)
            expected_rows = [(str(i), f"{b:.0f}", f"{a:.0f}") for i, (b, a) in enumerate(zip(before["scores"], after["scores"]), 1)]
            self.check(numeric_rows == expected_rows, run, "notebook_score_table", "Saved notebook's five displayed score rows do not match comparison.json")
            means = re.findall(r"^\s*Mean\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$", comparison_text, re.MULTILINE)
            self.check(means == [(f"{before['mean']:.1f}", f"{after['mean']:.1f}")], run, "notebook_score_means",
                       "Saved notebook's displayed means do not match comparison.json")
        return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "code_cells": len(codes), "saved_outputs": output_count, "saved_image_outputs": rich_count,
                "unexecuted_code_cells": unexecuted, "error_outputs": error_outputs, "parameter_assignments": settings}

    def run(self, folder, budget, notebook_path):
        label = str(budget)
        result = {"path": str(folder.resolve()), "episode_budget": budget}
        if not self.check(folder.is_dir(), label, "run_directory", f"Directory does not exist: {folder}"):
            return result
        config = self.read_json(folder / "config.json", label)
        baseline = self.read_json(folder / "baseline.json", label)
        comparison = self.read_json(folder / "comparison.json", label)
        summary = self.read_json(folder / "training_summary.json", label)
        result["config"] = config
        result["training_summary"] = summary
        if config:
            self.check(config.get("episodes_requested") == budget, label, "episode_budget", f"episodes_requested must be {budget}")
            for key, value in FIXED_CONFIG.items():
                self.check(config.get(key) == value and type(config.get(key)) is type(value), label, "fixed_config",
                           f"{key}: expected {value!r}, found {config.get(key)!r}")
            for key in ENVIRONMENT_FIELDS:
                self.check(bool(config.get(key)), label, "environment_metadata", f"Missing {key}")
        completed = 0
        if summary:
            self.check(summary.get("status") == "completed", label, "training_status", f"Run status is {summary.get('status')!r}, expected completed")
            completed_value = summary.get("completed_episodes")
            if self.check(whole_number(completed_value) and 0 <= completed_value <= budget, label, "completed_episodes", "Invalid completed_episodes"):
                completed = completed_value
            self.check(completed_value == budget, label, "completed_budget", f"Completed {completed_value} of {budget} requested episodes")
            decisions, updates = summary.get("total_decisions"), summary.get("learning_updates")
            if self.check(whole_number(decisions) and decisions >= 0, label, "training_decisions", "Invalid total_decisions"):
                self.check(whole_number(updates) and updates == expected_updates(decisions), label, "learning_updates",
                           f"Expected {expected_updates(decisions)} updates from {decisions} decisions; found {updates}")
            self.check(whole_number(updates) and updates > 0, label, "nonzero_learning", "Run has no valid positive learning-update count")
            self.check(finite_number(summary.get("elapsed_seconds_including_periodic_demos")) and summary["elapsed_seconds_including_periodic_demos"] > 0,
                       label, "elapsed_seconds", "Elapsed training time must be finite and positive")
        result["training_csv"] = self.training_csv(folder / "training.csv", label, summary)
        before, after = None, None
        baseline_validated = self.evaluation(baseline, label, "baseline.json")
        if comparison:
            self.check(comparison.get("baseline_kind") == "untrained network", label, "baseline_kind", "Baseline must be an untrained network")
            self.check(same_number(comparison.get("evaluation_exploration"), 0.05), label, "comparison_settings", "Evaluation exploration must be .05")
            self.check(comparison.get("max_decisions_per_game") == 3000, label, "comparison_settings", "Evaluation limit must be 3000")
            before = self.evaluation(comparison.get("before"), label, "comparison.before")
            after = self.evaluation(comparison.get("after"), label, "comparison.after")
            self.check(baseline is not None and comparison.get("before") == baseline, label, "baseline_consistency", "baseline.json differs from comparison.before")
        result["before"] = before or baseline_validated
        result["after"] = after
        if before and after:
            result["within_run_change"] = delta_summary(before["scores"], after["scores"])
        periodic = list(range(25, completed + 1, 25))
        result["expected_periodic_episodes"] = periodic
        self.image_file(folder / "training_dashboard.png", label, "png")
        for name in ["episode_0000.gif", "final_best.gif", *[f"episode_{e:04d}.gif" for e in periodic]]:
            self.image_file(folder / "demos" / name, label, "gif")
        for name in ["untrained.pt", "trained.pt", *[f"episode_{e:04d}.pt" for e in periodic]]:
            self.nonempty_file(folder / name, label)
        if periodic:
            try:
                demos = json.loads((folder / "demo_scores.json").read_text(encoding="utf-8"))
                demo_episodes = [d.get("episode") for d in demos] if isinstance(demos, list) and all(isinstance(d, dict) for d in demos) else []
                self.check(demo_episodes == periodic, label, "periodic_demo_scores", f"Expected demonstration records for {periodic}; found {demo_episodes}")
                if demo_episodes == periodic:
                    for record in demos:
                        self.check(record.get("seeds") == [101] and isinstance(record.get("scores"), list)
                                   and len(record["scores"]) == 1 and finite_number(record["scores"][0])
                                   and same_number(record.get("mean"), record["scores"][0]), label, "periodic_demo_scores",
                                   f"Invalid score record for demo episode {record['episode']}")
            except (OSError, ValueError, TypeError) as exc:
                self.issue("error", label, "periodic_demo_scores", str(exc))
        result["notebook"] = self.notebook(notebook_path, label, budget, before, after)
        return result


def compare_runs(audit, short, long):
    comparison = {}
    ca, cb = short.get("config"), long.get("config")
    if ca and cb:
        all_keys = set(ca) | set(cb)
        differences = {k: {"run50": ca.get(k), "run200": cb.get(k)} for k in sorted(all_keys) if ca.get(k) != cb.get(k)}
        comparison["configuration_differences"] = differences
        experimental = set(differences) - ENVIRONMENT_FIELDS - {"episodes_requested"}
        audit.check(not experimental, "comparison", "config_comparability", f"Experimental settings besides episode budget differ: {sorted(experimental)}")
        environment_changes = set(differences) & ENVIRONMENT_FIELDS
        if environment_changes:
            audit.issue("warning", "comparison", "environment_changed", f"Runtime metadata differs: {sorted(environment_changes)}")
    before50, before200 = short.get("before"), long.get("before")
    if before50 and before200:
        equal = all(before50[key] == before200[key] for key in ("scores", "steps", "time_limited"))
        comparison["baseline_scores_steps_and_caps_identical"] = equal
        if not equal:
            audit.issue("warning", "comparison", "baseline_mismatch", "Fresh-run baselines differ; inspect runtime/initialization before attributing changes only to episode budget")
    after50, after200 = short.get("after"), long.get("after")
    if after50 and after200:
        comparison.update(delta_summary(after50["scores"], after200["scores"]))
        comparison["seeds"] = SEEDS
    return comparison


def markdown(report):
    issues = report["issues"]
    errors = sum(i["severity"] == "error" for i in issues)
    warnings = sum(i["severity"] == "warning" for i in issues)
    lines = ["# Pac-Man evidence audit", "", f"**{'PASS' if not errors else 'FAIL'}** — {errors} errors; {warnings} warnings.", "",
             f"Generated: {report['generated_at_utc']}", "",
             "This is a structural and numerical audit. It does not establish that gameplay was watched or that public GitHub links work.", ""]
    for name, run in report["runs"].items():
        lines += [f"## {name}-episode run", "", f"Folder: `{run['path']}`", ""]
        summary = run.get("training_summary") or {}
        config = run.get("config") or {}
        if summary:
            lines += [f"Status: **{summary.get('status')}**. Completed {summary.get('completed_episodes')} episodes, {summary.get('total_decisions')} decisions, and {summary.get('learning_updates')} learning updates.", "",
                      f"Device: `{config.get('device')}`. Elapsed training seconds including periodic demos: {summary.get('elapsed_seconds_including_periodic_demos')}.", ""]
        before, after = run.get("before"), run.get("after")
        if before and after:
            lines += ["| Seed | Untrained | Trained | Change |", "|---|---:|---:|---:|"]
            for seed, b, a in zip(SEEDS, before["scores"], after["scores"]):
                lines.append(f"| {seed} | {b:g} | {a:g} | {a-b:+g} |")
            lines += [f"| **Mean** | **{before['mean']:.1f}** | **{after['mean']:.1f}** | **{after['mean']-before['mean']:+.1f}** |", "",
                      f"Time-limited games: {before['time_limited_count']} before; {after['time_limited_count']} after.", ""]
        csv_stats = run.get("training_csv")
        if csv_stats:
            lines += [f"Training scores: first up-to-25-game mean {csv_stats['first_up_to_25_training_score_mean']}; last up-to-25-game mean {csv_stats['last_up_to_25_training_score_mean']}. These use training exploration, unlike the evaluation table.", ""]
        if run.get("notebook"):
            notebook = run["notebook"]
            lines += [f"Executed notebook: `{notebook['path']}`; {notebook['saved_outputs']} saved outputs, including {notebook['saved_image_outputs']} images.", ""]
    cross = report.get("between_runs")
    if cross and "mean_delta" in cross:
        lines += ["## 200 versus 50", "", f"Mean evaluation-score change: **{cross['mean_delta']:+.1f}**. Paired deltas: {cross['paired_deltas']}.", "",
                  f"Seeds improved/tied/worse: **{cross['improved_seeds']}/{cross['tied_seeds']}/{cross['worse_seeds']}**. Median paired delta: {cross['median_paired_delta']:g}.", ""]
        percent = cross.get("relative_mean_change_percent")
        lines += [f"Relative mean change: {percent:+.1f}%." if percent is not None else "Relative mean change undefined because the 50-episode mean is zero.", ""]
    lines += ["## Findings", ""]
    lines += [f"- **{i['severity'].upper()}** [{i['run']} / {i['check']}]: {i['message']}" for i in issues] or ["- No failed structural or numerical checks."]
    lines += ["", "## Verification boundaries", ""]
    lines += [f"- {limitation}" for limitation in report["verification_boundaries"]]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run50", type=Path, required=True)
    parser.add_argument("--run200", type=Path)
    parser.add_argument("--notebook50", type=Path)
    parser.add_argument("--notebook200", type=Path)
    parser.add_argument("--json-out", "--json-output", type=Path)
    parser.add_argument("--markdown-out", "--markdown-output", type=Path)
    args = parser.parse_args(argv)
    if args.notebook200 and not args.run200:
        parser.error("--notebook200 requires --run200")
    input_dirs = [p.resolve() for p in (args.run50, args.run200) if p is not None]
    input_notebooks = [p.resolve() for p in (args.notebook50, args.notebook200) if p is not None]
    output_paths = [p.resolve() for p in (args.json_out, args.markdown_out) if p is not None]
    if len(output_paths) != len(set(output_paths)):
        parser.error("JSON and Markdown outputs must have different paths")
    for output in output_paths:
        if output in input_notebooks or any(output.is_relative_to(folder) for folder in input_dirs):
            parser.error("Write audit outputs outside the read-only input directories and notebook paths")
    audit = Audit()
    runs = {"50": audit.run(args.run50, 50, args.notebook50)}
    between = None
    if args.run200:
        runs["200"] = audit.run(args.run200, 200, args.notebook200)
        between = compare_runs(audit, runs["50"], runs["200"])
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not any(i["severity"] == "error" for i in audit.issues),
        "notebooks_audited_for_all_supplied_runs": all(r.get("notebook") is not None for r in runs.values()),
        "runs": runs, "between_runs": between, "issues": audit.issues,
        "verification_boundaries": [
            "Files and notebooks were read only. No training or PyTorch/pickle deserialization was performed.",
            "Checkpoint checks establish presence and nonempty files only; the notebook itself must verify model reload.",
            "GIF/PNG checks establish signatures and positive dimensions, not complete decoding, animation playback, or gameplay interpretation.",
            "No ZIP archive was supplied to this command; independently verify the preserved ZIP's integrity.",
            "Notebook checks examine saved execution counts, error outputs, parameters, five-score table, means, and final images; they do not rerun cells.",
            "One training seed and five fixed evaluation games do not establish reliable research-level generalization. Seed 101 is reused in periodic demos.",
            "The 200-episode model starts fresh. GPU kernels are not forced deterministic. Equal seeds do not make independent training replications.",
            "Training runtime includes periodic demos and excludes setup, baseline evaluation, final evaluation, and ZIP download.",
            "Public GitHub access, rendered notebook visibility, README embeds, and course submission must be verified separately.",
        ],
    }
    md = markdown(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(md, encoding="utf-8")
    sys.stdout.write(md)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
