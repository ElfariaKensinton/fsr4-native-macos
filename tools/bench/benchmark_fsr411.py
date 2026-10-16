#!/usr/bin/env python3
"""Benchmark FSR 4.1.1 serially, reversing variant order on alternate rounds.

Usage: benchmark_fsr411.py plan.json output --rounds 6 --bench 30
Plan: {"input": "/capture", "variants": [{"name": "baseline-full",
       "root": "/source", "builddir": "/cache", "profile": "exact-fast",
       "binary": "/source/tools/bin/fsr4mac", "source_hash": "optional"}]}
The plan may supply a default binary; each variant may override input.
Relative plan paths resolve against the plan's directory. Output must be
empty. Use --dry-run to inspect commands without starting a GPU workload.

Each CLI run warms one sequence, then measures complete command buffers.
No compilation, stage counters, captures or snapshots are requested. Run
this without other GPU benchmarks/tests. Full excludes reset/reused frames;
reported summary times are medians of round means, not pooled frame means.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time


PROTOCOL = ("Serial interleaved rounds, reversed on alternate rounds; median of round means; "
            "one warmup sequence per run; full command buffer timestamps; no diagnostic GPU work")
GROUPS = ("all", "reset", "full", "reuse")


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan(path):
    plan = json.loads(path.read_text())
    variants = plan.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("plan must contain a nonempty variants array")
    normalized, names = [], set()
    for item in variants:
        variant = dict(item)
        name = variant.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
            raise ValueError("variant name must use letters, numbers, dots, underscores or hyphens")
        if name in names:
            raise ValueError("duplicate variant: " + name)
        names.add(name)
        if variant.get("profile") not in ("exact", "exact-fast", "lossy"):
            raise ValueError(name + ": profile must be exact, exact-fast or lossy")
        for key in ("root", "builddir", "binary", "input"):
            value = variant.get(key, plan.get(key) if key in ("binary", "input") else None)
            if not isinstance(value, str) or not value:
                raise ValueError(name + ": missing " + key)
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            resolved = resolved.resolve()
            if key == "binary":
                if not resolved.is_file() or not os.access(resolved, os.X_OK):
                    raise ValueError(name + ": binary is not executable: " + str(resolved))
            elif not resolved.is_dir():
                raise ValueError(name + ": directory is missing: " + str(resolved))
            variant[key] = str(resolved)
        if not (Path(variant["input"]) / "meta.json").is_file():
            raise ValueError(name + ": input has no meta.json")
        if "source_hash" not in variant and "source_hash" in plan:
            variant["source_hash"] = plan["source_hash"]
        normalized.append(variant)
    return {**plan, "variants": normalized}


def command_for(variant, output, bench):
    backend = variant.get("backend", "i8mpp")
    if backend not in ("i8mpp", "simd", "scalar", "reference"):
        raise ValueError(variant.get("name", "variant") + ": unsupported backend " + str(backend))
    return [variant["binary"], "--fsr-version", "4.1.1", "--root", variant["root"],
            "--builddir", variant["builddir"], "--no-compile", "--" + backend,
            "--profile", variant["profile"], variant["input"], str(output), "--bench", str(bench)]


def summarize(rows):
    values = [row["gpu_ms"] for row in rows]
    if not values:
        return {"count": 0}
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("GPU durations must be finite positive numbers")
    return {"count": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values)}


def read_measurements(path, variant, bench):
    run = json.loads(path.read_text())
    expected_backend = {"i8mpp": "int8-mpp", "simd": "int8-simd",
                        "scalar": "int8-scalar", "reference": "int8-reference"}[variant.get("backend", "i8mpp")]
    if (run.get("fsr_version") != "4.1.1" or run.get("profile") != variant["profile"]
            or run.get("backend") != expected_backend):
        raise ValueError(str(path) + ": unexpected model, profile or backend")
    if (run.get("timing_scope") != "full frame command buffer"
            or run.get("diagnostic_gpu_counters") or run.get("diagnostic_gpu_copies")
            or run.get("capture_stages")):
        raise ValueError(str(path) + ": timings include diagnostic GPU work or have an unknown scope")
    if run.get("benchmark_iterations") != bench or run.get("warmup_sequences") != 1:
        raise ValueError(str(path) + ": unexpected measured/warmup sequence count")
    rows = run.get("measurements", [])
    if not rows or len(rows) != run.get("measured_frames"):
        raise ValueError(str(path) + ": incomplete measurements")
    for row in rows:
        if (row.get("effective_reset") not in (False, True, 0, 1)
                or row.get("skipped_inference") not in (False, True, 0, 1)
                or row.get("stage_profile") or row.get("captured_stages")):
            raise ValueError(str(path) + ": invalid frame classification or diagnostic measurements")
    return run, rows


def power_state():
    try:
        result = subprocess.run(["pmset", "-g", "custom"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def round_summary(results, variants):
    summary = {}
    for variant in variants:
        selected = [result for result in results if result["variant"] == variant["name"]]
        value = {"profile": variant["profile"], "rounds": len(selected)}
        for group in GROUPS:
            means = [result[group]["mean_ms"] for result in selected if result[group]["count"]]
            value[group] = {"count": sum(result[group]["count"] for result in selected),
                            "rounds": len(means)}
            if means:
                median = statistics.median(means)
                bounds = [min(means), max(means)]
                value[group].update(median_round_mean_ms=median, round_mean_range_ms=bounds)
                # Keep the concise fields used by the earlier local harness.
                value["mean_ms" if group == "all" else group + "_mean_ms"] = median
                if group == "all":
                    value["round_mean_range_ms"] = bounds
        equal = [result["steady_equal_mean_ms"] for result in selected if "steady_equal_mean_ms" in result]
        if equal:
            value["steady_equal_mean_ms"] = statistics.median(equal)
        summary[variant["name"]] = value
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--bench", type=int, default=30, help="measured sequences per variant per round")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.bench