#!/usr/bin/env python3
"""Aggregate retention scores and paired uncertainty estimates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from scipy.stats import binomtest


BACKBONES = ("qwen1.5", "qwen3")
ROUTES = ("original", "fair", "cpt")
MCQ_TASKS = ("mmlu", "ceval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mcq-bootstrap", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_accuracy_test(base: np.ndarray, other: np.ndarray, replicates: int,
                         rng: np.random.Generator) -> dict[str, float | int]:
    delta = other.astype(float) - base.astype(float)
    values = np.empty(replicates, dtype=float)
    batch = 500
    for start in range(0, replicates, batch):
        size = min(batch, replicates - start)
        indices = rng.integers(0, len(delta), size=(size, len(delta)))
        values[start:start + size] = delta[indices].mean(axis=1)
    base_only = int(np.sum(base & ~other))
    other_only = int(np.sum(~base & other))
    discordant = base_only + other_only
    p = 1.0 if discordant == 0 else binomtest(
        min(base_only, other_only), discordant, 0.5, alternative="two-sided"
    ).pvalue
    return {
        "delta": float(delta.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "base_only_correct": base_only,
        "other_only_correct": other_only,
        "mcnemar_p": float(p),
    }


def holm_adjust(rows: list[dict[str, Any]], p_key: str, out_key: str) -> None:
    order = sorted(range(len(rows)), key=lambda index: float(rows[index][p_key]))
    running = 0.0
    total = len(rows)
    for rank, index in enumerate(order):
        adjusted = min(1.0, (total - rank) * float(rows[index][p_key]))
        running = max(running, adjusted)
        rows[index][out_key] = running


def aligned_correct(root: Path, model: str, task: str) -> tuple[list[str], np.ndarray]:
    rows = read_jsonl(root / "evaluation" / model / f"{task}_predictions.jsonl")
    return [str(row["item_id"]) for row in rows], np.asarray([row["correct"] for row in rows], dtype=bool)


def retention_tables(root: Path, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    score_rows, test_rows = [], []
    rng = np.random.default_rng(args.seed)
    for backbone in BACKBONES:
        for task in MCQ_TASKS:
            arrays = {}
            ids = None
            for route in ROUTES:
                model = f"{backbone}-{route}"
                current_ids, correct = aligned_correct(root, model, task)
                if ids is not None and current_ids != ids:
                    raise ValueError(f"Item order mismatch for {model}/{task}")
                ids, arrays[route] = current_ids, correct
                score_rows.append({
                    "backbone": backbone, "route": route, "task": task,
                    "items": len(correct), "accuracy": float(correct.mean()),
                    "retained_fraction": float(correct.mean() / arrays["original"].mean())
                    if "original" in arrays else 1.0,
                })
            for route in ("fair", "cpt"):
                stats = paired_accuracy_test(arrays["original"], arrays[route], args.mcq_bootstrap, rng)
                test_rows.append({"backbone": backbone, "task": task,
                                  "comparison": f"{route}-original", **stats})
    holm_adjust(test_rows, "mcnemar_p", "holm_p")
    return score_rows, test_rows














def main() -> None:
    args = parse_args()
    analysis = args.root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    retention, retention_tests = retention_tables(args.root, args)
    write_csv(analysis / "retention_scores.csv", retention)
    write_csv(analysis / "retention_paired_tests.csv", retention_tests)
    print(json.dumps({"mcq_bootstrap": args.mcq_bootstrap, "seed": args.seed,
                      "holm_family": len(retention_tests)}))


if __name__ == "__main__":
    main()
