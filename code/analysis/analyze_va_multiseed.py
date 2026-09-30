#!/usr/bin/env python3
"""Aggregate three VA-SFT seeds and recompute endpoint route interactions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU

import analyze_generation_behavior as behavior


SEEDS = (42, 17, 123)
MODELS = ("qwen1.5", "qwen3")
TASKS = ("tlue", "gsm", "bele", "zh2bo", "bo2zh")
E0_NAMES = {"qwen1.5": "qwen1.5-7bbo-sft", "qwen3": "qwen3-8bbo-sft"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_seed(seed: int, label: str) -> int:
    return (seed + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")) % (2**32)


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda index: rows[index]["p_raw"])
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = min(1.0, (len(rows) - rank) * rows[index]["p_raw"])
        running = max(running, adjusted)
        rows[index]["p_holm"] = running


def bootstrap_p(samples: np.ndarray) -> float:
    low = (np.count_nonzero(samples <= 0) + 1) / (len(samples) + 1)
    high = (np.count_nonzero(samples >= 0) + 1) / (len(samples) + 1)
    return min(1.0, 2 * min(low, high))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def va_dir(args, model: str, seed: int) -> Path:
    if seed == 42:
        return args.seed42_root / f"{model}-vocab-aware-sft"
    return args.root / "evaluation" / "results" / f"{model}-va-sft-seed{seed}"


def task_path(directory: Path, task: str) -> Path:
    return directory / ("mt_predictions.jsonl" if task in {"zh2bo", "bo2zh"} else f"{task}_predictions.jsonl")


def prediction_text(row: dict, task: str) -> str:
    return row[f"{task}_pred"]


def assert_alignment(collection: list[list[dict]], task: str) -> None:
    reference = collection[0]
    for rows in collection[1:]:
        if [row["item_id"] for row in rows] != [row["item_id"] for row in reference]:
            raise ValueError(f"{task}: item IDs not aligned")
        if task in {"zh2bo", "bo2zh"}:
            for left, right in zip(rows, reference, strict=True):
                if left["zh"] != right["zh"] or left["bo"] != right["bo"]:
                    raise ValueError(f"{task}: sources/references not aligned")
        elif [row["gold"] for row in rows] != [row["gold"] for row in reference]:
            raise ValueError(f"{task}: gold labels not aligned")


def accuracy_interaction(args, task: str, draws: int) -> tuple[dict, list[float]]:
    cpt = {
        model: read_jsonl(args.e0_root / "results" / E0_NAMES[model] / f"{task}_predictions.jsonl")
        for model in MODELS
    }
    va = {
        (model, seed): read_jsonl(task_path(va_dir(args, model, seed), task))
        for model in MODELS for seed in SEEDS
    }
    assert_alignment(list(cpt.values()) + list(va.values()), task)
    cpt_values = {model: np.asarray([row["correct"] for row in rows], dtype=float) for model, rows in cpt.items()}
    va_values = {(model, seed): np.asarray([row["correct"] for row in rows], dtype=float) for (model, seed), rows in va.items()}
    cpt_upgrade = (cpt_values["qwen3"].mean() - cpt_values["qwen1.5"].mean()) * 100
    seedwise = [
        cpt_upgrade - (va_values[("qwen3", seed)].mean() - va_values[("qwen1.5", seed)].mean()) * 100
        for seed in SEEDS
    ]
    point = float(np.mean(seedwise))
    rng = np.random.default_rng(stable_seed(args.seed, task))
    boot = np.empty(draws)
    n = len(cpt_values["qwen3"])
    for start in range(0, draws, 1000):
        stop = min(draws, start + 1000)
        index = rng.integers(0, n, size=(stop - start, n))
        cpt_boot = (cpt_values["qwen3"][index].mean(axis=1) - cpt_values["qwen1.5"][index].mean(axis=1)) * 100
        va_boot = np.mean([
            (va_values[("qwen3", seed)][index].mean(axis=1) - va_values[("qwen1.5", seed)][index].mean(axis=1)) * 100
            for seed in SEEDS
        ], axis=0)
        boot[start:stop] = cpt_boot - va_boot
    low, high = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    return {
        "task": task,
        "cpt_endpoint_upgrade": float(cpt_upgrade),
        "va_seed_average_endpoint_upgrade": float(cpt_upgrade - point),
        "seed_average_interaction": point,
        "ci_low_item_bootstrap": low,
        "ci_high_item_bootstrap": high,
        "p_raw": bootstrap_p(boot),
        "seedwise_interactions": ";".join(f"{value:.6f}" for value in seedwise),
        "seed_sd_interaction": float(np.std(seedwise, ddof=1)),
    }, seedwise


def score_stats(metric: BLEU, stats: np.ndarray) -> float:
    return float(metric._compute_score_from_stats(stats.astype(np.int64).tolist()).score)


def mt_interaction(args, task: str, draws: int) -> tuple[dict, list[float]]:
    cpt = {
        model: read_jsonl(args.e0_root / "results_explicit_mt" / E0_NAMES[model] / "mt_predictions.jsonl")
        for model in MODELS
    }
    va = {
        (model, seed): read_jsonl(task_path(va_dir(args, model, seed), task))
        for model in MODELS for seed in SEEDS
    }
    assert_alignment(list(cpt.values()) + list(va.values()), task)
    metric = BLEU(tokenize="flores200")
    refs = [row["bo" if task == "zh2bo" else "zh"] for row in cpt["qwen1.5"]]
    cpt_stats = {
        model: np.asarray(metric._extract_corpus_statistics([prediction_text(row, task) for row in rows], [refs]), dtype=np.int64)
        for model, rows in cpt.items()
    }
    va_stats = {
        key: np.asarray(metric._extract_corpus_statistics([prediction_text(row, task) for row in rows], [refs]), dtype=np.int64)
        for key, rows in va.items()
    }
    cpt_scores = {model: score_stats(metric, stats.sum(axis=0)) for model, stats in cpt_stats.items()}
    va_scores = {key: score_stats(metric, stats.sum(axis=0)) for key, stats in va_stats.items()}
    cpt_upgrade = cpt_scores["qwen3"] - cpt_scores["qwen1.5"]
    seedwise = [cpt_upgrade - (va_scores[("qwen3", seed)] - va_scores[("qwen1.5", seed)]) for seed in SEEDS]
    point = float(np.mean(seedwise))
    rng = np.random.default_rng(stable_seed(args.seed, task))
    boot = np.empty(draws)
    n = len(refs)
    for start in range(0, draws, 50):
        stop = min(draws, start + 50)
        indices = rng.integers(0, n, size=(stop - start, n))
        for offset, index in enumerate(indices):
            cpt_boot = score_stats(metric, cpt_stats["qwen3"][index].sum(axis=0)) - score_stats(metric, cpt_stats["qwen1.5"][index].sum(axis=0))
            va_boot = np.mean([
                score_stats(metric, va_stats[("qwen3", seed)][index].sum(axis=0))
                - score_stats(metric, va_stats[("qwen1.5", seed)][index].sum(axis=0))
                for seed in SEEDS
            ])
            boot[start + offset] = cpt_boot - va_boot
    low, high = (float(x) for x in np.quantile(boot, [0.025, 0.975]))
    return {
        "task": task,
        "cpt_endpoint_upgrade": float(cpt_upgrade),
        "va_seed_average_endpoint_upgrade": float(cpt_upgrade - point),
        "seed_average_interaction": point,
        "ci_low_item_bootstrap": low,
        "ci_high_item_bootstrap": high,
        "p_raw": bootstrap_p(boot),
        "seedwise_interactions": ";".join(f"{value:.6f}" for value in seedwise),
        "seed_sd_interaction": float(np.std(seedwise, ddof=1)),
    }, seedwise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed42-root", type=Path, required=True)
    parser.add_argument("--e0-root", type=Path, required=True)
    parser.add_argument("--accuracy-bootstrap", type=int, default=50_000)
    parser.add_argument("--mt-bootstrap", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    out = args.root / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    seed_scores = []
    for model in MODELS:
        for seed in SEEDS:
            summary = json.loads((va_dir(args, model, seed) / "summary.json").read_text(encoding="utf-8"))
            tasks = summary["tasks"]
            seed_scores.append({
                "backbone": model, "seed": seed,
                "tlue": tasks["tlue"]["accuracy"] * 100,
                "gsm": tasks["gsm"]["accuracy"] * 100,
                "bele": tasks["bele"]["accuracy"] * 100,
                "zh2bo": tasks["mt"]["zh2bo"]["spbleu"],
                "bo2zh": tasks["mt"]["bo2zh"]["spbleu"],
            })
    write_csv(out / "va_seed_scores.csv", seed_scores)
    aggregates = []
    for model in MODELS:
        model_rows = [row for row in seed_scores if row["backbone"] == model]
        for task in TASKS:
            values = np.asarray([row[task] for row in model_rows])
            aggregates.append({
                "backbone": model, "task": task, "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)), "min": float(values.min()), "max": float(values.max()),
            })
    write_csv(out / "va_seed_aggregates.csv", aggregates)

    interactions = []
    for task in TASKS:
        row, _ = (
            accuracy_interaction(args, task, args.accuracy_bootstrap)
            if task in {"tlue", "gsm", "bele"}
            else mt_interaction(args, task, args.mt_bootstrap)
        )
        interactions.append(row)
    holm(interactions)
    write_csv(out / "seed_average_endpoint_interactions.csv", interactions)

    q3_behavior = []
    for seed in SEEDS:
        rows = read_jsonl(va_dir(args, "qwen3", seed) / "mt_predictions.jsonl")
        for direction in ("zh2bo", "bo2zh"):
            q3_behavior.append({"seed": seed, "direction": direction, **behavior.summarize(rows, direction)})
    write_csv(out / "qwen3_generation_behavior_by_seed.csv", q3_behavior)

    lines = ["# VA-SFT multi-seed stability", "", "## Seed scores", "",
             "| Backbone | Seed | Ti-MMLU | GSM8K | Belebele | zh→bo | bo→zh |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in seed_scores:
        lines.append(f"| {row['backbone']} | {row['seed']} | {row['tlue']:.2f} | {row['gsm']:.2f} | {row['bele']:.2f} | {row['zh2bo']:.2f} | {row['bo2zh']:.2f} |")
    lines += ["", "## Seed-average endpoint interactions", ""]
    for row in interactions:
        lines.append(
            f"- {row['task']}: {row['seed_average_interaction']:+.2f} "
            f"(item/corpus-bootstrap 95% CI {row['ci_low_item_bootstrap']:+.2f} to "
            f"{row['ci_high_item_bootstrap']:+.2f}; Holm p={row['p_holm']:.4g}; "
            f"between-seed SD={row['seed_sd_interaction']:.2f})."
        )
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "analysis_manifest.json").write_text(json.dumps({
        "training_seeds": list(SEEDS),
        "evaluation_seed": 42,
        "seed_average_interaction": "CPT endpoint upgrade minus mean of same-seed VA endpoint upgrades",
        "confidence_interval": "paired item/corpus bootstrap conditional on the three observed training seeds",
        "training_variance_note": "between-seed SD is reported separately; the bootstrap CI is not a population CI over training runs",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
