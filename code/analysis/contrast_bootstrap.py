#!/usr/bin/env python3
"""Paired item bootstrap for arbitrary linear contrasts within the matched evaluation block.

A contrast is a dict {run_name: weight}; e.g. the six-epoch route contrast
[Q3 CPT - Q1.5 CPT] - [Q3 VA6 - Q1.5 VA6]. Conventions follow horizon_bootstrap.py:
50,000 accuracy draws, 5,000 corpus-spBLEU draws, Holm within each named family.
Intervals are conditional on the observed (single) training runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
BLOCK = ROOT / "results/matched_block"
OUT = ROOT / "results/ipm/e14_contrasts.csv"
TASKS = ("tlue", "gsm", "bele", "zh2bo", "bo2zh")
ACC_DRAWS, MT_DRAWS = 50_000, 5_000

C15, C3 = "qwen1.5-instruct-cpt-sft-reference", "qwen3-instruct-cpt-sft-reference"
V15_2, V15_2L = "qwen1.5-instruct-va-sft-reference", "qwen1.5-instruct-va-sft-2epoch-l20-seed42"
V15_6 = "qwen1.5-instruct-va-sft-6epoch-seed42"
V3_2, V3_6 = "qwen3-instruct-va-sft-reference", "qwen3-instruct-va-sft-6epoch-seed42"

FAMILIES = {
    "route_contrast_6ep": {C3: 1, C15: -1, V3_6: -1, V15_6: 1},
    "route_contrast_2ep_seed42": {C3: 1, C15: -1, V3_2: -1, V15_2: 1},
    "va6_upgrade_q15_to_q3": {V3_6: 1, V15_6: -1},
    "va2_upgrade_q15_to_q3": {V3_2: 1, V15_2: -1},
    "cpt_upgrade_q15_to_q3": {C3: 1, C15: -1},
    "q15_horizon_gain_l20": {V15_6: 1, V15_2L: -1},
    "q3_horizon_gain": {V3_6: 1, V3_2: -1},
}
for _m in ("qwen1.5", "qwen2", "qwen2.5", "qwen3"):
    _cpt, _va2, _va6 = (f"{_m}-instruct-cpt-sft-reference", f"{_m}-instruct-va-sft-reference",
                        f"{_m}-instruct-va-sft-6epoch-seed42")
    FAMILIES[f"route_gap_{_m}_2ep"] = {_cpt: 1, _va2: -1}
    FAMILIES[f"route_gap_{_m}_6ep"] = {_cpt: 1, _va6: -1}
    FAMILIES[f"horizon_gain_{_m}"] = {_va6: 1, _va2: -1}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_seed(label: str) -> int:
    return (42 + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")) % (2**32)


def bootstrap_p(samples: np.ndarray) -> float:
    low = (np.count_nonzero(samples <= 0) + 1) / (len(samples) + 1)
    high = (np.count_nonzero(samples >= 0) + 1) / (len(samples) + 1)
    return min(1.0, 2 * min(low, high))


def holm(rows: list[dict]) -> None:
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_raw"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["p_raw"]))
        rows[index]["p_holm"] = running


def load(runs: list[str], task: str) -> dict:
    name = "mt_predictions.jsonl" if task in {"zh2bo", "bo2zh"} else f"{task}_predictions.jsonl"
    data = {run: read_jsonl(BLOCK / run / name) for run in runs}
    ref = data[runs[0]]
    for run, rows in data.items():
        if [str(r["item_id"]) for r in rows] != [str(r["item_id"]) for r in ref]:
            raise ValueError(f"{task}/{run}: item IDs not aligned")
        if task not in {"zh2bo", "bo2zh"} and [r["gold"] for r in rows] != [r["gold"] for r in ref]:
            raise ValueError(f"{task}/{run}: gold labels not aligned")
    return data


def flag(value) -> float:
    return float(value in (True, "True", "true", 1, "1"))


def evaluate(task: str, weights: dict, label: str) -> dict:
    runs = list(weights)
    data = load(runs, task)
    rng = np.random.default_rng(stable_seed(f"{label}-{task}"))
    if task in {"zh2bo", "bo2zh"}:
        metric = BLEU(tokenize="flores200")
        refs = [r["bo" if task == "zh2bo" else "zh"] for r in data[runs[0]]]
        stats = {run: np.asarray(metric._extract_corpus_statistics([r[f"{task}_pred"] for r in rows], [refs]),
                                 dtype=np.int64) for run, rows in data.items()}

        def score(s):
            return float(metric._compute_score_from_stats(s.sum(axis=0).tolist()).score)

        point = sum(w * score(stats[run]) for run, w in weights.items())
        n = len(refs)
        boot = np.empty(MT_DRAWS)
        for d in range(MT_DRAWS):
            index = rng.integers(0, n, size=n)
            boot[d] = sum(w * score(stats[run][index]) for run, w in weights.items())
    else:
        values = {run: np.asarray([flag(r["correct"]) for r in rows]) for run, rows in data.items()}
        point = sum(w * values[run].mean() * 100 for run, w in weights.items())
        n = len(next(iter(values.values())))
        chunks = []
        for start in range(0, ACC_DRAWS, 1000):
            index = rng.integers(0, n, size=(min(1000, ACC_DRAWS - start), n))
            chunks.append(sum(w * values[run][index].mean(axis=1) * 100 for run, w in weights.items()))
        boot = np.concatenate(chunks)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {"family": label, "task": task, "estimate": round(float(point), 3),
            "ci_low": round(float(low), 3), "ci_high": round(float(high), 3), "p_raw": bootstrap_p(boot)}


def main() -> None:
    rows = []
    for label, weights in FAMILIES.items():
        family = [evaluate(task, weights, label) for task in TASKS]
        holm(family)
        rows.extend(family)
        for r in family:
            print(f"{label:28s} {r['task']:6s} {r['estimate']:+8.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}] Holm p={r['p_holm']:.4g}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
