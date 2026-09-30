#!/usr/bin/env python3
"""Character-level chrF counterpart of contrast_bootstrap.py for the two translation directions.

Uses sacrebleu CHRF(word_order=0), which does not require word segmentation.
It uses the same families, paired item resampling (5,000 draws) and seeds as the spBLEU analysis. Intervals are conditional on the
trained models.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contrast_bootstrap as cb  # noqa: E402

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
cb.BLOCK = ROOT / "results/matched_block"
OUT = ROOT / "results/ipm/e14_contrasts_chrf.csv"
DRAWS = 5_000
METRIC = CHRF(word_order=0)


def stats_for(run: str, task: str) -> tuple[np.ndarray, list[str]]:
    rows = cb.read_jsonl(cb.BLOCK / run / "mt_predictions.jsonl")
    refs = [r["bo" if task == "zh2bo" else "zh"] for r in rows]
    hyps = [r[f"{task}_pred"] for r in rows]
    stats = np.asarray(METRIC._extract_corpus_statistics(hyps, [refs]), dtype=np.int64)
    return stats, [str(r["item_id"]) for r in rows]


def score(stats: np.ndarray) -> float:
    return float(METRIC._compute_score_from_stats(stats.sum(axis=0).tolist()).score)


def main() -> None:
    cache: dict = {}
    for label, weights in cb.FAMILIES.items():
        for run in weights:
            for task in ("zh2bo", "bo2zh"):
                if (run, task) not in cache:
                    cache[(run, task)] = stats_for(run, task)
    ids = {k: v[1] for k, v in cache.items()}
    ref_ids = next(iter(ids.values()))
    assert all(v == ref_ids for v in ids.values()), "item IDs not aligned"
    runs16 = [f"{m}-instruct-{c}" for m in ("qwen1.5", "qwen2", "qwen2.5", "qwen3")
              for c in ("original", "va-sft-reference", "va-sft-6epoch-seed42", "cpt-sft-reference")]
    with (ROOT / "results/ipm/chrf_scores.csv").open("w", encoding="utf-8", newline="") as h:
        wr = csv.writer(h)
        wr.writerow(["run", "zh2bo_chrf", "bo2zh_chrf"])
        for run in runs16:
            wr.writerow([run] + [round(score(stats_for(run, t)[0]), 2) for t in ("zh2bo", "bo2zh")])
    rows = []
    for label, weights in cb.FAMILIES.items():
        for task in ("zh2bo", "bo2zh"):
            point = sum(w * score(cache[(run, task)][0]) for run, w in weights.items())
            rng = np.random.default_rng(cb.stable_seed(f"chrf0-{label}-{task}"))
            n = len(ref_ids)
            boot = np.empty(DRAWS)
            for d in range(DRAWS):
                idx = rng.integers(0, n, size=n)
                boot[d] = sum(w * score(cache[(run, task)][0][idx]) for run, w in weights.items())
            low, high = np.quantile(boot, [0.025, 0.975])
            rows.append({"family": label, "task": task, "estimate": round(point, 3), "ci_low": round(float(low), 3),
                         "ci_high": round(float(high), 3), "p_raw": cb.bootstrap_p(boot)})
            print(f"{label:28s} {task:6s} {point:+8.2f} [{low:+.2f}, {high:+.2f}]", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
