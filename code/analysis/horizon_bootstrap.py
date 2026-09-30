#!/usr/bin/env python3
"""Paired item bootstrap for the matched two-/six-epoch VA-SFT block (Qwen2.5 vs Qwen3).

Horizon contrast: I = [Y(Q3,6) - Y(Q2.5,6)] - [Y(Q3,2) - Y(Q2.5,2)], conditional on the
observed single training runs. Conventions follow scripts/supp/analyze_va_multiseed.py.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sacrebleu.metrics import BLEU

ROOT = Path(__file__).resolve().parents[1]
BLOCK = ROOT / "results/matched_block"
OUT = ROOT / "results/ipm"
RUNS = {
    ("qwen2.5", 2): "qwen2.5-instruct-va-sft-reference",
    ("qwen2.5", 6): "qwen2.5-instruct-va-sft-6epoch-seed42",
    ("qwen3", 2): "qwen3-instruct-va-sft-reference",
    ("qwen3", 6): "qwen3-instruct-va-sft-6epoch-seed42",
}
TASKS = ("tlue", "gsm", "bele", "zh2bo", "bo2zh")
SEED = 42
ACC_DRAWS, MT_DRAWS = 50_000, 5_000


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stable_seed(label: str) -> int:
    return (SEED + int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")) % (2**32)


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


def load(task: str) -> dict:
    name = "mt_predictions.jsonl" if task in {"zh2bo", "bo2zh"} else f"{task}_predictions.jsonl"
    data = {key: read_jsonl(BLOCK / run / name) for key, run in RUNS.items()}
    ref = next(iter(data.values()))
    for rows in data.values():
        if [r["item_id"] for r in rows] != [r["item_id"] for r in ref]:
            raise ValueError(f"{task}: item IDs not aligned")
        if task in {"zh2bo", "bo2zh"}:
            if any(a["zh"] != b["zh"] or a["bo"] != b["bo"] for a, b in zip(rows, ref)):
                raise ValueError(f"{task}: sources not aligned")
        elif [r["gold"] for r in rows] != [r["gold"] for r in ref]:
            raise ValueError(f"{task}: gold labels not aligned")
    return data


def contrasts(score) -> dict:
    gap2 = score[("qwen3", 2)] - score[("qwen2.5", 2)]
    gap6 = score[("qwen3", 6)] - score[("qwen2.5", 6)]
    return {"gap_2ep": gap2, "gap_6ep": gap6, "horizon_contrast": gap6 - gap2,
            "gain_qwen2.5": score[("qwen2.5", 6)] - score[("qwen2.5", 2)],
            "gain_qwen3": score[("qwen3", 6)] - score[("qwen3", 2)]}


def correct_flag(value) -> float:
    return float(value in (True, "True", "true", 1, "1"))


def run_accuracy(task: str) -> dict:
    data = load(task)
    values = {k: np.asarray([correct_flag(r["correct"]) for r in rows]) for k, rows in data.items()}
    point = contrasts({k: v.mean() * 100 for k, v in values.items()})
    rng = np.random.default_rng(stable_seed(f"horizon-{task}"))
    n = len(next(iter(values.values())))
    boot = {key: [] for key in point}
    for start in range(0, ACC_DRAWS, 1000):
        index = rng.integers(0, n, size=(min(1000, ACC_DRAWS - start), n))
        res = contrasts({k: v[index].mean(axis=1) * 100 for k, v in values.items()})
        for key in point:
            boot[key].append(res[key])
    return summarize(task, point, {k: np.concatenate(v) for k, v in boot.items()}, n)


def run_mt(task: str) -> dict:
    data = load(task)
    metric = BLEU(tokenize="flores200")
    ref_key = "bo" if task == "zh2bo" else "zh"
    refs = [r[ref_key] for r in next(iter(data.values()))]
    stats = {k: np.asarray(metric._extract_corpus_statistics([r[f"{task}_pred"] for r in rows], [refs]), dtype=np.int64)
             for k, rows in data.items()}

    def score(s: np.ndarray) -> float:
        return float(metric._compute_score_from_stats(s.sum(axis=0).tolist()).score)

    point = contrasts({k: score(s) for k, s in stats.items()})
    rng = np.random.default_rng(stable_seed(f"horizon-{task}"))
    n = len(refs)
    boot = {key: np.empty(MT_DRAWS) for key in point}
    for d in range(MT_DRAWS):
        index = rng.integers(0, n, size=n)
        res = contrasts({k: score(s[index]) for k, s in stats.items()})
        for key in point:
            boot[key][d] = res[key]
    return summarize(task, point, boot, n)


def summarize(task: str, point: dict, boot: dict, n: int) -> dict:
    row = {"task": task, "items": n}
    for key, value in point.items():
        low, high = np.quantile(boot[key], [0.025, 0.975])
        row[key] = round(float(value), 4)
        row[f"{key}_ci_low"] = round(float(low), 4)
        row[f"{key}_ci_high"] = round(float(high), 4)
    row["p_raw"] = bootstrap_p(boot["horizon_contrast"])
    return row


def main() -> None:
    rows = [run_mt(t) if t in {"zh2bo", "bo2zh"} else run_accuracy(t) for t in TASKS]
    holm(rows)
    OUT.mkdir(exist_ok=True)
    path = OUT / "horizon_contrast_bootstrap.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{r['task']:6s} gap2 {r['gap_2ep']:+7.2f}  gap6 {r['gap_6ep']:+7.2f} "
              f"[{r['gap_6ep_ci_low']:+.2f},{r['gap_6ep_ci_high']:+.2f}]  "
              f"I {r['horizon_contrast']:+7.2f} [{r['horizon_contrast_ci_low']:+.2f},{r['horizon_contrast_ci_high']:+.2f}]  "
              f"Holm p={r['p_holm']:.4g}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
