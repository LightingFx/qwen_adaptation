#!/usr/bin/env python3
"""Freeze the valid-item Ti-MMLU scores without modifying historical predictions.

Read copied VA predictions in results/ipm_inputs and existing CPT predictions.
Only the known empty item is excluded. Other E5 tasks retain their prior
estimates; Holm correction is recomputed over the complete five-task family.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "results/ipm_inputs"
OUTPUT = ROOT / "results/ipm"
EXCLUDE = "global_facts2"
MODELS = ("qwen1.5", "qwen2", "qwen2.5", "qwen3")
SEEDS = (42, 17, 123)


def main():
    sources, scores, arrays = [], [], {}
    signature = None
    for model in MODELS:
        size = "8b" if model == "qwen3" else "7b"
        paths = [("CPT", None, ROOT / f"results/e0/results/{model}-{size}bo-sft/tlue_predictions.jsonl")]
        paths.extend(("VA", seed, INPUTS / f"{model}-va-seed{seed}-tlue.jsonl")
                     for seed in (SEEDS if model in ("qwen1.5", "qwen3") else (42,)))
        for route, seed, path in paths:
            raw = path.read_bytes()
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            assert len(rows) == 670, (path, len(rows))
            excluded = [row for row in rows if row["item_id"] == EXCLUDE]
            assert len(excluded) == 1, path
            # The stored prompt independently exposes the empty question.
            assert "请回答：\n\n答案：" in excluded[0]["prompt"], path
            rows = [row for row in rows if row["item_id"] != EXCLUDE]
            observed = [(row["item_id"], row["gold"]) for row in rows]
            if signature is None:
                signature = observed
            assert observed == signature, path
            values = np.array([row["correct"] for row in rows], dtype=float)
            arrays[(model, route, seed)] = values
            scores.append(dict(backbone=model, route=route, seed=seed, n=len(rows),
                               correct=int(values.sum()), accuracy_pct=float(values.mean() * 100)))
            sources.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest()))

    cpt_delta = arrays[("qwen3", "CPT", None)] - arrays[("qwen1.5", "CPT", None)]
    va_deltas = [arrays[("qwen3", "VA", seed)] - arrays[("qwen1.5", "VA", seed)] for seed in SEEDS]
    delta = 100 * (cpt_delta - np.mean(va_deltas, axis=0))
    seedwise = [float(100 * (cpt_delta - va_delta).mean()) for va_delta in va_deltas]
    rng_seed = (20260903 + int.from_bytes(hashlib.sha256(b"tlue").digest()[:4], "little")) % 2**32
    rng = np.random.default_rng(rng_seed)
    bootstrap = np.concatenate([delta[rng.integers(0, 669, size=(1000, 669))].mean(axis=1) for _ in range(50)])
    p = min(1.0, 2 * min((np.count_nonzero(bootstrap <= 0) + 1) / 50001,
                         (np.count_nonzero(bootstrap >= 0) + 1) / 50001))
    low, high = np.quantile(bootstrap, [.025, .975])
    source_csv = INPUTS / "e5_interactions_original.csv"
    with source_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        interactions = list(reader)
    assert [row["task"] for row in interactions] == ["tlue", "gsm", "bele", "zh2bo", "bo2zh"]
    interactions[0].update(cpt_endpoint_upgrade=float(cpt_delta.mean() * 100),
                           va_seed_average_endpoint_upgrade=float(np.mean(va_deltas) * 100),
                           seed_average_interaction=float(delta.mean()),
                           ci_low_item_bootstrap=float(low), ci_high_item_bootstrap=float(high),
                           p_raw=p, seedwise_interactions=";".join(f"{x:.6f}" for x in seedwise),
                           seed_sd_interaction=float(np.std(seedwise, ddof=1)))
    running = 0.0
    for rank, index in enumerate(sorted(range(5), key=lambda i: float(interactions[i]["p_raw"]))):
        running = max(running, min(1.0, (5-rank) * float(interactions[index]["p_raw"])))
        interactions[index]["p_holm"] = running
    OUTPUT.mkdir(exist_ok=True)
    for name, rows, columns in (("ti_mmlu_scores_669.csv", scores, list(scores[0])),
                                ("seed_average_endpoint_interactions.csv", interactions, fields)):
        with (OUTPUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    manifest = dict(excluded_item=EXCLUDE, exclusion_reason="empty question and options",
                    valid_items=669, prediction_files=len(sources), aligned=True,
                    source_files=sources,
                    other_tasks_source_sha256=hashlib.sha256(source_csv.read_bytes()).hexdigest(),
                    bootstrap_draws=50000, bootstrap_seed=rng_seed,
                    note="Only Ti-MMLU is rescored here; other task estimates are preserved. Holm is recomputed for all five tasks.")
    (OUTPUT / "ti_mmlu_freeze_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(valid_items=669, aligned_files=len(sources), corrected_ti_interaction=interactions[0]), indent=2))


if __name__ == "__main__":
    main()
