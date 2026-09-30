#!/usr/bin/env python3
"""Helpers for quantifying translation output behavior."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MODELS = ["qwen1.5", "qwen2", "qwen2.5", "qwen3"]
E0_NAMES = {
    "qwen1.5": "qwen1.5-7bbo-sft",
    "qwen2": "qwen2-7bbo-sft",
    "qwen2.5": "qwen2.5-7bbo-sft",
    "qwen3": "qwen3-8bbo-sft",
}
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TIBETAN_RE = re.compile(r"[\u0F00-\u0FFF]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
DIGIT_MAP = str.maketrans("༠༡༢༣༤༥༦༧༨༩", "0123456789")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text: str) -> str:
    text = THINK_RE.sub("", unicodedata.normalize("NFKC", text)).strip()
    return re.sub(r"\s+", "", text)


def digits(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text.translate(DIGIT_MAP))


def repeated_4gram_rate(text: str) -> float:
    text = normalize(text)
    grams = [text[index : index + 4] for index in range(max(0, len(text) - 3))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def item_metrics(row: dict[str, Any], direction: str) -> dict[str, Any]:
    if direction == "zh2bo":
        source, reference, raw = row["zh"], row["bo"], row["zh2bo_raw"]
        target_re, source_re = TIBETAN_RE, CJK_RE
    else:
        source, reference, raw = row["bo"], row["zh"], row["bo2zh_raw"]
        target_re, source_re = CJK_RE, TIBETAN_RE
    cleaned = THINK_RE.sub("", raw).strip()
    target_count = len(target_re.findall(cleaned))
    intrusion_count = len(source_re.findall(cleaned))
    script_total = target_count + intrusion_count
    source_norm, output_norm, reference_norm = normalize(source), normalize(cleaned), normalize(reference)
    source_digits, output_digits = digits(source), digits(cleaned)
    length_ratio = len(output_norm) / max(1, len(reference_norm))
    return {
        "valid_target": target_count > 0,
        "target_script_ratio": target_count / script_total if script_total else 0.0,
        "source_script_intrusion": intrusion_count > 0,
        "source_copy_similarity": SequenceMatcher(None, output_norm, source_norm).ratio(),
        "repeated_4gram_rate": repeated_4gram_rate(cleaned),
        "output_length": len(output_norm),
        "reference_length": len(reference_norm),
        "length_ratio": length_ratio,
        "abnormal_length": length_ratio < 0.25 or length_ratio > 4.0,
        "has_source_number": bool(source_digits),
        "numeric_preserved": source_digits == output_digits if source_digits else None,
    }


def summarize(rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    metrics = [item_metrics(row, direction) for row in rows]
    numeric = [row for row in metrics if row["has_source_number"]]
    return {
        "items": len(metrics),
        "valid_target_rate": sum(row["valid_target"] for row in metrics) / len(metrics),
        "mean_target_script_ratio": sum(row["target_script_ratio"] for row in metrics) / len(metrics),
        "source_script_intrusion_rate": sum(row["source_script_intrusion"] for row in metrics) / len(metrics),
        "mean_source_copy_similarity": sum(row["source_copy_similarity"] for row in metrics) / len(metrics),
        "mean_repeated_4gram_rate": sum(row["repeated_4gram_rate"] for row in metrics) / len(metrics),
        "abnormal_length_rate": sum(row["abnormal_length"] for row in metrics) / len(metrics),
        "mean_length_ratio": sum(row["length_ratio"] for row in metrics) / len(metrics),
        "numeric_items": len(numeric),
        "numeric_preservation_rate": (
            sum(row["numeric_preserved"] for row in numeric) / len(numeric) if numeric else None
        ),
    }



