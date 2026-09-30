#!/usr/bin/env python3
"""Evaluate English and Chinese capability retention."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from sacrebleu.metrics import BLEU, CHRF
from vllm import LLM, SamplingParams


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TIBETAN_RE = re.compile(r"[\u0F00-\u0FFF]")
CJK_RE = re.compile(r"[\u3400-\u9FFF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def render_chat(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate(llm: LLM, prompts: list[str], max_tokens: int) -> list[str]:
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params, use_tqdm=True)
    return [output.outputs[0].text if output.outputs else "" for output in outputs]


def extract_choice(text: str) -> str | None:
    cleaned = strip_thinking(text).upper()
    matches = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", cleaned)
    return matches[-1] if matches else None


def mcq_prompt(row: dict[str, Any], language: str) -> tuple[str, str]:
    choices = "\n".join(f"{label}. {text}" for label, text in zip("ABCD", row["choices"], strict=True))
    if language == "en":
        system = "Answer the multiple-choice question. Output only A, B, C, or D."
        user = f"Question: {row['question']}\n{choices}\nAnswer:"
    else:
        system = "请回答选择题。只输出 A、B、C 或 D，不要解释。"
        user = f"问题：{row['question']}\n{choices}\n答案："
    return system, user


def evaluate_mcq(llm: LLM, tokenizer: Any, path: Path, task: str, language: str,
                 out_dir: Path) -> dict[str, Any]:
    examples = read_jsonl(path)
    prompts = [render_chat(tokenizer, *mcq_prompt(row, language)) for row in examples]
    outputs = generate(llm, prompts, max_tokens=16)
    rows = []
    for example, prompt, raw in zip(examples, prompts, outputs, strict=True):
        pred = extract_choice(raw)
        rows.append({
            "item_id": example["item_id"], "subject": example["subject"],
            "gold": example["answer"], "pred": pred, "correct": pred == example["answer"],
            "raw_output": raw, "prompt": prompt,
        })
    write_jsonl(out_dir / f"{task}_predictions.jsonl", rows)
    total = len(rows)
    return {
        "items": total,
        "accuracy": sum(row["correct"] for row in rows) / total,
        "valid_answer_rate": sum(row["pred"] is not None for row in rows) / total,
    }












def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True, dtype="auto",
    )
    tokenizer = llm.get_tokenizer()
    summary = {"model": args.model, "model_name": args.model_name, "tasks": {}}
    summary["tasks"]["mmlu"] = evaluate_mcq(
        llm, tokenizer, args.data_dir / "mmlu_test.jsonl", "mmlu", "en", out_dir
    )
    write_json(out_dir / "summary.json", summary)
    summary["tasks"]["ceval"] = evaluate_mcq(
        llm, tokenizer, args.data_dir / "ceval_val.jsonl", "ceval", "zh", out_dir
    )
    write_json(out_dir / "summary.json", summary)
    write_json(out_dir / "run_config.json", {
        "model": args.model, "model_name": args.model_name, "data_dir": str(args.data_dir),
        "max_model_len": args.max_model_len,
        "decoding": {"temperature": 0.0, "top_p": 1.0, "thinking": False,
                     "mcq_max_tokens": 16, "translation_max_tokens": 256},
    })


if __name__ == "__main__":
    main()
