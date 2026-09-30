#!/usr/bin/env python3
"""Evaluate Tibetan-to-Chinese direction control under alternative prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from sacrebleu.metrics import BLEU, CHRF

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
TIBETAN_RE = re.compile(r"[\u0F00-\u0FFF]")
PROMPT_STYLES = ("source_only", "user_only", "tibetan_instruction")
CHINESE_SYSTEM = "你是中藏翻译器。把藏文翻译成中文，只输出中文译文。"
GENERIC_SYSTEM = "你是一个有帮助的助手。"
CHINESE_USER = "请将以下藏文翻译成中文，只输出中文译文：\n{}"
TIBETAN_SYSTEM = (
    "ཁྱེད་ནི་བོད་རྒྱའི་སྐད་སྒྱུར་པ་ཞིག་ཡིན། "
    "བོད་ཡིག་རྒྱ་ཡིག་ཏུ་སྒྱུར་ནས་རྒྱ་ཡིག་གི་འགྱུར་ཡིག་ཁོ་ན་འདོན་དགོས།"
)
TIBETAN_USER = (
    "གཤམ་གྱི་བོད་ཡིག་རྒྱ་ཡིག་ཏུ་སྒྱུར་རོགས། "
    "རྒྱ་ཡིག་གི་འགྱུར་ཡིག་ཁོ་ན་འདོན་རོགས།\n{}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mt-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def strip_thinking(text: str) -> str:
    return THINK_RE.sub("", text).strip()


def normalize(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", strip_thinking(text)))


def clean_chinese(text: str) -> str:
    cleaned = strip_thinking(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    selected = [line for line in lines if CJK_RE.search(line)]
    return " ".join(selected or lines).strip()


def render(tokenizer: Any, system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt(style: str, source: str) -> tuple[str, str]:
    if style == "source_only":
        return CHINESE_SYSTEM, source
    if style == "user_only":
        return GENERIC_SYSTEM, CHINESE_USER.format(source)
    if style == "tibetan_instruction":
        return TIBETAN_SYSTEM, TIBETAN_USER.format(source)
    raise ValueError(style)


def behavior(raw: str, reference: str) -> dict[str, Any]:
    cleaned = strip_thinking(raw)
    cjk = len(CJK_RE.findall(cleaned))
    tibetan = len(TIBETAN_RE.findall(cleaned))
    output_length = len(normalize(cleaned))
    reference_length = len(normalize(reference))
    ratio = output_length / max(1, reference_length)
    if cjk > tibetan:
        label = "chinese_dominant"
    elif tibetan > cjk:
        label = "tibetan_dominant"
    elif cjk or tibetan:
        label = "mixed_equal"
    else:
        label = "neither_script"
    return {
        "has_chinese": cjk > 0,
        "has_tibetan": tibetan > 0,
        "chinese_characters": cjk,
        "tibetan_characters": tibetan,
        "script_label": label,
        "output_length": output_length,
        "reference_length": reference_length,
        "length_ratio": ratio,
        "abnormal_length": ratio < 0.25 or ratio > 4.0,
        "empty": output_length == 0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bleu = BLEU(tokenize="flores200")
    chrf = CHRF(word_order=2)
    predictions = [row["prediction"] for row in rows]
    references = [row["reference"] for row in rows]
    metrics = [row["behavior"] for row in rows]
    labels = {label: sum(row["script_label"] == label for row in metrics) / len(metrics)
              for label in ("chinese_dominant", "tibetan_dominant", "mixed_equal", "neither_script")}
    return {
        "items": len(rows),
        "spbleu": bleu.corpus_score(predictions, [references]).score,
        "chrfpp": chrf.corpus_score(predictions, [references]).score,
        "chinese_output_rate": sum(row["has_chinese"] for row in metrics) / len(metrics),
        "tibetan_intrusion_rate": sum(row["has_tibetan"] for row in metrics) / len(metrics),
        "abnormal_length_rate": sum(row["abnormal_length"] for row in metrics) / len(metrics),
        "empty_rate": sum(row["empty"] for row in metrics) / len(metrics),
        "mean_length_ratio": sum(row["length_ratio"] for row in metrics) / len(metrics),
        "script_distribution": labels,
    }


def main() -> None:
    args = parse_args()
    data = json.loads(args.mt_file.read_text(encoding="utf-8"))[: args.samples]
    if len(data) != args.samples:
        raise RuntimeError(f"Requested {args.samples} items, got {len(data)}")
    out = args.output_root / args.name
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", {
        "model": args.model, "name": args.name, "mt_file": str(args.mt_file),
        "samples": args.samples, "prompt_styles": PROMPT_STYLES,
    })
    llm = LLM(
        model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True,
        dtype="auto", attention_config={"backend": "TRITON_ATTN"},
    )
    tokenizer = llm.get_tokenizer()
    eos = tokenizer.eos_token_id
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=256,
        stop_token_ids=[int(eos)] if eos is not None else None,
    )
    summary = {"model": args.model, "name": args.name, "items": len(data), "styles": {}}
    for style in PROMPT_STYLES:
        started = time.time()
        prompts = [render(tokenizer, *prompt(style, row["bo"])) for row in data]
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        raw = [value.outputs[0].text if value.outputs else "" for value in outputs]
        rows = []
        for index, (item, text) in enumerate(zip(data, raw, strict=True)):
            rows.append({
                "item_id": index, "source": item["bo"], "reference": item["zh"],
                "raw": text, "prediction": clean_chinese(text),
                "behavior": behavior(text, item["zh"]),
            })
        write_jsonl(out / f"{style}_predictions.jsonl", rows)
        summary["styles"][style] = summarize(rows) | {"elapsed_seconds": time.time() - started}
        write_json(out / "summary.json", summary)
    (out / "eval_complete").write_text("complete\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
