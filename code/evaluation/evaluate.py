#!/usr/bin/env python3
"""Unified deterministic evaluation on the frozen study inputs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

from sacrebleu.metrics import BLEU, CHRF

# Greedy decoding does not need the optional FlashInfer top-k/top-p sampler.
# Disable it per evaluation process because this node's optional cubin package
# is not version-matched with the installed FlashInfer Python package.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
from vllm import LLM, SamplingParams


TIBETAN_RE = re.compile(r"[\u0F00-\u0FFF]")
CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
CHOICE_MAP = {"ཀ": "A", "ཁ": "B", "ག": "C", "ང": "D"}
TSHEG = "\u0F0B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tlue-test-dir", type=Path, required=True)
    parser.add_argument("--tlue-demo-dir", type=Path, required=True)
    parser.add_argument("--gsm-file", type=Path, required=True)
    parser.add_argument("--bele-file", type=Path, required=True)
    parser.add_argument("--mt-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gsm-samples", type=int, default=500)
    parser.add_argument("--mt-samples", type=int, default=3000)
    parser.add_argument(
        "--mt-prompt-style",
        choices=["source-only", "explicit-user"],
        default="explicit-user",
    )
    parser.add_argument("--tasks", nargs="+", default=["tlue", "gsm", "bele", "mt"],
                        choices=["tlue", "gsm", "bele", "mt"])
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
    # Keep stopping behavior tied to the tokenizer actually used for rendering.
    eos_id = llm.get_tokenizer().eos_token_id
    params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        stop_token_ids=[int(eos_id)] if eos_id is not None else None,
    )
    outputs = llm.generate(prompts, params, use_tqdm=True)
    return [output.outputs[0].text if output.outputs else "" for output in outputs]


def extract_choice(text: str) -> str | None:
    cleaned = strip_thinking(text).upper()
    for tibetan, latin in CHOICE_MAP.items():
        if tibetan in cleaned:
            return latin
    matches = re.findall(r"(?<![A-Z])[ABCD](?![A-Z])", cleaned)
    return matches[-1] if matches else None


def load_tlue(test_dir: Path, demo_dir: Path, shots: int = 2) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for test_path in sorted(test_dir.glob("*.jsonl")):
        demos_path = demo_dir / test_path.name
        demos = read_jsonl(demos_path)[:shots] if demos_path.exists() else []
        for item in read_jsonl(test_path):
            # This frozen item has no question or options and was excluded from
            # the accepted 669-item score after the earlier E0/E1 evaluations.
            if item.get("loc") == "global_facts2":
                if str(item.get("polished_ti_content", "")).strip():
                    raise RuntimeError("The frozen Ti-MMLU exclusion is no longer empty")
                continue
            rows.append({"subject": test_path.stem, "item": item, "demos": demos})
    return rows


def tlue_prompt(row: dict[str, Any]) -> str:
    parts = []
    for demo in row["demos"]:
        parts.append(f"示例题：\n{demo['polished_ti_content']}\n答案：{demo['answer']}")
    parts.append(f"请回答：\n{row['item']['polished_ti_content']}\n答案：")
    return "\n\n".join(parts)


def evaluate_tlue(llm: LLM, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    rows = load_tlue(args.tlue_test_dir, args.tlue_demo_dir, shots=2)
    system = "请根据藏文问题和选项作答，只输出正确选项的字母 A、B、C 或 D。"
    prompts = [render_chat(tokenizer, system, tlue_prompt(row)) for row in rows]
    raw_outputs = generate(llm, prompts, max_tokens=16)
    results = []
    for index, (row, prompt, raw) in enumerate(zip(rows, prompts, raw_outputs)):
        pred = extract_choice(raw)
        gold = str(row["item"]["answer"]).upper()
        results.append({
            "item_id": row["item"].get("loc", f"{row['subject']}:{index}"),
            "subject": row["subject"], "gold": gold, "pred": pred,
            "correct": pred == gold, "raw_output": raw, "prompt": prompt,
        })
    write_jsonl(out_dir / "tlue_predictions.jsonl", results)
    return {
        "items": len(results),
        "correct": sum(row["correct"] for row in results),
        "accuracy": sum(row["correct"] for row in results) / len(results),
        "valid_answer_rate": sum(row["pred"] is not None for row in results) / len(results),
        "shots": 2,
    }


def normalize_number(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.replace(",", "").strip().rstrip(".")
    try:
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".12g")
    except ValueError:
        return value


def extract_number(text: str) -> str | None:
    cleaned = strip_thinking(text)
    match = re.search(r"\{\s*answer\s*[:：]\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", cleaned, re.I)
    if match:
        return normalize_number(match.group(1))
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", cleaned)
    return normalize_number(numbers[-1]) if numbers else None


def gsm_prompt(sample: dict[str, Any], demos: list[dict[str, Any]]) -> str:
    parts = ["གཤམ་གྱི་རྩིས་གཞི་ལ་རིམ་པ་བཞིན་དཔྱད་ནས། མཐའ་མའི་ལན་དེ་{answer:ཨང་གྲངས} རྣམ་པར་བྲིས།"]
    for demo in demos:
        parts.append(
            f"དྲི་བ། {demo['question_bo']}\nལན། {demo['answer_bo']}\n"
            f"{{answer:{demo['answer_only']}}}"
        )
    parts.append(f"དྲི་བ། {sample['question_bo']}\nལན།")
    return "\n\n".join(parts)


def evaluate_gsm(llm: LLM, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    dataset = json.loads(args.gsm_file.read_text(encoding="utf-8"))
    if isinstance(dataset, dict) and dataset.get("selection") == "frozen-study-subset":
        if args.gsm_samples != len(dataset["test"]):
            raise ValueError("Use the complete frozen GSM8K-bo subset for study reproduction")
        demo_pairs = [(row["item_id"], row) for row in dataset["demos"]]
        sample_pairs = [(row["item_id"], row) for row in dataset["test"]]
    else:
        train = [(index, row) for index, row in enumerate(dataset) if row.get("split") == "train"]
        test = [(index, row) for index, row in enumerate(dataset) if row.get("split") == "test"]
        rng = random.Random(args.seed)
        demo_pairs = rng.sample(train, 2)
        sample_pairs = rng.sample(test, args.gsm_samples)
    demos = [row for _, row in demo_pairs]
    samples = [row for _, row in sample_pairs]
    system = "请完成藏文数学应用题，并按要求给出最终数字答案。"
    prompts = [render_chat(tokenizer, system, gsm_prompt(sample, demos)) for sample in samples]
    raw_outputs = generate(llm, prompts, max_tokens=1024)
    results = []
    for (dataset_index, sample), prompt, raw in zip(sample_pairs, prompts, raw_outputs):
        pred = extract_number(raw)
        gold = normalize_number(str(sample["answer_only"]))
        results.append({
            "item_id": dataset_index, "question": sample["question_bo"], "gold": gold,
            "pred": pred, "correct": pred == gold, "raw_output": raw, "prompt": prompt,
        })
    write_jsonl(out_dir / "gsm_predictions.jsonl", results)
    return {
        "items": len(results),
        "correct": sum(row["correct"] for row in results),
        "accuracy": sum(row["correct"] for row in results) / len(results),
        "valid_answer_rate": sum(row["pred"] is not None for row in results) / len(results),
        "shots": 2,
        "seed": args.seed,
        "demo_item_ids": [index for index, _ in demo_pairs],
    }


def bele_prompt(item: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {item[f'mc_answer{index}']}"
        for index, letter in enumerate("ABCD", start=1)
    )
    return f"文章：\n{item['flores_passage']}\n\n问题：\n{item['question']}\n\n选项：\n{options}\n\n答案："


def evaluate_bele(llm: LLM, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    samples = read_jsonl(args.bele_file)
    system = "请阅读藏文文章并回答选择题，只输出 {answer: X}，其中 X 为 A、B、C 或 D。"
    prompts = [render_chat(tokenizer, system, bele_prompt(sample)) for sample in samples]
    raw_outputs = generate(llm, prompts, max_tokens=16)
    results = []
    for index, (sample, prompt, raw) in enumerate(zip(samples, prompts, raw_outputs)):
        pred = extract_choice(raw)
        answer_number = int(sample["correct_answer_num"])
        if answer_number not in range(1, 5):
            raise ValueError(f"Invalid Belebele answer number at item {index}: {answer_number}")
        gold = "ABCD"[answer_number - 1]
        results.append({
            "item_id": index, "question_number": sample.get("question_number"),
            "gold": gold, "pred": pred, "correct": pred == gold,
            "raw_output": raw, "prompt": prompt,
        })
    write_jsonl(out_dir / "bele_predictions.jsonl", results)
    return {
        "items": len(results),
        "correct": sum(row["correct"] for row in results),
        "accuracy": sum(row["correct"] for row in results) / len(results),
        "valid_answer_rate": sum(row["pred"] is not None for row in results) / len(results),
        "shots": 0,
    }


def clean_tibetan(text: str) -> str:
    cleaned = strip_thinking(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    selected = [line for line in lines if TIBETAN_RE.search(line)]
    return " ".join(selected or lines).strip()


def clean_chinese(text: str) -> str:
    cleaned = strip_thinking(text)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    selected = [line for line in lines if CJK_RE.search(line)]
    return " ".join(selected or lines).strip()


def tibetan_syllable_tokenize(text: str) -> str:
    pieces = []
    for whitespace_piece in strip_thinking(text).split():
        pieces.extend(piece for piece in whitespace_piece.split(TSHEG) if piece)
    return " ".join(pieces)


def evaluate_mt(llm: LLM, tokenizer: Any, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    dataset = json.loads(args.mt_file.read_text(encoding="utf-8"))[: args.mt_samples]
    zh_system = "你是中藏翻译器。把中文翻译成藏文，只输出藏文译文。"
    bo_system = "你是中藏翻译器。把藏文翻译成中文，只输出中文译文。"
    if args.mt_prompt_style == "explicit-user":
        zh_users = [f"请将以下中文翻译成藏文，只输出藏文译文：\n{row['zh']}" for row in dataset]
        bo_users = [f"请将以下藏文翻译成中文，只输出中文译文：\n{row['bo']}" for row in dataset]
    else:
        zh_users = [row["zh"] for row in dataset]
        bo_users = [row["bo"] for row in dataset]
    zh_prompts = [render_chat(tokenizer, zh_system, user) for user in zh_users]
    bo_prompts = [render_chat(tokenizer, bo_system, user) for user in bo_users]
    zh_raw = generate(llm, zh_prompts, max_tokens=256)
    bo_raw = generate(llm, bo_prompts, max_tokens=256)
    zh2bo = [clean_tibetan(text) for text in zh_raw]
    bo2zh = [clean_chinese(text) for text in bo_raw]
    refs_bo = [row["bo"] for row in dataset]
    refs_zh = [row["zh"] for row in dataset]
    bleu = BLEU(tokenize="flores200")
    syllable_bleu = BLEU(tokenize="none")
    chrf = CHRF(word_order=2)
    results = []
    for index, row in enumerate(dataset):
        results.append({
            "item_id": index, "zh": row["zh"], "bo": row["bo"],
            "zh2bo_raw": zh_raw[index], "zh2bo_pred": zh2bo[index],
            "bo2zh_raw": bo_raw[index], "bo2zh_pred": bo2zh[index],
        })
    write_jsonl(out_dir / "mt_predictions.jsonl", results)
    return {
        "items": len(dataset), "shots": 0, "prompt_style": args.mt_prompt_style,
        "selection": f"head:{len(dataset)}",
        "zh2bo": {
            "spbleu": bleu.corpus_score(zh2bo, [refs_bo]).score,
            "tibetan_syllable_bleu": syllable_bleu.corpus_score(
                [tibetan_syllable_tokenize(text) for text in zh2bo],
                [[tibetan_syllable_tokenize(text) for text in refs_bo]],
            ).score,
            "chrfpp": chrf.corpus_score(zh2bo, [refs_bo]).score,
            "chrf": CHRF(word_order=0).corpus_score(zh2bo, [refs_bo]).score,
            "tibetan_output_rate": sum(bool(TIBETAN_RE.search(text)) for text in zh2bo) / len(zh2bo),
        },
        "bo2zh": {
            "spbleu": bleu.corpus_score(bo2zh, [refs_zh]).score,
            "chrfpp": chrf.corpus_score(bo2zh, [refs_zh]).score,
            "chrf": CHRF(word_order=0).corpus_score(bo2zh, [refs_zh]).score,
            "chinese_output_rate": sum(bool(CJK_RE.search(text)) for text in bo2zh) / len(bo2zh),
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "run_config.json", vars(args) | {"output_dir": str(args.output_dir), "tlue_test_dir": str(args.tlue_test_dir), "tlue_demo_dir": str(args.tlue_demo_dir), "gsm_file": str(args.gsm_file), "bele_file": str(args.bele_file), "mt_file": str(args.mt_file)})
    started = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
        # Select the installed Triton implementation explicitly; this avoids
        # importing a mismatched optional FlashInfer package on the GPU node.
        attention_config={"backend": "TRITON_ATTN"},
    )
    tokenizer = llm.get_tokenizer()
    summary: dict[str, Any] = {"model": args.model, "model_name": args.model_name, "tasks": {}}
    evaluators = {
        "tlue": evaluate_tlue,
        "gsm": evaluate_gsm,
        "bele": evaluate_bele,
        "mt": evaluate_mt,
    }
    for task in args.tasks:
        task_started = time.time()
        summary["tasks"][task] = evaluators[task](llm, tokenizer, args, out_dir)
        summary["tasks"][task]["elapsed_seconds"] = time.time() - task_started
        write_json(out_dir / "summary.json", summary)
    summary["elapsed_seconds"] = time.time() - started
    write_json(out_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
