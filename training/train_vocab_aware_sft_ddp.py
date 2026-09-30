#!/usr/bin/env python3
"""Train vocabulary-aware direct supervised fine-tuning.

The model starts from the original Qwen checkpoint, adopts the tokenizer used by
the corresponding final Tibetan model, and updates only:
  1. LoRA parameters on the seven attention/MLP projections;
  2. input-embedding rows for tokens absent from the original tokenizer; and
  3. the matching rows of the untied LM head.

PEFT's TrainableTokensWrapper is used for both vocabulary matrices, so old rows
remain frozen and the selective rows are saved together with the LoRA adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


@dataclass
class RunManifest:
    run_name: str
    base_model: str
    extended_tokenizer: str
    data_path: str
    data_sha256: str
    original_tokenizer_size: int
    extended_tokenizer_size: int
    model_vocab_before_resize: int
    new_token_count: int
    new_token_id_min: int
    new_token_id_max: int
    embedding_module: str
    lm_head_module: str
    max_length: int
    epochs: float
    learning_rate: float
    per_device_batch_size: int
    gradient_accumulation_steps: int
    effective_batch_size: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    seed: int
    dataset_rows_raw: int = 0
    dataset_rows_used: int = 0
    supervised_tokens: int = 0
    trainable_parameters: int = 0
    total_parameters: int = 0
    trainable_parameter_groups: dict[str, int] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--extended-tokenizer", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-proc", type=int, default=12)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--skip-merged-save", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def module_name(model: torch.nn.Module, target: torch.nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target:
            return name
    raise RuntimeError(f"Could not resolve module name for {target!r}")


@torch.no_grad()
def row_mean(weight: torch.Tensor, row_ids: list[int], chunk_size: int = 4096) -> torch.Tensor:
    accumulator = torch.zeros(weight.shape[1], dtype=torch.float32, device=weight.device)
    for start in range(0, len(row_ids), chunk_size):
        index = torch.tensor(row_ids[start : start + chunk_size], dtype=torch.long, device=weight.device)
        accumulator.add_(weight.index_select(0, index).float().sum(dim=0))
    return (accumulator / len(row_ids)).to(dtype=weight.dtype)


@torch.no_grad()
def initialize_new_rows(
    model: torch.nn.Module,
    original_ids: list[int],
    new_ids: list[int],
) -> None:
    """Use deterministic means instead of architecture-dependent random rows."""
    index = torch.tensor(new_ids, dtype=torch.long, device=model.get_input_embeddings().weight.device)
    embedding = model.get_input_embeddings().weight
    embedding.index_copy_(0, index, row_mean(embedding, original_ids).expand(len(new_ids), -1))

    lm_head = model.get_output_embeddings().weight
    head_index = index.to(lm_head.device)
    lm_head.index_copy_(0, head_index, row_mean(lm_head, original_ids).expand(len(new_ids), -1))


def infer_sequence_lengths(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """LLaMA-Factory-style balanced truncation that preserves target supervision."""
    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * target_len / (source_len + target_len))
    new_target_len = min(max_target_len, target_len)
    new_source_len = min(cutoff_len - new_target_len, source_len)
    return new_source_len, new_target_len


class ConversationEncoder:
    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: dict[str, list[str]]) -> dict[str, list[Any]]:
        output: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "supervised_tokens": [],
        }
        for instruction, query, response in zip(
            batch["instruction"], batch["input"], batch["output"], strict=True
        ):
            instruction = instruction or ""
            query = query or ""
            response = response or ""
            user_content = instruction + query

            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = prompt_messages + [{"role": "assistant", "content": response}]
            template_kwargs = {"enable_thinking": False}
            source_encoding = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                **template_kwargs,
            )
            full_encoding = self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
                **template_kwargs,
            )
            # transformers>=5 returns BatchEncoding here, while older releases
            # returned the input-id list directly.
            source_ids = (
                source_encoding
                if isinstance(source_encoding, list)
                else source_encoding["input_ids"]
            )
            full_ids = (
                full_encoding
                if isinstance(full_encoding, list)
                else full_encoding["input_ids"]
            )
            if full_ids[: len(source_ids)] != source_ids:
                raise RuntimeError("Full conversation does not begin with the generation prompt")
            target_ids = full_ids[len(source_ids) :]
            source_len, target_len = infer_sequence_lengths(
                len(source_ids), len(target_ids), self.max_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            input_ids = source_ids + target_ids
            labels = [-100] * len(source_ids) + target_ids

            output["input_ids"].append(input_ids)
            output["attention_mask"].append([1] * len(input_ids))
            output["labels"].append(labels)
            output["supervised_tokens"].append(len(target_ids))
        return output


class CausalLMCollator:
    def __init__(self, tokenizer: Any, pad_to_multiple_of: int = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        input_ids, attention_mask, labels = [], [], []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            input_ids.append(feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len)
            attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def trainable_breakdown(model: torch.nn.Module) -> dict[str, int]:
    groups = {"lora": 0, "input_token_rows": 0, "lm_head_token_rows": 0, "other": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        count = parameter.numel()
        if "lora_" in name:
            groups["lora"] += count
        elif "trainable_tokens_delta" in name and "lm_head" in name:
            groups["lm_head_token_rows"] += count
        elif "trainable_tokens_delta" in name:
            groups["input_token_rows"] += count
        else:
            groups["other"] += count
    return groups


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    is_primary = rank == 0
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    original_tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.extended_tokenizer, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    original_vocab = original_tokenizer.get_vocab()
    extended_vocab = tokenizer.get_vocab()
    remapped = {
        token: (token_id, extended_vocab[token])
        for token, token_id in original_vocab.items()
        if token in extended_vocab and extended_vocab[token] != token_id
    }
    if remapped:
        sample = list(remapped.items())[:5]
        raise RuntimeError(f"Tokenizer remaps original token IDs: {sample}")
    new_ids = sorted(token_id for token, token_id in extended_vocab.items() if token not in original_vocab)
    if not new_ids:
        raise RuntimeError("No vocabulary-extension tokens were found")
    if len(set(new_ids)) != len(new_ids):
        raise RuntimeError("New token IDs are not unique")
    original_ids = sorted(set(original_vocab.values()))

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model_vocab_before_resize = model.get_input_embeddings().weight.shape[0]
    if max(new_ids) >= len(tokenizer):
        raise RuntimeError("A new token ID lies outside the tokenizer length")
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    initialize_new_rows(model, original_ids, new_ids)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.use_cache = False

    embedding_name = module_name(model, model.get_input_embeddings())
    lm_head_name = module_name(model, model.get_output_embeddings())
    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        bias="none",
        trainable_token_indices={
            embedding_name: new_ids,
            lm_head_name: new_ids,
        },
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    groups = trainable_breakdown(model)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if groups["other"] != 0 or groups["input_token_rows"] == 0 or groups["lm_head_token_rows"] == 0:
        raise RuntimeError(f"Unexpected trainable parameter partition: {groups}")

    manifest = RunManifest(
        run_name=args.run_name,
        base_model=str(Path(args.base_model).resolve()),
        extended_tokenizer=str(Path(args.extended_tokenizer).resolve()),
        data_path=str(Path(args.data_path).resolve()),
        data_sha256=sha256_file(args.data_path),
        original_tokenizer_size=len(original_tokenizer),
        extended_tokenizer_size=len(tokenizer),
        model_vocab_before_resize=model_vocab_before_resize,
        new_token_count=len(new_ids),
        new_token_id_min=min(new_ids),
        new_token_id_max=max(new_ids),
        embedding_module=embedding_name,
        lm_head_module=lm_head_name,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        effective_batch_size=args.per_device_batch_size * args.gradient_accumulation_steps * world_size,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=TARGET_MODULES,
        seed=args.seed,
        trainable_parameters=trainable,
        total_parameters=total,
        trainable_parameter_groups=groups,
    )
    if is_primary:
        save_json(output_dir / "run_manifest.pretrain.json", asdict(manifest))
        print(json.dumps(asdict(manifest), ensure_ascii=False, indent=2), flush=True)

    dataset = Dataset.from_json(args.data_path)
    manifest.dataset_rows_raw = len(dataset)
    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    cache_suffix = f".rank{rank}" if world_size > 1 else ""
    cache_file = cache_dir / f"{args.run_name}{cache_suffix}.arrow"
    dataset = dataset.map(
        ConversationEncoder(tokenizer, args.max_length),
        batched=True,
        num_proc=max(1, args.num_proc),
        remove_columns=dataset.column_names,
        cache_file_name=str(cache_file),
        desc=f"Tokenizing {args.run_name}",
    )
    dataset = dataset.filter(
        lambda supervised_tokens: supervised_tokens > 0,
        input_columns=["supervised_tokens"],
        num_proc=max(1, args.num_proc),
        desc="Dropping empty responses",
    )
    manifest.dataset_rows_used = len(dataset)
    manifest.supervised_tokens = int(sum(dataset["supervised_tokens"]))
    dataset = dataset.remove_columns(["supervised_tokens"])
    if is_primary:
        save_json(output_dir / "run_manifest.json", asdict(manifest))

    steps_per_epoch = math.ceil(
        len(dataset) / (args.per_device_batch_size * args.gradient_accumulation_steps * world_size)
    )
    planned_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.epochs * steps_per_epoch)
    warmup_steps = math.ceil(args.warmup_ratio * planned_steps)
    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        optim="adamw_torch_fused",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CausalLMCollator(tokenizer),
    )

    old_embedding_sample = None
    old_head_sample = None
    if args.smoke_test:
        sample_ids = torch.tensor(original_ids[:32], dtype=torch.long)
        old_embedding_sample = model.get_base_model().get_input_embeddings().weight.detach().cpu()[sample_ids].clone()
        old_head_sample = model.get_base_model().get_output_embeddings().weight.detach().cpu()[sample_ids].clone()

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    if trainer.is_world_process_zero():
        model.save_pretrained(output_dir / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output_dir / "adapter")

    if args.smoke_test and trainer.is_world_process_zero():
        embedding_after = model.get_base_model().get_input_embeddings().weight.detach().cpu()[sample_ids]
        head_after = model.get_base_model().get_output_embeddings().weight.detach().cpu()[sample_ids]
        smoke = {
            "loss_finite": bool(torch.isfinite(torch.tensor(train_result.training_loss))),
            "old_embedding_rows_exactly_unchanged": bool(torch.equal(old_embedding_sample, embedding_after)),
            "old_lm_head_rows_exactly_unchanged": bool(torch.equal(old_head_sample, head_after)),
            "training_loss": train_result.training_loss,
        }
        save_json(output_dir / "smoke_acceptance.json", smoke)
        if not all(value for key, value in smoke.items() if key != "training_loss"):
            raise RuntimeError(f"Smoke-test invariants failed: {smoke}")

    if not args.skip_merged_save and trainer.is_world_process_zero():
        merged = model.merge_and_unload(safe_merge=True)
        merged.config.use_cache = True
        merged_dir = output_dir / "final_merged"
        merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="5GB")
        tokenizer.save_pretrained(merged_dir)
    if trainer.is_world_process_zero():
        print(f"E1 run complete: {args.run_name}", flush=True)


if __name__ == "__main__":
    main()
