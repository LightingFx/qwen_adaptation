# Anonymous review code

This supplement contains the vocabulary-aware supervised fine-tuning (VA-SFT),
evaluation, statistical analysis, manual-assessment workflow, and numerical
figure code used in the study. It contains no model weights, predictions,
completed annotations, training corpora, credentials, or author metadata.
The two-stage route's CPT driver is not included in this code attachment.

## Layout and environment

Unzip `code.zip` and `data.zip` into the same directory. Run the commands below
from `code/`; the evaluation inputs are then in `../data/`.
Python 3.10 or newer is required. The training/evaluation dependency files
record the currently inspected reference environment, not a guarantee of every
historical run's environment. Install a compatible CUDA/PyTorch stack on GPU
hardware. Training and vLLM may be installed in separate virtual environments.
The analysis dependencies are independent of GPU inference.

```bash
python3 verify_package.py --data-dir ../data
python3 -m pip install -r requirements-analysis.txt
# GPU inference environment:
python3 -m pip install -r requirements-evaluation.txt
# GPU training environment, when required:
python3 -m pip install -r requirements-training.txt
```

`verify_package.py` needs only the Python standard library. It checks input
counts, IDs, fields, and Python syntax without training or GPU inference.
The archive builder additionally checked frozen IDs and references against
the study's stored predictions. A new end-to-end GPU run was not performed
as part of packaging.

## Evaluate a model

Supply a locally available **merged causal language model with its matching
tokenizer**, or an official released model identifier. Adapted model weights
and the extended tokenizers will be distributed separately. An adapter alone
is not an inference model; do not load it as if it were a merged checkpoint.

```bash
python3 evaluation/evaluate.py \
  --model MODEL_PATH --model-name qwen3-instruct-va-sft-6epoch-seed42 \
  --output-dir results/matched_block \
  --tlue-test-dir ../data/ti_mmlu/test \
  --tlue-demo-dir ../data/ti_mmlu/demos \
  --gsm-file ../data/gsm8k_bo.json \
  --bele-file ../data/belebele_bod.jsonl \
  --mt-file ../data/zh_bo_pairs.json \
  --mt-prompt-style explicit-user
```

The packaged GSM8K-bo file fixes the study's 500 test questions and two
demonstrations, including their original IDs and order. The evaluator does
not resample this file. Ti-MMLU contains 669 items and the actual two-shot
demonstrations per subject. Belebele contains 900 items. The 3,000 translation
pairs are shared by both directions. Chat rendering disables thinking when
supported. Decoding is greedy, with 16 tokens for multiple-choice tasks,
1,024 for mathematics, and 256 for translation.

spBLEU uses SacreBLEU's `flores200` tokenizer, whose first use can download
the public tokenizer model. Character-level chrF (`word_order=0`) is the
paper's complementary metric. The inference summary also retains the
historical `chrfpp` field (`word_order=2`) as an auxiliary metric. These are
not interchangeable. Answer extraction and translation cleanup are retained
from the original evaluation implementation.

Additional prompt conditions are evaluated by:

```bash
python3 evaluation/evaluate_prompt_robustness.py \
  --model MODEL_PATH --name RUN_NAME --mt-file ../data/zh_bo_pairs.json \
  --output-root results/prompt_robustness
```

This script adds source-only, user-only, and Tibetan-instruction conditions.
The explicit-user condition comes from the main evaluator, giving the four
conditions discussed in the paper. Output-script presence is an automatic
indicator, not a complete translation-quality or language-adherence score.

```bash
python3 evaluation/evaluate_retention.py \
  --model MODEL_PATH --model-name qwen3-cpt \
  --data-dir ../data/retention --output-dir results/retention/evaluation
```

Retention uses the frozen English MMLU test and Chinese C-Eval validation
inputs, zero-shot. The unrelated exploratory language-transfer branch has
been omitted from this attachment.

## VA-SFT training

Provide your own licensed instruction corpus as a JSON list. Each item has
`instruction`, `input`, and `output` strings. `configs/sft_schema_example.json`
is a synthetic illustration of the schema, not a sample from the private
training corpus. CPT/SFT training data and tokenized caches are withheld.

```bash
python3 training/train_vocab_aware_sft.py \
  --run-name RUN_NAME --base-model BASE_MODEL_PATH \
  --extended-tokenizer TOKENIZER_PATH --data-path PRIVATE_SFT_JSON \
  --output-dir outputs/RUN_NAME --cache-dir cache/RUN_NAME \
  --epochs 2 --per-device-batch-size 4 --gradient-accumulation-steps 16

torchrun --nproc_per_node=2 training/train_vocab_aware_sft_ddp.py \
  --run-name RUN_NAME --base-model BASE_MODEL_PATH \
  --extended-tokenizer TOKENIZER_PATH --data-path PRIVATE_SFT_JSON \
  --output-dir outputs/RUN_NAME --cache-dir cache/RUN_NAME \
  --epochs 6 --per-device-batch-size 4 --gradient-accumulation-steps 8
```

`configs/study.json` records the model identifiers, horizons, batch settings,
LoRA settings, and evaluation configuration. VA-SFT updates LoRA and added
input-embedding/LM-head rows while retaining the original rows. The CPT route
updates full embedding/head matrices during CPT; its supervised stage uses
the same instruction corpus, LoRA settings, and added-row policy as VA-SFT.
The private corpora and separately supplied model artifacts are required to
recreate adapted endpoints, rather than merely evaluate them.

## Analysis and figures

Experimental predictions and completed manual labels are **not** included in
this attachment. They will be shared separately. Statistical and figure
scripts need those outputs or compatible outputs from your own evaluation.
The main prediction layout is `results/matched_block/RUN_NAME/` with
`tlue_predictions.jsonl`, `gsm_predictions.jsonl`, `bele_predictions.jsonl`,
`mt_predictions.jsonl`, and `summary.json`.

Run names use backbone prefixes `qwen1.5`, `qwen2`, `qwen2.5`, and `qwen3`,
followed by `-instruct-original`, `-instruct-va-sft-reference`,
`-instruct-va-sft-6epoch-seed42`, or `-instruct-cpt-sft-reference`.

```bash
python3 analysis/contrast_bootstrap.py
python3 analysis/chrf_bootstrap.py
python3 analysis/horizon_bootstrap.py
python3 figures/build_figures.py
```

Contrasts preserve the paired bootstrap, sampling seeds, and Holm families
used in the study. Figures additionally require
`results/ipm/bo2zh_manual_check_summary.json`, produced from completed manual
labels by `manual_check/analyze_bo2zh_manual.py`. Only numerical figure code
is included; the separately drawn overview figure is not a code resource.

The repeated-seed analysis operates on the earlier evaluation pass rather
than mixing those outputs with the matched main matrix. Its entry point is
`analysis/analyze_va_multiseed.py --root REPEATED_SEED_ROOT
--seed42-root VA_REFERENCE_ROOT --e0-root CPT_REFERENCE_ROOT`.
`analysis/freeze_ti_mmlu.py` applies the documented 669-item mask to those
historical outputs and recomputes the five-task Holm family; it expects the
relative `results/ipm_inputs/` and `results/e0/` layout used by that analysis.
The path labels `fair` and `e0` in legacy repeated-seed/retention interfaces
are compatibility identifiers for VA-SFT and the reference evaluation, not
additional scientific conditions.

The blinded assessment can be generated with the matched two-stage outputs
and the six-epoch VA-SFT output:

```bash
python3 manual_check/build_bo2zh_sheet.py \
  --extra qwen3_va_sft_6epoch=results/matched_block/qwen3-instruct-va-sft-6epoch-seed42/mt_predictions.jsonl
```

The generated workbook is randomized and stratified as in the study. After
annotation, save it as `manual_check/bo2zh_manual_check_evaluated.xlsx` and
run `python3 manual_check/analyze_bo2zh_manual.py` to summarize the labels.
For retention statistics, use `python3 analysis/analyze_retention.py
--root results/retention`; that directory contains `evaluation/qwen1.5-original`,
`qwen1.5-fair`, `qwen1.5-cpt`, and the corresponding Qwen3 runs.

## Review distribution

This is an anonymous submission attachment, not a new blanket open-source
license grant. Public-release licensing and model access will be specified
separately. Third-party dataset sources and their existing licenses are
listed in the accompanying data attachment. Official upstream model names
and third-party attribution do not identify the manuscript's authors.
