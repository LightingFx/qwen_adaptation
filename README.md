# Tibetan Adaptation Across Qwen Generations

Code and evaluation data for **Adaptation Choices Shape the Returns to Backbone
Upgrades: Evidence from Tibetan Across Four Qwen Generations**.

The study compares Qwen1.5, Qwen2, Qwen2.5, and Qwen3 under vocabulary-aware
direct supervised fine-tuning at two training horizons and continual
pretraining followed by supervised fine-tuning.

## Repository structure

```text
code/
  training/       VA-SFT training, single GPU and DDP
  evaluation/     Main tasks, prompt checks, and knowledge retention
  analysis/       Statistical comparisons and output analysis
  manual_check/   Blinded assessment preparation and aggregation
  figures/        Numerical figure generation
  configs/        Study configuration and synthetic input schema example
  README.md       Dependencies and detailed running instructions
data/
  ti_mmlu/        669 test items and the study's few-shot demonstrations
  gsm8k_bo.json   500 frozen test items and two demonstrations
  belebele_bod.jsonl
                  900 Tibetan reading-comprehension items
  zh_bo_pairs.json
                  3,000 aligned Chinese--Tibetan evaluation pairs
  retention/     English MMLU and Chinese C-Eval evaluation inputs
  licenses/      Upstream license texts
  README.md      Dataset schemas, sources, and licensing details
```

## Quick start

Verify the supplied data and code syntax without third-party dependencies:

```bash
python3 code/verify_package.py
```

Run the training, evaluation, and analysis commands in [code/README.md](code/README.md)
from the `code/` directory. Their data paths point to the sibling `data/` directory.

```bash
cd code
python3 -m pip install -r requirements-analysis.txt
```

GPU training and inference have separate dependency files and require the
corresponding model and tokenizer artifacts. Model weights, experimental
predictions, and completed manual labels will be shared separately. The CPT
and SFT training corpora are withheld because of third-party copyright
restrictions. The included training implementation covers VA-SFT; the
historical CPT driver is not included.

## Data and licensing

The included data are evaluation resources with fixed study selections.
See [data/README.md](data/README.md) for per-dataset attribution, licenses,
and selection details. Existing upstream licenses apply individually;
the repository does not apply a blanket license to third-party datasets.
Code licensing will be specified separately.
