# Evaluation Data

These are the frozen inputs used for the manuscript's evaluation. They are
stored alongside `code/` in this repository. From the repository root, run
`python3 code/verify_package.py --data-dir data` for a standard-library-only
validation. The directory does not contain model predictions or CPT/SFT
training corpora.

| Resource | Included evaluation items | Prompt examples | File |
|---|---:|---|---|
| Ti-MMLU | 669 | Two per subject, supplied separately | `ti_mmlu/test/*.jsonl`, `ti_mmlu/demos/*.jsonl` |
| GSM8K-bo | 500 | Two frozen demonstrations | `gsm8k_bo.json` |
| Belebele, bod_Tibt | 900 | Zero-shot | `belebele_bod.jsonl` |
| Chinese--Tibetan translation | 3,000 aligned pairs, used bidirectionally | Zero-shot | `zh_bo_pairs.json` |
| English MMLU | 14,042 | Zero-shot | `retention/mmlu_test.jsonl` |
| Chinese C-Eval | 1,346 | Zero-shot | `retention/ceval_val.jsonl` |

## Format and selection

JSON and JSONL files use UTF-8. Content, answers, ordering, and study IDs are
retained. Nonessential source metadata is excluded.

- Ti-MMLU: files are grouped by subject. `loc` is the original item ID,
  `polished_ti_content` contains the Tibetan question and choices, and
  `answer` is the gold choice. Test files form the valid-item study subset.
  Demonstration files contain only the examples actually rendered in prompts.
- GSM8K-bo: the JSON object has `demos` and `test` arrays. `item_id` is the
  original source index, not a renumbered index. `question_bo`, `answer_bo`,
  and `answer_only` supply the question, explanation, and numerical answer.
  The 500 test questions and two examples are fixed in their study order.
  Changing the decoding seed does not resample this packaged subset.
- Belebele: passage, question, four choices, one-based gold answer number,
  and public passage/question identifiers are preserved. Item IDs in the
  evaluator are the zero-based row indices, with the original order intact.
- Translation: each list item has `zh` and `bo`. The index is the fixed pair
  ID. Neither direction uses a different set of pairs.
- Retention: each row has `item_id`, `subject`, `question`, four `choices`,
  and an A--D `answer`. C-Eval uses the publicly labeled validation split
  from revision `617524a00b307ff6f9933702f724131fe12ca7ce`.

No results, annotator details, credentials, machine paths, repository history,
model weights, private instruction corpus, or pretraining text are included.
Benchmark prompt demonstrations are evaluation resources, not a release of
the private model-training corpus.

## Sources, attribution, and licensing

Existing upstream licenses apply separately; there is no blanket license for
the entire bundle.
Third-party attribution is retained rather than replaced with anonymous
authorship. The license texts are in `licenses/`.

- Ti-MMLU is from TLUE: **TLUE: A Tibetan Language Understanding Evaluation
  Benchmark**, EMNLP 2025. The source paper's dataset footnote specifies
  CC-BY-NC-SA 4.0. This bundle selects the valid test subset and two examples
  per subject and retains the original Tibetan content.
  [Source repository](https://github.com/Vicentvankor/TLUE),
  [published source and attribution](https://aclanthology.org/2025.emnlp-main.1777/),
  [license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Belebele is from **The Belebele Benchmark: a Parallel Reading Comprehension
  Dataset in 122 Language Variants**, ACL 2024, by Lucas Bandarkar, Davis Liang,
  Benjamin Muller, Mikel Artetxe, and others as listed by the upstream
  publication. The test dataset is CC-BY-SA 4.0. Only the Tibetan variant and
  evaluation fields are included; the assembled training dataset is not.
  [Source and full attribution](https://github.com/facebookresearch/belebele),
  [license](https://creativecommons.org/licenses/by-sa/4.0/).
- GSM8K originates from OpenAI's **Training Verifiers to Solve Math Word
  Problems** and the `grade-school-math` repository (MIT, copyright 2021
  OpenAI). The attachment contains the Tibetan rendering used in this study
  and the fixed evaluation subset. The upstream MIT notice is preserved;
  it does not establish a separate blanket license for the Tibetan rendering.
  [Source](https://github.com/openai/grade-school-math).
- English MMLU originates from Dan Hendrycks and collaborators,
  **Measuring Massive Multitask Language Understanding**, ICLR 2021.
  The repository's MIT notice (copyright 2020 Dan Hendrycks) is preserved.
  [Source](https://github.com/hendrycks/test).
- C-Eval originates from Yuzhen Huang and collaborators,
  **C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for
  Foundation Models**, NeurIPS 2023. The dataset card specifies
  CC-BY-NC-SA 4.0. The validation rows are reformatted without altering
  question, choices, or answers.
  [Source and full attribution](https://huggingface.co/datasets/ceval/ceval-exam),
  [license](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- Chinese--Tibetan aligned evaluation pairs are supplied as the study's
  frozen evaluation resource for peer review. Their public-release license
  and the Tibetan GSM8K rendering's license will be specified separately.
  This attachment does not grant rights to the withheld training corpora.

Source/licensing links were checked on 2026-09-30. Evaluation code and
configuration are available in [`../code/`](../code/).
