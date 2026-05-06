"""
PromptSource-Select metadata used by the released Geofield-LoRA experiments.

Each entry specifies:
  - hf_name:    HuggingFace dataset name
  - hf_config:  dataset configuration (or None)
  - task_type:  classification | generation
  - metric:     evaluation metric
  - max_samples: max training samples to use (for balance across datasets)
  - text_cols:  column names to use as task input (dataset-specific)
  - label_col:  column name for the target
"""

from typing import Optional

PROMPTSOURCE_SELECT = [
    # ── Classification: Sentiment ───────────────────────────────────────────
    {
        "id": "sst2",
        "hf_name": "glue",
        "hf_config": "sst2",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["sentence"],
        "label_col": "label",
        "label_map": {0: "negative", 1: "positive"},
    },
    {
        "id": "imdb",
        "hf_name": "imdb",
        "hf_config": None,
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["text"],
        "label_col": "label",
        "label_map": {0: "negative", 1: "positive"},
        "eval_split": "test",
        "test_split": "test",
    },
    # ── Classification: NLI ────────────────────────────────────────────────
    {
        "id": "rte",
        "hf_name": "super_glue",
        "hf_config": "rte",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["premise", "hypothesis"],
        "label_col": "label",
        "label_map": {0: "entailment", 1: "not_entailment"},
    },
    {
        "id": "cb",
        "hf_name": "super_glue",
        "hf_config": "cb",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["premise", "hypothesis"],
        "label_col": "label",
        "label_map": {0: "entailment", 1: "contradiction", 2: "neutral"},
    },
    {
        "id": "snli",
        "hf_name": "snli",
        "hf_config": None,
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["premise", "hypothesis"],
        "label_col": "label",
        "label_map": {0: "entailment", 2: "neutral", 1: "contradiction"},
    },
    {
        "id": "mnli",
        "hf_name": "multi_nli",
        "hf_config": None,
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["premise", "hypothesis"],
        "label_col": "label",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "eval_split": "validation_matched",
    },
    # ── Classification: Paraphrase ─────────────────────────────────────────
    {
        "id": "mrpc",
        "hf_name": "glue",
        "hf_config": "mrpc",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["sentence1", "sentence2"],
        "label_col": "label",
        "label_map": {0: "not_equivalent", 1: "equivalent"},
    },
    {
        "id": "qqp",
        "hf_name": "glue",
        "hf_config": "qqp",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["question1", "question2"],
        "label_col": "label",
        "label_map": {0: "not_duplicate", 1: "duplicate"},
    },
    # ── Classification: QA / Boolean ──────────────────────────────────────
    {
        "id": "boolq",
        "hf_name": "super_glue",
        "hf_config": "boolq",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["passage", "question"],
        "label_col": "label",
        "label_map": {0: "false", 1: "true"},
    },
    {
        "id": "qnli",
        "hf_name": "glue",
        "hf_config": "qnli",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["question", "sentence"],
        "label_col": "label",
        "label_map": {0: "entailment", 1: "not_entailment"},
    },
    # ── Classification: Commonsense ────────────────────────────────────────
    {
        "id": "copa",
        "hf_name": "super_glue",
        "hf_config": "copa",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["premise", "choice1", "choice2", "question"],
        "label_col": "label",
        "label_map": {0: "choice1", 1: "choice2"},
    },
    {
        "id": "wic",
        "hf_name": "super_glue",
        "hf_config": "wic",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["word", "sentence1", "sentence2"],
        "label_col": "label",
        "label_map": {0: "false", 1: "true"},
    },
    {
        "id": "wsc",
        "hf_name": "super_glue",
        "hf_config": "wsc.fixed",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["text", "span1_text", "span2_text"],
        "label_col": "label",
        "label_map": {0: "false", 1: "true"},
    },
    {
        "id": "winogrande",
        "hf_name": "winogrande",
        "hf_config": "winogrande_xl",
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["sentence", "option1", "option2"],
        "label_col": "answer",
        "label_map": {"1": "option1", "2": "option2"},
    },
    # ── Generation: Summarization ─────────────────────────────────────────
    {
        "id": "xsum",
        "hf_name": "xsum",
        "hf_config": None,
        "task_type": "generation",
        "metric": "rouge",
        "max_samples": 2000,
        "text_cols": ["document"],
        "label_col": "summary",
        "label_map": None,
    },
    # ── NLI: ANLI R1 (harder NLI) ────────────────────────────────────────
    {
        "id": "anli_r1",
        "hf_name": "anli",
        "hf_config": None,
        "task_type": "classification",
        "metric": "accuracy",
        "max_samples": 2000,
        "text_cols": ["premise", "hypothesis"],
        "label_col": "label",
        "label_map": {0: "entailment", 1: "neutral", 2: "contradiction"},
        "train_split": "train_r1",
        "eval_split": "dev_r1",
        "test_split": "test_r1",
    },
]

# ── Template splits ────────────────────────────────────────────────────────────

def get_template_splits(
    templates: list,
    n_train: int = 3,
    n_held_out: int = 2,
) -> tuple[list, list]:
    """
    Split a list of prompt templates into train and held-out sets.
    Takes the first n_train for training, last n_held_out for evaluation.
    Ensures no overlap.

    Args:
        templates:  list of template objects (from promptsource)
        n_train:    number of training templates
        n_held_out: number of held-out templates

    Returns:
        (train_templates, held_out_templates)
    """
    if len(templates) < n_train + n_held_out:
        # Fallback: use all for training, none held out
        return templates, []

    train_templates = templates[:n_train]
    held_out_templates = templates[-(n_held_out):]
    return train_templates, held_out_templates


DATASET_ID_MAP = {d["id"]: d for d in PROMPTSOURCE_SELECT}
