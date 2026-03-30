"""
Build a custom evaluation test set by sampling from existing test CSVs.

Sampling rules:
- 250 random single-table samples from test_finqa.csv
- 20 multi_table_medium + 30 multi_table_hard from multi-table test_finqa.csv
- ALL single-table test negatives (test_finqa_negative.csv)
- ALL multi-table test negatives (multi_table_data/test_finqa_negative.csv)
"""

import json

import pandas as pd

from projects.finqa import constants as C


RANDOM_SEED = 42


def _parse_json_list(value):
    """Decode columns stored as JSON strings, defaulting to [] for empty values."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = stripped
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            cleaned = parsed.strip()
            return [cleaned] if cleaned else []
        return []
    return []


def _load_csv(path) -> pd.DataFrame:
    return pd.read_csv(path)


def _question_type_series(df: pd.DataFrame) -> pd.Series:
    return df.get("question_type", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()


def prepare_eval_data(
    n_single: int = 250,
    n_multi_medium: int = 20,
    n_multi_hard: int = 30,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Build and return the evaluation dataset as a list of dicts."""
    parts = []

    # --- 1. Single-table positive: sample n_single ---
    single_df = _load_csv(C.TEST_QUESTIONS_PATH)
    if len(single_df) > n_single:
        single_df = single_df.sample(n=n_single, random_state=seed)
    single_df = single_df.copy()
    single_df["curriculum_stage"] = "single_table"
    single_df["data_source"] = "single_table"
    parts.append(single_df)

    # --- 2. Multi-table positive: sample n_multi_medium medium + n_multi_hard hard ---
    if C.MULTI_TABLE_TEST_PATH.exists():
        multi_df = _load_csv(C.MULTI_TABLE_TEST_PATH)
        qtype = _question_type_series(multi_df)

        medium_mask = qtype.isin(["multi_table_medium", "multi_table_easy"])
        hard_mask = qtype.eq("multi_table_hard")

        medium_df = multi_df[medium_mask].copy()
        if len(medium_df) > n_multi_medium:
            medium_df = medium_df.sample(n=n_multi_medium, random_state=seed)
        medium_df["curriculum_stage"] = "multi_table_medium"
        medium_df["data_source"] = "multi_table"
        parts.append(medium_df)

        hard_df = multi_df[hard_mask].copy()
        if len(hard_df) > n_multi_hard:
            hard_df = hard_df.sample(n=n_multi_hard, random_state=seed)
        hard_df["curriculum_stage"] = "multi_table_hard"
        hard_df["data_source"] = "multi_table"
        parts.append(hard_df)

    # --- 3. Negative single-table: all ---
    if C.NEGATIVE_SINGLE_TABLE_TEST_PATH.exists():
        neg_single_df = _load_csv(C.NEGATIVE_SINGLE_TABLE_TEST_PATH).copy()
        neg_single_df["data_source"] = "negative_single_table"
        neg_single_df["curriculum_stage"] = "negative_single_table"
        parts.append(neg_single_df)

    # --- 4. Negative multi-table: all ---
    if C.NEGATIVE_MULTI_TABLE_TEST_PATH.exists():
        neg_multi_df = _load_csv(C.NEGATIVE_MULTI_TABLE_TEST_PATH).copy()
        neg_multi_df["data_source"] = "negative_multi_table"
        neg_multi_df["curriculum_stage"] = "negative_multi_table"
        parts.append(neg_multi_df)

    combined_df = pd.concat(parts, axis=0, ignore_index=True)

    # --- Preprocess (same logic as prepare_finqa_data.py) ---
    def preprocess_fn(example):
        source = example.get("data_source") if hasattr(example, "get") else None
        source = source if isinstance(source, str) and source else "single_table"
        raw_id = str(example["id"])

        # Negative samples use 'ground_truth' column; positive samples use 'answer'
        ground_truth = example.get("ground_truth")
        if pd.isna(ground_truth) or ground_truth is None:
            ground_truth = example["answer"]

        return {
            "question": example["user_query"],
            "ground_truth": ground_truth,
            "data_source": source,
            "company": example["company"],
            "question_id": f"{source}_{raw_id}",
            "question_type": example["question_type"],
            "curriculum_stage": example.get("curriculum_stage", "single_table"),
            "core_question": example.get("question", example["user_query"]),
            "table_name": _parse_json_list(example.get("table_name")),
            "columns_used": _parse_json_list(example.get("columns_used_json")),
            "rows_used": _parse_json_list(example.get("rows_used_json")),
            "explanation": example.get("explanation", ""),
        }

    processed = [preprocess_fn(row) for _, row in combined_df.iterrows()]

    # Print summary
    source_counts = {}
    for item in processed:
        src = item["data_source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    print(f"Eval dataset: {len(processed)} total samples")
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt}")

    return processed


if __name__ == "__main__":
    data = prepare_eval_data()
    print(f"\nTotal eval samples: {len(data)}")
