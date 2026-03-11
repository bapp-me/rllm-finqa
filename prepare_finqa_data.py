import json
import tarfile

import pandas as pd
from huggingface_hub import hf_hub_download

from projects.finqa import constants as C
from rllm.data.dataset import DatasetRegistry

HF_REPO_ID = "rLLM/finqa"
HF_FILENAME = "data.tar.gz"


def download_data():
    """Download and extract finqa data from HuggingFace if not present."""
    data_dir = C.DATA_DIR

    # Check if data already exists
    if data_dir.exists() and (data_dir / "train_finqa.csv").exists():
        print(f"Data already exists at {data_dir}")
        return

    print(f"Downloading finqa data from {HF_REPO_ID}...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download tar.gz from HuggingFace
    tar_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset")

    # Extract to parent directory (tar contains data/ prefix)
    print(f"Extracting to {data_dir}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir.parent)

    print("Done.")


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _question_type_series(df: pd.DataFrame) -> pd.Series:
    return df.get("question_type", pd.Series([""] * len(df))).fillna("").astype(str).str.lower()


def _build_curriculum_split(single_df: pd.DataFrame, multi_df: pd.DataFrame) -> pd.DataFrame:
    """Construct curriculum order: single-table -> multi-table medium -> multi-table hard."""
    single = single_df.copy()
    single["curriculum_stage"] = "single_table"
    single["data_source"] = "single_table"

    if multi_df.empty:
        return single.reset_index(drop=True)

    multi = multi_df.copy()
    qtype = _question_type_series(multi)

    # Keep only 3 curriculum phases. Rare easy samples are grouped into medium.
    medium_mask = qtype.isin(["multi_table_medium", "multi_table_easy"])
    hard_mask = qtype.eq("multi_table_hard")

    medium = multi[medium_mask].copy()
    medium["curriculum_stage"] = "multi_table_medium"
    medium["data_source"] = "multi_table"

    hard = multi[hard_mask].copy()
    hard["curriculum_stage"] = "multi_table_hard"
    hard["data_source"] = "multi_table"

    # Keep unexpected labels at the end rather than dropping them silently.
    other = multi[~(medium_mask | hard_mask)].copy()
    if not other.empty:
        other["curriculum_stage"] = "multi_table_other"
        other["data_source"] = "multi_table"

    ordered = [single, medium, hard]
    if not other.empty:
        ordered.append(other)
    return pd.concat(ordered, axis=0, ignore_index=True)


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


def prepare_finqa_data():
    single_train_df = _load_csv(C.TRAIN_QUESTIONS_PATH)
    single_val_df = _load_csv(C.VAL_QUESTIONS_PATH)
    single_test_df = _load_csv(C.TEST_QUESTIONS_PATH)

    multi_train_df = _load_csv(C.MULTI_TABLE_TRAIN_PATH) if C.MULTI_TABLE_TRAIN_PATH.exists() else pd.DataFrame()
    multi_val_df = _load_csv(C.MULTI_TABLE_VAL_PATH) if C.MULTI_TABLE_VAL_PATH.exists() else pd.DataFrame()
    multi_test_df = _load_csv(C.MULTI_TABLE_TEST_PATH) if C.MULTI_TABLE_TEST_PATH.exists() else pd.DataFrame()

    train_df = _build_curriculum_split(single_train_df, multi_train_df)
    val_df = _build_curriculum_split(single_val_df, multi_val_df)
    test_df = _build_curriculum_split(single_test_df, multi_test_df)

    print(
        "Curriculum split sizes (single -> medium -> hard): "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    def preprocess_fn(example):
        source = example.get("data_source") if hasattr(example, "get") else None
        source = source if isinstance(source, str) and source else "single_table"
        raw_id = str(example["id"])
        return {
            "question": example["user_query"],
            "ground_truth": example["answer"],
            "data_source": source,
            "company": example["company"],
            "question_id": f"{source}_{raw_id}",
            "question_type": example["question_type"],
            "curriculum_stage": example.get("curriculum_stage", "single_table"),
            "core_question": example["question"],
            "table_name": _parse_json_list(example.get("table_name")),
            "columns_used": _parse_json_list(example.get("columns_used_json")),
            "rows_used": _parse_json_list(example.get("rows_used_json")),
            "explanation": example["explanation"],
        }

    train_processed = [preprocess_fn(row) for _, row in train_df.iterrows()]
    val_processed = [preprocess_fn(row) for _, row in val_df.iterrows()]
    test_processed = [preprocess_fn(row) for _, row in test_df.iterrows()]

    train_dataset = DatasetRegistry.register_dataset("finqa", train_processed, "train")
    val_dataset = DatasetRegistry.register_dataset("finqa", val_processed, "val")
    test_dataset = DatasetRegistry.register_dataset("finqa", test_processed, "test")
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    download_data()
    train_dataset, val_dataset, test_dataset = prepare_finqa_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")
    print(f"Test dataset path: {test_dataset.get_data_path()}")
