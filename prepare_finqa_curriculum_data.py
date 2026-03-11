import argparse
import json

import pandas as pd

from projects.finqa import constants as C
from rllm.data.dataset import DatasetRegistry


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _parse_json_list(value):
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


def _safe_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    replace = len(df) < n
    return df.sample(n=n, replace=replace, random_state=seed).reset_index(drop=True)


def _linear_multi_count(epoch: int, total_epochs: int, max_multi: int) -> int:
    half = total_epochs // 2
    if epoch <= half:
        return 0

    second_half_steps = total_epochs - half
    step_in_second_half = epoch - half
    return int(round(max_multi * (step_in_second_half / second_half_steps)))


def _build_epoch_train(
    single_train: pd.DataFrame,
    multi_train: pd.DataFrame,
    epoch: int,
    total_epochs: int,
    train_total: int,
    max_multi: int,
    seed: int,
) -> pd.DataFrame:
    multi_n = _linear_multi_count(epoch, total_epochs, max_multi)
    multi_n = min(max_multi, max(0, multi_n))
    single_n = train_total - multi_n

    single_part = _safe_sample(single_train, single_n, seed + epoch * 101)
    single_part["curriculum_stage"] = "single_table"
    single_part["data_source"] = "single_table"

    multi_part = _safe_sample(multi_train, multi_n, seed + epoch * 103)
    if not multi_part.empty:
        qtype = multi_part["question_type"].fillna("").astype(str).str.lower()
        multi_part["curriculum_stage"] = qtype.where(
            qtype.isin(["multi_table_medium", "multi_table_hard"]),
            other="multi_table_other",
        )
        multi_part["data_source"] = "multi_table"

    return pd.concat([single_part, multi_part], axis=0, ignore_index=True)


def _build_epoch_val(
    single_val: pd.DataFrame,
    multi_val: pd.DataFrame,
    epoch: int,
    total_epochs: int,
    seed: int,
) -> pd.DataFrame:
    half = total_epochs // 2

    if epoch <= half:
        val_single = _safe_sample(single_val, 256, seed + epoch * 107)
        val_single["curriculum_stage"] = "single_table"
        val_single["data_source"] = "single_table"
        return val_single

    qtype = multi_val["question_type"].fillna("").astype(str).str.lower()
    medium_pool = multi_val[qtype == "multi_table_medium"]
    hard_pool = multi_val[qtype == "multi_table_hard"]

    single_part = _safe_sample(single_val, 200, seed + epoch * 109)
    single_part["curriculum_stage"] = "single_table"
    single_part["data_source"] = "single_table"

    medium_part = _safe_sample(medium_pool, 25, seed + epoch * 113)
    medium_part["curriculum_stage"] = "multi_table_medium"
    medium_part["data_source"] = "multi_table"

    hard_part = _safe_sample(hard_pool, 31, seed + epoch * 127)
    hard_part["curriculum_stage"] = "multi_table_hard"
    hard_part["data_source"] = "multi_table"

    return pd.concat([single_part, medium_part, hard_part], axis=0, ignore_index=True)


def _preprocess_rows(df: pd.DataFrame):
    processed = []
    for _, row in df.iterrows():
        source = row.get("data_source") if hasattr(row, "get") else None
        source = source if isinstance(source, str) and source else "single_table"
        raw_id = str(row["id"])
        processed.append(
            {
                "question": row["user_query"],
                "ground_truth": row["answer"],
                "data_source": source,
                "company": row["company"],
                "question_id": f"{source}_{raw_id}",
                "question_type": row["question_type"],
                "curriculum_stage": row.get("curriculum_stage", "single_table"),
                "core_question": row["question"],
                "table_name": _parse_json_list(row.get("table_name")),
                "columns_used": _parse_json_list(row.get("columns_used_json")),
                "rows_used": _parse_json_list(row.get("rows_used_json")),
                "explanation": row["explanation"],
            }
        )
    return processed


def prepare_epoch_curriculum_data(
    epoch: int,
    total_epochs: int,
    train_total: int = 4000,
    val_total: int = 256,
    max_multi_train: int = 400,
    seed: int = 42,
):
    if val_total != 256:
        raise ValueError("This curriculum implementation expects val_total=256.")

    single_train = _load_csv(C.TRAIN_QUESTIONS_PATH)
    single_val = _load_csv(C.VAL_QUESTIONS_PATH)
    multi_train = _load_csv(C.MULTI_TABLE_TRAIN_PATH)
    multi_val = _load_csv(C.MULTI_TABLE_VAL_PATH)
    test_df = _load_csv(C.TEST_QUESTIONS_PATH)

    train_df = _build_epoch_train(
        single_train=single_train,
        multi_train=multi_train,
        epoch=epoch,
        total_epochs=total_epochs,
        train_total=train_total,
        max_multi=max_multi_train,
        seed=seed,
    )
    val_df = _build_epoch_val(
        single_val=single_val,
        multi_val=multi_val,
        epoch=epoch,
        total_epochs=total_epochs,
        seed=seed,
    )

    test_df = test_df.copy()
    test_df["curriculum_stage"] = "single_table"
    test_df["data_source"] = "single_table"

    train_processed = _preprocess_rows(train_df)
    val_processed = _preprocess_rows(val_df)
    test_processed = _preprocess_rows(test_df)

    train_dataset = DatasetRegistry.register_dataset("finqa", train_processed, "train")
    val_dataset = DatasetRegistry.register_dataset("finqa", val_processed, "val")
    test_dataset = DatasetRegistry.register_dataset("finqa", test_processed, "test")

    train_stage_counts = train_df["curriculum_stage"].value_counts(dropna=False).to_dict()
    val_stage_counts = val_df["curriculum_stage"].value_counts(dropna=False).to_dict()

    print(
        f"[epoch {epoch}/{total_epochs}] train={len(train_df)} {train_stage_counts}; "
        f"val={len(val_df)} {val_stage_counts}"
    )
    print(f"Train dataset path: {train_dataset.get_data_path()}")
    print(f"Validation dataset path: {val_dataset.get_data_path()}")
    print(f"Test dataset path: {test_dataset.get_data_path()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare epoch-specific FinQA curriculum data")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--train-total", type=int, default=4000)
    parser.add_argument("--val-total", type=int, default=256)
    parser.add_argument("--max-multi-train", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.epoch < 1 or args.epoch > args.total_epochs:
        raise ValueError("epoch must be in [1, total-epochs]")

    prepare_epoch_curriculum_data(
        epoch=args.epoch,
        total_epochs=args.total_epochs,
        train_total=args.train_total,
        val_total=args.val_total,
        max_multi_train=args.max_multi_train,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
