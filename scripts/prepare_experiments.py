import os
import pandas as pd

from preprocessing import (
    preprocess_raw,
    preprocess_minimal,
    preprocess_traditional,
    preprocess_albanian_aware
)


# ============================================================
# KONFIGURIMI
# ============================================================

INPUT_FILES = {
    "train": "data/train.xlsx",
    "validation": "data/validation.xlsx",
    "test": "data/test.xlsx"
}

PIPELINES = {
    "raw": preprocess_raw,
    "minimal": preprocess_minimal,
    "traditional": preprocess_traditional,
    "albanian_aware": preprocess_albanian_aware
}

OUTPUT_FOLDER = "data/experiments"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# FUNKSION PËR KONTROLLIN E DATASET-IT
# ============================================================

def validate_dataset(df, split_name):

    required_columns = [
        "comment_id",
        "raw_text",
        "label"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Mungon kolona '{column}' në {split_name}"
            )

    if df["comment_id"].isna().any():
        raise ValueError(
            f"Ka comment_id bosh në {split_name}"
        )

    if df["raw_text"].isna().any():
        raise ValueError(
            f"Ka raw_text bosh në {split_name}"
        )

    if df["label"].isna().any():
        raise ValueError(
            f"Ka label bosh në {split_name}"
        )

    if df["comment_id"].duplicated().any():
        raise ValueError(
            f"Ka comment_id të përsëritura në {split_name}"
        )


# ============================================================
# GJENERIMI I DATASET-EVE
# ============================================================

print("=" * 70)
print("PËRGATITJA E DATASET-EVE EKSPERIMENTALE")
print("=" * 70)


for split_name, input_path in INPUT_FILES.items():

    print("\n" + "=" * 70)
    print(f"SPLIT: {split_name.upper()}")
    print("=" * 70)

    df = pd.read_excel(input_path)

    validate_dataset(
        df,
        split_name
    )

    print(f"Rreshta origjinalë: {len(df)}")

    print("\nShpërndarja e klasave:")

    print(
        df["label"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # APLIKIMI I 4 PIPELINE-VE
    # ========================================================

    for pipeline_name, pipeline_function in PIPELINES.items():

        experiment_df = df.copy()

        experiment_df["text"] = (
            experiment_df["raw_text"]
            .apply(pipeline_function)
        )

        # Numri i teksteve bosh pas preprocessing
        empty_count = (
            experiment_df["text"]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        if empty_count > 0:

            print(
                f"KUJDES: {pipeline_name} krijoi "
                f"{empty_count} tekste bosh."
            )

        # Dataset-i final për modelin
        output_df = experiment_df[
            [
                "comment_id",
                "text",
                "label"
            ]
        ].copy()

        output_path = os.path.join(
            OUTPUT_FOLDER,
            f"{split_name}_{pipeline_name}.xlsx"
        )

        output_df.to_excel(
            output_path,
            index=False
        )

        print(
            f"{pipeline_name:16s} "
            f"| rows={len(output_df):5d} "
            f"| empty={empty_count:3d}"
        )


# ============================================================
# KONTROLLI FINAL
# ============================================================

print("\n" + "=" * 70)
print("KONTROLLI FINAL")
print("=" * 70)


EXPECTED_SIZES = {
    "train": 7241,
    "validation": 1552,
    "test": 1552
}


for pipeline_name in PIPELINES.keys():

    print(
        f"\nPipeline: {pipeline_name.upper()}"
    )

    datasets = {}

    for split_name in INPUT_FILES.keys():

        path = os.path.join(
            OUTPUT_FOLDER,
            f"{split_name}_{pipeline_name}.xlsx"
        )

        dataset = pd.read_excel(path)

        datasets[split_name] = dataset

        expected_size = EXPECTED_SIZES[split_name]

        if len(dataset) != expected_size:

            raise ValueError(
                f"{pipeline_name}/{split_name}: "
                f"pritëm {expected_size}, "
                f"por gjetëm {len(dataset)}"
            )

        print(
            f"{split_name:10s}: {len(dataset)}"
        )


    # ========================================================
    # DATA LEAKAGE CHECK
    # ========================================================

    train_ids = set(
        datasets["train"]["comment_id"]
    )

    validation_ids = set(
        datasets["validation"]["comment_id"]
    )

    test_ids = set(
        datasets["test"]["comment_id"]
    )

    train_validation_overlap = (
        train_ids & validation_ids
    )

    train_test_overlap = (
        train_ids & test_ids
    )

    validation_test_overlap = (
        validation_ids & test_ids
    )

    if train_validation_overlap:
        raise ValueError(
            f"Leakage Train/Validation në {pipeline_name}"
        )

    if train_test_overlap:
        raise ValueError(
            f"Leakage Train/Test në {pipeline_name}"
        )

    if validation_test_overlap:
        raise ValueError(
            f"Leakage Validation/Test në {pipeline_name}"
        )

    print("Leakage: 0")


# ============================================================
# KONTROLLI QË TË GJITHA PIPELINE-T KANË TË NJËJTAT IDs
# ============================================================

print("\n" + "=" * 70)
print("KONTROLLI I CONSISTENCY MIDIS PIPELINE-VE")
print("=" * 70)


for split_name in INPUT_FILES.keys():

    reference_path = os.path.join(
        OUTPUT_FOLDER,
        f"{split_name}_raw.xlsx"
    )

    reference_df = pd.read_excel(
        reference_path
    )

    reference_ids = (
        reference_df["comment_id"]
        .astype(str)
        .tolist()
    )

    for pipeline_name in PIPELINES.keys():

        path = os.path.join(
            OUTPUT_FOLDER,
            f"{split_name}_{pipeline_name}.xlsx"
        )

        current_df = pd.read_excel(
            path
        )

        current_ids = (
            current_df["comment_id"]
            .astype(str)
            .tolist()
        )

        if reference_ids != current_ids:

            raise ValueError(
                f"ID mismatch: "
                f"{split_name}/{pipeline_name}"
            )

    print(
        f"{split_name:10s}: "
        "të njëjtat comment_id në të 4 pipeline-t"
    )


# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 70)
print("TË GJITHA DATASET-ET U KRIJUAN ME SUKSES")
print("=" * 70)

print("\nFolder:")
print(OUTPUT_FOLDER)

print("\nNumri total i file-ve të krijuar: 12")