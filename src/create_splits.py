import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. LOAD MASTER DATASET
# ============================================================

df = pd.read_excel("data/master_dataset.xlsx")

print("Master dataset:", len(df))

print("\nLabel distribution:")
print(df["label"].value_counts())


# ============================================================
# 2. FIRST SPLIT: 70% TRAIN / 30% TEMP
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)


# ============================================================
# 3. SECOND SPLIT: 15% VALIDATION / 15% TEST
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# ============================================================
# 4. ADD SPLIT COLUMN
# ============================================================

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"


# ============================================================
# 5. RECOMBINE
# ============================================================

final_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)


# ============================================================
# 6. CHECK RESULTS
# ============================================================

print("\n=========================")
print("FINAL SPLITS")
print("=========================")

print("\nTrain:", len(train_df))
print(train_df["label"].value_counts())

print("\nValidation:", len(val_df))
print(val_df["label"].value_counts())

print("\nTest:", len(test_df))
print(test_df["label"].value_counts())


print("\nPercentages:")

print(
    "Train:",
    round(len(train_df) / len(df) * 100, 2),
    "%"
)

print(
    "Validation:",
    round(len(val_df) / len(df) * 100, 2),
    "%"
)

print(
    "Test:",
    round(len(test_df) / len(df) * 100, 2),
    "%"
)


# ============================================================
# 7. CHECK FOR DATA LEAKAGE
# ============================================================

train_ids = set(train_df["comment_id"])
val_ids = set(val_df["comment_id"])
test_ids = set(test_df["comment_id"])

print("\n=========================")
print("DATA LEAKAGE CHECK")
print("=========================")

print(
    "Train / Validation overlap:",
    len(train_ids.intersection(val_ids))
)

print(
    "Train / Test overlap:",
    len(train_ids.intersection(test_ids))
)

print(
    "Validation / Test overlap:",
    len(val_ids.intersection(test_ids))
)


# ============================================================
# 8. SAVE
# ============================================================

final_df.to_excel(
    "data/master_dataset_split.xlsx",
    index=False
)

train_df.to_excel(
    "data/train.xlsx",
    index=False
)

val_df.to_excel(
    "data/validation.xlsx",
    index=False
)

test_df.to_excel(
    "data/test.xlsx",
    index=False
)


print("\nFiles created:")
print("data/master_dataset_split.xlsx")
print("data/train.xlsx")
print("data/validation.xlsx")
print("data/test.xlsx")

print("\nDONE.")