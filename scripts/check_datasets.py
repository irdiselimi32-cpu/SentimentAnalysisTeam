import pandas as pd

# =========================
# 1. LOAD BOTH DATASETS
# =========================

df1 = pd.read_excel(
    "data/dataset-sentiment-analysis-preprocessed.xlsx"
)

df2 = pd.read_excel(
    "data/Dataset_Ready_ForPreprocessing_FGJH.xlsx"
)

print("\n=========================")
print("DATASET 1")
print("=========================")
print("Rows:", len(df1))
print("Columns:", len(df1.columns))
print(df1.columns.tolist())

print("\n=========================")
print("DATASET 2")
print("=========================")
print("Rows:", len(df2))
print("Columns:", len(df2.columns))
print(df2.columns.tolist())


# =========================
# 2. SENTIMENT LABELS
# =========================

print("\n=========================")
print("LABELS - DATASET 1")
print("=========================")

print(
    df1["Sent. Anal. Category"]
    .value_counts(dropna=False)
)

print("\n=========================")
print("LABELS - DATASET 2")
print("=========================")

print(
    df2["Sent. Anal. Category"]
    .value_counts(dropna=False)
)


# =========================
# 3. MISSING COMMENTS
# =========================

print("\n=========================")
print("MISSING COMMENTS")
print("=========================")

print(
    "Dataset 1:",
    df1["Comment"].isna().sum()
)

print(
    "Dataset 2:",
    df2["Comment"].isna().sum()
)


# =========================
# 4. EXACT DUPLICATES INSIDE EACH DATASET
# =========================

print("\n=========================")
print("DUPLICATE COMMENTS")
print("=========================")

print(
    "Dataset 1:",
    df1["Comment"].duplicated().sum()
)

print(
    "Dataset 2:",
    df2["Comment"].duplicated().sum()
)


# =========================
# 5. NORMALIZE TEXT ONLY FOR DUPLICATE CHECKING
# =========================

def normalize_for_check(text):
    if pd.isna(text):
        return None

    return " ".join(
        str(text)
        .lower()
        .strip()
        .split()
    )

df1["check_text"] = df1["Comment"].apply(normalize_for_check)
df2["check_text"] = df2["Comment"].apply(normalize_for_check)


# =========================
# 6. OVERLAP BETWEEN DATASETS
# =========================

comments1 = set(df1["check_text"].dropna())
comments2 = set(df2["check_text"].dropna())

overlap = comments1.intersection(comments2)

print("\n=========================")
print("OVERLAP BETWEEN DATASETS")
print("=========================")

print("Unique Dataset 1:", len(comments1))
print("Unique Dataset 2:", len(comments2))
print("Comments present in BOTH:", len(overlap))


# =========================
# 7. CHECK CLEANED COLUMN
# =========================

if "Comment i Pastruar" in df2.columns:

    print("\n=========================")
    print("CLEANED COMMENTS - DATASET 2")
    print("=========================")

    print(
        "Available cleaned comments:",
        df2["Comment i Pastruar"].notna().sum()
    )

    print(
        "Missing cleaned comments:",
        df2["Comment i Pastruar"].isna().sum()
    )


# =========================
# 8. COMMENT IDs
# =========================

for name, data in [
    ("Dataset 1", df1),
    ("Dataset 2", df2)
]:

    print(f"\n{name} ID columns:")

    for col in ["comment_id", "Comment ID"]:
        if col in data.columns:
            print(
                col,
                "- non-null:",
                data[col].notna().sum(),
                "- unique:",
                data[col].nunique()
            )

print("\nCHECK FINISHED.")

print("\n=========================")
print("DATASET 2 - CORRECTED LABELS")
print("=========================")

print(
    df2["Sent. Anal. Category4"]
    .value_counts(dropna=False)
)

print("\n=========================")
print("RAW MISSING BUT CLEAN EXISTS")
print("=========================")

problem_rows = df2[
    df2["Comment"].isna() &
    df2["Comment i Pastruar"].notna()
]

print("Rows:", len(problem_rows))

print(
    problem_rows[
        [
            "comment_id",
            "Comment ID",
            "Comment",
            "Comment i Pastruar",
            "Sent. Anal. Category",
            "Sent. Anal. Category4"
        ]
    ].head(20).to_string()
)


print("\n=========================")
print("ROWS WITH BOTH RAW AND CLEAN")
print("=========================")

both = df2[
    df2["Comment"].notna() &
    df2["Comment i Pastruar"].notna()
]

print("Rows:", len(both))


print("\n=========================")
print("SAMPLE RAW -> CLEAN")
print("=========================")

for _, row in both.sample(min(10, len(both)), random_state=42).iterrows():

    print("\nRAW:")
    print(row["Comment"])

    print("CLEAN:")
    print(row["Comment i Pastruar"])

    print("OLD LABEL:")
    print(row["Sent. Anal. Category"])

    print("CORRECTED LABEL:")
    print(row["Sent. Anal. Category4"])

    print("-" * 70)