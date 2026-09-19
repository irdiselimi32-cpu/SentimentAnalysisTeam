import pandas as pd
import re
import unicodedata


# ============================================================
# 1. LOAD DATASETS
# ============================================================

df1 = pd.read_excel(
    "data/dataset-sentiment-analysis-preprocessed.xlsx"
)

df2 = pd.read_excel(
    "data/Dataset_Ready_ForPreprocessing_FGJH.xlsx"
)

print("Dataset 1:", df1.shape)
print("Dataset 2:", df2.shape)


# ============================================================
# 2. NORMALIZE LABELS
# ============================================================

def normalize_label(label):

    # Missing label remains missing
    if pd.isna(label):
        return None

    label = str(label).strip().lower()

    label_map = {
        "positive": "POSITIVE",
        "ositive": "POSITIVE",

        "negative": "NEGATIVE",
        "negativ": "NEGATIVE",

        "neutral": "NEUTRAL",
        "neutrale": "NEUTRAL",
        "neutre": "NEUTRAL",
        "neut": "NEUTRAL",

        # Methodological decision:
        # Mixed is treated as Neutral
        "mixed": "NEUTRAL"
    }

    return label_map.get(label)


df1["label"] = df1["Sent. Anal. Category"].apply(normalize_label)
df2["label"] = df2["Sent. Anal. Category"].apply(normalize_label)


print("\n=========================")
print("NORMALIZED LABELS")
print("=========================")

print("\nDataset 1:")
print(df1["label"].value_counts(dropna=False))

print("\nDataset 2:")
print(df2["label"].value_counts(dropna=False))


# ============================================================
# 3. FUNCTION USED ONLY TO MATCH COMMENTS
# ============================================================

def normalize_for_matching(text):

    if pd.isna(text):
        return None

    text = str(text)

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Lowercase only for matching
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df1["match_text"] = df1["Comment"].apply(normalize_for_matching)
df2["match_text"] = df2["Comment"].apply(normalize_for_matching)


# ============================================================
# 4. START MASTER DATASET FROM DATASET 1
# ============================================================

master = df1[
    [
        "comment_id",
        "Comment",
        "label"
    ]
].copy()

master.columns = [
    "comment_id",
    "raw_text",
    "label"
]


# Remove rows without text or valid label
master = master.dropna(
    subset=["raw_text", "label"]
)

master["match_text"] = master["raw_text"].apply(
    normalize_for_matching
)


print("\nRows after removing missing text/labels:", len(master))


# ============================================================
# 5. PREPARE CLEAN TEXT FROM DATASET 2
# ============================================================

clean_lookup = df2[
    [
        "Comment",
        "Comment i Pastruar"
    ]
].copy()

clean_lookup = clean_lookup.dropna(
    subset=["Comment", "Comment i Pastruar"]
)

clean_lookup["match_text"] = clean_lookup["Comment"].apply(
    normalize_for_matching
)

clean_lookup = clean_lookup[
    [
        "match_text",
        "Comment i Pastruar"
    ]
]

clean_lookup.columns = [
    "match_text",
    "clean_text"
]


# ============================================================
# 6. REMOVE DUPLICATE MATCH KEYS FROM CLEAN LOOKUP
# ============================================================

# If the same RAW comment appears multiple times in Dataset 2,
# keep the first available corrected version for now.
clean_lookup = clean_lookup.drop_duplicates(
    subset=["match_text"],
    keep="first"
)


# ============================================================
# 7. ADD CLEAN VERSION TO MASTER
# ============================================================

master = master.merge(
    clean_lookup,
    on="match_text",
    how="left"
)


# ============================================================
# 8. CHECK DUPLICATE RAW COMMENTS
# ============================================================

print("\n=========================")
print("DUPLICATES IN MASTER")
print("=========================")

print(
    "Duplicate raw texts:",
    master["match_text"].duplicated().sum()
)


# ============================================================
# 9. CHECK LABEL CONFLICTS
# ============================================================

label_conflicts = (
    master.groupby("match_text")["label"]
    .nunique()
)

label_conflicts = label_conflicts[
    label_conflicts > 1
]


print("\n=========================")
print("LABEL CONFLICTS")
print("=========================")

print(
    "Same text with different labels:",
    len(label_conflicts)
)


# ============================================================
# 10. REMOVE EXACT DUPLICATES
# ============================================================

# IMPORTANT:
# We only remove duplicate texts that have the same label.
# Conflicting cases will be inspected separately.

conflicting_texts = set(label_conflicts.index)

safe_rows = master[
    ~master["match_text"].isin(conflicting_texts)
].copy()

conflict_rows = master[
    master["match_text"].isin(conflicting_texts)
].copy()


safe_rows = safe_rows.drop_duplicates(
    subset=["match_text", "label"],
    keep="first"
)


# ============================================================
# 11. FINAL STATISTICS
# ============================================================

print("\n=========================")
print("MASTER DATASET")
print("=========================")

print("Safe unique rows:", len(safe_rows))
print("Conflicting rows:", len(conflict_rows))

print("\nFinal label distribution:")
print(safe_rows["label"].value_counts())

print("\nClean text available:")
print(safe_rows["clean_text"].notna().sum())

print("\nClean text missing:")
print(safe_rows["clean_text"].isna().sum())


# ============================================================
# 12. SAVE
# ============================================================

# match_text was created only for matching,
# so we don't need it in the experimental dataset.

final_master = safe_rows[
    [
        "comment_id",
        "raw_text",
        "clean_text",
        "label"
    ]
].copy()


final_master.to_excel(
    "data/master_dataset.xlsx",
    index=False
)


# Save conflicts separately so that nothing is silently lost
conflict_rows[
    [
        "comment_id",
        "raw_text",
        "clean_text",
        "label"
    ]
].to_excel(
    "data/label_conflicts.xlsx",
    index=False
)


print("\nFiles created:")
print("data/master_dataset.xlsx")
print("data/label_conflicts.xlsx")
print("\nDONE.")