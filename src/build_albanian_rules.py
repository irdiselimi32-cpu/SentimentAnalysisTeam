import pandas as pd
import re
from collections import Counter, defaultdict


# ============================================================
# 1. LOAD TRAIN DATASET
# ============================================================

df = pd.read_excel("data/train.xlsx")

# Na duhen vetëm çiftet RAW-CLEAN
df = df[
    df["raw_text"].notna() &
    df["clean_text"].notna()
].copy()

print("=" * 70)
print("ALBANIAN NORMALIZATION RULE ANALYSIS")
print("=" * 70)

print("\nTrain RAW-CLEAN pairs:", len(df))


# ============================================================
# 2. SIMPLE WORD TOKENIZATION
# ============================================================

def get_words(text):
    """
    Nxjerr fjalët duke ruajtur shkronjat ë dhe ç.
    Përdoret vetëm për analizën e kandidatëve.
    """

    text = str(text).lower()

    return re.findall(
        r"[a-zA-ZëËçÇ]+",
        text
    )


# ============================================================
# 3. COLLECT RAW AND CLEAN VOCABULARY
# ============================================================

raw_counter = Counter()
clean_counter = Counter()

for _, row in df.iterrows():

    raw_words = get_words(row["raw_text"])
    clean_words = get_words(row["clean_text"])

    raw_counter.update(raw_words)
    clean_counter.update(clean_words)


print("\nUnique RAW words:", len(raw_counter))
print("Unique CLEAN words:", len(clean_counter))


# ============================================================
# 4. WORD-LEVEL CANDIDATE PAIRS
# ============================================================

candidate_rules = defaultdict(Counter)

for _, row in df.iterrows():

    raw_words = get_words(row["raw_text"])
    clean_words = get_words(row["clean_text"])

    # Përdorim vetëm rastet ku numri i fjalëve është i njëjtë.
    # Kjo shmang shumë lidhje të pasakta kur CLEAN është
    # rishkruar ndjeshëm.
    if len(raw_words) != len(clean_words):
        continue

    for raw_word, clean_word in zip(
        raw_words,
        clean_words
    ):

        if raw_word != clean_word:
            candidate_rules[raw_word][clean_word] += 1


# ============================================================
# 5. BUILD RULE TABLE
# ============================================================

rows = []

for raw_word, replacements in candidate_rules.items():

    total_changes = sum(replacements.values())

    best_clean, best_count = replacements.most_common(1)[0]

    confidence = best_count / total_changes

    rows.append({
        "raw_word": raw_word,
        "clean_word": best_clean,
        "count": best_count,
        "total_changes": total_changes,
        "confidence": confidence,
        "raw_frequency": raw_counter[raw_word],
        "clean_frequency": clean_counter[best_clean]
    })


rules_df = pd.DataFrame(rows)

if not rules_df.empty:

    rules_df = rules_df.sort_values(
        by=["count", "confidence"],
        ascending=[False, False]
    )


# ============================================================
# 6. HIGH-CONFIDENCE CANDIDATES
# ============================================================

# Për momentin nuk i aplikojmë automatikisht.
# Vetëm identifikojmë kandidatët e fortë.

if not rules_df.empty:

    strong_rules = rules_df[
        (rules_df["count"] >= 3) &
        (rules_df["confidence"] >= 0.80)
    ].copy()

else:

    strong_rules = pd.DataFrame()


print("\n=========================")
print("RULE STATISTICS")
print("=========================")

print("All candidate rules:", len(rules_df))
print("Strong candidate rules:", len(strong_rules))


# ============================================================
# 7. PRINT TOP RULES
# ============================================================

print("\n=========================")
print("TOP NORMALIZATION CANDIDATES")
print("=========================")

if not strong_rules.empty:

    print(
        strong_rules[
            [
                "raw_word",
                "clean_word",
                "count",
                "confidence"
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

else:

    print("No strong rules found.")


# ============================================================
# 8. SPECIFIC ë / ç RULES
# ============================================================

if not strong_rules.empty:

    albanian_letters = strong_rules[
        (
            strong_rules["clean_word"].str.contains(
                "ë|ç",
                regex=True
            )
        )
        &
        (
            ~strong_rules["raw_word"].str.contains(
                "ë|ç",
                regex=True
            )
        )
    ].copy()

else:

    albanian_letters = pd.DataFrame()


print("\n=========================")
print("ë / ç NORMALIZATION CANDIDATES")
print("=========================")

if not albanian_letters.empty:

    print(
        albanian_letters[
            [
                "raw_word",
                "clean_word",
                "count",
                "confidence"
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

else:

    print("No ë/ç candidates found.")


# ============================================================
# 9. REPEATED LETTER WORDS
# ============================================================

def contains_repetition(word):

    return bool(
        re.search(
            r"([a-zëç])\1{2,}",
            word,
            flags=re.IGNORECASE
        )
    )


repeated_words = []

for word, frequency in raw_counter.items():

    if contains_repetition(word):

        repeated_words.append({
            "word": word,
            "frequency": frequency
        })


repeated_df = pd.DataFrame(repeated_words)

if not repeated_df.empty:

    repeated_df = repeated_df.sort_values(
        "frequency",
        ascending=False
    )


print("\n=========================")
print("TOP REPEATED-LETTER WORDS")
print("=========================")

if not repeated_df.empty:

    print(
        repeated_df
        .head(30)
        .to_string(index=False)
    )

else:

    print("No repeated-letter words found.")


# ============================================================
# 10. SAVE RESULTS
# ============================================================

rules_df.to_excel(
    "data/all_normalization_candidates.xlsx",
    index=False
)

strong_rules.to_excel(
    "data/strong_normalization_candidates.xlsx",
    index=False
)

albanian_letters.to_excel(
    "data/albanian_letter_candidates.xlsx",
    index=False
)

repeated_df.to_excel(
    "data/repeated_letter_words.xlsx",
    index=False
)


print("\n=========================")
print("FILES CREATED")
print("=========================")

print("data/all_normalization_candidates.xlsx")
print("data/strong_normalization_candidates.xlsx")
print("data/albanian_letter_candidates.xlsx")
print("data/repeated_letter_words.xlsx")

print("\nDONE.")