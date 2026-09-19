import pandas as pd
import re


# ============================================================
# 1. LOAD MASTER DATASET
# ============================================================

df = pd.read_excel("data/master_dataset.xlsx")

# Marrim vetëm rreshtat ku ekzistojnë RAW dhe CLEAN
pairs = df[
    df["raw_text"].notna() &
    df["clean_text"].notna()
].copy()

print("=" * 70)
print("ANALIZA RAW -> CLEAN")
print("=" * 70)

print("\nTotal master dataset:", len(df))
print("RAW-CLEAN pairs:", len(pairs))


# ============================================================
# 2. CHECK HOW MANY TEXTS WERE ACTUALLY CHANGED
# ============================================================

pairs["changed"] = (
    pairs["raw_text"].astype(str).str.strip()
    !=
    pairs["clean_text"].astype(str).str.strip()
)

print("\n=========================")
print("TEXT CHANGES")
print("=========================")

print("Changed:", pairs["changed"].sum())
print("Unchanged:", (~pairs["changed"]).sum())

print(
    "Changed percentage:",
    round(pairs["changed"].mean() * 100, 2),
    "%"
)


# ============================================================
# 3. ë ANALYSIS
# ============================================================

pairs["raw_e_diaeresis"] = (
    pairs["raw_text"]
    .astype(str)
    .str.count("ë|Ë")
)

pairs["clean_e_diaeresis"] = (
    pairs["clean_text"]
    .astype(str)
    .str.count("ë|Ë")
)

e_increased = (
    pairs["clean_e_diaeresis"]
    >
    pairs["raw_e_diaeresis"]
)

print("\n=========================")
print("LETTER ë")
print("=========================")

print(
    "Rows where CLEAN contains more ë:",
    e_increased.sum()
)


# ============================================================
# 4. ç ANALYSIS
# ============================================================

pairs["raw_c_cedilla"] = (
    pairs["raw_text"]
    .astype(str)
    .str.count("ç|Ç")
)

pairs["clean_c_cedilla"] = (
    pairs["clean_text"]
    .astype(str)
    .str.count("ç|Ç")
)

c_increased = (
    pairs["clean_c_cedilla"]
    >
    pairs["raw_c_cedilla"]
)

print("\n=========================")
print("LETTER ç")
print("=========================")

print(
    "Rows where CLEAN contains more ç:",
    c_increased.sum()
)


# ============================================================
# 5. EMOJI / NON-ALPHANUMERIC SYMBOL ANALYSIS
# ============================================================

def has_nonstandard_symbol(text):
    """
    Kontroll i thjeshtë për simbole jashtë
    shkronjave, numrave, hapësirave dhe pikësimit bazë.
    Përdoret si indikator për emoji/simbole.
    """

    text = str(text)

    return bool(
        re.search(
            r"[^\w\s.,!?;:'\"()\-ëËçÇ]",
            text,
            flags=re.UNICODE
        )
    )


pairs["raw_symbol"] = pairs["raw_text"].apply(
    has_nonstandard_symbol
)

pairs["clean_symbol"] = pairs["clean_text"].apply(
    has_nonstandard_symbol
)

symbols_removed = (
    pairs["raw_symbol"] &
    ~pairs["clean_symbol"]
)

print("\n=========================")
print("EMOJI / SYMBOLS")
print("=========================")

print(
    "RAW rows containing non-standard symbols:",
    pairs["raw_symbol"].sum()
)

print(
    "CLEAN rows containing non-standard symbols:",
    pairs["clean_symbol"].sum()
)

print(
    "Rows where symbols appear removed:",
    symbols_removed.sum()
)


# ============================================================
# 6. REPEATED LETTER ANALYSIS
# ============================================================

def has_repeated_letters(text):
    """
    Detecton 3 ose më shumë përsëritje
    të së njëjtës shkronjë.

    Shembull:
    duaaaaaaa
    bravoooo
    shummmm
    """

    text = str(text)

    return bool(
        re.search(
            r"([A-Za-zËÇëç])\1{2,}",
            text,
            flags=re.IGNORECASE
        )
    )


pairs["raw_repeated"] = pairs["raw_text"].apply(
    has_repeated_letters
)

pairs["clean_repeated"] = pairs["clean_text"].apply(
    has_repeated_letters
)

repetitions_removed = (
    pairs["raw_repeated"] &
    ~pairs["clean_repeated"]
)

print("\n=========================")
print("REPEATED LETTERS")
print("=========================")

print(
    "RAW rows with repeated letters:",
    pairs["raw_repeated"].sum()
)

print(
    "CLEAN rows with repeated letters:",
    pairs["clean_repeated"].sum()
)

print(
    "Rows where repetitions were removed:",
    repetitions_removed.sum()
)


# ============================================================
# 7. TEXT LENGTH
# ============================================================

pairs["raw_length"] = (
    pairs["raw_text"]
    .astype(str)
    .str.len()
)

pairs["clean_length"] = (
    pairs["clean_text"]
    .astype(str)
    .str.len()
)

print("\n=========================")
print("TEXT LENGTH")
print("=========================")

print(
    "Average RAW characters:",
    round(pairs["raw_length"].mean(), 2)
)

print(
    "Average CLEAN characters:",
    round(pairs["clean_length"].mean(), 2)
)


# ============================================================
# 8. PRINT EXAMPLES: ë
# ============================================================

print("\n=========================")
print("EXAMPLES - ë RESTORATION")
print("=========================")

sample_e = pairs[e_increased].head(5)

for _, row in sample_e.iterrows():

    print("\nRAW:")
    print(row["raw_text"])

    print("CLEAN:")
    print(row["clean_text"])

    print("-" * 70)


# ============================================================
# 9. PRINT EXAMPLES: ç
# ============================================================

print("\n=========================")
print("EXAMPLES - ç RESTORATION")
print("=========================")

sample_c = pairs[c_increased].head(5)

for _, row in sample_c.iterrows():

    print("\nRAW:")
    print(row["raw_text"])

    print("CLEAN:")
    print(row["clean_text"])

    print("-" * 70)


# ============================================================
# 10. PRINT EXAMPLES: REPEATED LETTERS
# ============================================================

print("\n=========================")
print("EXAMPLES - REPEATED LETTERS")
print("=========================")

sample_repeat = pairs[repetitions_removed].head(10)

for _, row in sample_repeat.iterrows():

    print("\nRAW:")
    print(row["raw_text"])

    print("CLEAN:")
    print(row["clean_text"])

    print("-" * 70)


# ============================================================
# 11. PRINT EXAMPLES: SYMBOL / EMOJI REMOVAL
# ============================================================

print("\n=========================")
print("EXAMPLES - SYMBOL / EMOJI REMOVAL")
print("=========================")

sample_symbols = pairs[symbols_removed].head(10)

for _, row in sample_symbols.iterrows():

    print("\nRAW:")
    print(row["raw_text"])

    print("CLEAN:")
    print(row["clean_text"])

    print("-" * 70)


# ============================================================
# 12. SAVE ANALYSIS
# ============================================================

pairs.to_excel(
    "data/raw_clean_analysis.xlsx",
    index=False
)

print("\n=========================")
print("FILE CREATED")
print("=========================")

print("data/raw_clean_analysis.xlsx")

print("\nANALYSIS FINISHED.")