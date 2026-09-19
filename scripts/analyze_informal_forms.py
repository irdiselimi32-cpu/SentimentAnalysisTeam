import pandas as pd
import re
from collections import Counter


# ============================================================
# 1. LOAD TRAIN SET ONLY
# ============================================================

df = pd.read_excel("data/train.xlsx")

df = df[
    df["raw_text"].notna() &
    df["clean_text"].notna()
].copy()

print("=" * 70)
print("ANALIZA E FORMAVE JOFORMALE / SLANG")
print("=" * 70)

print("\nRAW-CLEAN train pairs:", len(df))


# ============================================================
# 2. TOKENIZATION E THJESHTË PËR ANALIZË
# ============================================================

def get_words(text):
    """
    Nxjerr fjalët për analizë.
    Nuk është tokenizimi që do përdorë XLM-RoBERTa.
    """

    text = str(text).lower()

    return re.findall(
        r"[a-zA-ZëËçÇ]+",
        text
    )


# ============================================================
# 3. RAW VOCABULARY
# ============================================================

raw_counter = Counter()

for text in df["raw_text"]:
    raw_counter.update(get_words(text))


# ============================================================
# 4. LISTË KANDIDATËSH TË FORMAVE JOFORMALE
# ============================================================
#
# Këto NUK janë rregulla preprocessing-u.
# Janë vetëm forma që kërkojmë në TRAIN për të parë
# nëse ekzistojnë realisht në dataset.
# ============================================================

informal_candidates = [
    # Shkurtime të zakonshme shqip
    "flm",
    "falem",
    "dmth",
    "dmt",
    "psh",
    "p.sh",
    "ska",
    "seshte",
    "sesht",
    "sdi",
    "sdu",
    "sdo",
    "ska",
    "skam",
    "ske",
    "kena",
    "kemi",
    "kom",
    "jom",
    "osht",
    "asht",
    "esh",
    "eshte",
    "esht",

    # Forma joformale
    "shum",
    "mir",
    "keq",
    "kte",
    "kta",
    "ato",
    "cfar",
    "cfare",
    "qka",
    "cka",
    "qfar",
    "qfare",
    "qdo",
    "cdo",
    "naj",
    "najse",
    "nejse",
    "njerz",
    "njerez",

    # Internet / English slang
    "lol",
    "omg",
    "wtf",
    "bro",
    "bruh",
    "wow",
    "super",
    "nice",
    "love",
    "hate",
    "sorry",
    "thanks",
    "thank",
    "fake"
]


# ============================================================
# 5. FREQUENCY OF INFORMAL CANDIDATES
# ============================================================

found_candidates = []

for word in informal_candidates:

    frequency = raw_counter[word.lower()]

    if frequency > 0:

        found_candidates.append({
            "form": word,
            "frequency": frequency
        })


informal_df = pd.DataFrame(found_candidates)

if not informal_df.empty:

    informal_df = informal_df.sort_values(
        "frequency",
        ascending=False
    )


print("\n=========================")
print("INFORMAL FORMS FOUND")
print("=========================")

if not informal_df.empty:
    print(
        informal_df.to_string(index=False)
    )
else:
    print("No predefined informal candidates found.")


# ============================================================
# 6. SHORT WORDS
# ============================================================
#
# Shkurtesat/slang shpesh janë forma shumë të shkurtra.
# I nxjerrim për inspektim, por NUK i normalizojmë.
# ============================================================

short_words = []

for word, frequency in raw_counter.items():

    if (
        2 <= len(word) <= 4
        and frequency >= 5
    ):

        short_words.append({
            "word": word,
            "frequency": frequency
        })


short_df = pd.DataFrame(short_words)

if not short_df.empty:

    short_df = short_df.sort_values(
        "frequency",
        ascending=False
    )


print("\n=========================")
print("TOP SHORT FORMS")
print("=========================")

if not short_df.empty:

    print(
        short_df
        .head(100)
        .to_string(index=False)
    )


# ============================================================
# 7. FIND CONTEXT FOR INFORMAL FORMS
# ============================================================

context_rows = []

if not informal_df.empty:

    # Marrim format më të rëndësishme
    forms_to_check = informal_df[
        informal_df["frequency"] >= 2
    ]["form"].tolist()

    for form in forms_to_check:

        pattern = rf"\b{re.escape(form)}\b"

        matches = df[
            df["raw_text"]
            .astype(str)
            .str.contains(
                pattern,
                case=False,
                regex=True,
                na=False
            )
        ]

        # Maksimumi 10 shembuj për çdo formë
        for _, row in matches.head(10).iterrows():

            context_rows.append({
                "form": form,
                "raw_text": row["raw_text"],
                "clean_text": row["clean_text"],
                "label": row["label"]
            })


context_df = pd.DataFrame(context_rows)


# ============================================================
# 8. FIND VERY SHORT UPPERCASE / INTERNET EXPRESSIONS
# ============================================================

def extract_special_short_forms(text):

    text = str(text)

    words = re.findall(
        r"\b[A-Za-z]{2,5}\b",
        text
    )

    return words


special_counter = Counter()

for text in df["raw_text"]:

    words = extract_special_short_forms(text)

    for word in words:

        # Fokus te forma të shkurtra
        if len(word) <= 5:
            special_counter[word.lower()] += 1


special_rows = []

for word, frequency in special_counter.most_common():

    if frequency >= 3:

        special_rows.append({
            "word": word,
            "frequency": frequency
        })


special_df = pd.DataFrame(special_rows)


# ============================================================
# 9. SAVE RESULTS
# ============================================================

informal_df.to_excel(
    "data/informal_forms_found.xlsx",
    index=False
)

short_df.to_excel(
    "data/short_forms_train.xlsx",
    index=False
)

context_df.to_excel(
    "data/informal_forms_context.xlsx",
    index=False
)

special_df.to_excel(
    "data/special_short_forms.xlsx",
    index=False
)


print("\n=========================")
print("FILES CREATED")
print("=========================")

print("data/informal_forms_found.xlsx")
print("data/short_forms_train.xlsx")
print("data/informal_forms_context.xlsx")
print("data/special_short_forms.xlsx")

print("\nDONE.")