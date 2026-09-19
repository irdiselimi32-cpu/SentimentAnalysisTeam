import re
import unicodedata
import pandas as pd


# ============================================================
# RREGULLAT E NORMALIZIMIT PËR GJUHËN SHQIPE
# ============================================================
#
# Rregullat janë zgjedhur nga analiza e TRAIN set-it.
# Nuk përdoren validation/test për ndërtimin e tyre.
#
# Janë shmangur forma shumë të paqarta si:
# te, ne, qe, ma, kom, jom, naj, ska etj.
#
# Qëllimi nuk është kthimi i çdo komenti në shqip standard,
# por reduktimi selektiv i zhurmës pa humbur informacionin
# emocional të dobishëm për analizën e sentimentit.
# ============================================================

ALBANIAN_NORMALIZATION = {

    # --------------------------------------------------------
    # Forma të zakonshme ku mungon ë / ç
    # --------------------------------------------------------

    "per": "për",
    "nje": "një",

    "eshte": "është",
    "esht": "është",

    "shume": "shumë",
    "shum": "shumë",

    "mire": "mirë",
    "mir": "mirë",

    "kete": "këtë",
    "kte": "këtë",

    "vetem": "vetëm",

    "ben": "bën",

    "jane": "janë",
    "jan": "janë",

    "gjithe": "gjithë",

    "kenge": "këngë",

    "cdo": "çdo",
    "qdo": "çdo",

    "djal": "djalë",
    "djale": "djalë",

    "lumte": "lumtë",
    "lumt": "lumtë",

    "interviste": "intervistë",

    "une": "unë",

    "cfare": "çfarë",
    "cfar": "çfarë",
    "qfar": "çfarë",
    "qfare": "çfarë",

    "gje": "gjë",

    "gjithmon": "gjithmonë",
    "gjithmone": "gjithmonë",

    "vajze": "vajzë",

    "njerez": "njerëz",
    "njerz": "njerëz",

    "pelqen": "pëlqen",

    "zemer": "zemër",

    "qene": "qenë",
    "qen": "qenë",

    "tjeter": "tjetër",

    "degjuar": "dëgjuar",

    # --------------------------------------------------------
    # Forma joformale / dialektore të sigurta
    # --------------------------------------------------------

    "asht": "është",
    "osht": "është",

    # --------------------------------------------------------
    # Shkurtime të zakonshme
    # --------------------------------------------------------

    "flm": "faleminderit",
    "dmth": "domethënë"
}


# ============================================================
# NORMALIZIMI I FJALËVE SHQIPE
# ============================================================

def normalize_albanian_words(text):
    """
    Normalizon vetëm fjalët që gjenden në dictionary-n
    ALBANIAN_NORMALIZATION.

    Kapitalizimi ruhet kur është e mundur.

    Shembuj:
        eshte  -> është
        Eshte  -> Është
        ESHTE  -> ËSHTË
        shume  -> shumë
        cfare  -> çfarë
        flm    -> faleminderit
        osht   -> është
    """

    def replace_word(match):

        original_word = match.group(0)
        lower_word = original_word.lower()

        # Nëse fjala nuk është në dictionary,
        # nuk bëjmë asnjë ndryshim.
        if lower_word not in ALBANIAN_NORMALIZATION:
            return original_word

        replacement = ALBANIAN_NORMALIZATION[lower_word]

        # Shembull:
        # SHUME -> SHUMË
        if original_word.isupper():
            return replacement.upper()

        # Shembull:
        # Shume -> Shumë
        if original_word[0].isupper():
            return replacement.capitalize()

        # Shembull:
        # shume -> shumë
        return replacement

    return re.sub(
        r"\b[A-Za-zËÇëç]+\b",
        replace_word,
        text
    )


# ============================================================
# NORMALIZIMI I PËRSËRITJEVE TË SHKRONJAVE
# ============================================================

def normalize_repeated_letters(text):
    """
    Redukton vetëm përsëritjet ekstreme.

    Tre ose më shumë përsëritje të së njëjtës shkronjë
    reduktohen në dy.

    Shembuj:
        duaaaaaaa -> duaa
        bravooooo -> bravoo
        superrr   -> superr

    Përsëritja nuk reduktohet në vetëm një shkronjë,
    sepse mund të përmbajë informacion mbi intensitetin
    emocional të komentit.
    """

    return re.sub(
        r"([A-Za-zËÇëç])\1{2,}",
        r"\1\1",
        text,
        flags=re.IGNORECASE
    )


# ============================================================
# PIPELINE 0
# RAW BASELINE
# ============================================================

def preprocess_raw(text):
    """
    RAW BASELINE

    Teksti përdoret në formën origjinale.

    Nuk bëhet:
    - normalizim
    - lowercase
    - heqje emoji
    - heqje pikësimi
    - korrigjim drejtshkrimor
    - heqje URL

    Ky variant shërben si kontroll eksperimental.
    """

    if pd.isna(text):
        return ""

    return str(text)


# ============================================================
# PIPELINE 1
# MINIMAL PREPROCESSING
# ============================================================

def preprocess_minimal(text):
    """
    MINIMAL PREPROCESSING

    Kryen vetëm pastrim teknik:

    1. Unicode NFC
    2. Heq URL
    3. Heq HTML
    4. Normalizon whitespace

    Ruhen:
    - emoji
    - pikësimi
    - kapitalizimi
    - përsëritjet e shkronjave
    - format joformale
    - negacioni
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # 1. Unicode normalization
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFC",
        text
    )

    # --------------------------------------------------------
    # 2. Heq URL
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # --------------------------------------------------------
    # 3. Heq HTML tags
    # --------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # --------------------------------------------------------
    # 4. Normalizon whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PIPELINE 2
# TRADITIONAL PREPROCESSING
# ============================================================

def preprocess_traditional(text):
    """
    TRADITIONAL PREPROCESSING

    Përfaqëson një qasje më agresive të preprocessing-ut.

    Kryen:

    1. Unicode NFC
    2. Heq URL
    3. Heq HTML
    4. Lowercase
    5. Heq emoji
    6. Heq pikësimin
    7. Heq simbolet
    8. Normalizon whitespace

    Nuk aplikohet stopword removal, sepse fjalë si
    'nuk' dhe 'mos' janë të rëndësishme për sentimentin.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # 1. Unicode normalization
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFC",
        text
    )

    # --------------------------------------------------------
    # 2. Heq URL
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # --------------------------------------------------------
    # 3. Heq HTML
    # --------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # --------------------------------------------------------
    # 4. Lowercase
    # --------------------------------------------------------

    text = text.lower()

    # --------------------------------------------------------
    # 5-7. Heq emoji, pikësim dhe simbole
    #
    # Ruhen shkronjat latine, karakteret e alfabetit shqip,
    # numrat dhe whitespace.
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-zA-ZÀ-ÖØ-öø-ÿ0-9\s]",
        " ",
        text
    )

    # --------------------------------------------------------
    # 8. Normalizon whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PIPELINE 3
# ALBANIAN-AWARE PREPROCESSING
# ============================================================

def preprocess_albanian_aware(text):
    """
    ALBANIAN-AWARE PREPROCESSING

    Pipeline i propozuar për komentet joformale
    në gjuhën shqipe.

    Kryen:

    1. Unicode NFC
    2. Heq URL
    3. Heq HTML
    4. Normalizim selektiv të formave shqipe
    5. Normalizim të disa formave dialektore
    6. Zgjerim të disa shkurtimeve të sigurta
    7. Reduktim të kontrolluar të përsëritjeve
    8. Normalizim whitespace

    Ndryshe nga Traditional, RUHEN:

    - emoji
    - pikësimi
    - kapitalizimi
    - negacioni
    - fjalët angleze / code-switching
    - shprehjet e internetit si wow, lol, omg etj.

    Qëllimi është reduktimi i zhurmës pa eliminuar
    sinjale potencialisht të rëndësishme për sentimentin.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # --------------------------------------------------------
    # 1. Unicode normalization
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFC",
        text
    )

    # --------------------------------------------------------
    # 2. Heq URL
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # --------------------------------------------------------
    # 3. Heq HTML
    # --------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # --------------------------------------------------------
    # 4-6. Normalizim selektiv i shqipes
    # --------------------------------------------------------

    text = normalize_albanian_words(
        text
    )

    # --------------------------------------------------------
    # 7. Redukton përsëritjet ekstreme
    # --------------------------------------------------------

    text = normalize_repeated_letters(
        text
    )

    # --------------------------------------------------------
    # 8. Normalizon whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# TESTIMI I TË GJITHA STRATEGJIVE
# ============================================================

if __name__ == "__main__":

    examples = [

        # Emoji + repeated letters
        "Të duaaaaaaa❤❤❤",

        # Negacion + drejtshkrim jo standard
        "Mos lakmoni se keto ekan fundin e tmerrshem!!!",

        # URL + emoji
        "Shiko këtu https://example.com 😡😡",

        # Kapitalizim + repeated letters
        "BRAVOOO   SHUMË     MIRË!!!",

        # ë
        "Ky eshte nje emision shume i mire!!!",

        # ë + emoji
        "Lumte djal, je nje njeri shume i mire 👏👏",

        # ç + ë + English/emoji
        "Cfare interviste e bukur, me pelqen shume ❤️",

        # Forma jo standarde
        "Njerez shume te mire, respekt!!!",

        # Shkurtim
        "Flm shume, je super ❤️",

        # Formë dialektore
        "Ky osht njeri shume i mire",

        # Formë dialektore + English expression
        "Qfar interviste, wow!!!",

        # Shkurtim + emoji
        "Dmth kjo esht super 😂",

        # Code-switching
        "Love kete kenge, wow ❤️",

        # Negacion
        "Nuk me pelqen fare kjo kenge 😡",

        # Negacion tjetër
        "Mos e beni kete!!!"
    ]

    print("=" * 70)
    print("TESTIMI I 4 STRATEGJIVE TË PREPROCESSING")
    print("=" * 70)

    for text in examples:

        print("\nORIGINAL:")
        print(text)

        print("\nRAW:")
        print(
            preprocess_raw(text)
        )

        print("\nMINIMAL:")
        print(
            preprocess_minimal(text)
        )

        print("\nTRADITIONAL:")
        print(
            preprocess_traditional(text)
        )

        print("\nALBANIAN-AWARE:")
        print(
            preprocess_albanian_aware(text)
        )

        print("\n" + "-" * 70)