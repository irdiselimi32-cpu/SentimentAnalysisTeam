import os
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================
# CONFIG
# =========================

MODEL_NAME = "gpt-4o"          # or "gpt-4o-mini" for cheaper testing
DATASET_PATH = "data/dataset-sentiment-analysis-preprocessed.xlsx"
OUTPUT_DIR = "results/gpt4o_prompt_benchmark"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "gpt4o_test_predictions.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

client = OpenAI()  # reads OPENAI_API_KEY automatically


# =========================
# 1. LOAD DATA
# =========================

print("\nSTEP 1: Loading dataset...")

df = pd.read_excel(DATASET_PATH)
df = df[["Comment", "Sent. Anal. Category"]].dropna()
df.columns = ["text", "true_label"]

label_map = {
    "Positive": "POSITIVE",
    "Negative": "NEGATIVE",
    "Neutral": "NEUTRAL"
}

df["true_label"] = df["true_label"].map(label_map)
df = df.dropna(subset=["true_label"]).copy()

df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()

print("Dataset loaded:", len(df), "rows")


# =========================
# 2. SPLIT DATA
# =========================

print("\nSTEP 2: Splitting dataset...")

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["true_label"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["true_label"],
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

# For GPT-4o benchmark, we only need test_df
test_df = test_df.reset_index(drop=True).copy()


# =========================
# 3. GPT-4o CLASSIFIER
# =========================

VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

def classify_with_gpt4o(comment, max_retries=3):
    prompt = f"""
You are a sentiment analysis classifier for Albanian text.

Classify the following comment into exactly one of these labels:
POSITIVE
NEGATIVE
NEUTRAL

Return only the label. Do not explain.

Comment:
{comment}
""".strip()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict sentiment classification system."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            label = response.choices[0].message.content.strip().upper()

            # Clean possible formatting mistakes
            label = label.replace(".", "").replace("'", "").replace('"', "")

            if label in VALID_LABELS:
                return label

            # fallback normalization
            if "POSITIVE" in label:
                return "POSITIVE"
            if "NEGATIVE" in label:
                return "NEGATIVE"
            if "NEUTRAL" in label:
                return "NEUTRAL"

            return "INVALID"

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(2 * (attempt + 1))

    return "ERROR"


# =========================
# 4. RUN BENCHMARK
# =========================

print("\nSTEP 3: Running GPT-4o prompt-based benchmark...")

predictions = []

start_time = time.time()

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    comment = row["text"]
    prediction = classify_with_gpt4o(comment)

    predictions.append(prediction)

    # Save partial progress every 50 rows
    if (idx + 1) % 50 == 0:
        partial_df = test_df.iloc[:idx + 1].copy()
        partial_df["predicted_label"] = predictions
        partial_df.to_excel(OUTPUT_FILE, index=False)

end_time = time.time()

test_df["predicted_label"] = predictions

print(f"\nBenchmark completed in {end_time - start_time:.2f} seconds")


# =========================
# 5. CLEAN INVALID RESULTS
# =========================

invalid_rows = test_df[
    ~test_df["predicted_label"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"])
]

if len(invalid_rows) > 0:
    print("\nInvalid predictions found:", len(invalid_rows))
    print(invalid_rows[["text", "predicted_label"]].head())

# Keep only valid predictions for evaluation
eval_df = test_df[
    test_df["predicted_label"].isin(["POSITIVE", "NEGATIVE", "NEUTRAL"])
].copy()


# =========================
# 6. EVALUATION
# =========================

print("\nSTEP 4: Evaluating GPT-4o benchmark...")

y_true = eval_df["true_label"]
y_pred = eval_df["predicted_label"]

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")

print("\nGPT-4o TEST ACCURACY:", accuracy)
print("GPT-4o TEST WEIGHTED F1:", f1)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred))


# =========================
# 7. CONFUSION MATRIX
# =========================

print("\nSTEP 5: Confusion matrix...")

labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["POS", "NEG", "NEU"],
    yticklabels=["POS", "NEG", "NEU"]
)

plt.title("Confusion Matrix - GPT-4o Prompt-Based Benchmark")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


# =========================
# 8. SAVE RESULTS
# =========================

print("\nSTEP 6: Saving results...")

test_df.to_excel(OUTPUT_FILE, index=False)

summary = {
    "model": MODEL_NAME,
    "benchmark_type": "prompt-based zero-shot",
    "test_size": len(test_df),
    "valid_predictions": len(eval_df),
    "invalid_predictions": len(invalid_rows),
    "accuracy": accuracy,
    "weighted_f1": f1,
    "runtime_seconds": end_time - start_time
}

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print("Done.")
print("Results saved to:", OUTPUT_FILE)