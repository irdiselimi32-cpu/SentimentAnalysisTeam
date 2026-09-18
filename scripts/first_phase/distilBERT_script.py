#No training, just inference and evaluation on the test set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import os

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize
from transformers import pipeline

# START SECTION 1 LOAD DATA
print("\nSTEP 1: Loading dataset...")

df = pd.read_excel("data/dataset-sentiment-analysis-preprocessed.xlsx")

df = df[["Comment", "Sent. Anal. Category"]].dropna()
df.columns = ["text", "true_label"]

label_map = {
    "Positive": "POSITIVE",
    "Negative": "NEGATIVE",
    "Neutral": "NEUTRAL"
}

df["true_label"] = df["true_label"].map(label_map)
df = df.dropna(subset=["true_label"])

print("Dataset loaded:", len(df), "rows")
# END SECTION 1

# START SECTION 2 LOAD MODEL
print("\nSTEP 2: Loading pretrained model...")

model_id = "Yuu-Xie/distilbert-base-multilingual-cased-sentiment"

start_time = time.time()

classifier = pipeline(
    "sentiment-analysis",
    model=model_id,
    tokenizer=model_id,
    return_all_scores=True
)

print(f"Model loaded in {time.time() - start_time:.2f} seconds")
# END SECTION 2

# START SECTION 3 SAFE PREPROCESSING
print("\nSTEP 3: Preparing text safely...")

texts = (
    df["text"]
    .astype(str)
    .apply(lambda x: x[:300])
    .tolist()
)

print("Texts prepared:", len(texts))
# END SECTION 3

# START SECTION 4 INFERENCE / PREDICTION
print("\nSTEP 4: Running inference...")

start_time = time.time()
results = []
batch_size = 16

for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]

    preds = classifier(
        batch,
        truncation=True,
        max_length=256
    )

    results.extend(preds)

print(f"Inference completed in {time.time() - start_time:.2f} seconds")
# END SECTION 4

# START SECTION 5 PROCESS RESULTS
print("\nSTEP 5: Processing predictions...")

predicted_labels = []
probabilities = []

label_order = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

for r in results:
    # normalize possible output shapes
    if isinstance(r, dict):
        r = [r]
    if isinstance(r, list) and len(r) > 0 and isinstance(r[0], list):
        r = r[0]

    prob_dict = {
        "NEGATIVE": 0.0,
        "NEUTRAL": 0.0,
        "POSITIVE": 0.0
    }

    best_label = None
    best_score = -1

    for item in r:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "")).upper()
        score = float(item.get("score", 0.0))

        # flexible mapping for different label styles
        if "NEG" in label:
            mapped_label = "NEGATIVE"
        elif "NEU" in label:
            mapped_label = "NEUTRAL"
        elif "POS" in label:
            mapped_label = "POSITIVE"
        elif label in ["LABEL_0", "0"]:
            mapped_label = "NEGATIVE"
        elif label in ["LABEL_1", "1"]:
            mapped_label = "NEUTRAL"
        elif label in ["LABEL_2", "2"]:
            mapped_label = "POSITIVE"
        else:
            continue

        prob_dict[mapped_label] = score

        if score > best_score:
            best_score = score
            best_label = mapped_label

    ordered_probs = [prob_dict[lbl] for lbl in label_order]

    # fallback in case labels come back oddly
    if best_label is None:
        best_label = label_order[int(np.argmax(ordered_probs))]

    predicted_labels.append(best_label)
    probabilities.append(ordered_probs)

df["predicted_label"] = predicted_labels
y_scores = np.array(probabilities)

valid_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

mask = (
    df["true_label"].isin(valid_labels) &
    df["predicted_label"].isin(valid_labels)
)

df = df[mask].copy()
y_scores = y_scores[mask.values]

print("Predictions processed")
# END SECTION 5

# START SECTION ROC / AUC
print("\nSTEP X: Plotting ROC/AUC curve...")

label_to_int = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2
}

y_true = df["true_label"].map(label_to_int).values
y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

plt.figure()

for i, label_name in enumerate(["NEGATIVE", "NEUTRAL", "POSITIVE"]):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multiclass - DistilBERT)")
plt.legend()
plt.show()

macro_auc = roc_auc_score(y_true_bin, y_scores, average="macro")
print("Macro AUC:", macro_auc)
# END SECTION ROC / AUC

# START SECTION 6 EVALUATION
print("\nSTEP 6: Evaluating model...")

accuracy = accuracy_score(df["true_label"], df["predicted_label"])
f1 = f1_score(df["true_label"], df["predicted_label"], average="weighted")

print("\nACCURACY:", accuracy)
print("F1 SCORE:", f1)

print("\nCLASSIFICATION REPORT:")
print(classification_report(df["true_label"], df["predicted_label"]))
# END SECTION 6

# START SECTION 7 CONFUSION MATRIX
print("\nSTEP 7: Confusion matrix...")

cm = confusion_matrix(
    df["true_label"],
    df["predicted_label"],
    labels=["POSITIVE", "NEGATIVE", "NEUTRAL"]
)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["POS", "NEG", "NEU"],
    yticklabels=["POS", "NEG", "NEU"]
)

plt.title("Confusion Matrix - DistilBERT Sentiment")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
# END SECTION 7

# START SECTION 8 DISTRIBUTIONS
print("\nSTEP 8: Plots...")

plt.figure()
df["true_label"].value_counts().plot(kind="bar")
plt.title("True Sentiment Distribution")
plt.show()

plt.figure()
df["predicted_label"].value_counts().plot(kind="bar")
plt.title("Predicted Sentiment Distribution")
plt.show()

counts = pd.DataFrame({
    "True": df["true_label"].value_counts(),
    "Predicted": df["predicted_label"].value_counts()
})

counts.plot(kind="bar")
plt.title("True vs Predicted Sentiment Distribution")
plt.show()
# END SECTION 8

# START SECTION 9 FINAL SUMMARY
print("\nFINAL SUMMARY")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Macro AUC:", macro_auc)
print("\nPIPELINE COMPLETED SUCCESSFULLY")
# END SECTION 9

# SAVE RESULTS IN EXCEL
print("\nSTEP 9: Saving results to Excel...")

safe_model_name = model_id.replace("/", "_")
output_dir = f"results/first_phase_testing/{safe_model_name}"

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "predictions.xlsx")
df.to_excel(output_path, index=False)

print(f"Results saved to: {output_path}")