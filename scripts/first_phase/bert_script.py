#No training, just inference and evaluation on the test set

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from transformers import pipeline

# START SECTION 1 LOAD DATA
print("\nSTEP 1: Loading dataset...")

df = pd.read_excel("data/dataset-sentiment-analysis-preprocessed.xlsx")

# Take the relevant columns and drop rows with missing values
df = df[["Comment", "Sent. Anal. Category"]].dropna()
df.columns = ["text", "true_label"]

# Map original labels to model-compatible format
label_map = {
    "Positive": "POSITIVE",
    "Negative": "NEGATIVE",
    "Neutral": "NEUTRAL"
}

# Drop rows where mapping fails (has unexpected labels)
df["true_label"] = df["true_label"].map(label_map)
df = df.dropna(subset=["true_label"])

print("Dataset loaded:", len(df), "rows")
# END SECTION 1

# START SECTION 2 LOAD MODEL
print("\nSTEP 2: Loading pretrained model...")

model_id = "nlptown/bert-base-multilingual-uncased-sentiment"

start_time = time.time()

classifier = pipeline(
    "sentiment-analysis",
    model=model_id,
    tokenizer=model_id,
    return_all_scores=True
)

print(f"Model loaded in {time.time() - start_time:.2f} seconds")

# START SECTION 3 SAFE PREPROCESSING
print("\nSTEP 3: Preparing text safely...")

# HARD TRUNCATION
texts = (
    df["text"]
    .astype(str)
    .apply(lambda x: x[:300])   # safe character limit
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

    # r = list of 5 star probabilities
    star_probs = [0] * 5

    for item in r:
        stars = int(item["label"][0])
        star_probs[stars - 1] = item["score"]

    neg_prob = star_probs[0] + star_probs[1]
    neu_prob = star_probs[2]
    pos_prob = star_probs[3] + star_probs[4]

    prob_vector = [neg_prob, neu_prob, pos_prob]
    probabilities.append(prob_vector)

    # predicted label = max prob
    max_idx = prob_vector.index(max(prob_vector))

    if max_idx == 0:
        predicted_labels.append("NEGATIVE")
    elif max_idx == 1:
        predicted_labels.append("NEUTRAL")
    else:
        predicted_labels.append("POSITIVE")

df["predicted_label"] = predicted_labels
y_scores = np.array(probabilities)

valid_labels = ["POSITIVE", "NEGATIVE", "NEUTRAL"]

mask = (
    df["true_label"].isin(valid_labels) &
    df["predicted_label"].isin(valid_labels)
)

df = df[mask]
y_scores = y_scores[mask.values]

print("Predictions processed")
#END SECTION 5

#START SECTION ROC / AUC
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

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

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Multiclass - BERT)")
plt.legend()
plt.show()

macro_auc = roc_auc_score(y_true_bin, y_scores, average="macro")
print("Macro AUC:", macro_auc)
#END SECTION ROC / AUC

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

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["POS", "NEG", "NEU"],
    yticklabels=["POS", "NEG", "NEU"]
)

plt.title("Confusion Matrix - XLM-R Sentiment")
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
print("\nPIPELINE COMPLETED SUCCESSFULLY")
# END SECTION 9

# SAVE RESULTS IN EXCEL
print("\nSTEP 9: Saving results to Excel...")

import os
safe_model_name = model_id.replace("/", "_")
output_dir = f"results/first_phase_testing/{safe_model_name}"

os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "predictions.xlsx")
df.to_excel(output_path, index=False)

print(f"Results saved to: {output_path}")