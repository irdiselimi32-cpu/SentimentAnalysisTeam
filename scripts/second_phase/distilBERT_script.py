import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

def torch_softmax_numpy(logits):
    logits = np.array(logits)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# =========================
# 1. LOAD DATA
# =========================
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
df = df.dropna(subset=["true_label"]).copy()

# optional light preprocessing
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

# =========================
# 3. ENCODE LABELS
# =========================
print("\nSTEP 3: Encoding labels...")

label_to_id = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2
}
id_to_label = {v: k for k, v in label_to_id.items()}

for split_df in [train_df, val_df, test_df]:
    split_df["label"] = split_df["true_label"].map(label_to_id)

# keep only needed columns
train_df = train_df[["text", "label", "true_label"]].copy()
val_df = val_df[["text", "label", "true_label"]].copy()
test_df = test_df[["text", "label", "true_label"]].copy()

# =========================
# 4. CONVERT TO HF DATASETS
# =========================
print("\nSTEP 4: Converting to Hugging Face datasets...")

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

# =========================
# 5. LOAD TOKENIZER + MODEL
# =========================
print("\nSTEP 5: Loading tokenizer and model...")

model_id = "Yuu-Xie/distilbert-base-multilingual-cased-sentiment"
# You can replace with:
# model_id = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
# model_id = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    num_labels=3,
    id2label=id_to_label,
    label2id=label_to_id
)

# =========================
# 6. TOKENIZATION
# =========================
print("\nSTEP 6: Tokenizing datasets...")

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds = val_ds.map(tokenize_function, batched=True)
test_ds = test_ds.map(tokenize_function, batched=True)

columns_to_keep = ["input_ids", "attention_mask", "label"]
if "token_type_ids" in train_ds.column_names:
    columns_to_keep.append("token_type_ids")

train_ds.set_format(type="torch", columns=columns_to_keep)
val_ds.set_format(type="torch", columns=columns_to_keep)
test_ds.set_format(type="torch", columns=columns_to_keep)

# =========================
# 7. METRICS FUNCTION
# =========================
print("\nSTEP 7: Preparing metrics...")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_weighted": f1_score(labels, preds, average="weighted")
    }

# =========================
# 8. TRAINING ARGUMENTS
# =========================
print("\nSTEP 8: Setting training arguments...")

training_args = TrainingArguments(
    output_dir=f"models/second_phase/{model_id.replace('/', '_')}",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    dataloader_pin_memory=False
)

# =========================
# 9. TRAINER
# =========================
print("\nSTEP 9: Creating trainer...")

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# =========================
# 10. TRAIN
# =========================
print("\nSTEP 10: Fine-tuning model...")

start_time = time.time()
trainer.train()
print(f"Training completed in {time.time() - start_time:.2f} seconds")

# save best model + tokenizer
trainer.save_model(f"models/second_phase/{model_id.replace('/', '_')}/best_model")
tokenizer.save_pretrained(f"models/second_phase/{model_id.replace('/', '_')}/best_model")

# =========================
# 11. TEST EVALUATION
# =========================
print("\nSTEP 11: Evaluating on test set...")

test_output = trainer.predict(test_ds)

logits = test_output.predictions
y_scores = torch_softmax_numpy(logits)
y_pred_ids = np.argmax(logits, axis=1)
y_true_ids = test_output.label_ids

test_df = test_df.reset_index(drop=True).copy()
test_df["predicted_label"] = [id_to_label[i] for i in y_pred_ids]

accuracy = accuracy_score(y_true_ids, y_pred_ids)
f1 = f1_score(y_true_ids, y_pred_ids, average="weighted")

print("\nTEST ACCURACY:", accuracy)
print("TEST F1 SCORE:", f1)

print("\nCLASSIFICATION REPORT:")
print(classification_report(
    [id_to_label[i] for i in y_true_ids],
    [id_to_label[i] for i in y_pred_ids]
))

# =========================
# 12. ROC / AUC
# =========================
print("\nSTEP 12: Plotting ROC/AUC...")

y_true_bin = label_binarize(y_true_ids, classes=[0, 1, 2])

plt.figure()

for i, label_name in enumerate(["NEGATIVE", "NEUTRAL", "POSITIVE"]):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{label_name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Fine-tuned DistilBERT)")
plt.legend()
plt.show()

macro_auc = roc_auc_score(y_true_bin, y_scores, average="macro")
print("Macro AUC:", macro_auc)

# =========================
# 13. CONFUSION MATRIX
# =========================
print("\nSTEP 13: Confusion matrix...")

cm = confusion_matrix(
    [id_to_label[i] for i in y_true_ids],
    [id_to_label[i] for i in y_pred_ids],
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
plt.title("Confusion Matrix - Fine-tuned DistilBERT")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# =========================
# 14. SAVE TEST RESULTS
# =========================
print("\nSTEP 14: Saving test predictions...")

output_dir = f"results/second_phase_testing/{model_id.replace('/', '_')}"
os.makedirs(output_dir, exist_ok=True)

test_df.to_excel(os.path.join(output_dir, "test_predictions.xlsx"), index=False)

print("Done.")