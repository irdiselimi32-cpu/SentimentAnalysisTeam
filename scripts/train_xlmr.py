import os
import json
import argparse
import random

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    set_seed
)


# ============================================================
# ARGUMENTET
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--pipeline",
    required=True,
    choices=[
        "raw",
        "minimal",
        "traditional",
        "albanian_aware"
    ]
)

args = parser.parse_args()

PIPELINE = args.pipeline


# ============================================================
# KONFIGURIMI I EKSPERIMENTIT
# ============================================================

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

DATA_FOLDER = "data/experiments"

OUTPUT_FOLDER = os.path.join(
    "results",
    PIPELINE
)

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# PARAMETRAT FIKS
# ============================================================

SEED = 42

MAX_LENGTH = 128

LEARNING_RATE = 2e-5

EPOCHS = 3

# GPU ka vetëm 4 GB VRAM.
# Përdorim batch të vogël fizik.
TRAIN_BATCH_SIZE = 4

EVAL_BATCH_SIZE = 8

# 4 * 4 = effective batch size 16
GRADIENT_ACCUMULATION_STEPS = 4

WEIGHT_DECAY = 0.01


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

set_seed(SEED)


# ============================================================
# INFORMACION MBI HARDWARE
# ============================================================

print("=" * 70)
print("XLM-RoBERTa SENTIMENT EXPERIMENT")
print("=" * 70)

print(f"\nPipeline: {PIPELINE}")
print(f"Model: {MODEL_NAME}")

print("\nHardware:")

print(
    "CUDA available:",
    torch.cuda.is_available()
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA version:",
        torch.version.cuda
    )

else:

    print("WARNING: Training will run on CPU.")


# ============================================================
# LABELS
# ============================================================

LABEL2ID = {
    "NEGATIVE": 0,
    "NEUTRAL": 1,
    "POSITIVE": 2
}

ID2LABEL = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE"
}


# ============================================================
# LOAD DATA
# ============================================================

def load_split(split):

    path = os.path.join(
        DATA_FOLDER,
        f"{split}_{PIPELINE}.xlsx"
    )

    df = pd.read_excel(path)

    # Excel mund t'i lexojë tekstet bosh si NaN.
    # Nuk i heqim, sepse duam të ruajmë të njëjtat IDs.
    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
    )

    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    unknown_labels = (
        set(df["label"].unique())
        - set(LABEL2ID.keys())
    )

    if unknown_labels:

        raise ValueError(
            f"Unknown labels: {unknown_labels}"
        )

    df["labels"] = (
        df["label"]
        .map(LABEL2ID)
    )

    return df


train_df = load_split("train")
validation_df = load_split("validation")
test_df = load_split("test")


print("\nDataset sizes:")

print(
    "Train:",
    len(train_df)
)

print(
    "Validation:",
    len(validation_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# HUGGING FACE DATASETS
# ============================================================

train_dataset = Dataset.from_pandas(
    train_df[
        [
            "text",
            "labels"
        ]
    ],
    preserve_index=False
)

validation_dataset = Dataset.from_pandas(
    validation_df[
        [
            "text",
            "labels"
        ]
    ],
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df[
        [
            "text",
            "labels"
        ]
    ],
    preserve_index=False
)


# ============================================================
# TOKENIZER
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


def tokenize_function(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH
    )


print("Tokenizing datasets...")


train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)


# ============================================================
# DYNAMIC PADDING
# ============================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "f1_macro": macro_f1,
        "f1_weighted": weighted_f1
    }


# ============================================================
# TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(

    output_dir=os.path.join(
        OUTPUT_FOLDER,
        "checkpoints"
    ),

    # -----------------------
    # Epochs
    # -----------------------

    num_train_epochs=EPOCHS,

    # -----------------------
    # Batch sizes
    # -----------------------

    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,

    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    # -----------------------
    # Optimizer
    # -----------------------

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    # -----------------------
    # Evaluation
    # -----------------------

    eval_strategy="epoch",

    save_strategy="epoch",

    # -----------------------
    # Best model
    # -----------------------

    load_best_model_at_end=True,

    metric_for_best_model="f1_macro",

    greater_is_better=True,

    # -----------------------
    # Logging
    # -----------------------

    logging_strategy="steps",

    logging_steps=100,

    # -----------------------
    # Reproducibility
    # -----------------------

    seed=SEED,

    data_seed=SEED,

    # -----------------------
    # Checkpoints
    # -----------------------

    save_total_limit=1,

    # -----------------------
    # GPU
    # -----------------------

    fp16=False,

    # -----------------------
    # Reporting
    # -----------------------

    report_to="none"
)


# ============================================================
# TRAINER
# ============================================================

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=validation_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print(f"STARTING TRAINING: {PIPELINE.upper()}")
print("=" * 70)

trainer.train()


# ============================================================
# TEST SET
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

test_output = trainer.predict(
    test_dataset
)

logits = test_output.predictions

true_labels = test_output.label_ids

predictions = np.argmax(
    logits,
    axis=-1
)


# ============================================================
# FINAL METRICS
# ============================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)

macro_f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)


precision, recall, f1, support = (
    precision_recall_fscore_support(
        true_labels,
        predictions,
        labels=[0, 1, 2],
        zero_division=0
    )
)


print("\nFINAL RESULTS")

print(
    f"Accuracy:    {accuracy:.4f}"
)

print(
    f"Macro F1:    {macro_f1:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    true_labels,
    predictions,
    labels=[0, 1, 2],
    target_names=[
        "NEGATIVE",
        "NEUTRAL",
        "POSITIVE"
    ],
    digits=4,
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    true_labels,
    predictions,
    labels=[0, 1, 2]
)

print("\nCONFUSION MATRIX")
print(cm)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {

    "pipeline": PIPELINE,

    "model": MODEL_NAME,

    "seed": SEED,

    "max_length": MAX_LENGTH,

    "epochs": EPOCHS,

    "learning_rate": LEARNING_RATE,

    "train_batch_size": TRAIN_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "effective_batch_size":
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS,

    "accuracy": float(accuracy),

    "macro_f1": float(macro_f1),

    "weighted_f1": float(weighted_f1)
}


with open(
    os.path.join(
        OUTPUT_FOLDER,
        "metrics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# SAVE CLASS RESULTS
# ============================================================

class_results = pd.DataFrame({

    "class": [
        "NEGATIVE",
        "NEUTRAL",
        "POSITIVE"
    ],

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "support": support
})


class_results.to_excel(
    os.path.join(
        OUTPUT_FOLDER,
        "class_results.xlsx"
    ),
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(

    cm,

    index=[
        "TRUE_NEGATIVE",
        "TRUE_NEUTRAL",
        "TRUE_POSITIVE"
    ],

    columns=[
        "PRED_NEGATIVE",
        "PRED_NEUTRAL",
        "PRED_POSITIVE"
    ]
)


cm_df.to_excel(
    os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.xlsx"
    )
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "comment_id",
        "text",
        "label"
    ]
].copy()


prediction_df["predicted_label"] = [
    ID2LABEL[int(x)]
    for x in predictions
]


prediction_df["correct"] = (
    prediction_df["label"]
    ==
    prediction_df["predicted_label"]
)


prediction_df.to_excel(
    os.path.join(
        OUTPUT_FOLDER,
        "test_predictions.xlsx"
    ),
    index=False
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

trainer.save_model(
    os.path.join(
        OUTPUT_FOLDER,
        "best_model"
    )
)

tokenizer.save_pretrained(
    os.path.join(
        OUTPUT_FOLDER,
        "best_model"
    )
)


print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nResults saved in: {OUTPUT_FOLDER}"
)