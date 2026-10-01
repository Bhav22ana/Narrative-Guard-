import os
import ast
import json
import time

import pandas as pd
import torch
import mlflow
import mlflow.pytorch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    XLMRobertaForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# NARRATIVEGUARD
# XLM-RoBERTa FINAL TRAINING + MLFLOW TRACKING
# ============================================================

MODEL_NAME = "xlm-roberta-base"

TRAIN_PATH = "dataset/tokenized/train_tokenized.csv"
VAL_PATH = "dataset/tokenized/validation_tokenized.csv"
TEST_PATH = "dataset/tokenized/test_tokenized.csv"

MODEL_OUTPUT_DIR = "models/narrativeguard_xlmr"

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
NUM_LABELS = 8
MAX_LENGTH = 256

MLFLOW_EXPERIMENT = "NarrativeGuard"
MLFLOW_RUN_NAME = "XLM-RoBERTa-Final"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

LABEL_MAPPING = {
    0: "bank_impersonation_scam",
    1: "genuine_financial_conversation",
    2: "investment_scam",
    3: "job_scam",
    4: "kyc_scam",
    5: "loan_scam",
    6: "otp_payment_scam",
    7: "refund_scam"
}


# ============================================================
# 1. MLFLOW SETUP
# ============================================================

# Use the same local tracking directory used by the MLflow UI.
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ============================================================
# 2. DATASET CLASS
# ============================================================

class NarrativeGuardDataset(Dataset):

    def __init__(self, dataframe):
        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for _, row in dataframe.iterrows():

            input_ids = ast.literal_eval(row["input_ids"])
            attention_mask = ast.literal_eval(row["attention_mask"])
            label = int(row["label"])

            if len(input_ids) != MAX_LENGTH:
                raise ValueError(
                    f"Input ID length {len(input_ids)} != MAX_LENGTH {MAX_LENGTH}"
                )

            if len(attention_mask) != MAX_LENGTH:
                raise ValueError(
                    f"Attention mask length {len(attention_mask)} != MAX_LENGTH {MAX_LENGTH}"
                )

            self.input_ids.append(
                torch.tensor(input_ids, dtype=torch.long)
            )

            self.attention_masks.append(
                torch.tensor(attention_mask, dtype=torch.long)
            )

            self.labels.append(
                torch.tensor(label, dtype=torch.long)
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index]
        }


# ============================================================
# 3. EVALUATION FUNCTION
# ============================================================

def evaluate(model, dataloader):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            predictions = torch.argmax(
                outputs.logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    average_loss = total_loss / len(dataloader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
        all_labels,
        all_predictions
    )


# ============================================================
# 4. MAIN
# ============================================================

print("=" * 70)
print("NARRATIVEGUARD - FINAL XLM-RoBERTa TRAINING")
print("=" * 70)

print()
print("Device        :", DEVICE)
print("Model         :", MODEL_NAME)
print("Batch size    :", BATCH_SIZE)
print("Learning rate :", LEARNING_RATE)
print("Epochs        :", EPOCHS)
print("Num labels    :", NUM_LABELS)
print("Max length    :", MAX_LENGTH)
print("MLflow run    :", MLFLOW_RUN_NAME)


# ============================================================
# 5. CHECK INPUT FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING DATA FILES")
print("=" * 70)

for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    print("FOUND:", path)


# ============================================================
# 6. LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print()
print("Training samples   :", len(train_df))
print("Validation samples :", len(val_df))
print("Testing samples    :", len(test_df))

if len(train_df) != 800:
    raise ValueError(
        f"Expected 800 training samples, found {len(train_df)}"
    )

if len(val_df) != 162:
    raise ValueError(
        f"Expected 162 validation samples, found {len(val_df)}"
    )

if len(test_df) != 162:
    raise ValueError(
        f"Expected 162 test samples, found {len(test_df)}"
    )


# ============================================================
# 7. VALIDATE LABELS
# ============================================================

expected_labels = set(range(NUM_LABELS))

for name, dataframe in [
    ("training", train_df),
    ("validation", val_df),
    ("testing", test_df)
]:

    labels = set(
        dataframe["label"].astype(int).unique()
    )

    print(f"{name.capitalize()} labels:", sorted(labels))

    if not labels.issubset(expected_labels):
        raise ValueError(
            f"{name.capitalize()} dataset contains invalid labels."
        )

if set(train_df["label"].astype(int).unique()) != expected_labels:
    raise ValueError(
        "Training dataset does not contain all 8 labels."
    )

print("PASS - Label validation completed.")


# ============================================================
# 8. CREATE DATASETS / DATALOADERS
# ============================================================

train_dataset = NarrativeGuardDataset(train_df)
val_dataset = NarrativeGuardDataset(val_df)
test_dataset = NarrativeGuardDataset(test_df)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Training batches   :", len(train_loader))
print("Validation batches :", len(val_loader))
print("Testing batches    :", len(test_loader))


# ============================================================
# 9. START MLFLOW RUN
# ============================================================

with mlflow.start_run(run_name=MLFLOW_RUN_NAME) as run:

    run_id = run.info.run_id

    print()
    print("=" * 70)
    print("MLFLOW RUN STARTED")
    print("=" * 70)
    print("Experiment :", MLFLOW_EXPERIMENT)
    print("Run name   :", MLFLOW_RUN_NAME)
    print("Run ID     :", run_id)

    # --------------------------------------------------------
    # Log parameters
    # --------------------------------------------------------

    mlflow.log_params({
        "model_name": MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        "optimizer": "AdamW",
        "device": str(DEVICE),
        "total_samples": len(train_df) + len(val_df) + len(test_df),
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "training_dataset": "balanced_800",
        "validation_dataset": "162",
        "test_dataset": "162"
    })

    mlflow.set_tags({
        "project": "NarrativeGuard",
        "task": "multilingual financial scam classification",
        "model_family": "XLM-RoBERTa",
        "dataset_version": "final_balanced",
        "stage": "final_training"
    })

    # --------------------------------------------------------
    # Save training configuration
    # --------------------------------------------------------

    os.makedirs("mlflow_outputs", exist_ok=True)

    config_path = "mlflow_outputs/training_config.json"

    training_config = {
        "model_name": MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_labels": NUM_LABELS,
        "max_length": MAX_LENGTH,
        "optimizer": "AdamW",
        "device": str(DEVICE),
        "train_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df)
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=4)

    mlflow.log_artifact(config_path, artifact_path="configuration")

    # --------------------------------------------------------
    # Save label mapping
    # --------------------------------------------------------

    label_mapping_path = "mlflow_outputs/label_mapping.json"

    with open(label_mapping_path, "w", encoding="utf-8") as f:
        json.dump(LABEL_MAPPING, f, indent=4)

    mlflow.log_artifact(
        label_mapping_path,
        artifact_path="configuration"
    )

    # ========================================================
    # LOAD TOKENIZER
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING TOKENIZER")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    print("Tokenizer loaded from local cache.")

    # ========================================================
    # LOAD MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING XLM-RoBERTa MODEL")
    print("=" * 70)

    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        local_files_only=True
    )

    model.to(DEVICE)

    print("XLM-R model loaded.")
    print("Model moved to:", DEVICE)

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # TRAINING
    # ========================================================

    best_f1 = -1.0
    training_start = time.time()

    for epoch in range(EPOCHS):

        print()
        print("=" * 70)
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 70)

        model.train()

        total_train_loss = 0.0

        for batch_index, batch in enumerate(train_loader):

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            if (
                (batch_index + 1) % 10 == 0
                or
                (batch_index + 1) == len(train_loader)
            ):
                print(
                    f"Batch {batch_index + 1:3d}/"
                    f"{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        average_train_loss = (
            total_train_loss / len(train_loader)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        (
            val_loss,
            val_accuracy,
            val_precision,
            val_recall,
            val_f1,
            _,
            _
        ) = evaluate(
            model,
            val_loader
        )

        print()
        print("-" * 70)
        print("EPOCH RESULTS")
        print("-" * 70)
        print(f"Training Loss       : {average_train_loss:.4f}")
        print(f"Validation Loss     : {val_loss:.4f}")
        print(f"Validation Accuracy : {val_accuracy:.4f}")
        print(f"Macro Precision     : {val_precision:.4f}")
        print(f"Macro Recall        : {val_recall:.4f}")
        print(f"Macro F1            : {val_f1:.4f}")
        print("-" * 70)

        # ----------------------------------------------------
        # Log epoch metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metrics({
            "train_loss": average_train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "validation_precision_macro": val_precision,
            "validation_recall_macro": val_recall,
            "validation_f1_macro": val_f1
        }, step=epoch + 1)

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_f1 > best_f1:

            best_f1 = val_f1

            print()
            print("NEW BEST MODEL")
            print(f"Best Validation Macro F1: {best_f1:.4f}")

            os.makedirs(
                MODEL_OUTPUT_DIR,
                exist_ok=True
            )

            model.save_pretrained(
                MODEL_OUTPUT_DIR
            )

            tokenizer.save_pretrained(
                MODEL_OUTPUT_DIR
            )

    # ========================================================
    # TRAINING DURATION
    # ========================================================

    training_duration = time.time() - training_start

    mlflow.log_metric(
        "training_duration_seconds",
        training_duration
    )

    # ========================================================
    # RELOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 70)
    print("LOADING BEST MODEL FOR FINAL TEST")
    print("=" * 70)

    best_model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_OUTPUT_DIR,
        local_files_only=True
    )

    best_model.to(DEVICE)

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_labels,
        test_predictions
    ) = evaluate(
        best_model,
        test_loader
    )

    print()
    print("=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Test Loss            : {test_loss:.4f}")
    print(f"Test Accuracy        : {test_accuracy:.4f}")
    print(f"Test Macro Precision : {test_precision:.4f}")
    print(f"Test Macro Recall    : {test_recall:.4f}")
    print(f"Test Macro F1        : {test_f1:.4f}")

    # --------------------------------------------------------
    # Log final test metrics
    # --------------------------------------------------------

    mlflow.log_metrics({
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_precision_macro": test_precision,
        "test_recall_macro": test_recall,
        "test_f1_macro": test_f1
    })

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    class_names = [
        LABEL_MAPPING[i]
        for i in range(NUM_LABELS)
    ]

    report = classification_report(
        test_labels,
        test_predictions,
        labels=list(range(NUM_LABELS)),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)

    report_path = "mlflow_outputs/classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    mlflow.log_artifact(
        report_path,
        artifact_path="evaluation"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    cm = confusion_matrix(
        test_labels,
        test_predictions,
        labels=list(range(NUM_LABELS))
    )

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest")
    plt.title("NarrativeGuard - XLM-RoBERTa Confusion Matrix")
    plt.colorbar()

    tick_marks = range(NUM_LABELS)

    plt.xticks(
        tick_marks,
        class_names,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    confusion_matrix_path = (
        "mlflow_outputs/confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    mlflow.log_artifact(
        confusion_matrix_path,
        artifact_path="evaluation"
    )

    # ========================================================
    # LOG FINAL MODEL ARTIFACTS
    # ========================================================

    mlflow.log_artifacts(
        MODEL_OUTPUT_DIR,
        artifact_path="narrativeguard_model"
    )

    # Also log the model through MLflow's PyTorch flavor.
    # This creates an MLflow model artifact for later loading/
    # registration while the Hugging Face files above remain
    # directly available.
    try:
        mlflow.pytorch.log_model(
            best_model,
            artifact_path="pytorch_model"
        )
        print("MLflow PyTorch model logged successfully.")
    except Exception as e:
        print()
        print("WARNING: PyTorch model flavor logging failed.")
        print("The Hugging Face model artifacts were still logged.")
        print("Reason:", e)

    # ========================================================
    # FINAL TAGS
    # ========================================================

    mlflow.set_tags({
        "best_validation_macro_f1": f"{best_f1:.4f}",
        "final_test_macro_f1": f"{test_f1:.4f}",
        "status": "completed"
    })

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("MLFLOW RUN COMPLETED")
    print("=" * 70)

    print("Run Name :", MLFLOW_RUN_NAME)
    print("Run ID   :", run_id)

    print()
    print(f"Best Validation Macro F1 : {best_f1:.4f}")
    print(f"Test Accuracy            : {test_accuracy:.4f}")
    print(f"Test Macro Precision     : {test_precision:.4f}")
    print(f"Test Macro Recall        : {test_recall:.4f}")
    print(f"Test Macro F1            : {test_f1:.4f}")

    print()
    print("Model directory:")
    print(MODEL_OUTPUT_DIR)

    print()
    print("Training dataset : 800")
    print("Validation       : 162")
    print("Testing          : 162")
    print("Classes          : 8")

    print("=" * 70)