import os
import ast
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    AutoTokenizer,
    XLMRobertaForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)


# ============================================================
# NARRATIVEGUARD
# XLM-RoBERTa FINE-TUNING
# ============================================================

MODEL_NAME = "xlm-roberta-base"

TRAIN_PATH = "dataset/tokenized/train_tokenized.csv"
VAL_PATH = "dataset/tokenized/validation_tokenized.csv"

MODEL_OUTPUT_DIR = "models/narrativeguard_xlmr"

BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
NUM_LABELS = 8
MAX_LENGTH = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NARRATIVEGUARD - XLM-RoBERTa FINE-TUNING")
print("=" * 70)

print()
print("Device       :", DEVICE)
print("Model        :", MODEL_NAME)
print("Batch size   :", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("Epochs       :", EPOCHS)
print("Num labels   :", NUM_LABELS)


# ============================================================
# 1. DATASET CLASS
# ============================================================

class NarrativeGuardDataset(Dataset):

    def __init__(self, dataframe):

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for _, row in dataframe.iterrows():

            input_ids = ast.literal_eval(
                row["input_ids"]
            )

            attention_mask = ast.literal_eval(
                row["attention_mask"]
            )

            label = int(row["label"])

            self.input_ids.append(
                torch.tensor(
                    input_ids,
                    dtype=torch.long
                )
            )

            self.attention_masks.append(
                torch.tensor(
                    attention_mask,
                    dtype=torch.long
                )
            )

            self.labels.append(
                torch.tensor(
                    label,
                    dtype=torch.long
                )
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
# 2. LOAD DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TOKENIZED DATA")
print("=" * 70)

if not os.path.exists(TRAIN_PATH):
    raise FileNotFoundError(
        f"Training file not found: {TRAIN_PATH}"
    )

if not os.path.exists(VAL_PATH):
    raise FileNotFoundError(
        f"Validation file not found: {VAL_PATH}"
    )

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)

print("Training samples   :", len(train_df))
print("Validation samples :", len(val_df))


# ============================================================
# 3. VALIDATE LABELS
# ============================================================

expected_labels = set(range(NUM_LABELS))

train_labels = set(
    train_df["label"].astype(int).unique()
)

val_labels = set(
    val_df["label"].astype(int).unique()
)

print()
print("Training labels   :", sorted(train_labels))
print("Validation labels :", sorted(val_labels))

if train_labels != expected_labels:
    raise ValueError(
        "Training dataset does not contain all 8 labels."
    )

if not val_labels.issubset(expected_labels):
    raise ValueError(
        "Validation dataset contains invalid labels."
    )

print("PASS - All 8 labels are valid.")


# ============================================================
# 4. CREATE PYTORCH DATASETS
# ============================================================

print()
print("=" * 70)
print("CREATING PYTORCH DATASETS")
print("=" * 70)

train_dataset = NarrativeGuardDataset(train_df)
val_dataset = NarrativeGuardDataset(val_df)

print("Training dataset created.")
print("Validation dataset created.")


# ============================================================
# 5. CREATE DATALOADERS
# ============================================================

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

print()
print("Training batches   :", len(train_loader))
print("Validation batches :", len(val_loader))


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

print()
print("=" * 70)
print("LOADING TOKENIZER")
print("=" * 70)

try:

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True
    )

    print("Tokenizer loaded from local cache.")

except Exception as e:

    print()
    print("WARNING: Local tokenizer cache was not found.")
    print()
    print("The tokenizer must be downloaded once while")
    print("internet access to Hugging Face is available.")
    print()
    print("Error:")
    print(e)

    raise


# ============================================================
# 7. LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING XLM-RoBERTa MODEL")
print("=" * 70)

try:

    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        local_files_only=True
    )

    print("XLM-R model loaded from local cache.")

except Exception as e:

    print()
    print("WARNING: XLM-R model is not available locally.")
    print()
    print("The model must be downloaded once while")
    print("internet access to Hugging Face is available.")
    print()
    print("Error:")
    print(e)

    raise


model.to(DEVICE)

print("Model moved to:", DEVICE)


# ============================================================
# 8. OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)

print()
print("Optimizer      : AdamW")
print("Learning rate  :", LEARNING_RATE)


# ============================================================
# 9. VALIDATION FUNCTION
# ============================================================

def evaluate(model, dataloader):

    model.eval()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(
                DEVICE
            )

            attention_mask = batch["attention_mask"].to(
                DEVICE
            )

            labels = batch["labels"].to(
                DEVICE
            )

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

    average_loss = (
        total_loss /
        len(dataloader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0
        )
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1
    )


# ============================================================
# 10. TRAINING
# ============================================================

print()
print("=" * 70)
print("STARTING XLM-R TRAINING")
print("=" * 70)

best_f1 = -1.0

for epoch in range(EPOCHS):

    print()
    print("=" * 70)
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("=" * 70)

    model.train()

    total_train_loss = 0.0

    for batch_index, batch in enumerate(
        train_loader
    ):

        input_ids = batch["input_ids"].to(
            DEVICE
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE
        )

        labels = batch["labels"].to(
            DEVICE
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_train_loss += loss.item()

        # Progress
        if (
            (batch_index + 1) % 10 == 0
            or
            (batch_index + 1) == len(train_loader)
        ):

            print(
                f"Batch "
                f"{batch_index + 1:3d}/"
                f"{len(train_loader)} | "
                f"Loss: {loss.item():.4f}"
            )

    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = evaluate(
        model,
        val_loader
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("-" * 70)
    print("EPOCH RESULTS")
    print("-" * 70)

    print(
        f"Training Loss       : "
        f"{average_train_loss:.4f}"
    )

    print(
        f"Validation Loss     : "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Macro Precision     : "
        f"{val_precision:.4f}"
    )

    print(
        f"Macro Recall        : "
        f"{val_recall:.4f}"
    )

    print(
        f"Macro F1            : "
        f"{val_f1:.4f}"
    )

    print("-" * 70)


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_f1:

        best_f1 = val_f1

        print()
        print("NEW BEST MODEL")

        print(
            f"Best Validation Macro F1: "
            f"{best_f1:.4f}"
        )

        os.makedirs(
            MODEL_OUTPUT_DIR,
            exist_ok=True
        )

        # Save model
        model.save_pretrained(
            MODEL_OUTPUT_DIR
        )

        # Save tokenizer WITHOUT downloading
        tokenizer.save_pretrained(
            MODEL_OUTPUT_DIR
        )

        print()
        print("Best model saved to:")
        print(MODEL_OUTPUT_DIR)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"Best Validation Macro F1: "
    f"{best_f1:.4f}"
)

print()
print("Model directory:")
print(MODEL_OUTPUT_DIR)

print()
print("Training dataset : 800")
print("Validation       : 162")
print("Classes          : 8")

print("=" * 70)