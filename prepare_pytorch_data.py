import pandas as pd
import ast
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# NARRATIVEGUARD
# PYTORCH DATASET + DATALOADER PREPARATION
# ============================================================

TRAIN_PATH = "dataset/tokenized/train_tokenized.csv"
VAL_PATH = "dataset/tokenized/validation_tokenized.csv"
TEST_PATH = "dataset/tokenized/test_tokenized.csv"

BATCH_SIZE = 8

print("=" * 70)
print("NARRATIVEGUARD - PYTORCH DATA PREPARATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. DATASET CLASS
# ------------------------------------------------------------

class NarrativeGuardDataset(Dataset):

    def __init__(self, dataframe):

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for _, row in dataframe.iterrows():

            # CSV stores lists as strings, so convert them back
            input_ids = ast.literal_eval(row["input_ids"])
            attention_mask = ast.literal_eval(row["attention_mask"])

            self.input_ids.append(
                torch.tensor(input_ids, dtype=torch.long)
            )

            self.attention_masks.append(
                torch.tensor(attention_mask, dtype=torch.long)
            )

            self.labels.append(
                torch.tensor(
                    int(row["label"]),
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


# ------------------------------------------------------------
# 2. LOAD TOKENIZED DATA
# ------------------------------------------------------------

print("\nLoading tokenized datasets...")

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print("Training samples   :", len(train_df))
print("Validation samples :", len(val_df))
print("Testing samples    :", len(test_df))


# ------------------------------------------------------------
# 3. CREATE PYTORCH DATASETS
# ------------------------------------------------------------

train_dataset = NarrativeGuardDataset(train_df)
val_dataset = NarrativeGuardDataset(val_df)
test_dataset = NarrativeGuardDataset(test_df)

print("\nPyTorch datasets created successfully.")


# ------------------------------------------------------------
# 4. CREATE DATALOADERS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 5. INSPECT ONE BATCH
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATALOADER BATCH INSPECTION")
print("=" * 70)

batch = next(iter(train_loader))

print("Input IDs shape      :", batch["input_ids"].shape)
print("Attention mask shape:", batch["attention_mask"].shape)
print("Labels shape         :", batch["labels"].shape)

print("\nSample labels:")
print(batch["labels"])

print("\nInput IDs datatype:")
print(batch["input_ids"].dtype)

print("\nAttention mask datatype:")
print(batch["attention_mask"].dtype)

print("\nLabels datatype:")
print(batch["labels"].dtype)


# ------------------------------------------------------------
# 6. VERIFY TOKEN LENGTH
# ------------------------------------------------------------

if batch["input_ids"].shape[1] == 256:
    print("\nPASS - Token sequence length is 256.")
else:
    print("\nWARNING - Unexpected token sequence length.")


# ------------------------------------------------------------
# 7. VERIFY LABEL RANGE
# ------------------------------------------------------------

all_labels = set(train_df["label"].unique())

print("\nTraining labels found:")
print(sorted(all_labels))

if all_labels == set(range(8)):
    print("PASS - All 8 labels are present.")
else:
    print("WARNING - Expected labels 0-7.")


# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PYTORCH DATA PREPARATION COMPLETED")
print("=" * 70)

print("\nBatch size:", BATCH_SIZE)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Testing batches:", len(test_loader))

print("\nReady for XLM-RoBERTa fine-tuning.")
print("=" * 70)