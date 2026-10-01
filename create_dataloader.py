import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast

DATA_DIR = "dataset/tokenized"


class NarrativeGuardDataset(Dataset):

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        input_ids = torch.tensor(
            ast.literal_eval(row["input_ids"]),
            dtype=torch.long
        )

        attention_mask = torch.tensor(
            ast.literal_eval(row["attention_mask"]),
            dtype=torch.long
        )

        label = torch.tensor(
            int(row["label"]),
            dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label
        }


# Load datasets
train_dataset = NarrativeGuardDataset(
    f"{DATA_DIR}/train_encoded.csv"
)

validation_dataset = NarrativeGuardDataset(
    f"{DATA_DIR}/validation_encoded.csv"
)

test_dataset = NarrativeGuardDataset(
    f"{DATA_DIR}/test_encoded.csv"
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False
)


print("=" * 60)
print("NARRATIVEGUARD PYTORCH DATASET")
print("=" * 60)

print("\nDataset sizes:")
print("Training:", len(train_dataset))
print("Validation:", len(validation_dataset))
print("Testing:", len(test_dataset))

# Test one batch
batch = next(iter(train_loader))

print("\nSample batch:")
print("input_ids shape:", batch["input_ids"].shape)
print("attention_mask shape:", batch["attention_mask"].shape)
print("labels shape:", batch["labels"].shape)

print("\nLabels:")
print(batch["labels"])

print("\n" + "=" * 60)
print("PYTORCH DATASET READY")
print("=" * 60)