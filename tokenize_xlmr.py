import pandas as pd
import json
import os

from transformers import AutoTokenizer

# ============================================================
# NARRATIVEGUARD
# XLM-RoBERTa TOKENIZATION
# ============================================================

MODEL_NAME = "xlm-roberta-base"

INPUTS = {
    "train": "dataset/processed/train_preprocessed.csv",
    "validation": "dataset/processed/validation_preprocessed.csv",
    "test": "dataset/processed/test_preprocessed.csv"
}

OUTPUT_DIR = "dataset/tokenized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH = 256

print("=" * 70)
print("NARRATIVEGUARD - XLM-R TOKENIZATION")
print("=" * 70)

print("\nLoading tokenizer:")
print(MODEL_NAME)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully.")

# ------------------------------------------------------------
# 1. CREATE LABEL MAPPING
# ------------------------------------------------------------

all_labels = set()

for input_path in INPUTS.values():

    df = pd.read_csv(input_path)

    all_labels.update(
        df["scam_type"].unique()
    )

labels = sorted(all_labels)

label_mapping = {
    label: index
    for index, label in enumerate(labels)
}

print("\n" + "=" * 70)
print("LABEL MAPPING")
print("=" * 70)

for label, index in label_mapping.items():
    print(f"{index} -> {label}")

mapping_path = os.path.join(
    OUTPUT_DIR,
    "label_mapping.json"
)

with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(
        label_mapping,
        f,
        indent=4,
        ensure_ascii=False
    )

# ------------------------------------------------------------
# 2. TOKENIZE EACH SPLIT
# ------------------------------------------------------------

for split_name, input_path in INPUTS.items():

    print("\n" + "=" * 70)
    print(f"TOKENIZING: {split_name.upper()}")
    print("=" * 70)

    df = pd.read_csv(input_path)

    # Convert labels to integers
    df["label"] = df["scam_type"].map(label_mapping)

    # Safety check
    if df["label"].isna().any():
        raise ValueError(
            f"Unknown labels found in {split_name}"
        )

    # Tokenize conversations
    encoded = tokenizer(
        df["conversation"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

    # Store tokenized representations
    df["input_ids"] = [
        ids for ids in encoded["input_ids"]
    ]

    df["attention_mask"] = [
        mask for mask in encoded["attention_mask"]
    ]

    # Save tokenized dataset
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_tokenized.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("Samples        :", len(df))
    print("Input ID length:", len(df["input_ids"].iloc[0]))
    print("Attention mask :", len(df["attention_mask"].iloc[0]))
    print("Saved to       :", output_path)

# ------------------------------------------------------------
# 3. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TOKENIZATION COMPLETED")
print("=" * 70)

print("\nLabel mapping saved to:")
print(mapping_path)

print("\nTokenized datasets saved in:")
print(OUTPUT_DIR)

print("\n" + "=" * 70)