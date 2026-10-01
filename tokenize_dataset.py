import pandas as pd
from transformers import AutoTokenizer
import os

MODEL_NAME = "xlm-roberta-base"

INPUT_DIR = "dataset/processed"
OUTPUT_DIR = "dataset/tokenized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("NARRATIVEGUARD TOKENIZATION")
print("=" * 60)

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded:", MODEL_NAME)


def tokenize_file(input_file, output_file):

    df = pd.read_csv(input_file)

    # Tokenize the combined narrative
    encoded = tokenizer(
        df["model_input"].tolist(),
        padding="max_length",
        truncation=True,
        max_length=256
    )

    # Store token IDs and attention masks
    df["input_ids"] = [
        str(ids) for ids in encoded["input_ids"]
    ]

    df["attention_mask"] = [
        str(mask) for mask in encoded["attention_mask"]
    ]

    df.to_csv(output_file, index=False)

    print(
        f"Tokenized {len(df)} records → {output_file}"
    )


tokenize_file(
    f"{INPUT_DIR}/train_processed.csv",
    f"{OUTPUT_DIR}/train_tokenized.csv"
)

tokenize_file(
    f"{INPUT_DIR}/validation_processed.csv",
    f"{OUTPUT_DIR}/validation_tokenized.csv"
)

tokenize_file(
    f"{INPUT_DIR}/test_processed.csv",
    f"{OUTPUT_DIR}/test_tokenized.csv"
)

print("\n" + "=" * 60)
print("TOKENIZATION COMPLETE")
print("=" * 60)