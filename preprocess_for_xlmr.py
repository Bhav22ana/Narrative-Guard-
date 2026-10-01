import pandas as pd
import re
import os

# ============================================================
# NARRATIVEGUARD
# LIGHTWEIGHT TEXT PREPROCESSING
# ============================================================

INPUTS = {
    "train": "dataset/splits/train_balanced.csv",
    "validation": "dataset/splits/validation_clean.csv",
    "test": "dataset/splits/test_clean.csv"
}

OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_conversation(text):
    """
    Lightweight cleaning.
    Important scam-related words and multilingual text
    are preserved.
    """

    text = str(text)

    # Normalize line breaks and tabs
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text


print("=" * 70)
print("NARRATIVEGUARD - TEXT PREPROCESSING")
print("=" * 70)

for split_name, input_path in INPUTS.items():

    print(f"\nProcessing: {split_name}")
    print("-" * 70)

    df = pd.read_csv(input_path)

    # Make sure conversation column exists
    if "conversation" not in df.columns:
        raise ValueError(
            f"'conversation' column not found in {input_path}"
        )

    # Clean conversation text
    df["conversation"] = (
        df["conversation"]
        .apply(clean_conversation)
    )

    # Remove completely empty conversations
    before = len(df)

    df = df[
        df["conversation"].str.len() > 0
    ].copy()

    removed = before - len(df)

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{split_name}_preprocessed.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("Original rows :", before)
    print("Removed empty :", removed)
    print("Final rows    :", len(df))
    print("Saved to      :", output_path)

print("\n" + "=" * 70)
print("TEXT PREPROCESSING COMPLETED")
print("=" * 70)