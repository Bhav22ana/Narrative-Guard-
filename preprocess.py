import pandas as pd
import os
import re

INPUT_DIR = "dataset/splits"
OUTPUT_DIR = "dataset/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def clean_text(text):
    """
    Lightweight preprocessing for NarrativeGuard.
    Preserves English, Tanglish, numbers and financial context.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Normalize escaped newline characters
    text = text.replace("\\n", " ")

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


def prepare_dataset(input_file, output_file):

    df = pd.read_csv(input_file)

    # Clean important textual fields
    text_columns = [
        "conversation",
        "claim",
        "target_action",
        "manipulation_tactics",
        "evidence_required",
        "recommended_action",
        "explanation"
    ]

    for column in text_columns:
        df[column] = df[column].apply(clean_text)

    # Create a combined narrative input for the model
    df["model_input"] = (
        "Conversation: " + df["conversation"] +
        " Claim: " + df["claim"] +
        " Target Action: " + df["target_action"] +
        " Manipulation Tactics: " + df["manipulation_tactics"] +
        " Evidence Required: " + df["evidence_required"]
    )

    # Save processed dataset
    df.to_csv(output_file, index=False)

    print(
        f"Processed {len(df)} records → {output_file}"
    )


print("=" * 60)
print("NARRATIVEGUARD TEXT PREPROCESSING")
print("=" * 60)

prepare_dataset(
    f"{INPUT_DIR}/train.csv",
    f"{OUTPUT_DIR}/train_processed.csv"
)

prepare_dataset(
    f"{INPUT_DIR}/validation.csv",
    f"{OUTPUT_DIR}/validation_processed.csv"
)

prepare_dataset(
    f"{INPUT_DIR}/test.csv",
    f"{OUTPUT_DIR}/test_processed.csv"
)

print("\nTEXT PREPROCESSING COMPLETE")