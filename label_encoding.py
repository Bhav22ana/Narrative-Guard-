import pandas as pd
import json
import os

INPUT_DIR = "dataset/tokenized"
OUTPUT_DIR = "dataset/tokenized"

LABEL_COLUMN = "scam_type"

# Load training data first
train_df = pd.read_csv(
    f"{INPUT_DIR}/train_tokenized.csv"
)

# Get labels from training data
labels = sorted(train_df[LABEL_COLUMN].unique())

# Create label mapping
label2id = {
    label: idx
    for idx, label in enumerate(labels)
}

id2label = {
    idx: label
    for label, idx in label2id.items()
}

print("=" * 60)
print("NARRATIVEGUARD LABEL ENCODING")
print("=" * 60)

print("\nLabel Mapping:")
for label, idx in label2id.items():
    print(f"{idx} -> {label}")


def encode_file(input_file, output_file):

    df = pd.read_csv(input_file)

    # Check for unknown labels
    unknown_labels = set(df[LABEL_COLUMN]) - set(label2id)

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found in {input_file}: {unknown_labels}"
        )

    # Convert scam type to numerical label
    df["label"] = df[LABEL_COLUMN].map(label2id)

    df.to_csv(output_file, index=False)

    print(
        f"\nEncoded {len(df)} records -> {output_file}"
    )


# Encode all datasets
encode_file(
    f"{INPUT_DIR}/train_tokenized.csv",
    f"{OUTPUT_DIR}/train_encoded.csv"
)

encode_file(
    f"{INPUT_DIR}/validation_tokenized.csv",
    f"{OUTPUT_DIR}/validation_encoded.csv"
)

encode_file(
    f"{INPUT_DIR}/test_tokenized.csv",
    f"{OUTPUT_DIR}/test_encoded.csv"
)


# Save label mapping
mapping = {
    "label2id": label2id,
    "id2label": {
        str(key): value
        for key, value in id2label.items()
    }
}

with open(
    f"{OUTPUT_DIR}/label_mapping.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        mapping,
        file,
        indent=4,
        ensure_ascii=False
    )

print("\nLabel mapping saved to:")
print("dataset/tokenized/label_mapping.json")

print("\n" + "=" * 60)
print("LABEL ENCODING COMPLETE")
print("=" * 60)