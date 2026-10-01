import pandas as pd
from sklearn.model_selection import train_test_split
import os

INPUT_FILE = "dataset/annotated/narrativeguard_dataset.csv"
OUTPUT_DIR = "dataset/splits"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("NARRATIVEGUARD DATASET SPLITTING")
print("=" * 60)

print("\nTotal samples:", len(df))

# Shuffle dataset using fixed random state
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# First split: 70% train, 30% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

# Second split: temporary into validation and test
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)

# Save datasets
train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

validation_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)

print("\nSPLIT RESULTS")
print("-" * 40)

print("Training samples:   ", len(train_df))
print("Validation samples: ", len(validation_df))
print("Testing samples:    ", len(test_df))

print("\nFiles created:")
print("dataset/splits/train.csv")
print("dataset/splits/validation.csv")
print("dataset/splits/test.csv")

print("\n" + "=" * 60)
print("DATASET SPLITTING COMPLETE")
print("=" * 60)