import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher

# ============================================================
# NARRATIVEGUARD - MASTER DATASET QUALITY & LEAKAGE CHECK
# ============================================================

MASTER_PATH = "dataset/processed/narrativeguard_master_final.csv"

print("=" * 70)
print("NARRATIVEGUARD MASTER DATASET")
print("QUALITY / LEAKAGE CHECK")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(MASTER_PATH)

print("\n[1] DATASET STRUCTURE")
print("-" * 70)

print("Rows    :", len(df))
print("Columns :", len(df.columns))

print("\nColumns:")
for col in df.columns:
    print(" -", col)

# ------------------------------------------------------------
# 2. MISSING VALUES
# ------------------------------------------------------------

print("\n[2] MISSING VALUE CHECK")
print("-" * 70)

missing = df.isnull().sum()

missing_found = False

for col, count in missing.items():
    if count > 0:
        print(f"{col}: {count}")
        missing_found = True

if not missing_found:
    print("PASS - No missing values found.")

# ------------------------------------------------------------
# 3. EMPTY VALUES
# ------------------------------------------------------------

print("\n[3] EMPTY VALUE CHECK")
print("-" * 70)

for col in df.columns:
    empty_count = (
        df[col]
        .astype(str)
        .str.strip()
        .isin(["", "nan", "None"])
        .sum()
    )

    if empty_count > 0:
        print(f"{col}: {empty_count}")

# ------------------------------------------------------------
# 4. DUPLICATE CONVERSATION IDS
# ------------------------------------------------------------

print("\n[4] DUPLICATE CONVERSATION ID CHECK")
print("-" * 70)

if "conversation_id" in df.columns:

    duplicate_ids = df[
        df["conversation_id"].duplicated(keep=False)
    ]

    if len(duplicate_ids) == 0:
        print("PASS - No duplicate conversation IDs.")
    else:
        print(
            "WARNING - Duplicate conversation IDs:",
            len(duplicate_ids)
        )

        print(
            duplicate_ids[
                ["conversation_id"]
            ].sort_values("conversation_id")
        )

# ------------------------------------------------------------
# 5. EXACT DUPLICATE CONVERSATIONS
# ------------------------------------------------------------

print("\n[5] EXACT DUPLICATE CONVERSATION CHECK")
print("-" * 70)

if "conversation" in df.columns:

    duplicate_conversations = df[
        df["conversation"].duplicated(keep=False)
    ]

    if len(duplicate_conversations) == 0:
        print("PASS - No exact duplicate conversations.")
    else:
        print(
            "WARNING - Exact duplicate conversations:",
            len(duplicate_conversations)
        )

        cols = [
            c for c in
            ["conversation_id", "language", "scam_type"]
            if c in df.columns
        ]

        print(duplicate_conversations[cols])

# ------------------------------------------------------------
# 6. NORMALIZED DUPLICATE CHECK
# ------------------------------------------------------------

print("\n[6] NORMALIZED DUPLICATE CHECK")
print("-" * 70)


def normalize_text(text):
    text = str(text).lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


if "conversation" in df.columns:

    df["_normalized_conversation"] = (
        df["conversation"]
        .apply(normalize_text)
    )

    normalized_duplicates = df[
        df["_normalized_conversation"]
        .duplicated(keep=False)
    ]

    if len(normalized_duplicates) == 0:
        print("PASS - No normalized duplicate conversations.")
    else:
        print(
            "WARNING - Normalized duplicates:",
            len(normalized_duplicates)
        )

        cols = [
            c for c in
            ["conversation_id", "language", "scam_type"]
            if c in df.columns
        ]

        print(
            normalized_duplicates[cols]
            .sort_values("conversation_id")
        )

# ------------------------------------------------------------
# 7. LANGUAGE DISTRIBUTION
# ------------------------------------------------------------

print("\n[7] LANGUAGE DISTRIBUTION")
print("-" * 70)

if "language" in df.columns:

    print(
        df["language"]
        .value_counts(dropna=False)
    )

# ------------------------------------------------------------
# 8. SCAM TYPE DISTRIBUTION
# ------------------------------------------------------------

print("\n[8] SCAM TYPE DISTRIBUTION")
print("-" * 70)

if "scam_type" in df.columns:

    distribution = (
        df["scam_type"]
        .value_counts(dropna=False)
    )

    print(distribution)

# ------------------------------------------------------------
# 9. LABEL CONSISTENCY
# ------------------------------------------------------------

print("\n[9] LABEL CONSISTENCY CHECK")
print("-" * 70)

if "conversation" in df.columns and "scam_type" in df.columns:

    label_conflicts = (
        df.groupby("_normalized_conversation")["scam_type"]
        .nunique()
    )

    conflicts = label_conflicts[
        label_conflicts > 1
    ]

    if len(conflicts) == 0:
        print("PASS - No conflicting labels for identical normalized text.")
    else:
        print(
            "WARNING - Conflicting labels found:",
            len(conflicts)
        )

# ------------------------------------------------------------
# 10. VERY SHORT CONVERSATIONS
# ------------------------------------------------------------

print("\n[10] VERY SHORT CONVERSATION CHECK")
print("-" * 70)

if "conversation" in df.columns:

    df["_text_length"] = (
        df["conversation"]
        .astype(str)
        .str.split()
        .str.len()
    )

    short_samples = df[
        df["_text_length"] < 5
    ]

    print(
        "Conversations with fewer than 5 words:",
        len(short_samples)
    )

# ------------------------------------------------------------
# 11. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("QUALITY CHECK SUMMARY")
print("=" * 70)

print("Total samples:", len(df))

if "conversation_id" in df.columns:
    print(
        "Unique conversation IDs:",
        df["conversation_id"].nunique()
    )

if "conversation" in df.columns:
    print(
        "Unique conversations:",
        df["_normalized_conversation"].nunique()
    )

print("\nQuality / leakage check completed.")

print("=" * 70)