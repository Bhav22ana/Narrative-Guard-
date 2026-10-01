import pandas as pd
from sklearn.utils import resample

# ============================================================
# NARRATIVEGUARD
# CONTROLLED TRAINING DATA BALANCING
# ============================================================

INPUT_PATH = "dataset/splits/train_clean.csv"
OUTPUT_PATH = "dataset/splits/train_balanced.csv"

TARGET_COUNT = 100
RANDOM_STATE = 42

print("=" * 70)
print("NARRATIVEGUARD - CONTROLLED TRAINING DATA BALANCING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD TRAINING DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nOriginal training samples:", len(df))

print("\nOriginal distribution:")
print(df["scam_type"].value_counts())

# ------------------------------------------------------------
# 2. BALANCE EACH CLASS
# ------------------------------------------------------------

balanced_parts = []

for scam_type, group in df.groupby("scam_type"):

    current_count = len(group)

    if current_count > TARGET_COUNT:

        # Undersample majority class
        balanced_group = resample(
            group,
            replace=False,
            n_samples=TARGET_COUNT,
            random_state=RANDOM_STATE
        )

        action = "undersampled"

    elif current_count < TARGET_COUNT:

        # Oversample minority class
        balanced_group = resample(
            group,
            replace=True,
            n_samples=TARGET_COUNT,
            random_state=RANDOM_STATE
        )

        action = "oversampled"

    else:

        balanced_group = group.copy()
        action = "unchanged"

    balanced_parts.append(balanced_group)

    print(
        f"{scam_type:35s} "
        f"{current_count:4d} -> {len(balanced_group):4d} "
        f"({action})"
    )

# ------------------------------------------------------------
# 3. COMBINE
# ------------------------------------------------------------

balanced_df = pd.concat(
    balanced_parts,
    ignore_index=True
)

# Shuffle final training dataset
balanced_df = balanced_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# ------------------------------------------------------------
# 4. SAVE
# ------------------------------------------------------------

balanced_df.to_csv(
    OUTPUT_PATH,
    index=False
)

# ------------------------------------------------------------
# 5. FINAL DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BALANCED TRAINING DISTRIBUTION")
print("=" * 70)

print(
    balanced_df["scam_type"]
    .value_counts()
    .sort_index()
)

print("\nTotal balanced samples:", len(balanced_df))

# ------------------------------------------------------------
# 6. CHECK DUPLICATE IDs
# ------------------------------------------------------------

duplicate_ids = balanced_df[
    balanced_df["conversation_id"].duplicated(keep=False)
]

print("\n" + "=" * 70)
print("BALANCING VALIDATION")
print("=" * 70)

print(
    "Unique conversation IDs:",
    balanced_df["conversation_id"].nunique()
)

print(
    "Total rows:",
    len(balanced_df)
)

print(
    "Repeated IDs due to oversampling:",
    len(balanced_df) -
    balanced_df["conversation_id"].nunique()
)

# ------------------------------------------------------------
# 7. FINAL MESSAGE
# ------------------------------------------------------------

print("\nSaved balanced training dataset:")
print(OUTPUT_PATH)

print("\nValidation and test datasets were NOT modified.")

print("\n" + "=" * 70)
print("CONTROLLED BALANCING COMPLETED")
print("=" * 70)