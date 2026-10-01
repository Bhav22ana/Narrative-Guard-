import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# NARRATIVEGUARD
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# ============================================================

INPUT_PATH = "dataset/processed/narrativeguard_master_clean.csv"

TRAIN_PATH = "dataset/splits/train_clean.csv"
VAL_PATH = "dataset/splits/validation_clean.csv"
TEST_PATH = "dataset/splits/test_clean.csv"

RANDOM_STATE = 42

print("=" * 70)
print("NARRATIVEGUARD - DATASET SPLITTING")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

print("\nTotal samples:", len(df))

# ------------------------------------------------------------
# 2. CHECK CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\nOriginal class distribution:")
print(df["scam_type"].value_counts())

# ------------------------------------------------------------
# 3. FIRST SPLIT
#    70% TRAIN
#    30% TEMPORARY
# ------------------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["scam_type"],
    random_state=RANDOM_STATE
)

# ------------------------------------------------------------
# 4. SECOND SPLIT
#    TEMPORARY → 15% VALIDATION + 15% TEST
# ------------------------------------------------------------

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["scam_type"],
    random_state=RANDOM_STATE
)

# ------------------------------------------------------------
# 5. SAVE
# ------------------------------------------------------------

train_df.to_csv(TRAIN_PATH, index=False)
validation_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

# ------------------------------------------------------------
# 6. SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SPLIT SUMMARY")
print("=" * 70)

print("Training samples   :", len(train_df))
print("Validation samples :", len(validation_df))
print("Testing samples    :", len(test_df))
print("Total               :", len(train_df) + len(validation_df) + len(test_df))

# ------------------------------------------------------------
# 7. CLASS DISTRIBUTION PER SPLIT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING DISTRIBUTION")
print("=" * 70)

print(train_df["scam_type"].value_counts())

print("\n" + "=" * 70)
print("VALIDATION DISTRIBUTION")
print("=" * 70)

print(validation_df["scam_type"].value_counts())

print("\n" + "=" * 70)
print("TEST DISTRIBUTION")
print("=" * 70)

print(test_df["scam_type"].value_counts())

# ------------------------------------------------------------
# 8. VERIFY NO OVERLAP
# ------------------------------------------------------------

train_ids = set(train_df["conversation_id"])
val_ids = set(validation_df["conversation_id"])
test_ids = set(test_df["conversation_id"])

print("\n" + "=" * 70)
print("LEAKAGE CHECK BETWEEN SPLITS")
print("=" * 70)

print("Train ∩ Validation:", len(train_ids & val_ids))
print("Train ∩ Test      :", len(train_ids & test_ids))
print("Validation ∩ Test :", len(val_ids & test_ids))

if (
    len(train_ids & val_ids) == 0
    and len(train_ids & test_ids) == 0
    and len(val_ids & test_ids) == 0
):
    print("\nPASS - No conversation IDs overlap between splits.")
else:
    print("\nWARNING - Overlap detected!")

print("\nFiles saved:")
print(TRAIN_PATH)
print(VAL_PATH)
print(TEST_PATH)

print("\n" + "=" * 70)
print("DATASET SPLITTING COMPLETED")
print("=" * 70)