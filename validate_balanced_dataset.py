import pandas as pd

TRAIN_PATH = "dataset/splits/train_balanced.csv"
VAL_PATH = "dataset/splits/validation_clean.csv"
TEST_PATH = "dataset/splits/test_clean.csv"

print("=" * 70)
print("NARRATIVEGUARD - FINAL BALANCED DATASET VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD DATASETS
# ------------------------------------------------------------

train = pd.read_csv(TRAIN_PATH)
val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

print("\nDataset sizes")
print("-" * 70)
print("Balanced training :", len(train))
print("Validation         :", len(val))
print("Testing            :", len(test))

# ------------------------------------------------------------
# 2. CHECK TRAINING CLASS DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)

train_distribution = train["scam_type"].value_counts()

print(train_distribution)

if train_distribution.nunique() == 1 and train_distribution.iloc[0] == 100:
    print("\nPASS - All 8 training classes contain exactly 100 rows.")
else:
    print("\nWARNING - Training classes are not equally represented.")

# ------------------------------------------------------------
# 3. CHECK ALL 8 CLASSES
# ------------------------------------------------------------

expected_classes = {
    "bank_impersonation_scam",
    "genuine_financial_conversation",
    "investment_scam",
    "otp_payment_scam",
    "kyc_scam",
    "refund_scam",
    "job_scam",
    "loan_scam"
}

actual_classes = set(train["scam_type"].unique())

print("\n" + "=" * 70)
print("CLASS COMPLETENESS CHECK")
print("=" * 70)

missing_classes = expected_classes - actual_classes
unexpected_classes = actual_classes - expected_classes

print("Expected classes:", len(expected_classes))
print("Actual classes  :", len(actual_classes))

if not missing_classes and not unexpected_classes:
    print("PASS - All 8 expected classes are present.")
else:
    print("WARNING")
    print("Missing classes   :", missing_classes)
    print("Unexpected classes:", unexpected_classes)

# ------------------------------------------------------------
# 4. CHECK REPEATED IDS INSIDE TRAINING
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING OVERSAMPLING CHECK")
print("=" * 70)

train_unique_ids = train["conversation_id"].nunique()
train_total = len(train)

repeated_training_rows = train_total - train_unique_ids

print("Training rows       :", train_total)
print("Unique training IDs :", train_unique_ids)
print("Repeated rows       :", repeated_training_rows)

print(
    "\nNOTE: Repeated IDs are expected because minority "
    "classes were oversampled."
)

# ------------------------------------------------------------
# 5. CREATE ID SETS
# ------------------------------------------------------------

train_ids = set(train["conversation_id"])
val_ids = set(val["conversation_id"])
test_ids = set(test["conversation_id"])

# ------------------------------------------------------------
# 6. CHECK CROSS-SPLIT LEAKAGE
# ------------------------------------------------------------

train_val_overlap = train_ids & val_ids
train_test_overlap = train_ids & test_ids
val_test_overlap = val_ids & test_ids

print("\n" + "=" * 70)
print("CROSS-SPLIT LEAKAGE CHECK")
print("=" * 70)

print("Train ∩ Validation:", len(train_val_overlap))
print("Train ∩ Test      :", len(train_test_overlap))
print("Validation ∩ Test :", len(val_test_overlap))

if (
    len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):
    print("\nPASS - No cross-split conversation-ID leakage.")
else:
    print("\nFAIL - Cross-split overlap detected!")

# ------------------------------------------------------------
# 7. CHECK VALIDATION / TEST WERE NOT BALANCED
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION / TEST INTEGRITY")
print("=" * 70)

print("\nValidation distribution:")
print(val["scam_type"].value_counts())

print("\nTest distribution:")
print(test["scam_type"].value_counts())

# ------------------------------------------------------------
# 8. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

checks = {
    "Training size = 800": len(train) == 800,
    "All classes = 100": (
        train_distribution.nunique() == 1
        and train_distribution.iloc[0] == 100
    ),
    "All 8 classes present": (
        not missing_classes and not unexpected_classes
    ),
    "No train-validation leakage": len(train_val_overlap) == 0,
    "No train-test leakage": len(train_test_overlap) == 0,
    "No validation-test leakage": len(val_test_overlap) == 0
}

for check, result in checks.items():
    print(f"{'PASS' if result else 'FAIL'} - {check}")

print("\n" + "=" * 70)