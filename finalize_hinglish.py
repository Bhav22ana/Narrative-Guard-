import pandas as pd

INPUT = "dataset/processed/narrativeguard_hinglish_corrected.csv"
OUTPUT = "dataset/processed/narrativeguard_hinglish_final.csv"
EXCLUDED_OUTPUT = "dataset/processed/hinglish_excluded_review.csv"

print("=" * 70)
print("NARRATIVEGUARD FINAL HINGLISH DATASET CLEANING")
print("=" * 70)

df = pd.read_csv(INPUT)

print(f"\nOriginal samples: {len(df)}")

# ---------------------------------------------------------
# 1. Show REVIEW samples
# ---------------------------------------------------------

review = df[df["correction_status"] == "REVIEW"].copy()

print(f"REVIEW samples found: {len(review)}")

if len(review) > 0:
    print("\nThese samples belong to unsupported scam categories.")
    print("They will be excluded from model training.")

    review.to_csv(EXCLUDED_OUTPUT, index=False)

# ---------------------------------------------------------
# 2. Remove REVIEW samples
# ---------------------------------------------------------

df_final = df[df["correction_status"] != "REVIEW"].copy()

# ---------------------------------------------------------
# 3. Keep only the 8 NarrativeGuard classes
# ---------------------------------------------------------

VALID_CLASSES = [
    "bank_impersonation_scam",
    "genuine_financial_conversation",
    "investment_scam",
    "otp_payment_scam",
    "refund_scam",
    "kyc_scam",
    "job_scam",
    "loan_scam"
]

before_filter = len(df_final)

df_final = df_final[
    df_final["narrativeguard_label"].isin(VALID_CLASSES)
].copy()

removed_invalid = before_filter - len(df_final)

# ---------------------------------------------------------
# 4. Remove empty conversations
# ---------------------------------------------------------

df_final = df_final[
    df_final["conversation"].notna() &
    (df_final["conversation"].str.strip() != "")
].copy()

# ---------------------------------------------------------
# 5. Remove exact duplicate conversations
# ---------------------------------------------------------

before_duplicates = len(df_final)

df_final = df_final.drop_duplicates(
    subset=["conversation"],
    keep="first"
).copy()

duplicates_removed = before_duplicates - len(df_final)

# ---------------------------------------------------------
# 6. Final summary
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATASET SUMMARY")
print("=" * 70)

print(f"\nFinal samples: {len(df_final)}")
print(f"REVIEW samples excluded: {len(review)}")
print(f"Invalid-class samples removed: {removed_invalid}")
print(f"Duplicate conversations removed: {duplicates_removed}")

print("\n=== FINAL CLASS DISTRIBUTION ===")
print(
    df_final["narrativeguard_label"]
    .value_counts()
    .to_string()
)

print("\n=== LANGUAGE DISTRIBUTION ===")
print(
    df_final["language_style"]
    .value_counts()
    .to_string()
)

print("\n=== CORRECTION STATUS ===")
print(
    df_final["correction_status"]
    .value_counts()
    .to_string()
)

print("\n=== MISSING VALUES ===")
print(
    df_final.isnull().sum().to_string()
)

# ---------------------------------------------------------
# 7. Save
# ---------------------------------------------------------

df_final.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("FINAL HINGLISH DATASET SAVED")
print("=" * 70)

print(f"\nSaved to:")
print(OUTPUT)