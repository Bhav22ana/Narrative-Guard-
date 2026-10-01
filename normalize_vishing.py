import pandas as pd
import os
import re

INPUT = r"dataset\raw\vishing\dataset_fixed_v3.csv"
OUTPUT = r"dataset\processed\narrativeguard_external.csv"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df = pd.read_csv(INPUT)

print("=" * 70)
print("NARRATIVEGUARD VISHING DATASET NORMALIZATION")
print("=" * 70)

print(f"Original samples: {len(df)}")

records = []

for _, row in df.iterrows():

    text = str(row["dialogue"]).strip()
    source_category = str(row["scam_type"]).strip()
    language = str(row["language"]).strip()

    text_lower = text.lower()

    target_class = None

    # --------------------------------------------------
    # DIRECTLY MAPPED CATEGORIES
    # --------------------------------------------------

    if source_category == "refund_scam":
        target_class = "refund_scam"

    elif source_category == "investment_fraud":
        target_class = "investment_scam"

    elif source_category == "job_scam":
        target_class = "job_scam"

    elif source_category == "loan_scam":
        target_class = "loan_scam"

    elif source_category == "genuine_bank":
        target_class = "genuine_financial_conversation"

    # --------------------------------------------------
    # BANK FRAUD
    # Need to distinguish OTP / KYC / bank impersonation
    # --------------------------------------------------

    elif source_category == "bank_fraud":

        otp_pattern = (
            r"\botp\b|one[- ]time password|"
            r"one time pin|verification code|"
            r"upi pin|payment pin"
        )

        kyc_pattern = (
            r"\bkyc\b|know your customer|"
            r"aadhaar|aadhar|pan card|"
            r"pan[- ]aadhaar|identity verification"
        )

        if re.search(otp_pattern, text_lower):
            target_class = "otp_payment_scam"

        elif re.search(kyc_pattern, text_lower):
            target_class = "kyc_scam"

        else:
            target_class = "bank_impersonation_scam"

    # --------------------------------------------------
    # OTHER CATEGORIES
    # Do NOT force them into our 8 classes
    # --------------------------------------------------

    if target_class is None:
        continue

    records.append({
        "conversation_id": f"VISHING_{int(row['id']):04d}",
        "language": language,
        "source": "vishing_dataset",
        "scam_type": target_class,
        "conversation": text
    })


result = pd.DataFrame(records)

# ------------------------------------------------------
# ASSIGN NUMERIC LABELS
# ------------------------------------------------------

label_mapping = {
    "bank_impersonation_scam": 0,
    "genuine_financial_conversation": 1,
    "investment_scam": 2,
    "job_scam": 3,
    "kyc_scam": 4,
    "loan_scam": 5,
    "otp_payment_scam": 6,
    "refund_scam": 7
}

result["label"] = result["scam_type"].map(label_mapping)

# ------------------------------------------------------
# REMOVE DUPLICATES
# ------------------------------------------------------

before = len(result)

result = result.drop_duplicates(
    subset=["conversation"],
    keep="first"
).reset_index(drop=True)

removed = before - len(result)

# ------------------------------------------------------
# SAVE
# ------------------------------------------------------

result.to_csv(OUTPUT, index=False)

print("\nNormalization completed.")

print(f"Samples retained: {len(result)}")
print(f"Duplicates removed: {removed}")

print("\nClass distribution:")
print(
    result["scam_type"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nLanguage distribution:")
print(
    result["language"]
    .value_counts()
    .to_string()
)

print("\nSaved to:")
print(OUTPUT)

print("=" * 70)