import pandas as pd
import re
from pathlib import Path

INPUT = Path("dataset/raw/hinglish/India_Cyber_Scam_Hinglish_Dataset.csv")
OUTPUT = Path("dataset/processed/narrativeguard_hinglish.csv")

print("=" * 70)
print("NARRATIVEGUARD HINGLISH DATASET NORMALIZATION")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

print(f"Original samples: {len(df)}")

# ---------------------------------------------------------
# 2. Basic cleaning
# ---------------------------------------------------------

df = df.dropna(subset=["text", "label", "scam_category"])

df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# Remove empty text
df = df[df["text"].str.len() > 10]

# Remove exact duplicate conversations
before = len(df)
df = df.drop_duplicates(subset=["text"])
duplicates_removed = before - len(df)

print(f"Duplicates removed: {duplicates_removed}")
print(f"Samples after cleaning: {len(df)}")

# ---------------------------------------------------------
# 3. NarrativeGuard label mapping
# ---------------------------------------------------------
#
# IMPORTANT:
# We are deliberately conservative here.
# Some HF categories contain mixed scam scenarios.
# Those should NOT automatically be forced into a class.
# ---------------------------------------------------------

def map_category(row):

    category = str(row["scam_category"]).lower()
    text = str(row["text"]).lower()

    # -----------------------------
    # Genuine conversations
    # -----------------------------
    if category == "none" and row["label"] == 0:
        return "genuine_financial_conversation"

    # -----------------------------
    # KYC scams
    # -----------------------------
    if category == "bank_kyc":
        if any(x in text for x in [
            "kyc",
            "know your customer",
            "account block",
            "account blocked"
        ]):
            return "kyc_scam"

    # Aadhaar-related account verification
    if category == "aadhaar":
        return "kyc_scam"

    # -----------------------------
    # Digital arrest / authority
    # -----------------------------
    if category == "police_digital_arrest":
        return "bank_impersonation_scam"

    if category == "police_blackmail":
        return "bank_impersonation_scam"

    # -----------------------------
    # OTP / payment requests
    # -----------------------------
    if any(x in text for x in [
        "otp",
        "one time password",
        "upi pin",
        "pin share",
        "send money",
        "transfer money",
        "payment"
    ]):
        return "otp_payment_scam"

    # -----------------------------
    # Investment
    # -----------------------------
    if category == "lottery":
        return "investment_scam"

    # -----------------------------
    # Family/social engineering
    # -----------------------------
    if category == "relative":
        if any(x in text for x in [
            "send",
            "transfer",
            "money",
            "₹",
            "rs ",
            "rupees"
        ]):
            return "refund_scam"

    # -----------------------------
    # Amazon / delivery scams
    # -----------------------------
    if category == "amazon":
        if any(x in text for x in [
            "payment",
            "pay",
            "charge",
            "₹",
            "rs ",
            "rupees",
            "fee"
        ]):
            return "otp_payment_scam"

    # -------------------------------------------------
    # Anything that cannot be mapped confidently
    # -------------------------------------------------

    return None


df["narrativeguard_label"] = df.apply(map_category, axis=1)

# ---------------------------------------------------------
# 4. Keep only confidently mapped samples
# ---------------------------------------------------------

mapped = df[df["narrativeguard_label"].notna()].copy()

print(f"Samples retained: {len(mapped)}")
print(f"Samples excluded: {len(df) - len(mapped)}")

# ---------------------------------------------------------
# 5. Rename useful fields
# ---------------------------------------------------------

mapped = mapped.rename(columns={
    "text": "conversation",
    "urgency_level": "urgency",
    "contains_blackmail": "blackmail"
})

mapped["source"] = "huggingface_indian_hinglish"

# ---------------------------------------------------------
# 6. Select final columns
# ---------------------------------------------------------

final_columns = [
    "conversation",
    "narrativeguard_label",
    "scam_category",
    "caller_type",
    "urgency",
    "blackmail",
    "language_style",
    "source"
]

mapped = mapped[final_columns]

# ---------------------------------------------------------
# 7. Show class distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NARRATIVEGUARD CLASS DISTRIBUTION")
print("=" * 70)

print(
    mapped["narrativeguard_label"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# 8. Show language distribution
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LANGUAGE DISTRIBUTION")
print("=" * 70)

print(
    mapped["language_style"]
    .value_counts()
    .to_string()
)

# ---------------------------------------------------------
# 9. Save
# ---------------------------------------------------------

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

mapped.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("NORMALIZATION COMPLETED")
print("=" * 70)

print(f"Saved to:")
print(OUTPUT)