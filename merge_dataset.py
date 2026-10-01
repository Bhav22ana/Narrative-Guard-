import pandas as pd
from pathlib import Path

print("=" * 70)
print("NARRATIVEGUARD MASTER DATASET CREATION")
print("=" * 70)

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

HINGLISH_PATH = Path(
    "dataset/processed/narrativeguard_hinglish.csv"
)

VISHING_PATH = Path(
    "dataset/processed/narrativeguard_external.csv"
)

OUTPUT_PATH = Path(
    "dataset/processed/narrativeguard_master.csv"
)

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("\nLoading Hinglish dataset...")
hinglish = pd.read_csv(HINGLISH_PATH)

print(f"Hinglish samples: {len(hinglish)}")

print("\nLoading Vishing/Tanglish dataset...")
vishing = pd.read_csv(VISHING_PATH)

print(f"Vishing samples: {len(vishing)}")

# ---------------------------------------------------------
# Standardize Hinglish dataset
# ---------------------------------------------------------

hinglish_std = pd.DataFrame({
    "conversation": hinglish["conversation"].astype(str),
    "scam_type": hinglish["narrativeguard_label"].astype(str),
    "language": hinglish["language_style"].astype(str),
    "source": "huggingface_hinglish",
    "urgency": hinglish["urgency"].astype(str),
    "blackmail": hinglish["blackmail"].astype(str),
    "caller_type": hinglish["caller_type"].astype(str)
})

# ---------------------------------------------------------
# Standardize Vishing dataset
# ---------------------------------------------------------

vishing_std = pd.DataFrame({
    "conversation": vishing["conversation"].astype(str),
    "scam_type": vishing["scam_type"].astype(str),
    "language": vishing["language"].astype(str),
    "source": "vishing_tanglish",
    "urgency": "unknown",
    "blackmail": "unknown",
    "caller_type": "unknown"
})

# ---------------------------------------------------------
# Combine
# ---------------------------------------------------------

master = pd.concat(
    [hinglish_std, vishing_std],
    ignore_index=True
)

print("\nCombined samples:", len(master))

# ---------------------------------------------------------
# Clean
# ---------------------------------------------------------

master["conversation"] = (
    master["conversation"]
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Remove empty conversations
master = master[
    master["conversation"].str.len() > 10
].copy()

# Remove exact duplicate conversations
before_duplicates = len(master)

master = master.drop_duplicates(
    subset=["conversation", "scam_type"]
).reset_index(drop=True)

duplicates_removed = before_duplicates - len(master)

# ---------------------------------------------------------
# Validate classes
# ---------------------------------------------------------

expected_classes = {
    "refund_scam",
    "kyc_scam",
    "bank_impersonation_scam",
    "investment_scam",
    "otp_payment_scam",
    "job_scam",
    "loan_scam",
    "genuine_financial_conversation"
}

actual_classes = set(master["scam_type"].unique())

missing_classes = expected_classes - actual_classes
unexpected_classes = actual_classes - expected_classes

print("\nDuplicates removed:", duplicates_removed)

if missing_classes:
    print("\nWARNING - Missing classes:")
    for c in sorted(missing_classes):
        print("  ", c)

if unexpected_classes:
    print("\nWARNING - Unexpected classes:")
    for c in sorted(unexpected_classes):
        print("  ", c)

# ---------------------------------------------------------
# Add unique ID
# ---------------------------------------------------------

master.insert(
    0,
    "conversation_id",
    [
        f"NG_{i:05d}"
        for i in range(1, len(master) + 1)
    ]
)

# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

master.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Reports
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("MASTER DATASET SUMMARY")
print("=" * 70)

print("\nTotal samples:", len(master))

print("\n=== CLASS DISTRIBUTION ===")
print(
    master["scam_type"]
    .value_counts()
    .to_string()
)

print("\n=== LANGUAGE DISTRIBUTION ===")
print(
    master["language"]
    .value_counts()
    .to_string()
)

print("\n=== SOURCE DISTRIBUTION ===")
print(
    master["source"]
    .value_counts()
    .to_string()
)

print("\n=== MISSING VALUES ===")
print(
    master.isnull()
    .sum()
    .to_string()
)

print("\n=== DUPLICATE CONVERSATIONS ===")
print(
    master["conversation"]
    .duplicated()
    .sum()
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("MASTER DATASET CREATION COMPLETED")
print("=" * 70)