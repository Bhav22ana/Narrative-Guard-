import pandas as pd

PATH = "dataset/processed/narrativeguard_master.csv"

df = pd.read_csv(PATH)

print("=" * 70)
print("NARRATIVEGUARD MASTER DATASET INSPECTION")
print("=" * 70)

print("\nTotal samples:", len(df))

print("\n=== CLASS DISTRIBUTION ===")
print(df["scam_type"].value_counts().to_string())

print("\n=== CLASS × LANGUAGE ===")
print(
    pd.crosstab(
        df["scam_type"],
        df["language"]
    ).to_string()
)

print("\n=== CLASS × SOURCE ===")
print(
    pd.crosstab(
        df["scam_type"],
        df["source"]
    ).to_string()
)

print("\n=== SAMPLE TEXTS ===")

classes = [
    "refund_scam",
    "kyc_scam",
    "bank_impersonation_scam",
    "investment_scam",
    "otp_payment_scam",
    "job_scam",
    "loan_scam",
    "genuine_financial_conversation"
]

for scam_class in classes:

    subset = df[df["scam_type"] == scam_class]

    print("\n" + "-" * 70)
    print(f"{scam_class} ({len(subset)} samples)")
    print("-" * 70)

    for i, (_, row) in enumerate(subset.head(3).iterrows(), 1):

        print(f"\nSample {i}:")
        print("Language:", row["language"])
        print("Source:", row["source"])
        print("Conversation:")
        print(row["conversation"][:500])

print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)