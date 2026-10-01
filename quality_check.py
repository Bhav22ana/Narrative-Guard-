import pandas as pd

FILE_PATH = "dataset/annotated/narrativeguard_dataset.csv"

df = pd.read_csv(FILE_PATH)

print("=" * 60)
print("NARRATIVEGUARD DATASET QUALITY CHECK")
print("=" * 60)

# 1. Dataset size
print("\n1. DATASET SIZE")
print("Rows:", len(df))
print("Columns:", len(df.columns))

# 2. Missing values
print("\n2. MISSING VALUES")
missing = df.isnull().sum()
print(missing[missing > 0])

if missing.sum() == 0:
    print("No missing values found.")

# 3. Duplicate conversation IDs
print("\n3. DUPLICATE CONVERSATION IDs")
duplicates = df[df["conversation_id"].duplicated(keep=False)]

if duplicates.empty:
    print("No duplicate conversation IDs found.")
else:
    print(duplicates["conversation_id"])

# 4. Language distribution
print("\n4. LANGUAGE DISTRIBUTION")
print(df["language"].value_counts())

# 5. Scam type distribution
print("\n5. SCAM TYPE DISTRIBUTION")
print(df["scam_type"].value_counts())

# 6. Risk level distribution
print("\n6. RISK LEVEL DISTRIBUTION")
print(df["risk_level"].value_counts())

# 7. Verification status
print("\n7. VERIFICATION STATUS")
print(df["verification_status"].value_counts())

# 8. Source type
print("\n8. SOURCE TYPE")
print(df["source_type"].value_counts())

# 9. Empty conversations
print("\n9. EMPTY CONVERSATIONS")
empty_conversations = df["conversation"].isnull() | (
    df["conversation"].astype(str).str.strip() == ""
)

print("Empty conversations:", empty_conversations.sum())

# 10. Empty claims
print("\n10. EMPTY CLAIMS")
empty_claims = df["claim"].isnull() | (
    df["claim"].astype(str).str.strip() == ""
)

print("Empty claims:", empty_claims.sum())

# 11. Empty recommended actions
print("\n11. EMPTY RECOMMENDED ACTIONS")
empty_actions = df["recommended_action"].isnull() | (
    df["recommended_action"].astype(str).str.strip() == ""
)

print("Empty recommended actions:", empty_actions.sum())

# 12. Required columns
print("\n12. REQUIRED COLUMNS")

required_columns = [
    "conversation_id",
    "language",
    "source_type",
    "source_reference",
    "scam_type",
    "conversation",
    "claim",
    "target_action",
    "manipulation_tactics",
    "evidence_required",
    "evidence_available",
    "verification_status",
    "risk_level",
    "recommended_action",
    "explanation"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if not missing_columns:
    print("All required columns are present.")
else:
    print("Missing columns:", missing_columns)

print("\n" + "=" * 60)
print("QUALITY CHECK COMPLETE")
print("=" * 60)