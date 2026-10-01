import pandas as pd

FILE_PATH = "dataset/annotated/narrativeguard_dataset.csv"

# Read the malformed CSV
df = pd.read_csv(FILE_PATH)

# Correct NarrativeGuard schema
columns = [
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

# Keep only the actual 15 data columns
df = df.iloc[:, :15]

# Apply the correct column names
df.columns = columns

# Save repaired dataset
df.to_csv(FILE_PATH, index=False)

print("Dataset repaired successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Columns:", df.columns.tolist())