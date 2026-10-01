import pandas as pd
import os

print("=" * 70)
print("NARRATIVEGUARD FINAL MASTER DATASET CREATION")
print("=" * 70)

# ============================================================
# PATHS
# ============================================================

HINGLISH_PATH = "dataset/processed/narrativeguard_hinglish_final.csv"
EXTERNAL_PATH = "dataset/processed/narrativeguard_external.csv"

OUTPUT_PATH = "dataset/processed/narrativeguard_master_final.csv"

# ============================================================
# LOAD HINGLISH
# ============================================================

print("\nLoading FINAL Hinglish dataset...")

hinglish = pd.read_csv(HINGLISH_PATH)

print("Hinglish samples:", len(hinglish))

# ============================================================
# STANDARDIZE HINGLISH COLUMNS
# ============================================================

hinglish_final = pd.DataFrame({
    "conversation_id": [
        f"HF_{i+1:05d}" for i in range(len(hinglish))
    ],
    "conversation": hinglish["conversation"],
    "scam_type": hinglish["narrativeguard_label"],
    "language": hinglish["language_style"],
    "source": "huggingface_hinglish",
    "urgency": hinglish["urgency"],
    "blackmail": hinglish["blackmail"],
    "caller_type": hinglish["caller_type"]
})

# ============================================================
# LOAD EXTERNAL / VISHING
# ============================================================

print("\nLoading Vishing/Tanglish dataset...")

external = pd.read_csv(EXTERNAL_PATH)

print("Vishing/Tanglish samples:", len(external))

# ============================================================
# STANDARDIZE EXTERNAL COLUMNS
# ============================================================

external_final = pd.DataFrame({
    "conversation_id": [
        f"VT_{i+1:05d}" for i in range(len(external))
    ],
    "conversation": external["conversation"],
    "scam_type": external["scam_type"],
    "language": external["language"],
    "source": "vishing_tanglish",
    "urgency": external.get("urgency", "unknown"),
    "blackmail": external.get("blackmail", False),
    "caller_type": external.get("caller_type", "unknown")
})

# ============================================================
# COMBINE
# ============================================================

master = pd.concat(
    [hinglish_final, external_final],
    ignore_index=True
)

print("\nCombined samples:", len(master))

# ============================================================
# REMOVE EXACT DUPLICATES
# ============================================================

before_duplicates = len(master)

master = master.drop_duplicates(
    subset=["conversation"],
    keep="first"
).reset_index(drop=True)

duplicates_removed = before_duplicates - len(master)

print("Duplicates removed:", duplicates_removed)

# ============================================================
# REMOVE EMPTY CONVERSATIONS
# ============================================================

before_empty = len(master)

master["conversation"] = master["conversation"].astype(str).str.strip()

master = master[
    master["conversation"].str.len() > 0
].reset_index(drop=True)

empty_removed = before_empty - len(master)

print("Empty conversations removed:", empty_removed)

# ============================================================
# FINAL MISSING VALUE CHECK
# ============================================================

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
    master.isnull().sum().to_string()
)

print("\n=== DUPLICATE CONVERSATIONS ===")
print(
    master["conversation"].duplicated().sum()
)

# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

master.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved to:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("FINAL MASTER DATASET CREATION COMPLETED")
print("=" * 70)