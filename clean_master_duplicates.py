import pandas as pd

INPUT_PATH = "dataset/processed/narrativeguard_master_final.csv"
OUTPUT_PATH = "dataset/processed/narrativeguard_master_clean.csv"

REMOVE_IDS = [
    "VT_00376",
    "VT_00386"
]

df = pd.read_csv(INPUT_PATH)

print("Original samples:", len(df))

df_clean = df[
    ~df["conversation_id"].isin(REMOVE_IDS)
].copy()

df_clean.to_csv(OUTPUT_PATH, index=False)

print("Removed samples:", len(df) - len(df_clean))
print("Clean samples:", len(df_clean))

print("\nRemoved IDs:")
for cid in REMOVE_IDS:
    print("-", cid)

print("\nSaved to:")
print(OUTPUT_PATH)