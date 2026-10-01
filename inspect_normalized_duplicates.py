import pandas as pd
import re

PATH = "dataset/processed/narrativeguard_master_final.csv"

df = pd.read_csv(PATH)

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

df["_normalized"] = df["conversation"].apply(normalize_text)

duplicates = df[
    df["_normalized"].duplicated(keep=False)
].sort_values("_normalized")

print("=" * 70)
print("NORMALIZED DUPLICATE INVESTIGATION")
print("=" * 70)

for normalized_text, group in duplicates.groupby("_normalized"):

    print("\n" + "-" * 70)

    for _, row in group.iterrows():

        print("\nConversation ID :", row["conversation_id"])
        print("Language       :", row["language"])
        print("Scam Type      :", row["scam_type"])
        print("Source         :", row["source"])
        print("Conversation   :")
        print(row["conversation"])

print("\n" + "=" * 70)