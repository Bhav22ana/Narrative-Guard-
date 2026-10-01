import pandas as pd

PATH = "dataset/processed/narrativeguard_master_clean.csv"

df = pd.read_csv(PATH)

print("=" * 70)
print("NARRATIVEGUARD - CLASS DISTRIBUTION ANALYSIS")
print("=" * 70)

print("\nTotal samples:", len(df))

distribution = df["scam_type"].value_counts()

percentage = (
    df["scam_type"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

result = pd.DataFrame({
    "Sample Count": distribution,
    "Percentage": percentage
})

print("\nCLASS DISTRIBUTION")
print("-" * 70)
print(result)

majority_count = distribution.max()
minority_count = distribution.min()

majority_class = distribution.idxmax()
minority_class = distribution.idxmin()

print("\n" + "=" * 70)
print("IMBALANCE SUMMARY")
print("=" * 70)

print("Majority class :", majority_class)
print("Majority count :", majority_count)

print("Minority class :", minority_class)
print("Minority count :", minority_count)

print(
    "Imbalance ratio:",
    round(majority_count / minority_count, 2),
    ": 1"
)

print("=" * 70)