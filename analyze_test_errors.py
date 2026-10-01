import os
import pandas as pd


# ============================================================
# NARRATIVEGUARD - FINAL TEST ERROR ANALYSIS
# ============================================================

PREDICTIONS_PATH = "evaluation/test_predictions.csv"

OUTPUT_PATH = "evaluation/misclassified_samples.csv"


print("=" * 70)
print("NARRATIVEGUARD - FINAL TEST ERROR ANALYSIS")
print("=" * 70)


# ============================================================
# 1. LOAD PREDICTIONS
# ============================================================

if not os.path.exists(PREDICTIONS_PATH):
    raise FileNotFoundError(
        f"Prediction file not found: {PREDICTIONS_PATH}"
    )

df = pd.read_csv(PREDICTIONS_PATH)

print()
print("Total test samples:", len(df))


# ============================================================
# 2. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "actual_class",
    "predicted_class"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# 3. IDENTIFY ERRORS
# ============================================================

errors = df[
    df["actual_class"] != df["predicted_class"]
].copy()

correct = df[
    df["actual_class"] == df["predicted_class"]
].copy()


print()
print("=" * 70)
print("OVERALL ERROR SUMMARY")
print("=" * 70)

print("Total samples       :", len(df))
print("Correct predictions :", len(correct))
print("Incorrect predictions:", len(errors))

error_rate = (
    len(errors) / len(df)
) * 100

print(
    f"Error rate          : {error_rate:.2f}%"
)


# ============================================================
# 4. ACTUAL CLASS DISTRIBUTION OF ERRORS
# ============================================================

print()
print("=" * 70)
print("ERRORS BY ACTUAL CLASS")
print("=" * 70)

actual_errors = (
    errors["actual_class"]
    .value_counts()
)

print(actual_errors)


# ============================================================
# 5. PREDICTED CLASS DISTRIBUTION OF ERRORS
# ============================================================

print()
print("=" * 70)
print("ERRORS BY PREDICTED CLASS")
print("=" * 70)

predicted_errors = (
    errors["predicted_class"]
    .value_counts()
)

print(predicted_errors)


# ============================================================
# 6. CONFUSION PAIRS
# ============================================================

print()
print("=" * 70)
print("MOST COMMON MISCLASSIFICATION PAIRS")
print("=" * 70)

if len(errors) > 0:

    pairs = (
        errors
        .groupby(
            [
                "actual_class",
                "predicted_class"
            ]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    print(pairs.to_string(index=False))

else:

    print("No misclassified samples found.")


# ============================================================
# 7. PRINT INDIVIDUAL MISCLASSIFICATIONS
# ============================================================

print()
print("=" * 70)
print("INDIVIDUAL MISCLASSIFIED SAMPLES")
print("=" * 70)

if len(errors) == 0:

    print("No errors found.")

else:

    for index, row in errors.iterrows():

        print()
        print("-" * 70)

        if "conversation_id" in row:

            print(
                "Conversation ID :",
                row["conversation_id"]
            )

        print(
            "Actual class    :",
            row["actual_class"]
        )

        print(
            "Predicted class :",
            row["predicted_class"]
        )

        if "conversation" in row:

            print(
                "Conversation:"
            )

            print(
                row["conversation"]
            )


# ============================================================
# 8. SAVE MISCLASSIFIED SAMPLES
# ============================================================

errors.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print("=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print()
print(
    "Misclassified samples saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 70)