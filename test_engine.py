import pandas as pd

from modules.narrativeguard_engine import analyze_conversation


DATASET_PATH = "dataset/annotated/narrativeguard_dataset.csv"


print("=" * 70)
print("NARRATIVEGUARD ENGINE V2 EVALUATION")
print("=" * 70)


df = pd.read_csv(DATASET_PATH)

print(f"\nTotal conversations: {len(df)}")

results = []


for _, row in df.iterrows():

    result = analyze_conversation(
        row["conversation"],
        row["verification_status"]
    )

    results.append({

        "conversation_id":
            row["conversation_id"],

        "actual_scam_type":
            row["scam_type"],

        "predicted_scam_type":
            result["predicted_scam_type"],

        "actual_risk_level":
            row["risk_level"],

        "predicted_risk_level":
            result["risk_level"],

        "risk_score":
            result["risk_score"]
    })


results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# SCAM TYPE ACCURACY
# ------------------------------------------------------------

results_df["scam_type_correct"] = (
    results_df["actual_scam_type"]
    ==
    results_df["predicted_scam_type"]
)


scam_accuracy = (
    results_df["scam_type_correct"].mean()
    * 100
)


# ------------------------------------------------------------
# RISK LEVEL ACCURACY
# ------------------------------------------------------------

results_df["risk_correct"] = (
    results_df["actual_risk_level"]
    ==
    results_df["predicted_risk_level"]
)


risk_accuracy = (
    results_df["risk_correct"].mean()
    * 100
)


# ------------------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DETAILED RESULTS")
print("=" * 70)

print(
    results_df[
        [
            "conversation_id",
            "actual_scam_type",
            "predicted_scam_type",
            "actual_risk_level",
            "predicted_risk_level",
            "risk_score"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EVALUATION SUMMARY")
print("=" * 70)

print(
    f"\nScam-type accuracy: "
    f"{scam_accuracy:.2f}%"
)

print(
    f"Correct scam-type predictions: "
    f"{results_df['scam_type_correct'].sum()}/{len(results_df)}"
)

print(
    f"\nRisk-level accuracy: "
    f"{risk_accuracy:.2f}%"
)

print(
    f"Correct risk predictions: "
    f"{results_df['risk_correct'].sum()}/{len(results_df)}"
)


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

output_path = "dataset/engine_evaluation.csv"

results_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nResults saved to:\n{output_path}"
)

print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)