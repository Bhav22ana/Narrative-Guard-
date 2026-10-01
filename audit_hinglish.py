import pandas as pd
import re

INPUT = "dataset/processed/narrativeguard_hinglish.csv"
OUTPUT = "dataset/processed/hinglish_label_audit.csv"

df = pd.read_csv(INPUT)

# ---------------------------------------------------------
# Keyword patterns associated with each NarrativeGuard class
# ---------------------------------------------------------

patterns = {
    "kyc_scam": [
        "kyc",
        "know your customer",
        "aadhaar update",
        "aadhaar deactivate",
        "pan update",
        "account block",
        "account blocked"
    ],

    "otp_payment_scam": [
        "otp",
        "one time password",
        "upi pin",
        "upi",
        "pin share",
        "payment",
        "transfer",
        "send money",
        "bhej",
        "bhejiye",
        "paise bhejo",
        "paise bhejiye"
    ],

    "bank_impersonation_scam": [
        "bank se",
        "bank bol",
        "bank manager",
        "bank officer",
        "police",
        "cyber crime",
        "cyber cell",
        "crime branch",
        "cbi",
        "court",
        "legal action",
        "digital arrest",
        "arrest"
    ],

    "investment_scam": [
        "investment",
        "invest",
        "trading",
        "crypto",
        "bitcoin",
        "stock",
        "share market",
        "profit",
        "returns",
        "double money"
    ],

    "refund_scam": [
        "refund",
        "refund amount",
        "money return",
        "return money",
        "accidentally transferred",
        "mistakenly transferred",
        "wrong transfer",
        "galti se",
        "galti",
        "refund dena"
    ],

    "job_scam": [
        "job",
        "work from home",
        "wfh",
        "salary",
        "hr",
        "recruitment",
        "registration fee",
        "joining fee",
        "training fee"
    ],

    "loan_scam": [
        "loan",
        "lending",
        "emi",
        "processing fee",
        "loan approved",
        "loan app",
        "insurance premium"
    ]
}


def detect_signals(text):

    text = str(text).lower()

    detected = []

    for category, keywords in patterns.items():

        matched = []

        for keyword in keywords:

            if keyword.lower() in text:
                matched.append(keyword)

        if matched:
            detected.append(
                (category, matched)
            )

    return detected


# ---------------------------------------------------------
# Audit
# ---------------------------------------------------------

audit_rows = []

for _, row in df.iterrows():

    text = row["conversation"]
    actual_label = row["narrativeguard_label"]

    signals = detect_signals(text)

    predicted_candidates = [
        category
        for category, matches in signals
    ]

    # No useful signal
    if not predicted_candidates:

        audit_status = "REVIEW"

    # Label appears among detected categories
    elif actual_label in predicted_candidates:

        audit_status = "CONSISTENT"

    # Strong mismatch
    else:

        audit_status = "POSSIBLE_MISMATCH"

    audit_rows.append({
        "conversation": text,
        "current_label": actual_label,
        "detected_categories": "|".join(
            predicted_candidates
        ),
        "audit_status": audit_status
    })


audit_df = pd.DataFrame(audit_rows)

audit_df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print("=" * 70)
print("NARRATIVEGUARD HINGLISH LABEL QUALITY AUDIT")
print("=" * 70)

print("\nTotal samples:", len(audit_df))

print("\n=== AUDIT STATUS ===")
print(
    audit_df["audit_status"]
    .value_counts()
    .to_string()
)

print("\n=== POSSIBLE MISMATCHES ===")

mismatches = audit_df[
    audit_df["audit_status"] == "POSSIBLE_MISMATCH"
]

print(
    "Possible mismatches:",
    len(mismatches)
)

for _, row in mismatches.head(30).iterrows():

    print("\n" + "-" * 70)

    print("CURRENT LABEL:")
    print(row["current_label"])

    print("DETECTED:")
    print(row["detected_categories"])

    print("CONVERSATION:")
    print(row["conversation"])


print("\n" + "=" * 70)
print("Audit saved to:")
print(OUTPUT)
print("=" * 70)