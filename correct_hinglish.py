import pandas as pd
import re
from pathlib import Path

# ============================================================
# NARRATIVEGUARD HINGLISH LABEL CORRECTION
# ============================================================

INPUT_FILE = Path("dataset/processed/narrativeguard_hinglish.csv")
OUTPUT_FILE = Path("dataset/processed/narrativeguard_hinglish_corrected.csv")
REVIEW_FILE = Path("dataset/processed/hinglish_manual_review.csv")

VALID_LABELS = {
    "bank_impersonation_scam",
    "genuine_financial_conversation",
    "investment_scam",
    "job_scam",
    "kyc_scam",
    "loan_scam",
    "otp_payment_scam",
    "refund_scam",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize common variations
    text = text.replace("₹", " rs ")
    text = text.replace("rs.", " rs ")
    text = text.replace("rupees", " rs ")

    # Remove repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LABEL DETECTION
# ============================================================

def detect_strong_label(row):
    """
    Returns:
        (predicted_label, reason, confidence)

    confidence:
        HIGH   -> safe automated correction
        MEDIUM -> send to manual review
        NONE   -> retain existing label
    """

    text = normalize_text(row["conversation"])

    # --------------------------------------------------------
    # 1. JOB SCAM
    # --------------------------------------------------------

    job_patterns = [
        r"\bjob\b",
        r"\bwork from home\b",
        r"\bwfh\b",
        r"\bhr\b",
        r"\bjoining fee\b",
        r"\bregistration fee\b",
        r"\btraining fee\b",
        r"\bdata entry\b",
        r"\bappointment letter\b",
        r"\bresume\b",
        r"\bsalary\b",
    ]

    job_hits = sum(bool(re.search(p, text)) for p in job_patterns)

    if job_hits >= 2:
        return (
            "job_scam",
            "Multiple job-recruitment/fee indicators detected",
            "HIGH",
        )

    # --------------------------------------------------------
    # 2. LOAN SCAM
    # --------------------------------------------------------

    loan_patterns = [
        r"\bloan\b",
        r"\bloan app\b",
        r"\bprocessing fee\b",
        r"\binsurance premium\b",
        r"\bloan approved\b",
        r"\bdisbursal\b",
        r"\bdisbursement\b",
        r"\badvance fee\b",
    ]

    loan_hits = sum(bool(re.search(p, text)) for p in loan_patterns)

    if loan_hits >= 2:
        return (
            "loan_scam",
            "Multiple loan/advance-fee indicators detected",
            "HIGH",
        )

    # --------------------------------------------------------
    # 3. INVESTMENT SCAM
    # --------------------------------------------------------

    investment_patterns = [
        r"\binvestment\b",
        r"\binvest\b",
        r"\bcrypto\b",
        r"\bbitcoin\b",
        r"\bstock\b",
        r"\btrading\b",
        r"\bprofit\b",
        r"\breturns\b",
        r"\bdouble your money\b",
        r"\bguaranteed return\b",
        r"\bshare market\b",
    ]

    investment_hits = sum(
        bool(re.search(p, text)) for p in investment_patterns
    )

    if investment_hits >= 2:
        return (
            "investment_scam",
            "Multiple investment/profit indicators detected",
            "HIGH",
        )

    # --------------------------------------------------------
    # 4. KYC SCAM
    # --------------------------------------------------------

    kyc_patterns = [
        r"\bkyc\b",
        r"\baadhaar\b",
        r"\baadhar\b",
        r"\bpan card\b",
        r"\baccount.*block\b",
        r"\baccount.*deactivate\b",
        r"\baccount.*freeze\b",
        r"\bupdate.*kyc\b",
        r"\bkyc.*update\b",
        r"\bkyc.*pending\b",
    ]

    kyc_hits = sum(bool(re.search(p, text)) for p in kyc_patterns)

    if kyc_hits >= 1:
        # If the same conversation contains OTP request,
        # classify as OTP/payment scam instead.
        if re.search(r"\botp\b|\bpin\b|\bupi\b", text):
            return (
                "otp_payment_scam",
                "KYC/account claim combined with OTP/UPI/PIN request",
                "HIGH",
            )

        return (
            "kyc_scam",
            "KYC/Aadhaar/account verification indicator detected",
            "HIGH",
        )

    # --------------------------------------------------------
    # 5. OTP / PAYMENT SCAM
    # --------------------------------------------------------

    otp_patterns = [
        r"\botp\b",
        r"\bone time password\b",
        r"\bupi\b",
        r"\bupi pin\b",
        r"\bpin\b",
        r"\bverification code\b",
        r"\bpayment code\b",
        r"\bsend.*money\b",
        r"\btransfer.*money\b",
        r"\bpay.*immediately\b",
    ]

    otp_hits = sum(bool(re.search(p, text)) for p in otp_patterns)

    if otp_hits >= 1:

        # A genuine conversation may mention payment
        # without asking for OTP/PIN/credentials.
        credential_request = bool(
            re.search(
                r"\botp\b|\bupi pin\b|\bpin\b|\bverification code\b",
                text,
            )
        )

        payment_request = bool(
            re.search(
                r"\bsend.*money\b|\btransfer.*money\b|\bpay.*immediately\b",
                text,
            )
        )

        if credential_request or payment_request:
            return (
                "otp_payment_scam",
                "OTP/PIN/payment request detected",
                "HIGH",
            )

    # --------------------------------------------------------
    # 6. REFUND SCAM
    # --------------------------------------------------------

    refund_patterns = [
        r"\brefund\b",
        r"\brefund.*money\b",
        r"\bmoney.*sent.*by mistake\b",
        r"\bmistakenly.*transfer\b",
        r"\baccidentally.*transfer\b",
        r"\bsent.*wrong amount\b",
        r"\breturn.*money\b",
        r"\bsend.*back\b",
        r"\btransfer.*back\b",
    ]

    refund_hits = sum(
        bool(re.search(p, text)) for p in refund_patterns
    )

    if refund_hits >= 1:
        return (
            "refund_scam",
            "Refund/accidental-transfer money-return indicator detected",
            "HIGH",
        )

    # --------------------------------------------------------
    # 7. BANK IMPERSONATION
    # --------------------------------------------------------

    bank_patterns = [
        r"\bhdfc\b",
        r"\bsbi\b",
        r"\bstate bank\b",
        r"\bicici\b",
        r"\baxis bank\b",
        r"\bbank\b",
        r"\bpolice\b",
        r"\bcyber crime\b",
        r"\bcyber cell\b",
        r"\bcbi\b",
        r"\bacp\b",
        r"\binspector\b",
        r"\blaw enforcement\b",
        r"\binvestigation officer\b",
        r"\bspecial investigation team\b",
        r"\bdigital arrest\b",
        r"\bcriminal complaint\b",
        r"\blegal action\b",
    ]

    bank_hits = sum(bool(re.search(p, text)) for p in bank_patterns)

    if bank_hits >= 2:
        return (
            "bank_impersonation_scam",
            "Multiple authority/bank impersonation indicators detected",
            "HIGH",
        )

    # One very strong authority signal
    strong_authority = [
        r"\bdigital arrest\b",
        r"\bpolice.*call\b",
        r"\bcyber crime department\b",
        r"\bcyber cell\b",
        r"\bcbi\b",
        r"\blaw enforcement\b",
        r"\binvestigation officer\b",
    ]

    if any(re.search(p, text) for p in strong_authority):
        return (
            "bank_impersonation_scam",
            "Strong law-enforcement impersonation indicator detected",
            "HIGH",
        )

    # --------------------------------------------------------
    # 8. LOTTERY / PRIZE
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Lottery is NOT automatically refund_scam.
    #
    # Our 8-class taxonomy has no lottery class.
    # Therefore these cases should NOT be force-labelled
    # automatically. Send them for review.
    # --------------------------------------------------------

    lottery_patterns = [
        r"\blottery\b",
        r"\bwon\b.*\bprize\b",
        r"\bcongratulations\b.*\bwon\b",
        r"\bprocessing fee\b.*\bprize\b",
        r"\b₹.*lakh.*lottery\b",
    ]

    if any(re.search(p, text) for p in lottery_patterns):
        return (
            None,
            "Lottery/prize scam detected but no dedicated NarrativeGuard class",
            "MEDIUM",
        )

    # --------------------------------------------------------
    # No strong correction
    # --------------------------------------------------------

    return (
        None,
        "No strong semantic correction rule triggered",
        "NONE",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NARRATIVEGUARD HINGLISH LABEL CORRECTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        print(f"\nERROR: Input file not found:")
        print(INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    print(f"\nOriginal samples: {len(df)}")

    # Preserve original label
    df["original_label"] = df["narrativeguard_label"]

    corrected_labels = []
    correction_reasons = []
    correction_confidence = []
    correction_status = []

    for _, row in df.iterrows():

        original = row["narrativeguard_label"]

        detected, reason, confidence = detect_strong_label(row)

        # ----------------------------------------------------
        # HIGH confidence correction
        # ----------------------------------------------------

        if confidence == "HIGH" and detected in VALID_LABELS:

            if detected != original:

                corrected_labels.append(detected)
                correction_reasons.append(reason)
                correction_confidence.append(confidence)
                correction_status.append("CORRECTED")

            else:

                corrected_labels.append(original)
                correction_reasons.append("Existing label confirmed")
                correction_confidence.append(confidence)
                correction_status.append("CONFIRMED")

        # ----------------------------------------------------
        # MEDIUM confidence
        # ----------------------------------------------------

        elif confidence == "MEDIUM":

            corrected_labels.append(original)
            correction_reasons.append(reason)
            correction_confidence.append(confidence)
            correction_status.append("REVIEW")

        # ----------------------------------------------------
        # No correction
        # ----------------------------------------------------

        else:

            corrected_labels.append(original)
            correction_reasons.append(reason)
            correction_confidence.append(confidence)
            correction_status.append("UNCHANGED")

    df["narrativeguard_label"] = corrected_labels
    df["correction_reason"] = correction_reasons
    df["correction_confidence"] = correction_confidence
    df["correction_status"] = correction_status

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 70)
    print("CORRECTION SUMMARY")
    print("=" * 70)

    print("\nCorrection status:")
    print(df["correction_status"].value_counts().to_string())

    print("\nOriginal distribution:")
    print(
        df["original_label"]
        .value_counts()
        .to_string()
    )

    print("\nCorrected distribution:")
    print(
        df["narrativeguard_label"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # SHOW CORRECTIONS
    # ========================================================

    corrections = df[
        df["correction_status"] == "CORRECTED"
    ]

    print("\n" + "=" * 70)
    print(f"AUTOMATIC CORRECTIONS: {len(corrections)}")
    print("=" * 70)

    if len(corrections) > 0:

        for _, row in corrections.head(50).iterrows():

            print("\n" + "-" * 60)

            print(
                "OLD:",
                row["original_label"]
            )

            print(
                "NEW:",
                row["narrativeguard_label"]
            )

            print(
                "REASON:",
                row["correction_reason"]
            )

            print(
                "TEXT:",
                row["conversation"]
            )

    # ========================================================
    # MANUAL REVIEW FILE
    # ========================================================

    review_df = df[
        df["correction_status"] == "REVIEW"
    ].copy()

    review_columns = [
        "conversation",
        "original_label",
        "narrativeguard_label",
        "scam_category",
        "caller_type",
        "urgency",
        "blackmail",
        "correction_reason",
        "correction_confidence",
        "correction_status",
        "language_style",
        "source",
    ]

    review_df[review_columns].to_csv(
        REVIEW_FILE,
        index=False
    )

    # ========================================================
    # SAVE CORRECTED DATASET
    # ========================================================

    # Keep the full audit columns in corrected dataset.
    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)

    print(f"\nCorrected dataset:")
    print(OUTPUT_FILE)

    print(f"\nManual review dataset:")
    print(REVIEW_FILE)

    print("\n" + "=" * 70)
    print("LABEL CORRECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()