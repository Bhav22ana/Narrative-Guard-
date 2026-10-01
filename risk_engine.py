import re


# ============================================================
# NARRATIVEGUARD - RISK ASSESSMENT ENGINE
# ============================================================

SCAM_WEIGHTS = {
    "bank_impersonation_scam": 25,
    "genuine_financial_conversation": 0,
    "investment_scam": 30,
    "job_scam": 25,
    "kyc_scam": 30,
    "loan_scam": 30,
    "otp_payment_scam": 35,
    "refund_scam": 25
}


# ============================================================
# SIGNAL DETECTION
# ============================================================

def detect_signals(conversation):

    text = conversation.lower()

    signals = {
        "urgency": False,
        "credential_request": False,
        "otp_request": False,
        "payment_request": False,
        "threat_or_pressure": False,
        "impersonation": False,
        "suspicious_link": False
    }

    # --------------------------------------------------------
    # URGENCY
    # --------------------------------------------------------

    urgency_patterns = [
        r"\burgent\b",
        r"\bimmediately\b",
        r"\bimmediate\b",
        r"\btoday\b",
        r"\bright now\b",
        r"\basap\b",
        r"\bjaldi\b",
        r"\bturant\b",
        r"\bquickly\b",
        r"\bwithin \d+ minutes?\b"
    ]

    # --------------------------------------------------------
    # CREDENTIAL REQUEST
    # --------------------------------------------------------

    credential_patterns = [
        r"\bpassword\b",
        r"\bpin\b",
        r"\bpasscode\b",
        r"\bcard number\b",
        r"\bcard details?\b",
        r"\bcvv\b",
        r"\baccount details\b",
        r"\bbank details\b",
        r"\bcredentials?\b"
    ]

    # --------------------------------------------------------
    # OTP
    # --------------------------------------------------------

    otp_patterns = [
        r"\botp\b",
        r"\bone time password\b",
        r"\bverification code\b",
        r"\bverify code\b"
    ]

    # --------------------------------------------------------
    # PAYMENT
    # --------------------------------------------------------

    payment_patterns = [
        r"\bsend money\b",
        r"\bsend\s+(?:₹|rs\.?|inr)?\s*\d[\d,]*\b",
        r"\bpay\b",
        r"\bpayment\b",
        r"\btransfer\b",
        r"\bprocessing fee\b",
        r"\bclearance charge\b",
        r"\bfee\b",
        r"\bupi\b",
        r"\brefund\b"
    ]

    # --------------------------------------------------------
    # THREAT / PRESSURE
    # --------------------------------------------------------

    threat_patterns = [
        r"\bblocked\b",
        r"\bblock your account\b",
        r"\bblock the account\b",
        r"\blegal action\b",
        r"\bpolice\b",
        r"\barrest\b",
        r"\bpenalty\b",
        r"\bdeleted\b",
        r"\bsuspend\b",
        r"\bsuspended\b",
        r"\bblackmail\b",
        r"\botherwise\b",
        r"\blast warning\b"
    ]

    # --------------------------------------------------------
    # IMPERSONATION
    # --------------------------------------------------------

    impersonation_patterns = [
        r"\bfrom sbi\b",
        r"\bfrom hdfc\b",
        r"\bfrom icici\b",
        r"\bfrom axis bank\b",
        r"\bcustomer care\b",
        r"\bcustomer support\b",
        r"\bbank officer\b",
        r"\bbank representative\b",
        r"\bepfo\b",
        r"\bnps helpline\b",
        r"\brbi\b",
        r"\bsebi\b",
        r"\bgovernment officer\b",
        r"\bamazon\b",
        r"\bnetflix\b"
    ]

    # --------------------------------------------------------
    # SUSPICIOUS LINKS
    # --------------------------------------------------------

    link_patterns = [
        r"https?://",
        r"\bwww\.",
        r"\b[\w.-]+\.(?:com|in|net|org|co)\b"
    ]

    # --------------------------------------------------------
    # APPLY SIGNALS
    # --------------------------------------------------------

    if any(re.search(pattern, text) for pattern in urgency_patterns):
        signals["urgency"] = True

    if any(re.search(pattern, text) for pattern in credential_patterns):
        signals["credential_request"] = True

    if any(re.search(pattern, text) for pattern in otp_patterns):
        signals["otp_request"] = True

    if any(re.search(pattern, text) for pattern in payment_patterns):
        signals["payment_request"] = True

    if any(re.search(pattern, text) for pattern in threat_patterns):
        signals["threat_or_pressure"] = True

    if any(re.search(pattern, text) for pattern in impersonation_patterns):
        signals["impersonation"] = True

    if any(re.search(pattern, text) for pattern in link_patterns):
        signals["suspicious_link"] = True

    return signals


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk(
    conversation,
    predicted_class,
    confidence
):

    signals = detect_signals(conversation)

    score = 0

    # --------------------------------------------------------
    # 1. MODEL SIGNAL
    # --------------------------------------------------------

    score += SCAM_WEIGHTS.get(
        predicted_class,
        0
    )

    # --------------------------------------------------------
    # 2. MODEL CONFIDENCE
    # --------------------------------------------------------

    if confidence >= 0.90:
        score += 25

    elif confidence >= 0.75:
        score += 18

    elif confidence >= 0.60:
        score += 10

    # --------------------------------------------------------
    # 3. BEHAVIORAL SIGNALS
    # --------------------------------------------------------

    if signals["urgency"]:
        score += 10

    if signals["credential_request"]:
        score += 15

    if signals["otp_request"]:
        score += 20

    if signals["payment_request"]:
        score += 15

    if signals["threat_or_pressure"]:
        score += 10

    if signals["impersonation"]:
        score += 10

    if signals["suspicious_link"]:
        score += 10

    # --------------------------------------------------------
    # 4. STRONG BEHAVIORAL SIGNAL OVERRIDE
    #
    # Multiple dangerous signals should not result in LOW
    # risk merely because the ML confidence is low.
    # --------------------------------------------------------

    critical_signals = sum([
        signals["otp_request"],
        signals["credential_request"],
        signals["payment_request"],
        signals["threat_or_pressure"],
        signals["impersonation"]
    ])

    # Strong combination:
    # OTP + threat + impersonation
    if (
        signals["otp_request"]
        and signals["threat_or_pressure"]
        and signals["impersonation"]
    ):
        score = max(score, 85)

    # Very suspicious combination of 4+ critical signals
    elif critical_signals >= 4:
        score = max(score, 85)

    # 3 critical signals
    elif critical_signals >= 3:
        score = max(score, 70)

    # OTP + credential request is inherently high-risk
    elif (
        signals["otp_request"]
        and signals["credential_request"]
    ):
        score = max(score, 75)

    # --------------------------------------------------------
    # 5. CAP SCORE
    # --------------------------------------------------------

    score = min(score, 100)

    # --------------------------------------------------------
    # 6. RISK LEVEL
    # --------------------------------------------------------

    if predicted_class == "genuine_financial_conversation":

        # Strong behavioral evidence overrides
        # the genuine prediction.
        if critical_signals >= 3:
            risk_level = "HIGH"

        elif score >= 65:
            risk_level = "HIGH"

        elif score >= 40:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

    else:

        if score >= 70:
            risk_level = "HIGH"

        elif score >= 40:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

    # --------------------------------------------------------
    # 7. RECOMMENDATION
    # --------------------------------------------------------

    if risk_level == "HIGH":

        recommendation = (
            "Do not share OTP, passwords, card details "
            "or other credentials. Do not transfer money. "
            "Verify the claim through the official bank "
            "or service channel."
        )

    elif risk_level == "MEDIUM":

        recommendation = (
            "Do not act immediately. Verify the caller "
            "and claim through an official channel before "
            "sharing information or making any payment."
        )

    else:

        recommendation = (
            "No strong immediate risk signal was detected. "
            "Continue to verify financial claims before "
            "taking action."
        )

    # --------------------------------------------------------
    # 8. RETURN RESULT
    # --------------------------------------------------------

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "signals": signals,
        "recommendation": recommendation
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    conversation = """
    Hello sir, this is SBI customer care.

    Your account will be blocked today.

    Please share your OTP to verify your account.
    """

    predicted_class = "otp_payment_scam"
    confidence = 0.9337

    result = calculate_risk(
        conversation,
        predicted_class,
        confidence
    )

    print()
    print("=" * 70)
    print("NARRATIVEGUARD RISK ENGINE TEST")
    print("=" * 70)

    print()
    print("Predicted class:")
    print(predicted_class)

    print()
    print("Confidence:")
    print(f"{confidence * 100:.2f}%")

    print()
    print("Risk score:")
    print(result["risk_score"])

    print()
    print("Risk level:")
    print(result["risk_level"])

    print()
    print("Detected signals:")

    for signal, detected in result["signals"].items():
        print(f"{signal:25s}: {detected}")

    print()
    print("Recommended action:")
    print(result["recommendation"])

    print()
    print("=" * 70)