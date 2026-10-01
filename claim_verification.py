import re


# ============================================================
# NARRATIVEGUARD - CLAIM VERIFICATION ENGINE
# ============================================================

# Each claim type is mapped to:
# 1. Keywords used to identify the claim
# 2. Evidence expected to verify the claim
# 3. Safe action when the claim cannot be verified


CLAIM_RULES = {

    "money_transfer_claim": {

        "keywords": [
            "sent",
            "transferred",
            "transfer",
            "credited",
            "received",
            "mistakenly sent",
            "accidentally sent",
            "payment made",
            "money sent",
            "upi transfer",
            "transaction"
        ],

        "evidence_required": [
            "Bank transaction record",
            "UTR / transaction reference number",
            "Actual account credit"
        ],

        "safe_action": (
            "Do not return or transfer money based only on "
            "the caller's statement. Verify the actual "
            "account transaction independently."
        )
    },


    "otp_claim": {

        "keywords": [
            "otp",
            "one time password",
            "verification code",
            "verify code"
        ],

        "evidence_required": [
            "Official service verification process"
        ],

        "safe_action": (
            "Never share OTPs with callers. Verify the request "
            "through the official application or website."
        )
    },


    "kyc_claim": {

        "keywords": [
            "kyc",
            "re-kyc",
            "kyc pending",
            "kyc update",
            "verify your account",
            "account verification",
            "identity verification"
        ],

        "evidence_required": [
            "Official account status",
            "Official KYC notification",
            "Verified bank/application portal"
        ],

        "safe_action": (
            "Verify the KYC status through the official "
            "bank or service application. Do not share "
            "passwords, OTPs or sensitive credentials."
        )
    },


    "refund_claim": {

        "keywords": [
            "refund",
            "refund pending",
            "refund process",
            "refund amount",
            "money back",
            "return the money"
        ],

        "evidence_required": [
            "Original transaction record",
            "Official refund status",
            "Transaction/reference number"
        ],

        "safe_action": (
            "Verify the original transaction and refund status "
            "through the official merchant or payment platform."
        )
    },


    "loan_claim": {

        "keywords": [
            "loan approved",
            "loan approval",
            "loan sanctioned",
            "loan amount",
            "processing fee",
            "loan fee",
            "loan application"
        ],

        "evidence_required": [
            "Official loan application status",
            "Official lender documentation",
            "Verified lender information"
        ],

        "safe_action": (
            "Do not pay advance processing fees based only "
            "on a caller's claim. Verify the loan through "
            "the official lender."
        )
    },


    "investment_claim": {

        "keywords": [
            "guaranteed profit",
            "guaranteed return",
            "profit",
            "investment",
            "stock tips",
            "returns",
            "300% profit",
            "high return"
        ],

        "evidence_required": [
            "Official investment platform",
            "Regulatory registration where applicable",
            "Investment documentation"
        ],

        "safe_action": (
            "Do not invest based on guaranteed-return claims. "
            "Verify the investment platform and regulatory "
            "information independently."
        )
    },


    "account_block_claim": {

        "keywords": [
            "account blocked",
            "account will be blocked",
            "account block",
            "account suspended",
            "account freeze",
            "will be suspended",
            "legal action"
        ],

        "evidence_required": [
            "Official account status",
            "Official bank/service notification"
        ],

        "safe_action": (
            "Do not respond to pressure by sharing credentials "
            "or OTPs. Check the account status through the "
            "official bank or service channel."
        )
    }
}


# ============================================================
# CLAIM DETECTION
# ============================================================

def detect_claims(conversation):

    text = conversation.lower()

    detected_claims = []

    for claim_type, rule in CLAIM_RULES.items():

        matched_keywords = []

        for keyword in rule["keywords"]:

            if keyword.lower() in text:

                matched_keywords.append(keyword)

        if matched_keywords:

            detected_claims.append({
                "claim_type": claim_type,
                "matched_keywords": matched_keywords
            })

    return detected_claims


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

def extract_money_amounts(conversation):

    amounts = []

    # Matches:
    # ₹25,000
    # Rs 25,000
    # Rs. 25000
    # 25000 rupees

    patterns = [
        r"₹\s?[\d,]+(?:\.\d+)?",
        r"rs\.?\s?[\d,]+(?:\.\d+)?",
        r"[\d,]+(?:\.\d+)?\s?rupees"
    ]

    for pattern in patterns:

        matches = re.findall(
            pattern,
            conversation,
            flags=re.IGNORECASE
        )

        amounts.extend(matches)

    return list(dict.fromkeys(amounts))


# ============================================================
# EVIDENCE CHECK
# ============================================================

def check_evidence(conversation):

    text = conversation.lower()

    evidence_keywords = {

        "transaction_record": [
            "transaction record",
            "bank statement",
            "statement",
            "transaction history"
        ],

        "utr": [
            "utr",
            "transaction reference",
            "reference number",
            "transaction id"
        ],

        "account_credit": [
            "credited",
            "credit received",
            "amount credited",
            "money received"
        ],

        "official_portal": [
            "official website",
            "official app",
            "bank app",
            "official portal"
        ],

        "notification": [
            "official notification",
            "bank notification",
            "email notification",
            "sms notification"
        ]
    }

    evidence_found = []

    for evidence_type, keywords in evidence_keywords.items():

        for keyword in keywords:

            if keyword in text:

                evidence_found.append(
                    evidence_type
                )

                break

    return list(
        dict.fromkeys(evidence_found)
    )


# ============================================================
# CLAIM CONSISTENCY CHECK
# ============================================================

def analyze_consistency(
    conversation,
    detected_claims
):

    text = conversation.lower()

    inconsistencies = []

    # --------------------------------------------------------
    # MONEY CLAIM WITHOUT ACTUAL CREDIT
    # --------------------------------------------------------

    has_transfer_claim = any(
        claim["claim_type"] == "money_transfer_claim"
        for claim in detected_claims
    )

    has_credit = (
        "credited" in text
        or "credit received" in text
        or "money received" in text
    )

    if has_transfer_claim and not has_credit:

        inconsistencies.append(
            "Money transfer is claimed but actual "
            "account credit is not established."
        )

    # --------------------------------------------------------
    # GUARANTEED INVESTMENT RETURNS
    # --------------------------------------------------------

    if (
        "guaranteed profit" in text
        or "guaranteed return" in text
        or "300% profit" in text
    ):

        inconsistencies.append(
            "Guaranteed or unusually high investment "
            "returns are claimed without independent evidence."
        )

    # --------------------------------------------------------
    # OTP + ACCOUNT THREAT
    # --------------------------------------------------------

    if (
        "otp" in text
        and (
            "blocked" in text
            or "suspended" in text
            or "legal action" in text
        )
    ):

        inconsistencies.append(
            "OTP request is combined with account-blocking "
            "or legal-action pressure."
        )

    return inconsistencies


# ============================================================
# VERIFICATION STATUS
# ============================================================

def determine_status(
    detected_claims,
    evidence_found,
    inconsistencies
):

    if not detected_claims:

        return "NO_CLAIM_DETECTED"

    if inconsistencies:

        return "SUSPICIOUS"

    if evidence_found:

        return "PARTIALLY_VERIFIABLE"

    return "UNVERIFIED"


# ============================================================
# MAIN CLAIM VERIFICATION FUNCTION
# ============================================================

def verify_claim(conversation):

    if not isinstance(
        conversation,
        str
    ):

        raise ValueError(
            "Conversation must be a string."
        )

    conversation = conversation.strip()

    if not conversation:

        raise ValueError(
            "Conversation cannot be empty."
        )

    # --------------------------------------------------------
    # DETECT CLAIMS
    # --------------------------------------------------------

    detected_claims = detect_claims(
        conversation
    )

    # --------------------------------------------------------
    # EXTRACT MONEY
    # --------------------------------------------------------

    money_amounts = extract_money_amounts(
        conversation
    )

    # --------------------------------------------------------
    # FIND EVIDENCE
    # --------------------------------------------------------

    evidence_found = check_evidence(
        conversation
    )

    # --------------------------------------------------------
    # CONSISTENCY
    # --------------------------------------------------------

    inconsistencies = analyze_consistency(
        conversation,
        detected_claims
    )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    verification_status = determine_status(
        detected_claims,
        evidence_found,
        inconsistencies
    )

    # --------------------------------------------------------
    # CLAIM DETAILS
    # --------------------------------------------------------

    claim_details = []

    for claim in detected_claims:

        claim_type = claim[
            "claim_type"
        ]

        rule = CLAIM_RULES[
            claim_type
        ]

        claim_details.append({

            "claim_type":
                claim_type,

            "matched_keywords":
                claim["matched_keywords"],

            "evidence_required":
                rule["evidence_required"],

            "safe_action":
                rule["safe_action"]
        })

    # --------------------------------------------------------
    # FINAL ACTION
    # --------------------------------------------------------

    if verification_status == "SUSPICIOUS":

        final_action = (
            "Do not act on the caller's claim. "
            "Independently verify the information "
            "through an official channel."
        )

    elif verification_status == "UNVERIFIED":

        final_action = (
            "The claim could not be independently "
            "verified from the conversation. "
            "Do not transfer money or share sensitive "
            "information until verification is completed."
        )

    elif verification_status == "PARTIALLY_VERIFIABLE":

        final_action = (
            "Some verification-related information "
            "is present, but independent verification "
            "is still required before taking financial action."
        )

    else:

        final_action = (
            "No financial claim requiring verification "
            "was detected."
        )

    return {

        "verification_status":
            verification_status,

        "detected_claims":
            claim_details,

        "money_amounts":
            money_amounts,

        "evidence_found":
            evidence_found,

        "inconsistencies":
            inconsistencies,

        "final_action":
            final_action
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_conversation = """
    Sir, I accidentally sent ₹25,000 instead of ₹2,500.
    You received ₹25,000. Please return ₹22,500 immediately.
    """

    result = verify_claim(
        test_conversation
    )

    print()
    print("=" * 70)
    print("NARRATIVEGUARD CLAIM VERIFICATION ENGINE")
    print("=" * 70)

    print()
    print("Conversation:")
    print(test_conversation)

    print()
    print("Verification Status:")
    print(
        result["verification_status"]
    )

    print()
    print("Detected Claims:")

    for claim in result[
        "detected_claims"
    ]:

        print(
            " -",
            claim["claim_type"]
        )

    print()
    print("Money Amounts:")

    for amount in result[
        "money_amounts"
    ]:

        print(
            " -",
            amount
        )

    print()
    print("Evidence Found:")

    if result["evidence_found"]:

        for evidence in result[
            "evidence_found"
        ]:

            print(
                " -",
                evidence
            )

    else:

        print(
            " - None"
        )

    print()
    print("Inconsistencies:")

    if result["inconsistencies"]:

        for item in result[
            "inconsistencies"
        ]:

            print(
                " -",
                item
            )

    else:

        print(
            " - None"
        )

    print()
    print("Final Action:")
    print(
        result["final_action"]
    )

    print()
    print("=" * 70)