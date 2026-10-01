# ============================================================
# NARRATIVEGUARD - VERIFICATION GUIDANCE ENGINE
# ============================================================

"""
This module converts detected scam/risk/claim signals into
actionable verification guidance for the user.

Important:
NarrativeGuard does NOT directly access or verify a user's
bank account. Instead, it tells the user what evidence to
check through an independent official channel.
"""


# ============================================================
# GUIDANCE TEMPLATES
# ============================================================

GUIDANCE = {

    "money_transfer_claim": {

        "title": "Suspicious Money Transfer Claim",

        "what_to_verify": [
            "Check your own bank or payment application.",
            "Check your actual transaction history.",
            "Confirm whether the claimed amount was actually credited.",
            "Verify the sender and exact transaction amount.",
            "Check the transaction reference in your own account."
        ],

        "do_not": [
            "Do not return money only because the caller claims it was transferred.",
            "Do not trust screenshots or payment proofs sent by the caller.",
            "Do not click links provided by the caller.",
            "Do not share OTP, PIN, CVV, password or other credentials."
        ],

        "safe_action": (
            "Do not transfer or return money until the transaction "
            "is independently verified through your own banking "
            "or payment application."
        )
    },


    "otp_request": {

        "title": "OTP Request Detected",

        "what_to_verify": [
            "Check whether the request is actually associated "
            "with an action you initiated.",
            "Contact the bank or service through its official channel."
        ],

        "do_not": [
            "Do not share the OTP with the caller.",
            "Do not read OTPs aloud.",
            "Do not share PIN, CVV, password or card details.",
            "Do not click links provided during the call."
        ],

        "safe_action": (
            "End the conversation and contact the organization "
            "through its official application, website or "
            "verified customer-care channel."
        )
    },


    "credential_request": {

        "title": "Sensitive Credential Request",

        "what_to_verify": [
            "Verify the caller's identity through an official channel.",
            "Check whether the requested information is actually "
            "required for the service."
        ],

        "do_not": [
            "Do not share passwords.",
            "Do not share banking PINs.",
            "Do not share CVV or complete card details.",
            "Do not share OTPs."
        ],

        "safe_action": (
            "Do not provide credentials. Verify the request "
            "through the organization's official channel."
        )
    },


    "investment_claim": {

        "title": "Investment Claim Requires Verification",

        "what_to_verify": [
            "Verify the company or platform independently.",
            "Check whether the organization is officially registered.",
            "Verify the investment product through an appropriate "
            "official or regulatory source.",
            "Check whether the promised returns are realistic."
        ],

        "do_not": [
            "Do not transfer money based only on promised returns.",
            "Do not trust guaranteed-profit claims.",
            "Do not rely only on information supplied by the caller.",
            "Do not share banking credentials."
        ],

        "safe_action": (
            "Do not invest until the company, investment product "
            "and payment destination have been independently verified."
        )
    },


    "loan_claim": {

        "title": "Loan Offer Requires Verification",

        "what_to_verify": [
            "Verify the lender independently.",
            "Check the official loan application or website.",
            "Verify the loan terms and processing requirements.",
            "Confirm whether the requested payment is legitimate."
        ],

        "do_not": [
            "Do not pay an upfront fee solely because the caller demands it.",
            "Do not share OTPs or banking credentials.",
            "Do not install unknown applications sent by the caller."
        ],

        "safe_action": (
            "Verify the lender through an official channel before "
            "making any payment or sharing sensitive information."
        )
    },


    "kyc_claim": {

        "title": "KYC Verification Requires Caution",

        "what_to_verify": [
            "Verify the KYC request through the organization's official app or website.",
            "Check whether you actually initiated a KYC-related action.",
            "Confirm the identity of the organization."
        ],

        "do_not": [
            "Do not share OTPs.",
            "Do not share passwords or PINs.",
            "Do not provide sensitive information to an unverified caller.",
            "Do not click suspicious KYC links."
        ],

        "safe_action": (
            "Complete KYC only through the organization's official "
            "application, website or verified service channel."
        )
    },


    "refund_claim": {

        "title": "Refund Claim Requires Verification",

        "what_to_verify": [
            "Check whether you actually made the original payment.",
            "Check your own transaction history.",
            "Verify the refund status through the official merchant or service.",
            "Confirm the amount independently."
        ],

        "do_not": [
            "Do not share OTPs to receive a refund.",
            "Do not provide card PIN or CVV.",
            "Do not click refund links sent by unknown callers.",
            "Do not transfer money to receive a supposed refund."
        ],

        "safe_action": (
            "Verify the refund directly through the merchant's "
            "official application or website."
        )
    },


    "bank_impersonation": {

        "title": "Possible Bank Impersonation",

        "what_to_verify": [
            "Verify the caller through your bank's official channel.",
            "Check your account directly using your banking application.",
            "Check whether the claimed account problem actually exists."
        ],

        "do_not": [
            "Do not share OTPs.",
            "Do not share PINs or passwords.",
            "Do not provide complete card details.",
            "Do not follow instructions from an unverified caller."
        ],

        "safe_action": (
            "End the call and contact your bank using the official "
            "number or banking application."
        )
    }
}


# ============================================================
# DEFAULT GUIDANCE
# ============================================================

DEFAULT_GUIDANCE = {

    "title": "Independent Verification Recommended",

    "what_to_verify": [
        "Verify the caller's claim through an official channel.",
        "Check the relevant transaction or account information yourself.",
        "Confirm the information using an independent source."
    ],

    "do_not": [
        "Do not make an immediate payment.",
        "Do not share sensitive financial credentials.",
        "Do not rely solely on information provided by the caller."
    ],

    "safe_action": (
        "Pause before taking action and independently verify "
        "the claim through an official channel."
    )
}


# ============================================================
# NORMALIZE SCAM TYPE
# ============================================================

def normalize_scam_type(scam_type):

    if not scam_type:
        return ""

    scam_type = str(
        scam_type
    ).lower().strip()

    return scam_type


# ============================================================
# SELECT GUIDANCE
# ============================================================

def select_guidance(
    scam_type=None,
    detected_claims=None,
    signals=None
):
    """
    Select the most relevant verification guidance.

    Priority:
        1. Detected claim
        2. Risk signal
        3. Predicted scam class
        4. Default guidance
    """

    scam_type = normalize_scam_type(
        scam_type
    )

    detected_claims = (
        detected_claims
        if detected_claims
        else []
    )

    signals = (
        signals
        if signals
        else {}
    )

    # --------------------------------------------------------
    # 1. CLAIM-BASED GUIDANCE
    # --------------------------------------------------------

    claim_types = []

    for claim in detected_claims:

        if isinstance(claim, dict):

            claim_type = claim.get(
                "claim_type",
                ""
            )

        else:

            claim_type = str(
                claim
            )

        claim_types.append(
            str(claim_type).lower()
        )

    if "money_transfer_claim" in claim_types:

        return GUIDANCE[
            "money_transfer_claim"
        ]

    # --------------------------------------------------------
    # 2. RISK SIGNAL GUIDANCE
    # --------------------------------------------------------

    if signals.get(
        "otp_request",
        False
    ):

        return GUIDANCE[
            "otp_request"
        ]

    if signals.get(
        "credential_request",
        False
    ):

        return GUIDANCE[
            "credential_request"
        ]

    # --------------------------------------------------------
    # 3. SCAM CLASS GUIDANCE
    # --------------------------------------------------------

    if scam_type == "investment_scam":

        return GUIDANCE[
            "investment_claim"
        ]

    if scam_type == "loan_scam":

        return GUIDANCE[
            "loan_claim"
        ]

    if scam_type == "kyc_scam":

        return GUIDANCE[
            "kyc_claim"
        ]

    if scam_type == "refund_scam":

        return GUIDANCE[
            "refund_claim"
        ]

    if scam_type == "bank_impersonation_scam":

        return GUIDANCE[
            "bank_impersonation"
        ]

    if scam_type == "otp_payment_scam":

        return GUIDANCE[
            "otp_request"
        ]

    # --------------------------------------------------------
    # 4. DEFAULT
    # --------------------------------------------------------

    return DEFAULT_GUIDANCE


# ============================================================
# GENERATE GUIDANCE
# ============================================================

def generate_verification_guidance(
    scam_type=None,
    verification_status=None,
    detected_claims=None,
    signals=None
):
    """
    Generate complete user-facing verification guidance.
    """

    guidance = select_guidance(
        scam_type=scam_type,
        detected_claims=detected_claims,
        signals=signals
    )

    verification_status = (
        str(
            verification_status
        ).upper()
        if verification_status
        else "UNVERIFIED"
    )

    return {

        "title": guidance[
            "title"
        ],

        "verification_status":
            verification_status,

        "what_to_verify":
            guidance[
                "what_to_verify"
            ],

        "do_not":
            guidance[
                "do_not"
            ],

        "safe_action":
            guidance[
                "safe_action"
            ]
    }


# ============================================================
# DISPLAY GUIDANCE
# ============================================================

def display_guidance(result):

    print()
    print("=" * 70)
    print("NARRATIVEGUARD - VERIFICATION GUIDANCE")
    print("=" * 70)

    print()
    print(
        "Status :",
        result[
            "verification_status"
        ]
    )

    print()
    print(
        result[
            "title"
        ]
    )

    print()
    print("WHAT TO VERIFY")
    print("-" * 70)

    for item in result[
        "what_to_verify"
    ]:

        print(
            "  [✓]",
            item
        )

    print()
    print("DO NOT")
    print("-" * 70)

    for item in result[
        "do_not"
    ]:

        print(
            "  [!]",
            item
        )

    print()
    print("SAFE ACTION")
    print("-" * 70)

    print(
        result[
            "safe_action"
        ]
    )

    print()
    print("=" * 70)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("NARRATIVEGUARD VERIFICATION GUIDANCE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Test 1 - Money transfer claim
    # --------------------------------------------------------

    test_claims = [
        {
            "claim_type":
                "money_transfer_claim"
        }
    ]

    test_signals = {
        "urgency": True,
        "credential_request": False,
        "otp_request": False,
        "payment_request": True,
        "threat_or_pressure": True,
        "impersonation": False,
        "suspicious_link": False
    }

    result = generate_verification_guidance(

        scam_type="refund_scam",

        verification_status="SUSPICIOUS",

        detected_claims=test_claims,

        signals=test_signals
    )

    display_guidance(
        result
    )