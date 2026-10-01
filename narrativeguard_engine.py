# ============================================================
# NARRATIVEGUARD - MASTER ENGINE
# ============================================================
#
# Complete NarrativeGuard pipeline:
#
# Conversation
#      ↓
# XLM-RoBERTa Classification
#      ↓
# Risk Engine
#      ↓
# Claim Verification
#      ↓
# Verification Guidance
#      ↓
# Final Decision
#      ↓
# JSON Result
#
# ============================================================

import os
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# Existing NarrativeGuard modules
from risk_engine import calculate_risk
from claim_verification import verify_claim
from verification_guidance import generate_verification_guidance


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/narrativeguard_xlmr"

LABEL_MAPPING_PATH = (
    "dataset/tokenized/label_mapping.json"
)

OUTPUT_PATH = (
    "evaluation/narrativeguard_result.json"
)

MAX_LENGTH = 256


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

def load_label_mapping():

    if not os.path.exists(
        LABEL_MAPPING_PATH
    ):

        raise FileNotFoundError(
            f"\nLabel mapping not found:\n"
            f"{LABEL_MAPPING_PATH}"
        )

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        mapping = json.load(file)

    # --------------------------------------------------------
    # FORMAT 1:
    #
    # {
    #   "0": "bank_impersonation_scam",
    #   "1": "genuine_financial_conversation"
    # }
    # --------------------------------------------------------

    if all(
        str(key).isdigit()
        for key in mapping.keys()
    ):

        return {
            int(key): str(value)
            for key, value in mapping.items()
        }

    # --------------------------------------------------------
    # FORMAT 2:
    #
    # {
    #   "bank_impersonation_scam": 0,
    #   "genuine_financial_conversation": 1
    # }
    # --------------------------------------------------------

    if all(
        isinstance(value, int)
        for value in mapping.values()
    ):

        return {
            int(value): str(key)
            for key, value in mapping.items()
        }

    # --------------------------------------------------------
    # FORMAT 3:
    #
    # {
    #   "bank_impersonation_scam": "0",
    #   "genuine_financial_conversation": "1"
    # }
    # --------------------------------------------------------

    if all(
        str(value).isdigit()
        for value in mapping.values()
    ):

        return {
            int(value): str(key)
            for key, value in mapping.items()
        }

    raise ValueError(
        "\nUnsupported label_mapping.json format.\n"
        f"Contents:\n{mapping}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "Loading NarrativeGuard XLM-R tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )

    print(
        "Loading NarrativeGuard XLM-R model..."
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_PATH,
            local_files_only=True
        )
    )

    model.to(DEVICE)

    model.eval()

    print(
        "XLM-R model loaded successfully."
    )

    print(
        "Device:",
        DEVICE
    )

    return tokenizer, model


# ============================================================
# XLM-R CLASSIFICATION
# ============================================================

def classify_conversation(
    conversation,
    tokenizer,
    model,
    label_mapping
):

    encoded = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    encoded = {
        key: value.to(DEVICE)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = model(
            **encoded
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    predicted_id = (
        torch.argmax(
            probabilities
        ).item()
    )

    confidence = (
        probabilities[
            predicted_id
        ].item()
    )

    predicted_class = (
        label_mapping[
            predicted_id
        ]
    )

    class_probabilities = {}

    for class_id, probability in enumerate(
        probabilities
    ):

        class_name = label_mapping.get(
            class_id,
            f"class_{class_id}"
        )

        class_probabilities[
            class_name
        ] = float(
            probability.item()
        )

    return {

        "predicted_class":
            predicted_class,

        "predicted_label_id":
            predicted_id,

        "confidence":
            confidence,

        "class_probabilities":
            class_probabilities
    }


# ============================================================
# RISK ENGINE ADAPTER
# ============================================================

def run_risk_engine(
    conversation,
    predicted_class,
    confidence
):

    """
    Calls the existing risk engine.

    Your risk_engine.py may accept different
    argument combinations, so this adapter tries
    the common forms without changing the existing
    risk engine.
    """

    # Try the full interface first.
    try:

        result = calculate_risk(
            conversation,
            predicted_class,
            confidence
        )

        if result is not None:
            return result

    except TypeError:
        pass

    # Try conversation + predicted class.
    try:

        result = calculate_risk(
            conversation,
            predicted_class
        )

        if result is not None:
            return result

    except TypeError:
        pass

    # Try conversation only.
    try:

        result = calculate_risk(
            conversation
        )

        if result is not None:
            return result

    except TypeError:
        pass

    raise RuntimeError(
        "\nUnable to call calculate_risk(). "
        "Please check risk_engine.py."
    )


# ============================================================
# CLAIM VERIFICATION
# ============================================================

def run_claim_verification(
    conversation
):

    """
    Uses the existing verify_claim()
    function from claim_verification.py.
    """

    result = verify_claim(
        conversation
    )

    if result is None:

        return {}

    return result


# ============================================================
# SAFE DICTIONARY VALUE
# ============================================================

def get_value(
    data,
    keys,
    default=None
):

    if not isinstance(
        data,
        dict
    ):

        return default

    for key in keys:

        if key in data:

            return data[key]

    return default


# ============================================================
# EXTRACT CLAIMS
# ============================================================

def extract_claims(
    claim_result
):

    claims = get_value(
        claim_result,
        [
            "detected_claims",
            "claims"
        ],
        []
    )

    if claims is None:

        return []

    return claims


# ============================================================
# EXTRACT SIGNALS
# ============================================================

def extract_signals(
    risk_result
):

    signals = get_value(
        risk_result,
        [
            "signals",
            "detected_signals"
        ],
        {}
    )

    if signals is None:

        return {}

    return signals


# ============================================================
# DISPLAY CLASSIFICATION
# ============================================================

def display_classification(
    classification
):

    print()
    print(
        "SCAM CLASSIFICATION"
    )

    print(
        "-" * 70
    )

    print(
        "Predicted Scam Type :",
        classification[
            "predicted_class"
        ]
    )

    print(
        "Confidence          :",
        f"{classification['confidence'] * 100:.2f}%"
    )

    print()
    print(
        "Class Probabilities:"
    )

    for class_name, probability in (
        classification[
            "class_probabilities"
        ].items()
    ):

        print(
            f"  {class_name:<35}"
            f"{probability * 100:>7.2f}%"
        )


# ============================================================
# DISPLAY RISK
# ============================================================

def display_risk(
    risk_result
):

    print()
    print(
        "RISK ASSESSMENT"
    )

    print(
        "-" * 70
    )

    risk_score = get_value(
        risk_result,
        [
            "risk_score",
            "score"
        ],
        0
    )

    risk_level = get_value(
        risk_result,
        [
            "risk_level",
            "level"
        ],
        "UNKNOWN"
    )

    print(
        "Risk Score          :",
        risk_score,
        "/ 100"
    )

    print(
        "Risk Level          :",
        str(
            risk_level
        ).upper()
    )

    signals = extract_signals(
        risk_result
    )

    print()
    print(
        "Detected Risk Signals:"
    )

    if signals:

        for signal, value in (
            signals.items()
        ):

            if bool(value):

                print(
                    f"  [✓] {signal}"
                )

    else:

        print(
            "  None"
        )


# ============================================================
# DISPLAY CLAIM VERIFICATION
# ============================================================

def display_claim_verification(
    claim_result
):

    print()
    print(
        "CLAIM VERIFICATION"
    )

    print(
        "-" * 70
    )

    status = get_value(
        claim_result,
        [
            "verification_status",
            "status"
        ],
        "UNVERIFIED"
    )

    print(
        "Verification Status :",
        str(
            status
        ).upper()
    )

    # --------------------------------------------------------
    # CLAIMS
    # --------------------------------------------------------

    claims = extract_claims(
        claim_result
    )

    print()
    print(
        "Detected Claims:"
    )

    if claims:

        for claim in claims:

            if isinstance(
                claim,
                dict
            ):

                claim_name = (
                    claim.get(
                        "claim_type"
                    )
                    or
                    claim.get(
                        "type"
                    )
                    or
                    str(claim)
                )

            else:

                claim_name = str(
                    claim
                )

            print(
                "  -",
                claim_name
            )

    else:

        print(
            "  - None"
        )

    # --------------------------------------------------------
    # MONEY AMOUNTS
    # --------------------------------------------------------

    amounts = get_value(
        claim_result,
        [
            "money_amounts",
            "amounts"
        ],
        []
    )

    print()
    print(
        "Money Amounts:"
    )

    if amounts:

        for amount in amounts:

            print(
                "  -",
                amount
            )

    else:

        print(
            "  - None"
        )

    # --------------------------------------------------------
    # EVIDENCE
    # --------------------------------------------------------

    evidence = get_value(
        claim_result,
        [
            "evidence_found",
            "evidence"
        ],
        []
    )

    print()
    print(
        "Evidence Found:"
    )

    if evidence:

        for item in evidence:

            print(
                "  -",
                item
            )

    else:

        print(
            "  - None"
        )

    # --------------------------------------------------------
    # INCONSISTENCIES
    # --------------------------------------------------------

    inconsistencies = get_value(
        claim_result,
        [
            "inconsistencies",
            "inconsistency"
        ],
        []
    )

    print()
    print(
        "Inconsistencies:"
    )

    if inconsistencies:

        for item in inconsistencies:

            print(
                "  -",
                item
            )

    else:

        print(
            "  - None"
        )


# ============================================================
# VERIFICATION GUIDANCE
# ============================================================

def generate_guidance(
    predicted_class,
    verification_status,
    claims,
    signals
):

    return generate_verification_guidance(

        scam_type=predicted_class,

        verification_status=
            verification_status,

        detected_claims=
            claims,

        signals=
            signals
    )


# ============================================================
# DISPLAY GUIDANCE
# ============================================================

def display_guidance(
    guidance
):

    print()
    print(
        "VERIFICATION GUIDANCE"
    )

    print(
        "-" * 70
    )

    print(
        "Guidance:",
        guidance.get(
            "title",
            "Independent Verification Recommended"
        )
    )

    print()
    print(
        "WHAT TO VERIFY"
    )

    for item in guidance.get(
        "what_to_verify",
        []
    ):

        print(
            "  [✓]",
            item
        )

    print()
    print(
        "DO NOT"
    )

    for item in guidance.get(
        "do_not",
        []
    ):

        print(
            "  [!]",
            item
        )

    print()
    print(
        "SAFE ACTION"
    )

    print(
        guidance.get(
            "safe_action",
            "Independently verify the claim."
        )
    )


# ============================================================
# FINAL DECISION
# ============================================================

def determine_final_decision(
    risk_result,
    claim_result
):

    risk_score = get_value(
        risk_result,
        [
            "risk_score",
            "score"
        ],
        0
    )

    verification_status = str(
        get_value(
            claim_result,
            [
                "verification_status",
                "status"
            ],
            "UNVERIFIED"
        )
    ).upper()

    # --------------------------------------------------------
    # Suspicious/unverified claims require a protective
    # HIGH-RISK action.
    # --------------------------------------------------------

    if verification_status in [
        "SUSPICIOUS",
        "UNVERIFIED",
        "INCONSISTENT"
    ]:

        return {

            "decision":
                "HIGH RISK",

            "reason":
                (
                    "The financial claim could not "
                    "be independently verified."
                )
        }

    # --------------------------------------------------------
    # Numerical risk score
    # --------------------------------------------------------

    if risk_score >= 75:

        return {

            "decision":
                "HIGH RISK",

            "reason":
                (
                    "Multiple high-risk indicators "
                    "were detected."
                )
        }

    if risk_score >= 50:

        return {

            "decision":
                "MEDIUM RISK",

            "reason":
                (
                    "The conversation contains "
                    "suspicious financial risk indicators."
                )
        }

    if risk_score >= 25:

        return {

            "decision":
                "CAUTION",

            "reason":
                (
                    "Some suspicious indicators "
                    "were detected."
                )
        }

    return {

        "decision":
            "LOW RISK",

        "reason":
            (
                "No strong financial threat "
                "indicators were detected."
            )
    }


# ============================================================
# FINAL ACTION
# ============================================================

def generate_final_action(
    decision
):

    if decision == "HIGH RISK":

        return (
            "Do not transfer money or share OTPs, "
            "passwords, card details or other sensitive "
            "information. Independently verify the claim "
            "through an official channel."
        )

    if decision == "MEDIUM RISK":

        return (
            "Pause before taking financial action. "
            "Verify the caller and claim through an "
            "official channel."
        )

    if decision == "CAUTION":

        return (
            "Proceed carefully and independently verify "
            "the information before taking financial action."
        )

    return (
        "No immediate high-risk indicators were detected. "
        "Continue normal caution and verify important "
        "financial claims."
    )


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def analyze_conversation(
    conversation,
    tokenizer,
    model,
    label_mapping
):

    # --------------------------------------------------------
    # STEP 1: XLM-R
    # --------------------------------------------------------

    classification = classify_conversation(

        conversation,

        tokenizer,

        model,

        label_mapping
    )

    predicted_class = classification[
        "predicted_class"
    ]

    confidence = classification[
        "confidence"
    ]

    # --------------------------------------------------------
    # STEP 2: RISK ENGINE
    # --------------------------------------------------------

    risk_result = run_risk_engine(

        conversation,

        predicted_class,

        confidence
    )

    # --------------------------------------------------------
    # STEP 3: CLAIM VERIFICATION
    # --------------------------------------------------------

    claim_result = run_claim_verification(
        conversation
    )

    # --------------------------------------------------------
    # STEP 4: EXTRACT INFORMATION
    # --------------------------------------------------------

    claims = extract_claims(
        claim_result
    )

    signals = extract_signals(
        risk_result
    )

    verification_status = get_value(
        claim_result,
        [
            "verification_status",
            "status"
        ],
        "UNVERIFIED"
    )

    # --------------------------------------------------------
    # STEP 5: VERIFICATION GUIDANCE
    # --------------------------------------------------------

    guidance = generate_guidance(

        predicted_class,

        verification_status,

        claims,

        signals
    )

    # --------------------------------------------------------
    # STEP 6: FINAL DECISION
    # --------------------------------------------------------

    final_decision = (
        determine_final_decision(

            risk_result,

            claim_result
        )
    )

    final_action = (
        generate_final_action(
            final_decision[
                "decision"
            ]
        )
    )

    # --------------------------------------------------------
    # MASTER RESULT
    # --------------------------------------------------------

    return {

        "conversation":
            conversation,

        "classification":
            classification,

        "risk_assessment":
            risk_result,

        "claim_verification":
            claim_result,

        "verification_guidance":
            guidance,

        "final_decision":
            final_decision,

        "recommended_action":
            final_action
    }


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(
    result
):

    os.makedirs(
        "evaluation",
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    return OUTPUT_PATH


# ============================================================
# DISPLAY FINAL RESULT
# ============================================================

def display_final_result(
    result
):

    print()
    print(
        "=" * 70
    )

    print(
        "NARRATIVEGUARD - FINAL ANALYSIS"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    print()
    print(
        "INPUT CONVERSATION"
    )

    print(
        "-" * 70
    )

    print(
        result[
            "conversation"
        ].strip()
    )

    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    display_classification(
        result[
            "classification"
        ]
    )

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    display_risk(
        result[
            "risk_assessment"
        ]
    )

    # --------------------------------------------------------
    # CLAIM
    # --------------------------------------------------------

    display_claim_verification(
        result[
            "claim_verification"
        ]
    )

    # --------------------------------------------------------
    # GUIDANCE
    # --------------------------------------------------------

    display_guidance(
        result[
            "verification_guidance"
        ]
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    final_decision = result[
        "final_decision"
    ]

    print()
    print(
        "=" * 70
    )

    print(
        "FINAL NARRATIVEGUARD DECISION"
    )

    print(
        "=" * 70
    )

    print()
    print(
        "Decision:",
        final_decision[
            "decision"
        ]
    )

    print()
    print(
        "Reason:"
    )

    print(
        final_decision[
            "reason"
        ]
    )

    print()
    print(
        "Recommended Action:"
    )

    print(
        result[
            "recommended_action"
        ]
    )

    print()
    print(
        "=" * 70
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "NARRATIVEGUARD MASTER ENGINE"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load label mapping
    # --------------------------------------------------------

    label_mapping = (
        load_label_mapping()
    )

    print()
    print(
        "Label mapping loaded."
    )

    print(
        "Classes:",
        len(label_mapping)
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            f"\nModel directory not found:\n"
            f"{MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    tokenizer, model = (
        load_model()
    )

    # --------------------------------------------------------
    # TEST CONVERSATION
    # --------------------------------------------------------

    conversation = """

Sir, I accidentally sent ₹25,000 instead of ₹2,500.
You received ₹25,000. Please return ₹22,500 immediately.

"""

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    result = analyze_conversation(

        conversation,

        tokenizer,

        model,

        label_mapping
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    display_final_result(
        result
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_file = save_result(
        result
    )

    print()
    print(
        "Analysis saved to:",
        output_file
    )

    print()