import torch

from transformers import (
    AutoTokenizer,
    XLMRobertaForSequenceClassification
)


# ============================================================
# NARRATIVEGUARD XLM-R PREDICTOR
# ============================================================

MODEL_PATH = "models/narrativeguard_xlmr"

MAX_LENGTH = 256


# ============================================================
# LABEL MAPPING
# ============================================================

LABELS = {
    0: "bank_impersonation_scam",
    1: "genuine_financial_conversation",
    2: "investment_scam",
    3: "job_scam",
    4: "kyc_scam",
    5: "loan_scam",
    6: "otp_payment_scam",
    7: "refund_scam"
}


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("Loading NarrativeGuard XLM-R tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
    fix_mistral_regex=False
)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading NarrativeGuard XLM-R model...")

model = XLMRobertaForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print("XLM-R model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_conversation(conversation):

    if not isinstance(conversation, str):

        raise ValueError(
            "Conversation must be a string."
        )

    conversation = conversation.strip()

    if not conversation:

        raise ValueError(
            "Conversation cannot be empty."
        )

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    inputs = tokenizer(
        conversation,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predicted_id = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_id
    ].item()

    # --------------------------------------------------------
    # ALL CLASS PROBABILITIES
    # --------------------------------------------------------

    class_probabilities = {}

    for class_id, class_name in LABELS.items():

        class_probabilities[class_name] = round(
            probabilities[
                0,
                class_id
            ].item(),
            4
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    result = {

        "predicted_class":
            LABELS[predicted_id],

        "confidence":
            round(confidence, 4),

        "confidence_percent":
            round(confidence * 100, 2),

        "class_probabilities":
            class_probabilities
    }

    return result


# ============================================================
# TEST MODE
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("NARRATIVEGUARD XLM-R PREDICTION TEST")
    print("=" * 70)

    test_conversation = """
    Hello sir, this is SBI customer care.
    Your account will be blocked today.
    Please share your OTP to verify your account.
    """

    result = predict_conversation(
        test_conversation
    )

    print()
    print("Conversation:")
    print(test_conversation)

    print()
    print("Predicted class:")
    print(
        result["predicted_class"]
    )

    print()
    print("Confidence:")
    print(
        f'{result["confidence_percent"]}%'
    )

    print()
    print("Class probabilities:")

    for class_name, probability in (
        result["class_probabilities"].items()
    ):

        print(
            f"{class_name:35s} "
            f"{probability:.4f}"
        )

    print()
    print("=" * 70)