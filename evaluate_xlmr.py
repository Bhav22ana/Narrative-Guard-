import os
import ast
import json
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import XLMRobertaForSequenceClassification

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


# ============================================================
# NARRATIVEGUARD
# FINAL XLM-RoBERTa TEST EVALUATION
# ============================================================

MODEL_PATH = "models/narrativeguard_xlmr"

TEST_PATH = "dataset/tokenized/test_tokenized.csv"

LABEL_MAPPING_PATH = "dataset/tokenized/label_mapping.json"

RESULTS_DIR = "evaluation"

BATCH_SIZE = 8
NUM_LABELS = 8
MAX_LENGTH = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("NARRATIVEGUARD - FINAL XLM-RoBERTa TEST EVALUATION")
print("=" * 70)

print()
print("Device :", DEVICE)
print("Model  :", MODEL_PATH)
print("Test   :", TEST_PATH)


# ============================================================
# 1. CHECK FILES
# ============================================================

print()
print("=" * 70)
print("CHECKING FILES")
print("=" * 70)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model directory not found: {MODEL_PATH}"
    )

if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_PATH}"
    )

print("PASS - Model directory found.")
print("PASS - Test dataset found.")


# ============================================================
# 2. LOAD LABEL MAPPING
# ============================================================

label_names = [
    "bank_impersonation_scam",
    "genuine_financial_conversation",
    "investment_scam",
    "job_scam",
    "kyc_scam",
    "loan_scam",
    "otp_payment_scam",
    "refund_scam"
]

if os.path.exists(LABEL_MAPPING_PATH):

    with open(
        LABEL_MAPPING_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        mapping = json.load(f)

    print()
    print("Label mapping loaded.")

    # Handle JSON mappings such as:
    # {"0": "bank_impersonation_scam", ...}

    if all(str(i) in mapping for i in range(NUM_LABELS)):

        label_names = [
            mapping[str(i)]
            for i in range(NUM_LABELS)
        ]

    # Handle reverse mapping:
    # {"bank_impersonation_scam": 0, ...}

    elif all(
        name in mapping
        for name in label_names
    ):

        reverse_mapping = {
            int(value): key
            for key, value in mapping.items()
        }

        label_names = [
            reverse_mapping[i]
            for i in range(NUM_LABELS)
        ]


print()
print("=" * 70)
print("LABEL MAPPING")
print("=" * 70)

for i, name in enumerate(label_names):

    print(
        f"{i} -> {name}"
    )


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

test_df = pd.read_csv(TEST_PATH)

print(
    "Test samples :",
    len(test_df)
)

required_columns = [
    "input_ids",
    "attention_mask",
    "label"
]

for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )

print("PASS - Required columns found.")


# ============================================================
# 4. TEST DATASET CLASS
# ============================================================

class NarrativeGuardTestDataset(Dataset):

    def __init__(self, dataframe):

        self.input_ids = []
        self.attention_masks = []
        self.labels = []

        for _, row in dataframe.iterrows():

            input_ids = ast.literal_eval(
                row["input_ids"]
            )

            attention_mask = ast.literal_eval(
                row["attention_mask"]
            )

            label = int(row["label"])

            self.input_ids.append(
                torch.tensor(
                    input_ids,
                    dtype=torch.long
                )
            )

            self.attention_masks.append(
                torch.tensor(
                    attention_mask,
                    dtype=torch.long
                )
            )

            self.labels.append(
                torch.tensor(
                    label,
                    dtype=torch.long
                )
            )

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_masks[index],
            "labels": self.labels[index]
        }


# ============================================================
# 5. CREATE TEST DATASET
# ============================================================

test_dataset = NarrativeGuardTestDataset(
    test_df
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print()
print("Test dataset created.")
print("Test batches:", len(test_loader))


# ============================================================
# 6. CHECK TEST LABELS
# ============================================================

test_labels = set(
    test_df["label"].astype(int).unique()
)

expected_labels = set(
    range(NUM_LABELS)
)

print()
print("Test labels:")
print(sorted(test_labels))

if not test_labels.issubset(
    expected_labels
):

    raise ValueError(
        "Invalid label found in test dataset."
    )

print(
    "PASS - Test labels are valid."
)


# ============================================================
# 7. LOAD TRAINED MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING TRAINED MODEL")
print("=" * 70)

model = (
    XLMRobertaForSequenceClassification
    .from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
)

model.to(DEVICE)

model.eval()

print("Trained model loaded successfully.")
print("Model device:", DEVICE)


# ============================================================
# 8. TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("RUNNING FINAL TEST EVALUATION")
print("=" * 70)

all_predictions = []
all_true_labels = []

total_loss = 0.0

with torch.no_grad():

    for batch_index, batch in enumerate(
        test_loader
    ):

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        total_loss += outputs.loss.item()

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_true_labels.extend(
            labels.cpu().numpy()
        )

        print(
            f"Batch "
            f"{batch_index + 1:2d}/"
            f"{len(test_loader)} completed"
        )


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

average_test_loss = (
    total_loss /
    len(test_loader)
)

accuracy = accuracy_score(
    all_true_labels,
    all_predictions
)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        all_true_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        all_true_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )
)


# ============================================================
# 10. FINAL OVERALL RESULTS
# ============================================================

print()
print("=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

print(
    f"Test Loss              : "
    f"{average_test_loss:.4f}"
)

print(
    f"Test Accuracy          : "
    f"{accuracy:.4f}"
)

print(
    f"Test Accuracy (%)      : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision        : "
    f"{precision_macro:.4f}"
)

print(
    f"Macro Recall           : "
    f"{recall_macro:.4f}"
)

print(
    f"Macro F1               : "
    f"{f1_macro:.4f}"
)

print(
    f"Weighted Precision     : "
    f"{precision_weighted:.4f}"
)

print(
    f"Weighted Recall        : "
    f"{recall_weighted:.4f}"
)

print(
    f"Weighted F1            : "
    f"{f1_weighted:.4f}"
)


# ============================================================
# 11. CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("PER-CLASS CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    all_true_labels,
    all_predictions,
    labels=list(range(NUM_LABELS)),
    target_names=label_names,
    digits=4,
    zero_division=0
)

print(report)


# ============================================================
# 12. CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    all_true_labels,
    all_predictions,
    labels=list(range(NUM_LABELS))
)

cm_df = pd.DataFrame(
    cm,
    index=label_names,
    columns=label_names
)

print(cm_df)


# ============================================================
# 13. SAVE RESULTS
# ============================================================

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# Save classification report

with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# Save confusion matrix

cm_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.csv"
    )
)


# Save overall metrics

metrics = {
    "test_loss": average_test_loss,
    "accuracy": accuracy,
    "accuracy_percent": accuracy * 100,
    "macro_precision": precision_macro,
    "macro_recall": recall_macro,
    "macro_f1": f1_macro,
    "weighted_precision": precision_weighted,
    "weighted_recall": recall_weighted,
    "weighted_f1": f1_weighted
}

with open(
    os.path.join(
        RESULTS_DIR,
        "test_metrics.json"
    ),
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# Save predictions

prediction_df = test_df.copy()

prediction_df[
    "predicted_label"
] = all_predictions

prediction_df[
    "predicted_class"
] = [
    label_names[p]
    for p in all_predictions
]

prediction_df[
    "actual_class"
] = [
    label_names[y]
    for y in all_true_labels
]

prediction_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# ============================================================
# 14. COMPLETION
# ============================================================

print()
print("=" * 70)
print("TEST EVALUATION COMPLETED")
print("=" * 70)

print()
print("Results saved to:")
print(
    "evaluation/classification_report.txt"
)
print(
    "evaluation/confusion_matrix.csv"
)
print(
    "evaluation/test_metrics.json"
)
print(
    "evaluation/test_predictions.csv"
)

print()
print("Final Test Accuracy:")
print(
    f"{accuracy * 100:.2f}%"
)

print()
print("Final Test Macro F1:")
print(
    f"{f1_macro:.4f}"
)

print("=" * 70)