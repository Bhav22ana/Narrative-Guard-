import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

MODEL_DIR = "models/narrativeguard"
VALIDATION_FILE = "dataset/tokenized/validation_encoded.csv"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("NARRATIVEGUARD VALIDATION EVALUATION")
print("=" * 60)

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model.to(device)
model.eval()

# Load validation data
df = pd.read_csv(VALIDATION_FILE)

predictions = []
actual_labels = []

print("\nValidation Predictions:\n")

with torch.no_grad():

    for _, row in df.iterrows():

        input_ids = torch.tensor(
            eval(row["input_ids"]),
            dtype=torch.long
        ).unsqueeze(0).to(device)

        attention_mask = torch.tensor(
            eval(row["attention_mask"]),
            dtype=torch.long
        ).unsqueeze(0).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

        actual = int(row["label"])

        predictions.append(prediction)
        actual_labels.append(actual)

        print(
            f"Actual: {actual} | Predicted: {prediction}"
        )

# Accuracy
accuracy = sum(
    p == a
    for p, a in zip(predictions, actual_labels)
) / len(actual_labels)

print("\nValidation Accuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        actual_labels,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        actual_labels,
        predictions
    )
)

print("\n" + "=" * 60)
print("VALIDATION EVALUATION COMPLETE")
print("=" * 60)