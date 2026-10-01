import os
import json
import torch
import mlflow
import mlflow.transformers

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score


# ============================================================
# NARRATIVEGUARD CONFIGURATION
# ============================================================

MODEL_NAME = "xlm-roberta-base"

DATA_DIR = "dataset/tokenized"
MODEL_DIR = "models/narrativeguard"

LEARNING_RATE = 2e-5
BATCH_SIZE = 4
EPOCHS = 3

EXPERIMENT_NAME = "NarrativeGuard"


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# MLflow SETUP
# ============================================================

mlflow.set_experiment(EXPERIMENT_NAME)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

with open(
    f"{DATA_DIR}/label_mapping.json",
    "r",
    encoding="utf-8"
) as file:
    mapping = json.load(file)


label2id = mapping["label2id"]

id2label = {
    int(key): value
    for key, value in mapping["id2label"].items()
}

num_labels = len(label2id)


# ============================================================
# LOAD PYTORCH DATALOADERS
# ============================================================

from create_dataloader import (
    train_loader,
    validation_loader,
    test_loader,
    train_dataset,
    validation_dataset,
    test_dataset
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START MLFLOW RUN
# ============================================================

with mlflow.start_run(
    run_name="XLM-RoBERTa-Baseline"
) as run:

    print("=" * 65)
    print("NARRATIVEGUARD XLM-ROBERTA TRAINING")
    print("=" * 65)

    print("\nMLflow Run ID:")
    print(run.info.run_id)

    print("\nModel:", MODEL_NAME)
    print("Device:", device)
    print("Number of classes:", num_labels)

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    total_samples = (
        len(train_dataset)
        + len(validation_dataset)
        + len(test_dataset)
    )

    print("\nDataset:")
    print("Total samples:", total_samples)
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(validation_dataset))
    print("Testing samples:", len(test_dataset))

    # --------------------------------------------------------
    # LOG PARAMETERS
    # --------------------------------------------------------

    mlflow.log_params({
        "model_name": MODEL_NAME,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_labels": num_labels,
        "optimizer": "AdamW",
        "device": str(device),
        "total_samples": total_samples,
        "training_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset)
    })

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("Loading XLM-RoBERTa model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label
    )

    model.to(device)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        print("\n" + "-" * 65)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 65)

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch["labels"].to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()

        # ----------------------------------------------------
        # TRAINING LOSS
        # ----------------------------------------------------

        average_train_loss = (
            total_loss / len(train_loader)
        )

        print(
            "Training Loss:",
            round(average_train_loss, 4)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        predictions = []
        actual_labels = []

        validation_loss = 0.0

        with torch.no_grad():

            for batch in validation_loader:

                input_ids = batch[
                    "input_ids"
                ].to(device)

                attention_mask = batch[
                    "attention_mask"
                ].to(device)

                labels = batch[
                    "labels"
                ].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                validation_loss += (
                    outputs.loss.item()
                )

                preds = torch.argmax(
                    outputs.logits,
                    dim=1
                )

                predictions.extend(
                    preds.cpu().tolist()
                )

                actual_labels.extend(
                    labels.cpu().tolist()
                )

        average_validation_loss = (
            validation_loss / len(validation_loader)
        )

        validation_accuracy = accuracy_score(
            actual_labels,
            predictions
        )

        print(
            "Validation Loss:",
            round(average_validation_loss, 4)
        )

        print(
            "Validation Accuracy:",
            round(validation_accuracy, 4)
        )

        # ----------------------------------------------------
        # LOG EPOCH METRICS TO MLFLOW
        # ----------------------------------------------------

        mlflow.log_metric(
            "train_loss",
            average_train_loss,
            step=epoch + 1
        )

        mlflow.log_metric(
            "validation_loss",
            average_validation_loss,
            step=epoch + 1
        )

        mlflow.log_metric(
            "validation_accuracy",
            validation_accuracy,
            step=epoch + 1
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    print("\nSaving trained model...")

    model.save_pretrained(
        MODEL_DIR
    )

    tokenizer.save_pretrained(
        MODEL_DIR
    )

    # Save label mapping alongside model
    with open(
        f"{MODEL_DIR}/label_mapping.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            mapping,
            file,
            indent=4,
            ensure_ascii=False
        )

    # ========================================================
    # LOG MODEL ARTIFACTS
    # ========================================================

    mlflow.log_artifacts(
        MODEL_DIR,
        artifact_path="narrativeguard_model"
    )

    # ========================================================
    # FINAL INFORMATION
    # ========================================================

    print("\nModel saved to:")
    print(MODEL_DIR)

    print("\nMLflow Run ID:")
    print(run.info.run_id)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)