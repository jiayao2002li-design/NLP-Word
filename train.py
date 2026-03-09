"""
Train a BERT-based multi-label toxicity classifier on the Jigsaw dataset.
"""

import os
import sys
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from datasets import Dataset as HFDataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config


def load_and_prepare_data():
    """
    Load Kaggle Jigsaw CSV data and split into train/validation.
    Returns train and validation datasets with comment_text and multi-label targets.
    """
    train_csv_path = os.path.join(config.DATA_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            "Could not find the Jigsaw training file. "
            f"Expected: {train_csv_path}. "
            "Download the Kaggle competition files and place train.csv in DATA_DIR."
        )

    train_df = pd.read_csv(train_csv_path)
    required_columns = ["comment_text", *config.LABEL_NAMES]
    missing_columns = [column for column in required_columns if column not in train_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in train.csv: {missing_columns}")

    train_dataset = HFDataset.from_pandas(
        train_df[required_columns],
        preserve_index=False,
    )
    split = train_dataset.train_test_split(test_size=config.VALIDATION_SPLIT, seed=42)
    train_split = split["train"]
    val_split = split["test"]

    if config.MAX_TRAIN_SAMPLES is not None:
        train_limit = min(config.MAX_TRAIN_SAMPLES, len(train_split))
        train_split = train_split.select(range(train_limit))

    if config.MAX_VALIDATION_SAMPLES is not None:
        val_limit = min(config.MAX_VALIDATION_SAMPLES, len(val_split))
        val_split = val_split.select(range(val_limit))

    return train_split, val_split


def tokenize_examples(
    examples: dict[str, Any],
    tokenizer: BertTokenizer,
) -> dict[str, list]:
    """Tokenize comment_text and return input_ids and attention_mask."""
    return tokenizer(
        examples["comment_text"],
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
        return_tensors=None,
    )


def get_labels_array(dataset) -> np.ndarray:
    """Extract the 6 label columns as a numpy array of shape (n, 6)."""
    return np.column_stack([
        np.array(dataset[name], dtype=np.float32)
        for name in config.LABEL_NAMES
    ])


class ToxicityDataset(Dataset):
    """PyTorch Dataset yielding tokenized inputs and multi-label targets."""

    def __init__(
        self,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        labels: np.ndarray,
    ):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def create_dataloaders(train_ds, val_ds, tokenizer: BertTokenizer):
    """Tokenize datasets and create train/val DataLoaders."""
    # Tokenize train
    train_tokenized = train_ds.map(
        lambda x: tokenize_examples(x, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
        desc="Tokenize train",
    )
    train_labels = get_labels_array(train_ds)

    # Tokenize validation
    val_tokenized = val_ds.map(
        lambda x: tokenize_examples(x, tokenizer),
        batched=True,
        remove_columns=val_ds.column_names,
        desc="Tokenize val",
    )
    val_labels = get_labels_array(val_ds)

    train_dataset = ToxicityDataset(
        train_tokenized["input_ids"],
        train_tokenized["attention_mask"],
        train_labels,
    )
    val_dataset = ToxicityDataset(
        val_tokenized["input_ids"],
        val_tokenized["attention_mask"],
        val_labels,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def evaluate(model: torch.nn.Module, val_loader: DataLoader, device: torch.device):
    """Compute validation loss and multi-label metrics (F1, ROC-AUC)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(np.float32)
            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs, average="macro")
    except ValueError:
        roc_auc = 0.0
    return avg_loss, micro_f1, macro_f1, roc_auc


def train():
    """Load data, build model, train, evaluate, and save."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Load data
    train_ds, val_ds = load_and_prepare_data()
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

    # Step 2: Tokenizer and dataloaders
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, tokenizer)

    # Step 3: Model and loss
    model = BertForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
    )
    model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Step 4: Training loop
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.EPOCHS}")

        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        val_loss, micro_f1, macro_f1, roc_auc = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1} | Train loss: {epoch_loss / len(train_loader):.4f} | "
            f"Val loss: {val_loss:.4f} | Micro F1: {micro_f1:.4f} | "
            f"Macro F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}"
        )

    # Step 5: Save model and tokenizer
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"Model and tokenizer saved to {config.OUTPUT_DIR}")


if __name__ == "__main__":
    train()
