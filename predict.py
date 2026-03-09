"""
Load the trained BERT toxicity model and run inference (word filter).
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Union

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config


def load_model_and_tokenizer(model_dir: Union[str, Path] = None):
    """
    Load the saved BERT model and tokenizer from disk.
    Returns (model, tokenizer). Model is set to eval mode.
    """
    if model_dir is None:
        model_dir = config.OUTPUT_DIR
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=config.NUM_LABELS,
    )
    model.eval()
    return model, tokenizer


def load_texts_from_file(input_file: Union[str, Path], text_column: str = "text") -> list[str]:
    """
    Load texts from a .txt or .csv file.

    - .txt: one text per line
    - .csv: reads the column named by text_column
    """
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix.lower() == ".txt":
        texts = [
            line.strip()
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif input_path.suffix.lower() == ".csv":
        with input_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or text_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV file must contain a '{text_column}' column. "
                    f"Found columns: {reader.fieldnames}"
                )
            texts = [row[text_column].strip() for row in reader if row[text_column].strip()]
    else:
        raise ValueError("Only .txt and .csv files are supported.")

    if not texts:
        raise ValueError(f"No valid texts were found in {input_path}.")
    return texts


def predict(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    texts: Union[str, list[str]],
    device: torch.device = None,
) -> np.ndarray:
    """
    Run the model on one or more texts.
    Returns probabilities of shape (n_texts, 6) for the 6 toxicity labels.
    """
    if isinstance(texts, str):
        texts = [texts]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    all_probabilities = []

    with torch.no_grad():
        for start_idx in range(0, len(texts), config.INFERENCE_BATCH_SIZE):
            batch_texts = texts[start_idx : start_idx + config.INFERENCE_BATCH_SIZE]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probabilities.append(probs)

    return np.vstack(all_probabilities)


def is_toxic(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    threshold: float = 0.5,
    device: torch.device = None,
) -> bool:
    """
    Return True if any of the 6 toxicity labels is above the threshold.
    """
    probs = predict(model, tokenizer, text, device)
    return (probs[0] >= threshold).any()


def get_toxic_labels(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    text: str,
    threshold: float = 0.5,
    device: torch.device = None,
) -> list[str]:
    """
    Return the list of label names that are above the threshold for the given text.
    """
    probs = predict(model, tokenizer, text, device)
    return [
        config.LABEL_NAMES[i]
        for i in range(config.NUM_LABELS)
        if probs[0, i] >= threshold
    ]


def build_prediction_rows(
    texts: list[str],
    probabilities: np.ndarray,
    threshold: float,
) -> list[dict[str, object]]:
    """Build structured prediction rows for printing or CSV export."""
    rows = []
    for text, probs in zip(texts, probabilities):
        labels = [
            config.LABEL_NAMES[idx]
            for idx, score in enumerate(probs)
            if score >= threshold
        ]
        row = {
            "text": text,
            "is_toxic": bool((probs >= threshold).any()),
            "active_labels": ",".join(labels),
        }
        for idx, label_name in enumerate(config.LABEL_NAMES):
            row[f"{label_name}_probability"] = float(probs[idx])
        rows.append(row)
    return rows


def save_predictions(rows: list[dict[str, object]], output_file: Union[str, Path]) -> None:
    """Save prediction rows to a CSV file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for single-text or file-based inference."""
    parser = argparse.ArgumentParser(
        description="Run the trained BERT toxicity model on text or on a .txt/.csv file.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="A single text string to classify.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a .txt or .csv file containing texts to classify.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="CSV column name to read when --input-file points to a .csv file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Optional path to save predictions as a CSV file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DEFAULT_THRESHOLD,
        help="Probability threshold used to mark a label as active.",
    )
    return parser.parse_args()


def resolve_input_texts(args: argparse.Namespace) -> list[str]:
    """Resolve inference input from CLI arguments or fall back to example samples."""
    if args.text:
        return [args.text]
    if args.input_file:
        return load_texts_from_file(args.input_file, text_column=args.text_column)
    return [
        "Thank you for your help, that was very kind.",
        "You are an idiot and I hate you.",
    ]


def main():
    """Load the saved model and run inference on text or file input."""
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = resolve_input_texts(args)
    probabilities = predict(model, tokenizer, texts, device=device)
    rows = build_prediction_rows(texts, probabilities, threshold=args.threshold)

    for row in rows:
        text_preview = row["text"][:60]
        if len(row["text"]) > 60:
            text_preview += "..."
        print(f"Text: {text_preview}")
        print(f"  Is toxic: {row['is_toxic']}")
        print(f"  Active labels: {row['active_labels'] or 'none'}")
        print(
            "  Probabilities: "
            f"{ {label: row[f'{label}_probability'] for label in config.LABEL_NAMES} }"
        )
        print()

    if args.output_file:
        save_predictions(rows, args.output_file)
        print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()
