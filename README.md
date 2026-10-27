# Word Filter Model

A BERT-based multi-label toxicity classifier that detects harmful content across six categories. Train the model on the Jigsaw dataset, then use it as a word filter to flag toxic text in real time.

---

## Overview

The model fine-tunes `bert-base-uncased` for **multi-label sequence classification**. Given a piece of text, it independently predicts the probability of six toxicity types:

| Label | Description |
|---|---|
| `toxic` | Generally offensive or harmful content |
| `severe_toxic` | Highly offensive, extreme language |
| `obscene` | Vulgar or sexually explicit content |
| `threat` | Threats of violence or harm |
| `insult` | Personal attacks or demeaning language |
| `identity_hate` | Hatred targeting identity (race, religion, gender, etc.) |

A comment can belong to **multiple categories** at once (multi-label). Each label uses a sigmoid activation with a configurable probability threshold (default `0.5`).

---

## Project Structure

```
word_filter_model/
├── config.py               # All training and inference hyperparameters
├── train.py                # Fine-tuning script
├── predict.py              # Inference / word filter script
├── requirements.txt        # Python dependencies
├── scripts/
│   └── sklearn_breast_cancer_starter.py  # Sklearn import smoke-check
├── real_world_input.txt    # Sample input texts for batch inference
├── real_world_predictions.csv  # Saved predictions from a previous run
├── EXPLANATION.md          # Step-by-step code explanation (English & Chinese)
├── data/
│   └── jigsaw/
│       ├── train.csv       # Jigsaw training data (download separately)
│       ├── test.csv        # Jigsaw test data
│       └── test_labels.csv # Jigsaw test labels
└── saved_model/            # Fine-tuned model weights and tokenizer (auto-generated)
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    └── tokenizer_config.json
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Requirements: `datasets`, `transformers`, `torch`, `scikit-learn`, `pandas`, `matplotlib`, `tqdm`.

### 2. Download the dataset

Download the [Jigsaw Toxic Comment Classification dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data) from Kaggle and place the files in `data/jigsaw/`:

```
data/jigsaw/train.csv
data/jigsaw/test.csv
data/jigsaw/test_labels.csv
```

---

## Sklearn starter (scripts)

Smoke-check that `scikit-learn`, `pandas`, and `matplotlib` import cleanly:

```bash
python scripts/sklearn_breast_cancer_starter.py
```

---

## Training

Run the training script from the project root:

```bash
python train.py
```

The script will:
1. Load and split `data/jigsaw/train.csv` into train / validation sets (90 / 10 split).
2. Tokenize comments using the BERT tokenizer (max length 128 tokens).
3. Fine-tune `bert-base-uncased` with `BCEWithLogitsLoss` for multi-label classification.
4. Evaluate after each epoch: validation loss, micro F1, macro F1, and ROC-AUC.
5. Save the fine-tuned model and tokenizer to `./saved_model/`.

### Key training settings (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `MODEL_NAME` | `bert-base-uncased` | Base BERT checkpoint |
| `EPOCHS` | `1` | Number of training epochs |
| `BATCH_SIZE` | `8` | Samples per training batch |
| `MAX_LENGTH` | `128` | Maximum token sequence length |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `MAX_TRAIN_SAMPLES` | `2000` | Cap on training examples (set `None` for full dataset) |
| `MAX_VALIDATION_SAMPLES` | `500` | Cap on validation examples |
| `VALIDATION_SPLIT` | `0.1` | Fraction of data held out for validation |

To train on the full dataset, set `MAX_TRAIN_SAMPLES = None` and `MAX_VALIDATION_SAMPLES = None` in `config.py`.

---

## Inference

After training, use `predict.py` to classify text.

### Single text

```bash
python predict.py --text "You are an absolute idiot."
```

### Batch from a file

Pass a `.txt` file (one text per line):

```bash
python predict.py --input-file real_world_input.txt
```

Or a `.csv` file with a named column:

```bash
python predict.py --input-file comments.csv --text-column comment
```

### Save results to CSV

```bash
python predict.py --input-file real_world_input.txt --output-file predictions.csv
```

### Change the threshold

```bash
python predict.py --text "I hate you." --threshold 0.3
```

### Example output

```
Text: You are an idiot and I hate you.
  Is toxic: True
  Active labels: toxic,insult
  Probabilities: {'toxic': 0.93, 'severe_toxic': 0.12, 'obscene': 0.45, ...}
```

---

## Using as a Library

You can import the inference helpers directly in your own code:

```python
from predict import load_model_and_tokenizer, is_toxic, get_toxic_labels

model, tokenizer = load_model_and_tokenizer()

# Simple boolean check
print(is_toxic(model, tokenizer, "I will hurt you."))  # True

# Get specific active labels
labels = get_toxic_labels(model, tokenizer, "You're such a loser.")
print(labels)  # ['toxic', 'insult']
```

---

## How It Works

See [`EXPLANATION.md`](EXPLANATION.md) for a detailed step-by-step walkthrough of the training and inference pipeline in both English and Chinese.

In brief:
- **Tokenization** — raw text is converted to BERT token IDs and attention masks.
- **Model** — `BertForSequenceClassification` adds a linear head on top of BERT's `[CLS]` token, outputting 6 logits.
- **Loss** — `BCEWithLogitsLoss` treats each label independently (no softmax).
- **Prediction** — sigmoid is applied to logits; any label with probability ≥ threshold is flagged.

---

## Hardware Notes

- Training and inference automatically use a **CUDA GPU** if available, otherwise CPU.
- With the default capped dataset (`MAX_TRAIN_SAMPLES = 2000`) training takes a few minutes on CPU.
- For full dataset training, a GPU is strongly recommended.
