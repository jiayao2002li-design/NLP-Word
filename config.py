"""Configuration for BERT toxicity (word filter) training and inference."""

# Model
MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 6
LABEL_NAMES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# Training (tuned for faster runs)
BATCH_SIZE = 8
EPOCHS = 1
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
GRADIENT_CLIP = 1.0
VALIDATION_SPLIT = 0.1
MAX_TRAIN_SAMPLES = 2000
MAX_VALIDATION_SAMPLES = 500

# Inference
DEFAULT_THRESHOLD = 0.5
INFERENCE_BATCH_SIZE = 32

# Paths
OUTPUT_DIR = "./saved_model"
DATASET_NAME = "google/jigsaw_toxicity_pred"
DATA_DIR = "./data/jigsaw"
