"""
Constants Module for Model Development (Phase 5).
"""

TARGET_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_CLASSES = len(TARGET_LABELS)

DEFAULT_MODEL_DIR = "outputs/models/classifiers"
DEFAULT_REGISTRY_DIR = "outputs/models/registry"
DEFAULT_RANDOM_STATE = 42
DEFAULT_THRESHOLD = 0.5
