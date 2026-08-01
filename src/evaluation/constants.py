"""
Constants Module for Evaluation (Phase 6).
"""

TARGET_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_CLASSES = len(TARGET_LABELS)

DEFAULT_EVALUATION_DIR = "outputs/evaluation"
DEFAULT_FIGURES_DIR = "outputs/figures"
DEFAULT_REPORTS_DIR = "outputs/reports"
DEFAULT_THRESHOLD = 0.5
