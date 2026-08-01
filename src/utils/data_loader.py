"""
Data loader module for Toxic Comment Classification.

Provides robust data ingestion, synthetic data fallback for testing/verification,
and schema validation.
"""

import os
import logging
from typing import Optional, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_COLUMNS: List[str] = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]
TEXT_COLUMN: str = "comment_text"


def generate_synthetic_dataset(num_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generates a realistic synthetic Toxic Comment dataset for verification.

    Args:
        num_samples: Number of comment records to generate.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame containing comment_text and 6 target binary columns.
    """
    np.random.seed(seed)
    logger.info(f"Generating synthetic dataset with {num_samples} records...")

    sample_comments = [
        "Explanation\nWhy the edits made under my username Hardcore Metallica Fan were reverted? They weren't vandalisms.",
        "D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)",
        "Hey man, I'm really not trying to edit war. It's just that this guy is constantly removing relevant information.",
        "You are a terrible idiot and you should stop writing nonsense here!!! GO AWAY NOW!",
        "SHUT UP YOU STUPID FOOL! I WILL KILL YOU AND DESTROY YOUR PAGE! YOU ARE DISGUSTING!",
        "Fuck you and your stupid rules! You complete asshole!",
        "This article is completely biased and lacks neutral point of view references.",
        "All people from that country are filthy scum and subhumans. I hate them all!",
        "I will find where you live and end you. Consider this a real threat.",
        "You are a pathetic loser, get a life and stop editing Wikipedia.",
    ]

    # Generate texts with varying lengths, ALL CAPS, duplicates, and edge cases
    texts = []
    for i in range(num_samples):
        base_text = np.random.choice(sample_comments)
        if i % 25 == 0:
            # Introduce exact duplicate
            text = sample_comments[3]
        elif i % 40 == 0:
            # Introduce ALL CAPS SHOUTING
            text = "YOU ARE AN ABSOLUTE MONSTER AND A COMPLETE FAILURE! " * np.random.randint(1, 5)
        elif i % 50 == 0:
            # Extremely long comment
            text = "This is a very long repeating comment for testing length limits. " * 40
        elif i % 100 == 0:
            # Single word comment
            text = "Hello."
        else:
            # Add random noise
            text = f"{base_text} Ref_{i}."
        texts.append(text)

    df = pd.DataFrame({TEXT_COLUMN: texts})

    # Generate correlated multi-label targets
    # toxic is most common (~10%), obscene/insult (~5%), severe_toxic (~1%), threat (~0.5%), identity_hate (~1%)
    toxic = np.random.binomial(1, 0.12, num_samples)
    severe_toxic = toxic * np.random.binomial(1, 0.15, num_samples)
    obscene = toxic * np.random.binomial(1, 0.55, num_samples)
    insult = toxic * np.random.binomial(1, 0.50, num_samples)
    threat = toxic * np.random.binomial(1, 0.08, num_samples)
    identity_hate = toxic * np.random.binomial(1, 0.10, num_samples)

    df["toxic"] = toxic
    df["severe_toxic"] = severe_toxic
    df["obscene"] = obscene
    df["threat"] = threat
    df["insult"] = insult
    df["identity_hate"] = identity_hate

    # Introduce a few missing values in comment_text for data quality tests
    df.loc[12, TEXT_COLUMN] = np.nan
    df.loc[45, TEXT_COLUMN] = np.nan

    logger.info("Synthetic dataset generated successfully.")
    return df


def load_toxic_comment_data(
    file_path: Optional[str] = None, fallback_samples: int = 1000
) -> pd.DataFrame:
    """Loads Toxic Comment dataset from file path or generates synthetic fallback data.

    Args:
        file_path: Path to dataset CSV file.
        fallback_samples: Number of synthetic samples if file missing.

    Returns:
        pd.DataFrame containing loaded dataset.
    """
    candidate_paths = [
        file_path,
        "data/raw/train.csv",
        "data/train.csv",
        "../data/raw/train.csv",
    ]

    valid_path = None
    for path in candidate_paths:
        if path and os.path.exists(path):
            valid_path = path
            break

    if valid_path:
        logger.info(f"Loading raw dataset from {valid_path}...")
        df = pd.read_csv(valid_path)
    else:
        logger.warning("Raw dataset file not found in candidates. Creating synthetic fallback dataset...")
        os.makedirs("data/raw", exist_ok=True)
        df = generate_synthetic_dataset(num_samples=fallback_samples)
        df.to_csv("data/raw/train.csv", index=False)
        logger.info("Saved synthetic fallback dataset to data/raw/train.csv")

    # Validate required columns
    required_cols = [TEXT_COLUMN] + TARGET_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required schema columns: {missing_cols}")

    logger.info(f"Dataset loaded successfully with shape: {df.shape}")
    return df
