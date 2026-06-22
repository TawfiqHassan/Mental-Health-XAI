"""
Text preprocessing utilities for the Mental Health XAI project.
"""
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

TARGET_NAMES = [
    "Anxiety", "Bipolar", "Depression", "Normal",
    "Personality disorder", "Stress", "Suicidal",
]


def clean_text(text: str) -> str:
    """
    Clean a raw social media post for TF-IDF based models.

    Steps: lowercase, strip URLs, remove retweet markers, remove
    non-alphabetic characters, drop very short noise tokens (<=2 chars),
    and remove English stopwords.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\brt\b", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\b\w{1,2}\b", "", text)
    text = re.sub(r"\s+", " ", text)
    text = " ".join(w for w in text.split() if w not in STOP_WORDS)
    return text.strip()


def load_and_clean_dataset(csv_path: str):
    """
    Load the 'Sentiment Analysis for Mental Health' Kaggle CSV,
    drop nulls, and return a cleaned DataFrame with a `clean_text` column.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["statement"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.reset_index(drop=True)
    df["clean_text"] = df["statement"].apply(clean_text)
    return df
