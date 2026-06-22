"""
Train classical ML baselines (Logistic Regression, SVM, Random Forest,
XGBoost) on SMOTE-balanced TF-IDF features.

Usage:
    python -m src.train_ml --csv path/to/Combined_Data.csv --out models/
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from .preprocessing import load_and_clean_dataset, TARGET_NAMES


def main(csv_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and cleaning dataset...")
    df = load_and_clean_dataset(csv_path)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["status"])

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2,
        random_state=42, stratify=df["label"],
    )

    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    print("Balancing training set with SMOTE...")
    X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(
        X_train_tfidf, y_train
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "svm": LinearSVC(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            random_state=42, eval_metric="mlogloss",
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test_tfidf)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "macro_f1": f1_score(y_test, y_pred, average="macro"),
            "weighted_f1": f1_score(y_test, y_pred, average="weighted"),
        }
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open(out_dir / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    np.save(out_dir / "y_test.npy", np.array(y_test))

    results_df = pd.DataFrame(results).T.round(4)
    results_df.to_csv(out_dir / "ml_results.csv")
    print("\n===== Results =====")
    print(results_df)
    print(f"\nAll models and artifacts saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Combined Data.csv")
    parser.add_argument("--out", default="models", help="Output directory")
    args = parser.parse_args()
    main(args.csv, args.out)
