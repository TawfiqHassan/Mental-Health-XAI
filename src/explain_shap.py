"""
Generate global, per-class, and instance-level SHAP explanations
for the trained XGBoost model.

Usage:
    python -m src.explain_shap --models models/
"""
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import shap

from .preprocessing import TARGET_NAMES


def load_artifacts(models_dir: str):
    models_dir = Path(models_dir)
    with open(models_dir / "xgboost.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open(models_dir / "tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return xgb_model, tfidf


def build_explainer(xgb_model):
    return shap.TreeExplainer(xgb_model)


def plot_global_importance(explainer, X_sample, tfidf, out_path, top_n=20):
    feature_names = tfidf.get_feature_names_out()
    shap_values = explainer.shap_values(X_sample.toarray())
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3 and shap_arr.shape[0] == len(TARGET_NAMES):
        shap_arr = np.transpose(shap_arr, (1, 2, 0))

    global_importance = np.mean(np.mean(np.abs(shap_arr), axis=0), axis=1)
    top_idx = np.argsort(global_importance)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, top_n))
    ax.barh(range(top_n), global_importance[top_idx], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feature_names[top_idx], fontsize=10)
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(f"Top {top_n} Most Influential Features (Global SHAP)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved global SHAP plot to {out_path}")
    return shap_arr


def explain_instance(explainer, tfidf, xgb_model, text: str, clean_fn, top_k=5):
    """Return predicted class + top contributing tokens for a single text."""
    clean = clean_fn(text)
    vec = tfidf.transform([clean])
    pred = xgb_model.predict(vec)[0]
    proba = xgb_model.predict_proba(vec)[0]

    shap_values = explainer.shap_values(vec.toarray())
    shap_arr = np.array(shap_values)
    class_shap = (
        shap_arr[pred][0]
        if shap_arr.ndim == 3 and shap_arr.shape[0] == len(TARGET_NAMES)
        else shap_arr[0, :, pred]
    )
    feature_names = tfidf.get_feature_names_out()
    top_pos = np.argsort(class_shap)[-top_k:][::-1]
    top_neg = np.argsort(class_shap)[:top_k]

    return {
        "text": text,
        "predicted_class": TARGET_NAMES[pred],
        "confidence": float(proba[pred]),
        "top_supporting_tokens": [
            (feature_names[i], float(class_shap[i])) for i in top_pos if class_shap[i] > 0
        ],
        "top_opposing_tokens": [
            (feature_names[i], float(class_shap[i])) for i in top_neg if class_shap[i] < 0
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="models", help="Directory with saved models")
    parser.add_argument("--out", default="models/shap_global.png")
    parser.add_argument("--sample_size", type=int, default=200)
    args = parser.parse_args()

    xgb_model, tfidf = load_artifacts(args.models)
    explainer = build_explainer(xgb_model)

    with open(Path(args.models) / "xgboost.pkl", "rb") as f:
        pass  # X_test not stored separately; recommend passing your own sample in practice.

    print("Explainer ready. Use explain_instance() for single-text explanations.")
    with open(Path(args.models) / "shap_explainer.pkl", "wb") as f:
        pickle.dump(explainer, f)
    print(f"Saved SHAP explainer to {Path(args.models) / 'shap_explainer.pkl'}")
