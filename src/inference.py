"""
Unified inference utilities for both the XGBoost+SHAP pipeline and the
fine-tuned DistilBERT model. This is the module the Gradio app (app.py)
and any future API wrap around.
"""
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocessing import clean_text, TARGET_NAMES


class MentalHealthPredictor:
    """
    Loads both model families once and exposes simple predict methods.

        predictor = MentalHealthPredictor("models/")
        predictor.predict_xgboost("I feel hopeless...")
        predictor.predict_distilbert("I feel hopeless...")
    """

    def __init__(self, models_dir: str, distilbert_dir: str = None, device: str = None):
        models_dir = Path(models_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        with open(models_dir / "tfidf.pkl", "rb") as f:
            self.tfidf = pickle.load(f)
        with open(models_dir / "xgboost.pkl", "rb") as f:
            self.xgb_model = pickle.load(f)

        shap_path = models_dir / "shap_explainer.pkl"
        self.shap_explainer = None
        if shap_path.exists():
            with open(shap_path, "rb") as f:
                self.shap_explainer = pickle.load(f)

        self.bert_model = None
        self.bert_tokenizer = None
        if distilbert_dir:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(distilbert_dir)
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                distilbert_dir
            ).to(self.device)
            self.bert_model.eval()

    # ---------- XGBoost (lexical, explainable) ----------
    def predict_xgboost(self, text: str, top_k: int = 5):
        clean = clean_text(text)
        vec = self.tfidf.transform([clean])
        pred = self.xgb_model.predict(vec)[0]
        proba = self.xgb_model.predict_proba(vec)[0]

        result = {
            "model": "xgboost",
            "predicted_class": TARGET_NAMES[pred],
            "probabilities": {TARGET_NAMES[i]: float(p) for i, p in enumerate(proba)},
        }

        if self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(vec.toarray())
            shap_arr = np.array(shap_values)
            class_shap = (
                shap_arr[pred][0]
                if shap_arr.ndim == 3 and shap_arr.shape[0] == len(TARGET_NAMES)
                else shap_arr[0, :, pred]
            )
            feature_names = self.tfidf.get_feature_names_out()
            top_pos = np.argsort(class_shap)[-top_k:][::-1]
            result["top_supporting_tokens"] = [
                feature_names[i] for i in top_pos if class_shap[i] > 0
            ]
        return result

    # ---------- DistilBERT (contextual) ----------
    def predict_distilbert(self, text: str):
        if self.bert_model is None:
            raise RuntimeError("DistilBERT model not loaded. Pass distilbert_dir to the constructor.")

        encoding = self.bert_tokenizer(
            text, max_length=128, padding="max_length",
            truncation=True, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.bert_model(**encoding).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred = int(np.argmax(probs))
        return {
            "model": "distilbert",
            "predicted_class": TARGET_NAMES[pred],
            "probabilities": {TARGET_NAMES[i]: float(p) for i, p in enumerate(probs)},
        }
