https://huggingface.co/spaces/tawfiqhassan/mental-health-xai-demo
# Explainable Multi-Class Mental Health Detection from Social Media Text

Detects seven mental health conditions — **Normal, Depression, Suicidal,
Anxiety, Bipolar, Stress, Personality Disorder** — from social media text,
using both classical ML (TF-IDF + XGBoost, explained with SHAP) and a
fine-tuned **DistilBERT** transformer.

> ⚠️ **Disclaimer:** This is a research prototype, not a diagnostic tool.
> It must not be used as a substitute for professional mental health
> assessment or care.

## Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 0.7452 | 0.7074 | 0.7479 |
| Linear SVM | 0.7224 | 0.6920 | 0.7252 |
| Random Forest | 0.7389 | 0.6973 | 0.7344 |
| XGBoost | 0.7472 | 0.7123 | 0.7446 |
| BiLSTM | 0.7293 | 0.6902 | 0.7278 |
| **DistilBERT** | **0.7895** | **0.7656** | **0.7887** |

## Project Structure

```
mental-health-xai/
├── app.py                  # Gradio demo (local or HuggingFace Spaces)
├── requirements.txt
├── src/
│   ├── preprocessing.py    # text cleaning + dataset loading
│   ├── train_ml.py         # TF-IDF + SMOTE + 4 classical models
│   ├── train_distilbert.py # fine-tune DistilBERT
│   ├── explain_shap.py     # SHAP global/per-class/instance explanations
│   └── inference.py        # unified predictor used by app.py
├── notebooks/               # exploratory notebook(s)
├── models/                  # trained artifacts (gitignored, see below)
└── data/                    # dataset (gitignored, see below)
```

## Setup

```bash
git clone <this-repo>
cd mental-health-xai
pip install -r requirements.txt
```

## 1. Get the data

Download from Kaggle and place it at `data/Combined_Data.csv`:
https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

## 2. Train

```bash
python -m src.train_ml --csv data/Combined_Data.csv --out models/
python -m src.train_distilbert --csv data/Combined_Data.csv --out models/distilbert_mental
python -m src.explain_shap --models models/
```

## 3. Run the demo locally

```bash
python app.py
```

## Deployment (free)

Large model files (`models/distilbert_mental/`, `*.pkl`) are **not** pushed
to GitHub (see `.gitignore`) — GitHub isn't built for multi-hundred-MB binary
files. Instead:

1. **Push trained models to the HuggingFace Hub** (free, made for this):
   ```python
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo("your-username/mental-health-distilbert")
   api.upload_folder(folder_path="models/distilbert_mental",
                      repo_id="your-username/mental-health-distilbert")
   ```
   Then in `app.py` / `inference.py`, set `distilbert_dir="your-username/mental-health-distilbert"`
   instead of a local path — `transformers` will download it automatically.

2. **Deploy the Gradio app to HuggingFace Spaces** (free hosting):
   - Create a new Space (SDK: Gradio) at https://huggingface.co/new-space
   - Push this repo's `app.py` + `requirements.txt` + `src/` to the Space
   - The Space builds and gives you a public URL automatically

## Citation / Paper

See the accompanying IEEE-format paper draft for full methodology,
related work, and explainability analysis.
