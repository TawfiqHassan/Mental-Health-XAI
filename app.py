"""
Gradio demo for the Explainable Mental Health Detection project.

Run locally:
    python app.py

Or deploy free on HuggingFace Spaces by pushing this repo with an
`app.py` + `requirements.txt` at the root -- Spaces auto-detects Gradio apps.
"""
import gradio as gr

from src.inference import MentalHealthPredictor

# Update these paths: locally they point at ./models, on HF Spaces they
# should point at a HuggingFace Hub model repo (see README "Deployment" section).
predictor = MentalHealthPredictor(
    models_dir="models",
    distilbert_dir="models/distilbert_mental",
)


def predict(text):
    if not text or not text.strip():
        return "Please enter some text.", "", ""

    bert_result = predictor.predict_distilbert(text)
    xgb_result = predictor.predict_xgboost(text)

    bert_probs = "\n".join(
        f"{k}: {v*100:.1f}%"
        for k, v in sorted(bert_result["probabilities"].items(), key=lambda x: -x[1])
    )
    top_tokens = ", ".join(xgb_result.get("top_supporting_tokens", []))

    summary = (
        f"**Predicted condition (DistilBERT): {bert_result['predicted_class']}**\n\n"
        f"Lexical model (XGBoost) prediction: {xgb_result['predicted_class']}\n"
        f"Key words behind the lexical prediction: {top_tokens}"
    )
    return summary, bert_probs, top_tokens


with gr.Blocks(title="Explainable Mental Health Detection") as demo:
    gr.Markdown(
        "# Explainable Mental Health Detection from Social Media Text\n"
        "Enter a short piece of text below. This is a research prototype, "
        "**not a diagnostic tool** -- it should never replace professional "
        "mental health care. If you or someone you know is in crisis, please "
        "contact a local crisis line or mental health professional."
    )
    inp = gr.Textbox(
        label="Text",
        placeholder="e.g., I haven't been sleeping well and everything feels overwhelming lately...",
        lines=4,
    )
    btn = gr.Button("Analyze", variant="primary")
    out_summary = gr.Markdown(label="Summary")
    out_probs = gr.Textbox(label="DistilBERT class probabilities", lines=7)
    out_tokens = gr.Textbox(label="Top words driving the lexical prediction")

    btn.click(predict, inputs=inp, outputs=[out_summary, out_probs, out_tokens])

if __name__ == "__main__":
    demo.launch()
