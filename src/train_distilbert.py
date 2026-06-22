"""
Fine-tune DistilBERT for 7-class mental health text classification.

Usage:
    python -m src.train_distilbert --csv path/to/Combined_Data.csv --out models/distilbert_mental
"""
import argparse
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .preprocessing import load_and_clean_dataset

MODEL_NAME = "distilbert-base-uncased"


class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def main(csv_path: str, out_dir: str, epochs: int = 4, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = load_and_clean_dataset(csv_path)
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["status"])

    # Note: raw `statement` text is used here (not the lexically-cleaned
    # `clean_text`), since DistilBERT performs its own subword tokenization
    # and benefits from natural sentence structure. No SMOTE / class
    # weighting is applied -- see paper Section III-F for justification.
    X_train, X_test, y_train, y_test = train_test_split(
        df["statement"], df["label"], test_size=0.2,
        random_state=42, stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(le.classes_)
    ).to(device)

    train_loader = DataLoader(
        MentalHealthDataset(X_train, y_train, tokenizer),
        batch_size=batch_size, shuffle=True,
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Train Acc: {correct/total:.4f}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    print(f"\nModel + tokenizer saved to {out_path}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to Combined Data.csv")
    parser.add_argument("--out", default="models/distilbert_mental")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    main(args.csv, args.out, args.epochs, args.batch_size)
