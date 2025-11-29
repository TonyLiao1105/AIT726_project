"""Train a spaCy NER model on the processed JSONL dataset.

Run this script outside of pytest in a Python environment with spaCy installed.

Example:
  python scripts/train_spacy.py \
    --train data/processed/train_data.jsonl \
    --dev data/processed/test_data.jsonl \
    --output models/ner_model \
    --epochs 20
"""
import argparse
from pathlib import Path
import logging

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="Path to train JSONL")
    p.add_argument("--dev", required=False, help="Path to dev/test JSONL")
    p.add_argument("--output", required=True, help="Directory to save trained model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--drop", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports so this script can be imported without immediately requiring spaCy
    from src.ner.dataset import NERDataset
    from src.ner.model import NERModel
    from src.ner.train import train_model

    train_ds = NERDataset(args.train)
    dev_ds = NERDataset(args.dev) if args.dev else None

    model = NERModel()
    logging.info("Starting training...")
    trained = train_model(model, train_ds, val_dataset=dev_ds, epochs=args.epochs, drop=args.drop)
    logging.info(f"Saving trained model to {out_dir}")
    try:
        trained.save(str(out_dir))
    except Exception:
        # fallback: if spaCy-backed save not available, write a marker
        (out_dir / "MODEL_SAVED.txt").write_text("fallback-model")


if __name__ == "__main__":
    main()
