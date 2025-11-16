from spacy.training import Example
import spacy
import random
from spacy.training import Example
from spacy.util import minibatch, compounding

def train_model(model, train_dataset, val_dataset=None, epochs: int = 30, drop: float = 0.2):
    # collect labels
    labels = set()
    for i in range(len(train_dataset)):
        _, ann = train_dataset[i]
        for s, e, lab in ann.get("entities", []):
            labels.add(lab)
    model.add_labels(list(labels))

    other_pipes = [p for p in model.nlp.pipe_names if p != "ner"]
    with model.nlp.disable_pipes(*other_pipes):
        # spaCy v2/v3 compatibility for optimizer
        try:
            optimizer = model.nlp.resume_training()
        except Exception:
            optimizer = model.nlp.begin_training() if hasattr(model.nlp, "begin_training") else model.nlp.create_optimizer()

        for epoch in range(1, epochs + 1):
            losses = {"ner": 0.0}
            examples = []
            for i in range(len(train_dataset)):
                text, ann = train_dataset[i]
                doc = model.nlp.make_doc(text)
                examples.append(Example.from_dict(doc, {"entities": [(s, e, l) for s, e, l in ann.get("entities", [])]}))
            random.shuffle(examples)

            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                model.nlp.update(batch, sgd=optimizer, drop=drop, losses=losses)

            print(f"Epoch {epoch}/{epochs} - Loss: {losses['ner']:.4f}")

            # optional quick val check every 5 epochs
            if val_dataset is not None and (epoch % 5 == 0 or epoch == epochs):
                from src.evaluation import evaluate_model
                metrics = evaluate_model(model, val_dataset)
                print(f"  VAL (epoch {epoch}) - precision: {metrics['precision']:.3f} recall: {metrics['recall']:.3f} f1: {metrics['f1']:.3f}")

    return model