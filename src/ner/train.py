from spacy.training import Example
import spacy

def train_model(model, train_dataset, val_dataset, epochs: int = 10, drop: float = 0.5):
    """Train the NER model on training data."""
    
    # Collect all unique labels from training data
    labels = set()
    for i in range(len(train_dataset)):
        _, ann = train_dataset[i]
        for start, end, label in ann.get("entities", []):
            labels.add(label)
    
    # Add labels to model
    model.add_labels(list(labels))
    
    # Disable other pipes during training
    pipe_exceptions = ["ner"]
    other_pipes = [p for p in model.nlp.pipe_names if p not in pipe_exceptions]
    with model.nlp.disable_pipes(*other_pipes):
        optimizer = model.nlp.create_optimizer()
        
        for epoch in range(epochs):
            losses = {"ner": 0}
            
            for i in range(len(train_dataset)):
                text, ann = train_dataset[i]
                doc = model.nlp.make_doc(text)
                
                # Create Example with aligned entities
                ents = []
                for start, end, label in ann.get("entities", []):
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span:
                        ents.append(span)
                
                doc.ents = ents
                example = Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]})
                model.nlp.update([example], sgd=optimizer, drop=drop, losses=losses)
            
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {losses['ner']:.4f}")
    
    return model