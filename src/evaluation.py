from sklearn.metrics import precision_recall_fscore_support, classification_report

def evaluate_model(model, val_dataset):
    """Evaluate NER model on validation data."""
    
    y_true_labels = []
    y_pred_labels = []
    
    for i in range(len(val_dataset)):
        text, ann = val_dataset[i]
        true_entities = ann.get("entities", [])
        pred_results = model.predict(text)
        
        # Extract labels from true entities (start, end, label)
        true_labels = [ent[2] for ent in true_entities]
        # Extract labels from predictions (text, label, start_char, end_char)
        pred_labels = [ent[1] for ent in pred_results]
        
        y_true_labels.extend(true_labels)
        y_pred_labels.extend(pred_labels)
    
    # Compute metrics
    if y_true_labels and y_pred_labels:
        # Pad shorter list to match length
        max_len = max(len(y_true_labels), len(y_pred_labels))
        y_true_labels += ["O"] * (max_len - len(y_true_labels))
        y_pred_labels += ["O"] * (max_len - len(y_pred_labels))
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_labels, y_pred_labels, average='weighted', zero_division=0
        )
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    else:
        metrics = {"precision": 0, "recall": 0, "f1": 0}
    
    return metrics