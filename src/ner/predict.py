import json
import torch
from torch.utils.data import DataLoader
from src.ner.dataset import PressReleaseDataset
from src.ner.model import NERModel

def load_model(model_path):
    model = NERModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_entities(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(inputs, attention_mask)
            predictions.extend(outputs.argmax(dim=2).tolist())
    return predictions

def main(input_file, model_path):
    model = load_model(model_path)
    dataset = PressReleaseDataset(input_file)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    predictions = predict_entities(model, dataloader)
    
    # Process predictions to map to entity labels
    # This part will depend on how the dataset and model are structured
    # For example, you might want to convert indices to labels
    # Here, we assume a function `convert_predictions_to_labels` exists
    labels = convert_predictions_to_labels(predictions)
    
    return labels

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict entities in DOJ press releases.")
    parser.add_argument("input_file", type=str, help="Path to the input JSONL file containing press releases.")
    parser.add_argument("model_path", type=str, help="Path to the trained model file.")
    
    args = parser.parse_args()
    main(args.input_file, args.model_path)