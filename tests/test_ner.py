import pytest
from src.ner.model import NERModel
from src.ner.dataset import NERDataset

def test_ner_model_initialization():
    model = NERModel()
    assert model is not None
    assert hasattr(model, 'layers')

def test_ner_dataset_loading():
    dataset = NERDataset('data/processed/train_data.jsonl')
    assert dataset is not None
    assert len(dataset) > 0

def test_ner_model_prediction():
    model = NERModel()
    sample_text = "John Doe was sentenced by Judge Smith in the case of United States v. Doe."
    predictions = model.predict(sample_text)
    assert 'DEFENDANT' in predictions
    assert 'JUDGE' in predictions
    assert 'COURT CASE START DATE' in predictions
    assert 'DISTRICT COURT' in predictions

def test_ner_model_training():
    model = NERModel()
    dataset = NERDataset('data/processed/train_data.jsonl')
    model.train(dataset)
    assert model.trained

def test_ner_model_evaluation():
    model = NERModel()
    dataset = NERDataset('data/processed/test_data.jsonl')
    metrics = model.evaluate(dataset)
    assert metrics['precision'] >= 0.7
    assert metrics['recall'] >= 0.7
    assert metrics['f1_score'] >= 0.7