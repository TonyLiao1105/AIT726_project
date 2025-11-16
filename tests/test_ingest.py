import pytest
from src.ingest import load_data

def test_load_data():
    # Test loading data from the JSONL file
    data = load_data('data/raw/doj_press_releases.jsonl')
    assert isinstance(data, list), "Data should be a list"
    assert len(data) > 0, "Data should not be empty"
    assert all(isinstance(item, dict) for item in data), "Each item should be a dictionary"

def test_load_data_invalid_file():
    # Test loading data from an invalid file path
    with pytest.raises(FileNotFoundError):
        load_data('data/raw/invalid_file.jsonl')