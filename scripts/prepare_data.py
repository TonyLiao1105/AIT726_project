"""
Data Preparation Script for DOJ Press Release NER Model
Converts Prodigy JSONL format to spaCy training format
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
import spacy
from spacy.tokens import DocBin
from sklearn.model_selection import train_test_split


def load_prodigy_data(file_path: str) -> List[Dict]:
    """Load annotated data from Prodigy JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Only include accepted annotations
                if entry.get('answer') == 'accept':
                    data.append(entry)
            except json.JSONDecodeError:
                continue
    print(f"Loaded {len(data)} annotated examples from {file_path}")
    return data


def convert_to_spacy_format(prodigy_data: List[Dict]) -> List[Tuple[str, Dict]]:
    """
    Convert Prodigy format to spaCy training format
    
    Prodigy format: {"text": "...", "spans": [{"start": 0, "end": 10, "label": "DEFENDANT"}]}
    spaCy format: ("text", {"entities": [(start, end, label)]})
    """
    spacy_data = []
    
    for entry in prodigy_data:
        text = entry.get('text', '')
        spans = entry.get('spans', [])
        
        # Convert spans to spaCy entities format
        entities = []
        for span in spans:
            start = span['start']
            end = span['end']
            label = span['label']
            entities.append((start, end, label))
        
        # Sort entities by start position
        entities = sorted(entities, key=lambda x: x[0])
        
        # Check for overlapping entities
        valid_entities = []
        for i, ent in enumerate(entities):
            overlap = False
            for j, prev_ent in enumerate(valid_entities):
                if not (ent[1] <= prev_ent[0] or ent[0] >= prev_ent[1]):
                    overlap = True
                    break
            if not overlap:
                valid_entities.append(ent)
        
        if text:
            spacy_data.append((text, {"entities": valid_entities}))
    
    print(f"Converted {len(spacy_data)} examples to spaCy format")
    return spacy_data


def create_docbin(data: List[Tuple[str, Dict]], nlp) -> DocBin:
    """Create a DocBin object from training data"""
    db = DocBin()
    
    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is not None:
                ents.append(span)
        
        doc.ents = ents
        db.add(doc)
    
    return db


def main():
    """Main data preparation pipeline"""
    # Paths
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / "data" / "raw" / "2025_11_27.jsonl"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and convert data
    print("Loading Prodigy annotations...")
    prodigy_data = load_prodigy_data(raw_data_path)
    
    print("Converting to spaCy format...")
    spacy_data = convert_to_spacy_format(prodigy_data)
    
    # Split data into train, dev, and test sets (70%, 15%, 15%)
    print("Splitting data into train/dev/test sets...")
    train_data, temp_data = train_test_split(spacy_data, test_size=0.3, random_state=42)
    dev_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    print(f"Train set: {len(train_data)} examples")
    print(f"Dev set: {len(dev_data)} examples")
    print(f"Test set: {len(test_data)} examples")
    
    # Create blank spaCy model for conversion
    nlp = spacy.blank("en")
    
    # Create DocBin objects
    print("Creating DocBin files...")
    train_db = create_docbin(train_data, nlp)
    dev_db = create_docbin(dev_data, nlp)
    test_db = create_docbin(test_data, nlp)
    
    # Save DocBin files
    train_db.to_disk(output_dir / "train.spacy")
    dev_db.to_disk(output_dir / "dev.spacy")
    test_db.to_disk(output_dir / "test.spacy")
    
    print(f"\nData preparation complete!")
    print(f"Files saved to: {output_dir}")
    print(f"- train.spacy: {len(train_data)} examples")
    print(f"- dev.spacy: {len(dev_data)} examples")
    print(f"- test.spacy: {len(test_data)} examples")
    
    # Extract and save entity labels
    labels = set()
    for _, annotations in spacy_data:
        for _, _, label in annotations["entities"]:
            labels.add(label)
    
    labels_file = output_dir / "labels.txt"
    with open(labels_file, 'w') as f:
        for label in sorted(labels):
            f.write(f"{label}\n")
    
    print(f"\nEntity labels found: {sorted(labels)}")
    print(f"Labels saved to: {labels_file}")


if __name__ == "__main__":
    main()
