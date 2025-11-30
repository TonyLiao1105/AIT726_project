"""
Inference Script for DOJ Press Release NER Model
Uses trained model to extract entities from new press releases
"""

import spacy
from pathlib import Path
import json
from typing import List, Dict
import sys


def load_model(model_path: Path):
    """Load the trained spaCy NER model"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    print(f"Loading model from {model_path}...")
    nlp = spacy.load(model_path)
    return nlp


def extract_entities(nlp, text: str) -> Dict:
    """Extract entities from text using the trained model"""
    doc = nlp(text)
    
    entities = {
        "DEFENDANT": [],
        "PROSECUTOR": [],
        "JUDGE": [],
        "SENTENCE": [],
        "FRAUD_MECHANISM": [],
        "FRAUD_AMOUNT": [],
        "GOV_PROGRAM": [],
        "BUSINESS": []
    }
    
    for ent in doc.ents:
        # Normalize label (handle spaces in label names)
        label_key = ent.label_.replace(" ", "_").upper()
        if label_key in entities:
            entities[label_key].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
        else:
            # Handle any unexpected labels
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "label": ent.label_
            })
    
    return entities


def process_single_text(nlp, text: str) -> Dict:
    """Process a single press release text"""
    entities = extract_entities(nlp, text)
    
    result = {
        "text": text,
        "entities": entities,
        "summary": {
            "total_entities": sum(len(ents) for ents in entities.values())
        }
    }
    
    # Add counts per entity type
    for label, ents in entities.items():
        result["summary"][f"{label.lower()}_count"] = len(ents)
    
    return result


def process_jsonl_file(nlp, input_file: Path, output_file: Path):
    """Process a JSONL file containing press releases"""
    results = []
    
    print(f"Processing {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                text = data.get('text', '')
                
                if text:
                    result = process_single_text(nlp, text)
                    result['metadata'] = data.get('meta', {})
                    result['line_number'] = i
                    results.append(result)
                    
                    if i % 100 == 0:
                        print(f"  Processed {i} documents...")
            
            except json.JSONDecodeError:
                print(f"  Warning: Could not parse line {i}")
                continue
    
    # Save results
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Processed {len(results)} documents")
    return results


def print_entity_summary(entities: Dict):
    """Print a nice summary of extracted entities"""
    print("\nExtracted Entities:")
    print("=" * 80)
    
    for label, ents in entities.items():
        if ents:
            print(f"\n{label}:")
            for ent in ents:
                print(f"  - {ent['text']}")


def interactive_mode(nlp):
    """Interactive mode for testing the model"""
    print("\n" + "=" * 80)
    print("Interactive Mode")
    print("=" * 80)
    print("Enter press release text (or 'quit' to exit)")
    print("=" * 80 + "\n")
    
    while True:
        print("\nEnter text (or 'quit' to exit):")
        text = input("> ")
        
        if text.lower() == 'quit':
            break
        
        if text.strip():
            result = process_single_text(nlp, text)
            print_entity_summary(result['entities'])
            print(f"\nTotal entities found: {result['summary']['total_entities']}")


def main():
    """Main inference function"""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "ner_model" / "model-best"
    
    print("=" * 80)
    print("DOJ Press Release NER - Inference")
    print("=" * 80)
    
    # Load model
    try:
        nlp = load_model(model_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please run train_model.py first to train the model.")
        sys.exit(1)
    
    print(f"Model loaded successfully!")
    print(f"Entities recognized: {nlp.get_pipe('ner').labels}")
    print()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
        
        if not input_file.exists():
            print(f"ERROR: Input file not found: {input_file}")
            sys.exit(1)
        
        # Determine output file
        if len(sys.argv) > 2:
            output_file = Path(sys.argv[2])
        else:
            output_file = input_file.parent / f"{input_file.stem}_annotated.json"
        
        # Process file
        results = process_jsonl_file(nlp, input_file, output_file)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        total_entities = sum(r['summary']['total_entities'] for r in results)
        print(f"Total documents processed: {len(results)}")
        print(f"Total entities extracted: {total_entities}")
        print(f"Average entities per document: {total_entities/len(results):.2f}")
        
    else:
        # Interactive mode
        interactive_mode(nlp)


if __name__ == "__main__":
    main()
