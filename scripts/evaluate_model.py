"""
Evaluation Script for DOJ Press Release NER Model
Evaluates model performance on test data
"""

import spacy
from pathlib import Path
from spacy.tokens import DocBin
from spacy.scorer import Scorer
from collections import defaultdict
import json


def load_test_data(test_path: Path, nlp):
    """Load test data from .spacy file"""
    doc_bin = DocBin().from_disk(test_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    return docs


def evaluate_model(model_path: Path, test_data_path: Path):
    """Evaluate the trained model on test data"""
    print("=" * 80)
    print("DOJ Press Release NER Model Evaluation")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test data: {test_data_path}")
    print()
    
    # Load model
    print("Loading model...")
    nlp = spacy.load(model_path)
    
    # Load test data
    print("Loading test data...")
    test_docs = load_test_data(test_data_path, nlp)
    print(f"Test documents: {len(test_docs)}")
    print()
    
    # Make predictions
    print("Making predictions...")
    examples = []
    for gold_doc in test_docs:
        pred_doc = nlp(gold_doc.text)
        examples.append(spacy.training.Example(pred_doc, gold_doc))
    
    # Calculate scores
    print("Calculating scores...")
    scorer = Scorer()
    scores = scorer.score(examples)
    
    # Print overall results
    print("=" * 80)
    print("Overall Performance")
    print("=" * 80)
    print(f"Precision: {scores['ents_p']:.4f}")
    print(f"Recall:    {scores['ents_r']:.4f}")
    print(f"F1 Score:  {scores['ents_f']:.4f}")
    print()
    
    # Print per-entity results
    if 'ents_per_type' in scores:
        print("=" * 80)
        print("Performance by Entity Type")
        print("=" * 80)
        for label, metrics in sorted(scores['ents_per_type'].items()):
            print(f"\n{label}:")
            print(f"  Precision: {metrics['p']:.4f}")
            print(f"  Recall:    {metrics['r']:.4f}")
            print(f"  F1 Score:  {metrics['f']:.4f}")
    
    # Count entities in test data
    print("\n" + "=" * 80)
    print("Entity Distribution in Test Data")
    print("=" * 80)
    entity_counts = defaultdict(int)
    for doc in test_docs:
        for ent in doc.ents:
            entity_counts[ent.label_] += 1
    
    for label, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"{label}: {count}")
    
    # Save results
    results_dir = model_path.parent / "evaluation_results"
    results_dir.mkdir(exist_ok=True)
    
    results_file = results_dir / "evaluation_metrics.json"
    with open(results_file, 'w') as f:
        json.dump(scores, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Show some examples
    print("\n" + "=" * 80)
    print("Sample Predictions (First 3 Documents)")
    print("=" * 80)
    for i, (example, gold_doc) in enumerate(zip(examples[:3], test_docs[:3])):
        print(f"\n--- Document {i+1} ---")
        print(f"Text: {gold_doc.text[:200]}...")
        print("\nGold entities:")
        for ent in gold_doc.ents:
            print(f"  {ent.text} ({ent.label_})")
        print("\nPredicted entities:")
        for ent in example.predicted.ents:
            print(f"  {ent.text} ({ent.label_})")


def main():
    """Main evaluation function"""
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "ner_model" / "model-best"
    test_data_path = project_root / "data" / "processed" / "test.spacy"
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Please run train_model.py first to train the model.")
        return
    
    if not test_data_path.exists():
        print(f"ERROR: Test data not found at {test_data_path}")
        print("Please run prepare_data.py first to generate test data.")
        return
    
    evaluate_model(model_path, test_data_path)


if __name__ == "__main__":
    main()
