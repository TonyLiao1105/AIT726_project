"""
Training Script for DOJ Press Release NER Model
Trains a spaCy NER model using prepared training data
"""

import sys
from pathlib import Path
import spacy
from spacy.cli.train import train as spacy_train


def main():
    """Train the NER model using spaCy's training command"""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.cfg"
    output_dir = project_root / "models" / "ner_model"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("DOJ Press Release NER Model Training")
    print("=" * 80)
    print(f"Config file: {config_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if config exists
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}")
        print("Please run prepare_data.py first to generate training data.")
        sys.exit(1)
    
    # Train the model
    print("Starting training...")
    print("This may take several minutes...")
    print()
    
    try:
        spacy_train(
            config_path=str(config_path),
            output_path=str(output_dir),
            overrides={},
            use_gpu=-1  # Use -1 for CPU, 0 for GPU
        )
        
        print()
        print("=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Model saved to: {output_dir / 'model-best'}")
        print()
        print("Next steps:")
        print("1. Run evaluate_model.py to test the model")
        print("2. Run inference.py to use the model on new press releases")
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
