#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the training script
python src/ner/train.py --config configs/default.yaml

# Deactivate the virtual environment
deactivate