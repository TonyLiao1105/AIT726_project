# DOJ Press Release NER Model - Quick Reference Guide

## Entity Types
1. **DEFENDANT** - Names of defendants
2. **PROSECUTOR** - Names of prosecutors  
3. **JUDGE** - Names of judges
4. **SENTENCE** - Sentencing information
5. **FRAUD MECHANISM** - Types/descriptions of fraud
6. **FRAUD AMOUNT** - Monetary amounts
7. **GOV PROGRAM** - Government programs
8. **BUSINESS** - Business entities

## Quick Commands

### 1. Prepare Data
```bash
cd scripts
python prepare_data.py
```

### 2. Train Model
```bash
python train_model.py
```

### 3. Evaluate Model
```bash
python evaluate_model.py
```

### 4. Run Inference

**Interactive mode:**
```bash
python inference.py
```

**Batch processing:**
```bash
python inference.py path/to/input.jsonl output.json
```

## File Locations

- **Training data**: `data/raw/2025_11_27.jsonl`
- **Processed data**: `data/processed/` 
- **Trained model**: `models/ner_model/model-best/`
- **Config**: `config/config.cfg`

## Troubleshooting

**If training fails:**
1. Check that data exists in `data/raw/2025_11_27.jsonl`
2. Run prepare_data.py first
3. Check you have ~70+ MB free disk space

**If imports fail:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Performance issues:**
- Reduce max_steps in `config/config.cfg` (line 65)
- Use smaller batches (reduce batch_size on line 22)

## Example Usage

```python
import spacy

# Load model
nlp = spacy.load("models/ner_model/model-best")

# Process text
text = "John Smith was sentenced to 10 years by Judge Williams..."
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(f"{ent.text} -> {ent.label_}")
```
