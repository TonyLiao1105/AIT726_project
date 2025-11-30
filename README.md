# DOJ Press Release NER Model

A Named Entity Recognition (NER) model built with spaCy to extract key information from Department of Justice press releases.

## Project Overview

This project analyzes DOJ press releases to identify and extract:
- **DEFENDANT**: Names of defendants in cases
- **PROSECUTOR**: Names of prosecutors handling cases
- **JUDGE**: Names of judges presiding over cases
- **SENTENCE**: Sentencing information
- **FRAUD MECHANISM**: Types and descriptions of fraud
- **FRAUD AMOUNT**: Monetary amounts involved in fraud
- **GOV PROGRAM**: Government programs referenced
- **BUSINESS**: Business entities mentioned

## Project Structure

```
AIT726_project/
├── data/
│   ├── raw/
│   │   └── 2025_11_27.jsonl          # Prodigy-annotated training data
│   └── processed/
│       ├── train.spacy                # Training data (generated)
│       ├── dev.spacy                  # Development data (generated)
│       ├── test.spacy                 # Test data (generated)
│       └── labels.txt                 # Entity labels list
├── models/
│   └── ner_model/
│       ├── model-best/                # Best trained model
│       └── model-last/                # Last checkpoint
├── scripts/
│   ├── prepare_data.py                # Data preparation script
│   ├── train_model.py                 # Model training script
│   ├── evaluate_model.py              # Model evaluation script
│   └── inference.py                   # Inference on new data
├── config/
│   └── config.cfg                     # spaCy training configuration
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## Installation

### 1. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

For better performance, you can download the larger model:

```bash
python -m spacy download en_core_web_lg
```

## Usage

### Step 1: Prepare Training Data

Convert Prodigy annotations to spaCy training format:

```bash
cd scripts
python prepare_data.py
```

This will:
- Load annotated data from `data/raw/2025_11_27.jsonl`
- Convert Prodigy format to spaCy format
- Split data into train (70%), dev (15%), and test (15%) sets
- Save processed data to `data/processed/`

### Step 2: Train the Model

Train the NER model:

```bash
python train_model.py
```

This will:
- Use the configuration in `config/config.cfg`
- Train a custom NER model
- Save the best model to `models/ner_model/model-best/`
- Training typically takes 10-30 minutes depending on your hardware

### Step 3: Evaluate the Model

Evaluate model performance on test data:

```bash
python evaluate_model.py
```

This will:
- Load the trained model
- Run predictions on test data
- Calculate precision, recall, and F1 scores
- Show performance by entity type
- Display sample predictions

### Step 4: Use the Model (Inference)

#### Interactive Mode

Test the model interactively:

```bash
python inference.py
```

Then enter press release text to see extracted entities.

#### Batch Processing

Process a JSONL file:

```bash
python inference.py ../data/raw/2025_11_12.jsonl output_results.json
```

This will:
- Process all entries in the input file
- Extract entities from each press release
- Save results to the output JSON file

## Model Configuration

The model uses a custom spaCy pipeline with:
- **tok2vec**: Token-to-vector embedding layer
- **ner**: Named Entity Recognition component

Key training parameters in `config/config.cfg`:
- Max steps: 20,000
- Dropout: 0.1
- Learning rate: 0.001
- Batch size: Dynamic (100-1000 tokens)
- Evaluation frequency: Every 200 steps

## Expected Performance

Based on the training data characteristics:
- **Overall F1**: 0.85-0.92 (target)
- **High performers**: DEFENDANT, PROSECUTOR, JUDGE
- **Challenging**: FRAUD MECHANISM, GOV PROGRAM (more varied)

## Entity Categories Explained

1. **DEFENDANT**: Individual or entity being prosecuted
   - Example: "John Smith", "ABC Corporation"

2. **PROSECUTOR**: Attorney prosecuting the case
   - Example: "Assistant U.S. Attorney Jane Doe"

3. **JUDGE**: Judge presiding over the case
   - Example: "U.S. District Judge Robert Johnson"

4. **SENTENCE**: Punishment or sentencing information
   - Example: "15 years in prison", "5 years probation"

5. **FRAUD MECHANISM**: Description of how fraud was committed
   - Example: "Medicare fraud scheme", "false claims"

6. **FRAUD AMOUNT**: Monetary value involved
   - Example: "$2.5 million", "$500,000"

7. **GOV PROGRAM**: Government program referenced
   - Example: "Medicare", "Paycheck Protection Program"

8. **BUSINESS**: Business entity mentioned
   - Example: "XYZ Healthcare Services"

## Troubleshooting

### Common Issues

1. **Module not found errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate your virtual environment

2. **spaCy model not found**
   - Download the English model: `python -m spacy download en_core_web_sm`

3. **Training takes too long**
   - Reduce `max_steps` in `config/config.cfg`
   - Use a smaller subset of training data for testing

4. **Low model performance**
   - Increase training data size
   - Adjust learning rate or dropout in config
   - Try using a larger pre-trained model (en_core_web_lg)

## Advanced Usage

### Fine-tuning

To fine-tune the model with additional data:
1. Add new annotated data to `data/raw/`
2. Re-run `prepare_data.py`
3. Run `train_model.py` with the combined dataset

### Custom Configuration

Modify `config/config.cfg` to:
- Adjust model architecture
- Change hyperparameters
- Add additional pipeline components

## Future Enhancements

- [ ] Add COURT and DATE entity types
- [ ] Implement active learning for efficient annotation
- [ ] Create web interface for easy inference
- [ ] Add model ensemble for improved accuracy
- [ ] Support for multi-document extraction
- [ ] Integration with document management systems

## License

This project is for educational purposes as part of AIT 726 coursework.

## Authors

Tony Liao - George Mason University

## Acknowledgments

- Prodigy for annotation tool
- spaCy for NLP framework
- DOJ for public press release data
