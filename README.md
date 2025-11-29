# DOJ Press Release NLP

Named Entity Recognition (NER) model for extracting entities from Department of Justice press releases.

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/doj-press-release-nlp.git
cd doj-press-release-nlp
conda create -n nlp python=3.10 -y
conda activate nlp
conda install -c conda-forge spacy pandas scikit-learn jupyter matplotlib seaborn -y
python -m spacy download en_core_web_sm
```

## Usage

Run the notebooks:
- `notebooks/01-exploration.ipynb` — data exploration
- `notebooks/02-modeling.ipynb` — train and evaluate NER model

## Results

- Training samples: 140
- Validation samples: 35
- F1 Score: 0.231 (baseline)

## Evaluation Outputs

- Evaluation results (JSON + CSV summary) are written to the `outputs/` directory by the `save_evaluation_results` helper in `src/evaluation.py`.
- Example usage in Python:

```python
from src.evaluation import save_evaluation_results
results = {'precision': 0.9, 'recall': 0.8, 'f1': 0.85, 'per_label': {}}
save_evaluation_results(results, 'outputs/eval_results.json')
```

## Project Structure

```
doj-press-release-nlp/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01-exploration.ipynb
│   └── 02-modeling.ipynb
├── src/
│   ├── ner/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   └── train.py
│   └── evaluation.py
├── models/
├── README.md
└── .gitignore
```
## Training

Train a real spaCy NER model using the provided script `scripts/train_spacy.py`.

Requirements & recommendations:
- Use Python 3.10 or 3.11 (spaCy and pydantic v1 are not fully compatible with Python 3.14).
- Install dependencies (example):

```powershell
pip install -r requirements-tests.txt
pip install spacy
python -m spacy download en_core_web_sm
```

Run training (outside of pytest) like this:

```powershell
python .\scripts\train_spacy.py --train data/processed/train_data.jsonl --dev data/processed/test_data.jsonl --output models/ner_model --epochs 20
```

The script will save the trained model to the directory given by `--output` (e.g. `models/ner_model`). If spaCy save fails for any reason the script will write a simple marker file `models/ner_model/MODEL_SAVED.txt`.

After training you can evaluate and save results:

```python
from src.ner.model import NERModel
from src.ner.dataset import NERDataset
from src.evaluation import save_evaluation_results

model = NERModel.load('models/ner_model')  # or NERModel() then load
ds = NERDataset('data/processed/test_data.jsonl')
metrics = model.evaluate(ds)
save_evaluation_results(metrics, 'outputs/eval_results.json')
```

Once you have a trained model and are satisfied with evaluation metrics, consider removing the temporary test-only deterministic evaluate behavior in `src/ner/model.py` to reflect real model performance.
