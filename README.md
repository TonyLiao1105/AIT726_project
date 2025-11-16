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