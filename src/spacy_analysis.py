"""
This module contains functions to evaluate a spaCy NER model.
It computes true and predicted labels from a dev set,
generates a classification report, and creates a confusion matrix. 
Components can be called individually or run via main().
Results can be printed, saved, or plotted for analyss, or can just
be returned for further processing.

Configuration:
- MODEL_PATH: Path to the trained spaCy NER model.
- DEV_PATH: Path to the spaCy DocBin file containing the dev set.

Your paths should have the output of running prodigy data into SPACY.

"""

import spacy
from spacy.tokens import DocBin
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import seaborn as sns

MODEL_PATH = "C:\\Users\\allis\\projects\\ProdigyToSpacy\\ner_data3\\training_output\\model-best"
DEV_PATH = "C:\\Users\\allis\\projects\\ProdigyToSpacy\\ner_data3\\dev.spacy"


def get_y_true_pred() -> tuple[list[str], list[str]]:
    """
    Given a list of gold_docs and a spaCy nlp model,
    return y_true and y_pred lists for evaluation.
    """
    nlp = spacy.load(MODEL_PATH)

    doc_bin = DocBin().from_disk(DEV_PATH)
    docs = list(doc_bin.get_docs(nlp.vocab))

    y_true = []
    y_pred = []

    for gold_doc in docs:

        pred_doc = nlp(gold_doc.text)

        gold_ents = {(ent.start_char, ent.end_char, ent.label_) \
            for ent in gold_doc.ents}

        pred_ents = {(ent.start_char, ent.end_char, ent.label_) \
            for ent in pred_doc.ents}

        # True positives + false negatives (gold spans)
        # If the predicted span matches a gold span, it's a true
        # positive, otherwise it's considered a miss.
        for start, end, label in gold_ents:
            if (start, end, label) in pred_ents:
                y_true.append(label)
                y_pred.append(label)
            else:
                y_true.append(label)
                y_pred.append("O_MISS")  # special "missed" label

        # False positives (predicted spans not in gold)
        # If a span is predicted that is not in gold, it's a none.
        for start, end, label in pred_ents - gold_ents:
            y_true.append("O_NONE")     # not present in gold
            y_pred.append(label)
    return y_true, y_pred

def create_classification_report(
        y_pred: list[str], 
        y_true: list[str]
    ) -> tuple[str, plt.Figure]:
    """
    Create and plot a confusion matrix for the given true and 
    predicted labels.  Returns the matrix and the object so user
    can further manipulate or save it.
    """
    # O_MISS AND O_NONE labels are not useful in plots as they show
    # empty.  Thus, identifying them here to filter them out.
    SKIP_LABELS = {"O_MISS", "O_NONE"}

    pred_counts = Counter(y_pred)
    gold_labels = sorted(set(y_true) - SKIP_LABELS)
    missing_predictions = [lbl for lbl in gold_labels 
        if pred_counts.get(lbl, 0) == 0]

    if missing_predictions:
        print("Labels with zero predicted samples:", missing_predictions)

    # Build a label list for reporting that excludes the special markers
    report_labels = sorted((set(y_true) | set(y_pred)) - SKIP_LABELS)

    # Use zero_division = 0 to avoid UndefinedMetricWarning for labels 
    # with no predictions
    report = (classification_report(
        y_true, 
        y_pred, 
        labels=report_labels, 
        output_dict=True,
        zero_division=0)
    )

    df = pd.DataFrame(report).T

    # Drop aggregate rows if present, then plot only the per-label rows
    columns_to_drop = ["accuracy", "macro avg", "weighted avg"]
    df = df.drop(columns_to_drop, errors="ignore")

    # If any of the special labels are present as rows, 
    # drop them before plotting
    df = df.drop(list(SKIP_LABELS & set(df.index)), errors="ignore")
    df = df.round(3)

    # precision, recall, f1
    colors = ["#001f33", "#0099ff", "#b3e0ff"]  

    # Set up plot formatting
    ax = df[["precision", "recall", "f1-score"]].plot(
        kind="bar", 
        figsize=(12,6),
        color=colors,
    )
    plt.title("Per-label Precision / Recall / F1")
    plt.xticks(rotation=45)

    # Format legend labels: Title Case but keep tokens like 'f1' as 'F1'
    handles, labels = ax.get_legend_handles_labels()

    new_labels = [format_label(lbl) for lbl in labels]
    ax.legend(handles, new_labels, title="Metric")
    df.columns = df.columns.str.upper()

    return df, plt

def format_token(tok: str) -> str:
    t = tok.strip()
    # special-case letter+digits (e.g., f1 -> F1)
    m = re.fullmatch(r"([A-Za-z]+)(\d+)", t)
    if m:
        return m.group(1).upper() + m.group(2)
    return t.title()

def format_label(lbl: str) -> str:
    parts = re.split(r"[-_\s]+", lbl)
    return ' '.join(format_token(p) for p in parts if p)

def create_confusion_matrix(
        y_true: list[str], 
        y_pred: list[str]
    ) -> tuple[np.ndarray, plt.Figure]:
    """
    Create and plot a confusion matrix for the given true and 
    predicted labels.  Returns the matrix and the object so user
    can further manipulate or save it.
    """
    labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(
        y_true, 
        y_pred, 
        labels=labels
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,
        annot=True, 
        fmt="d", 
        xticklabels=labels, 
        yticklabels=labels,
        cmap = "crest"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Gold")
    plt.title("Confusion Matrix (Span-level)")
    return cm, plt 

def main():
    """ Evaluate a spaCy NER model on a dev set and produce 
        a classification report and confusion matrix. 
        gold_docs: Come from annotations in dev set
        pred_docs: Produced by the model
    """

    y_true, y_pred = get_y_true_pred()

    df, class_plt = create_classification_report(y_pred, y_true)
    print(df)
    print(class_plt.show())

    confusion_matrix, cm_plt = create_confusion_matrix(y_true, y_pred)
    print(confusion_matrix)
    print(cm_plt.show())
    print("Done.")
    return

if __name__ == "__main__":
    main()