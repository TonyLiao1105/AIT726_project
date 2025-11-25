from prodigy.components.db import connect
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd   # optional but handy

DATASET_NAME = "legal_roles"

def main():
    Database = connect()
 #   examples = db.get_dataset(DATASET_NAME) 
    examples = Database.get_dataset_examples(DATASET_NAME)

    print(f"Loaded {len(examples)} examples from dataset '{DATASET_NAME}'")

    # --- 1. Basic answer stats (accept / reject / ignore) ---
    answers = [ex.get("answer") for ex in examples]
    answer_counts = Counter(answers)
    print("\nAnswer counts:")
    for ans, cnt in answer_counts.items():
        print(f"  {ans}: {cnt}")

    # --- 2. Label counts (NER spans) ---
    span_label_counts = Counter()
    docs_with_label_counts = Counter()

    for ex in examples:
        spans = ex.get("spans", [])
        labels_in_doc = set()
        for s in spans:
            label = s.get("label")
            if label:
                span_label_counts[label] += 1
                labels_in_doc.add(label)
        # count documents that have at least one of each label
        for label in labels_in_doc:
            docs_with_label_counts[label] += 1

    print("\nSpan label counts (total annotated spans per label):")
    for label, cnt in span_label_counts.items():
        print(f"  {label}: {cnt}")

    print("\nDocs with label (number of examples containing at least one span of that label):")
    for label, cnt in docs_with_label_counts.items():
        print(f"  {label}: {cnt}")

    # --- 3. Optional: put stats into a DataFrame for nice table export ---
    df_labels = pd.DataFrame({
        "label": list(span_label_counts.keys()),
        "num_spans": [span_label_counts[l] for l in span_label_counts.keys()],
        "num_docs_with_label": [docs_with_label_counts[l] for l in span_label_counts.keys()],
    }).sort_values("num_spans", ascending=False)

    print("\nLabel stats table:")
    print(df_labels)

    # --- 4. Visualization: bar chart of # spans per label ---
    if span_label_counts:
        labels = list(span_label_counts.keys())
        values = [span_label_counts[l] for l in labels]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title(f"Number of annotated spans per label in '{DATASET_NAME}'")
        plt.xlabel("Label")
        plt.ylabel("Number of spans")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo spans found in this dataset – nothing to plot.")

if __name__ == "__main__":
    main()