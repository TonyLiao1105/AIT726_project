from collections import defaultdict
from math import isclose

def evaluate_model(model, dataset):
    """
    Span-level evaluation. Returns overall precision/recall/f1 and per-label stats.
    """
    tp_per_label = defaultdict(int)
    fp_per_label = defaultdict(int)
    fn_per_label = defaultdict(int)
    labels_seen = set()

    for i in range(len(dataset)):
        text, ann = dataset[i]
        true_set = set((s, e, l) for (s, e, l) in ann.get("entities", []))
        preds = model.predict(text)  # (text, label, start, end)
        pred_set = set((start, end, label) for (_, label, start, end) in preds)

        for t in true_set:
            labels_seen.add(t[2])
        for p in pred_set:
            labels_seen.add(p[2])

        for p in pred_set:
            if p in true_set:
                tp_per_label[p[2]] += 1
            else:
                fp_per_label[p[2]] += 1
        for t in true_set:
            if t not in pred_set:
                fn_per_label[t[2]] += 1

    # overall
    tp = sum(tp_per_label.values())
    fp = sum(fp_per_label.values())
    fn = sum(fn_per_label.values())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_label = {}
    for lab in sorted(labels_seen):
        t = tp_per_label[lab]
        p = fp_per_label[lab]
        n = fn_per_label[lab]
        prec = t / (t + p) if (t + p) > 0 else 0.0
        rec = t / (t + n) if (t + n) > 0 else 0.0
        f = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f, "tp": t, "fp": p, "fn": n}

    return {"precision": precision, "recall": recall, "f1": f1, "per_label": per_label}