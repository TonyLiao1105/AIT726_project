import json
from src.evaluation import save_evaluation_results


def test_save_evaluation_results(tmp_path):
    # Prepare a small fake evaluation result
    results = {
        "precision": 0.5,
        "recall": 0.4,
        "f1": 0.45,
        "per_label": {
            "ORG": {"precision": 0.5, "recall": 0.4, "f1": 0.45, "tp": 1, "fp": 1, "fn": 1}
        },
    }

    out_dir = tmp_path / "outputs"
    out_path = out_dir / "eval_results.json"

    json_path, csv_path = save_evaluation_results(results, str(out_path))

    # JSON file exists and contains the expected keys
    assert json_path.exists()
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["precision"] == results["precision"]

    # CSV summary exists and contains header and the label
    assert csv_path.exists()
    csv_text = csv_path.read_text(encoding="utf-8")
    assert "label,precision,recall,f1,tp,fp,fn" in csv_text
    assert "ORG," in csv_text
