from pathlib import Path
from collections import Counter
import statistics
from prodigy.components.db import connect
import pandas as pd
import matplotlib.pyplot as plt

MY_PATH = "C:/Users/allis/.prodigy/"
DATASET_NAME = "legal_roles"

def main():
    db = connect()
    # Use the newer API (get_dataset_examples / iter_dataset_examples)
    examples = db.get_dataset_examples(DATASET_NAME)
    # Ensure we have a list to allow multiple passes over the data
    examples = list(examples)

    total = len(examples)
    print("Total annotated:", total)

    counts = Counter([ex.get("answer") for ex in examples])
    # Print counts vertically
    def _print_dict_vertical(d, title=None):
        if title:
            print(title)
        if not d:
            print("  (none)")
            return
        for k, v in d.items():
            print(f"  {k}: {v}")

    _print_dict_vertical(counts, title="Answer counts:")

    label_counts = Counter()

    for ex in examples:
        spans = ex.get("spans", [])
        for s in spans:
            label_counts[s["label"]] += 1

    _print_dict_vertical(label_counts, title="Label counts:")

    # Collect annotation times robustly. Some datasets use different fields
    # (e.g. 'time', 'duration') or may include start/end timestamps.
    from collections import Counter as _Counter
    time_fields_found = _Counter()
    times = []
    for ex in examples:
        # direct numeric fields
        if isinstance(ex.get("time"), (int, float)):
            times.append(ex["time"])
            time_fields_found["time"] += 1
            continue
        if isinstance(ex.get("duration"), (int, float)):
            times.append(ex["duration"])
            time_fields_found["duration"] += 1
            continue

        # try to compute from common timestamp fields
        start = ex.get("start_timestamp") or ex.get("start_time") or ex.get("task_start") or ex.get("timestamp_start")
        end = ex.get("answer_timestamp") or ex.get("accept_timestamp") or ex.get("end_time") or ex.get("timestamp")
        if start is not None and end is not None:
            try:
                t = float(end) - float(start)
                times.append(t)
                time_fields_found["computed"] += 1
            except Exception:
                time_fields_found["bad_timestamps"] += 1

    if times:
        avg_time = statistics.mean(times)
        print("Average time per annotation (s):", round(avg_time, 2))

        total_time = sum(times)
        print("Total annotation time (seconds):", round(total_time, 2))
    else:
        print("No numeric time values found in examples; cannot compute average/total.")

    if time_fields_found:
        _print_dict_vertical(dict(time_fields_found), title="Time fields summary:")

    df = pd.DataFrame(examples)

    print(df.head())

if __name__ == "__main__":
    main()