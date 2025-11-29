"""
    This module converts NER examples in a JSONL file to span-based 
    examples.   It reads NER examples from 'ner.jsonl', extracts the 
    spans, and writes them to 'spans_from_ner.jsonl'.  These spans can
    then be used to see if they create better model results than NER.
    Each span set is marked with a specified key.
"""

import srsly
from pathlib import Path

SOURCE = Path("C:\\Users\\allis\\OneDrive\\code_repos\\AIT726_project\\data\\raw\\ner.jsonl")
DEST = Path("C:\\Users\\allis\\OneDrive\\code_repos\\AIT726_project\\data\\raw\\spans_from_ner.jsonl")
SPANS_KEY = "span_classification"  # key used to mark the span set in each example

examples_out = []

for eg in srsly.read_jsonl(SOURCE):
    # if the NER example has spans, we keep it
    if eg.get("spans"):
        # mark which span key these belong to
        eg["spans_key"] = SPANS_KEY
        examples_out.append(eg)

srsly.write_jsonl(DEST, examples_out)
print(f"Wrote {len(examples_out)} examples to {DEST}")