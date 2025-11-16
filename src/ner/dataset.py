# ...existing code...
import typing as t
from dataclasses import dataclass
import pandas as pd
import spacy
from spacy.tokens import DocBin

@dataclass
class ExampleAnn:
    start: int
    end: int
    label: str

class CustomDataset:
    """
    Lightweight dataset wrapper expected by the notebook.
    Expects a DataFrame with a text column (default 'text') and
    a spans column (default 'spans') where spans is a list of
    {"start": int, "end": int, "label": str} or tuples (start,end,label).
    """

    def __init__(self, df: pd.DataFrame, text_col: str = "text", spans_col: str = "spans"):
        self.df = df.reset_index(drop=True)
        self.text_col = text_col
        self.spans_col = spans_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = row.get(self.text_col, "") if isinstance(row, (pd.Series, dict)) else ""
        raw_spans = row.get(self.spans_col, []) if isinstance(row, (pd.Series, dict)) else []
        entities: t.List[t.Tuple[int, int, str]] = []
        if raw_spans:
            for s in raw_spans:
                if isinstance(s, dict):
                    start = s.get("start")
                    end = s.get("end")
                    label = s.get("label") or s.get("entity") or s.get("label_name")
                elif isinstance(s, (list, tuple)) and len(s) >= 3:
                    start, end, label = s[0], s[1], s[2]
                else:
                    continue
                if isinstance(start, int) and isinstance(end, int) and isinstance(label, str):
                    entities.append((start, end, label))
        return text, {"entities": entities}

    def to_spacy_docbin(self, nlp: t.Optional[spacy.language.Language] = None) -> DocBin:
        """Convert dataset to a spaCy DocBin (useful for training)."""
        if nlp is None:
            nlp = spacy.blank("en")
        db = DocBin()
        for i in range(len(self)):
            text, ann = self[i]
            doc = nlp.make_doc(text)
            spans = []
            for (start, end, label) in ann.get("entities", []):
                span = doc.char_span(start, end, label=label, alignment_mode="contract")
                if span is None:
                    # skip spans that don't align with tokenization
                    continue
                spans.append(span)
            doc.ents = spans
            db.add(doc)
        return db

    @classmethod
    def from_jsonl(cls, path: str, text_col: str = "text", spans_col: str = "spans"):
        """Load a DataFrame-backed dataset from a JSONL file (each line a JSON object)."""
        df = pd.read_json(path, lines=True)
        return cls(df, text_col=text_col, spans_col=spans_col)
# ...existing code...