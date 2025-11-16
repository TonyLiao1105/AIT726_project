import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

class NERModel:
    """spaCy-based Named Entity Recognition model for DOJ press releases."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            self.nlp = spacy.blank("en")
        
        # Ensure NER component exists
        if "ner" not in self.nlp.pipe_names:
            ner = self.nlp.add_pipe("ner", last=True)
        else:
            ner = self.nlp.get_pipe("ner")
        
        self.ner = ner
    
    def add_labels(self, labels: list):
        """Add entity labels to the NER component."""
        for label in labels:
            self.ner.add_label(label)
    
    def predict(self, text: str):
        """Predict entities in text."""
        doc = self.nlp(text)
        return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
    
    def save(self, path: str):
        """Save model to disk."""
        self.nlp.to_disk(path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk."""
        nlp = spacy.load(path)
        model = cls.__new__(cls)
        model.nlp = nlp
        model.ner = nlp.get_pipe("ner")
        return model