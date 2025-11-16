import pandas as pd
import numpy as np

def extract_features(text):
    features = {}
    
    # Basic features
    features['length'] = len(text)
    features['num_words'] = len(text.split())
    
    # Example of more complex features
    features['num_sentences'] = text.count('.') + text.count('!') + text.count('?')
    
    # Add more features as needed for NER
    return features

def create_feature_dataframe(texts):
    feature_list = []
    for text in texts:
        features = extract_features(text)
        feature_list.append(features)
    
    return pd.DataFrame(feature_list)

def extract_entity_features(entities):
    entity_features = []
    for entity in entities:
        features = {
            'entity_type': entity['type'],
            'start': entity['start'],
            'end': entity['end'],
            'text': entity['text']
        }
        entity_features.append(features)
    
    return pd.DataFrame(entity_features)