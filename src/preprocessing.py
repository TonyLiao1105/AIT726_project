import pandas as pd
import json
import re

def load_press_releases(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = text.strip()  # Remove leading and trailing whitespace
    return text

def preprocess_press_releases(press_releases):
    for release in press_releases:
        release['title'] = clean_text(release.get('title', ''))
        release['content'] = clean_text(release.get('content', ''))
        release['date'] = release.get('date', None)  # Keep date as is
    return press_releases

def save_processed_data(processed_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for release in processed_data:
            f.write(json.dumps(release) + '\n')