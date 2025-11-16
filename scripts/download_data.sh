#!/bin/bash

# Create directories if they do not exist
mkdir -p data/raw
mkdir -p data/interim
mkdir -p data/processed

# Download the Department of Justice press releases dataset
curl -o data/raw/doj_press_releases.jsonl https://example.com/path/to/doj_press_releases.jsonl

echo "Data downloaded to data/raw/doj_press_releases.jsonl"