import json
import pandas as pd
from pathlib import Path

def load_press_releases(file_path):
    """
    Load the raw Department of Justice press releases from a JSONL file.

    Args:
        file_path (str or Path): The path to the JSONL file containing press releases.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded press releases.
    """
    with Path(file_path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.json_normalize(rows)

def save_to_csv(df, output_path):
    """
    Save the DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (str or Path): The path where the CSV file will be saved.
    """
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    INPUT_PATH = "data/raw/doj_press_releases.jsonl"
    OUTPUT_PATH = "data/processed/doj_press_releases.csv"

    press_releases_df = load_press_releases(INPUT_PATH)
    save_to_csv(press_releases_df, OUTPUT_PATH)