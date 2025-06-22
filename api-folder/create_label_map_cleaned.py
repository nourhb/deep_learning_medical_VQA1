import pandas as pd
import json
import os

csv_path = "kvasir_data/kvasir_vqa_full.csv"

if not os.path.exists(csv_path):
    print(f"CSV file not found: {csv_path}")
    exit(1)

try:
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(df)} rows.")
    if 'answer' not in df.columns:
        print("Column 'answer' not found in CSV!")
        exit(1)
    answers = sorted(df['answer'].dropna().unique())
    print(f"Found {len(answers)} unique answers.")
    label_map = {ans: idx for idx, ans in enumerate(answers)}
    with open("label_map_cleaned.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"label_map_cleaned.json created with {len(label_map)} labels.")
except Exception as e:
    print(f"Error: {e}")
    exit(1) 