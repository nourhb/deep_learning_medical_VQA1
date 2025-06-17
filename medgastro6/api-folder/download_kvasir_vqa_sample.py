import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# API endpoint for the first 100 rows
url = "https://datasets-server.huggingface.co/rows?dataset=SimulaMet-HOST%2FKvasir-VQA&config=default&split=raw&offset=0&length=100"

print("Fetching sample data from Hugging Face API...")
response = requests.get(url)
data = response.json()

rows = []
for row in data['rows']:
    fields = row['row']
    rows.append({
        'img_id': fields['img_id'],
        'question': fields['question'],
        'answer': fields['answer'],
        'source': fields['source'],
        'image_url': fields['image']['src']
    })

sample_dir = "kvasir_data_sample"
img_dir = os.path.join(sample_dir, "images")
os.makedirs(img_dir, exist_ok=True)

print("Saving sample_100.csv...")
df = pd.DataFrame(rows)
df.to_csv(os.path.join(sample_dir, "sample_100.csv"), index=False)

print("Downloading images...")
for i, row in df.iterrows():
    img_url = row['image_url']
    img_id = row['img_id']
    img_path = os.path.join(img_dir, f"{img_id}.jpg")
    img_data = requests.get(img_url).content
    with open(img_path, 'wb') as handler:
        handler.write(img_data)
    if (i+1) % 10 == 0:
        print(f"Downloaded {i+1} images...")

print("Sample download complete!") 