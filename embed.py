import pandas as pd
import json
from sentence_transformers import SentenceTransformer

df = pd.read_csv('data/f1_laps.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = []
for i, row in df.iterrows():
    text = f"Driver {row['Driver']} completed lap {row['LapNumber']} in {row['LapTime']}. Position: {row['Position']}"
    chunks.append(text)

print(f"Creating embeddings for {len(chunks)} laps...")
embeddings = model.encode(chunks)

data = {
    "chunks": chunks,
    "embeddings": embeddings.tolist()
}

with open('data/f1_vectors.json', 'w') as f:
    json.dump(data, f)

print(f"Saved {len(chunks)} chunks to data/f1_vectors.json")