import pandas as pd
import json
from sentence_transformers import SentenceTransformer

df = pd.read_csv('data/f1_laps.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

chunks = []
for i, row in df.iterrows():
    driver_code = row['Driver']
    lap_num = row['LapNumber']
    lap_time = row['LapTime']
    position = row['Position']
    
    # Convert lap time to seconds for better comparison
    try:
        time_str = str(lap_time)
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                seconds = int(parts[0]) * 60 + float(parts[1])
            else:
                seconds = float(parts[0])
        else:
            seconds = float(lap_time)
    except:
        seconds = 0
    
    text = f"Driver {driver_code} lap {lap_num} time {lap_time} ({seconds:.1f} seconds). Finished position {position}."
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