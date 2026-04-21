import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import re

# Driver code to full name mapping
driver_names = {
    'NOR': 'Lando Norris',
    'VER': 'Max Verstappen',
    'HAM': 'Lewis Hamilton',
    'PER': 'Sergio Perez',
    'SAI': 'Carlos Sainz',
    'LEC': 'Charles Leclerc',
    'RUS': 'George Russell',
    'ALO': 'Fernando Alonso',
    'PIA': 'Oscar Piastri',
    'TSU': 'Yuki Tsunoda',
    'STR': 'Lance Stroll',
    'HUL': 'Nico Hulkenberg',
    'BOT': 'Valtteri Bottas',
    'ZHO': 'Guanyu Zhou',
    'GAS': 'Pierre Gasly',
    'OCO': 'Esteban Ocon',
    'ALB': 'Alexander Albon',
    'SAR': 'Logan Sargeant',
    'RIC': 'Daniel Ricciardo',
    'MAG': 'Kevin Magnussen'
}

# Load the CSV
df = pd.read_csv('data/f1_laps.csv')

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert each lap to text
chunks = []
for i, row in df.iterrows():
    driver_code = row['Driver']
    driver_name = driver_names.get(driver_code, driver_code)
    lap_num = row['LapNumber']
    lap_time = row['LapTime']
    position = row['Position']
    
        # Convert lap time to seconds (simple version)
    try:
        time_str = str(lap_time)
        # Handle format like "0 days 00:01:27.956000"
        if 'days' in time_str:
            # Extract the HH:MM:SS part
            import re
            match = re.search(r'(\d{2}):(\d{2}):(\d{2}\.\d+)', time_str)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds_val = float(match.group(3))
                seconds = hours * 3600 + minutes * 60 + seconds_val
            else:
                seconds = 0
        elif ':' in time_str:
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

    text = f"Driver {driver_name} ({driver_code}) lap {lap_num} time {lap_time} ({seconds:.1f} seconds). Finished position {position}."
    chunks.append(text)

print(f"Creating embeddings for {len(chunks)} laps...")
embeddings = model.encode(chunks)

# Save to JSON
data = {
    "chunks": chunks,
    "embeddings": embeddings.tolist()
}

with open('data/f1_vectors.json', 'w') as f:
    json.dump(data, f)

print(f"Saved {len(chunks)} chunks to data/f1_vectors.json")