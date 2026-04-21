import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

with open('data/f1_vectors.json', 'r') as f:
    data = json.load(f)

chunks = data['chunks']
embeddings = np.array(data['embeddings'])

# Find actual fastest lap from data
fastest_lap = None
fastest_time = float('inf')
fastest_driver = None

for chunk in chunks:
    match = re.search(r'\((\d+\.?\d*) seconds\)', chunk)
    if match:
        lap_seconds = float(match.group(1))
        if lap_seconds < fastest_time:
            fastest_time = lap_seconds
            fastest_lap = chunk
            driver_match = re.search(r'Driver (.+?) \(', chunk)
            if driver_match:
                fastest_driver = driver_match.group(1)

def cosine_similarity(a, b):
    dot = 0
    for i in range(len(a)):
        dot += a[i] * b[i]
    
    norm_a = 0
    for i in range(len(a)):
        norm_a += a[i] * a[i]
    norm_a = norm_a ** 0.5
    
    norm_b = 0
    for i in range(len(b)):
        norm_b += b[i] * b[i]
    norm_b = norm_b ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot / (norm_a * norm_b)

query = input("Ask about F1 race: ")
query_embedding = model.encode([query])[0]

# Check if asking about fastest lap
if "fastest" in query.lower() and fastest_lap:
    print("\n" + "="*60)
    print(f"Question: {query}")
    print("="*60)
    print(f"\n🏁 THE ACTUAL FASTEST LAP:")
    print(fastest_lap)
    print(f"\nDriver: {fastest_driver}")
    print(f"Time: {fastest_time:.1f} seconds")
    print("-"*60)
else:
    scores = []
    for i, emb in enumerate(embeddings):
        sim = cosine_similarity(query_embedding, emb)
        scores.append((sim, i))
    
    scores.sort(reverse=True)
    
    print("\n" + "="*60)
    print(f"Question: {query}")
    print("="*60)
    
    for sim, idx in scores[:3]:
        print(f"\n[Score: {sim:.4f}]")
        print(chunks[idx])
        print("-"*60)