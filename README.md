# F1 RAG CLI

A command-line application for semantic search across Formula 1 race data using Retrieval-Augmented Generation (RAG) with embeddings.

## Overview

F1 RAG CLI enables you to query Formula 1 race lap data semantically. It uses embeddings to find the most relevant laps based on your natural language questions, making it easy to explore F1 statistics and performance metrics without writing SQL queries.

## Features

- 🏎️ **Fetch F1 Data**: Automatically download and cache official F1 race session data using FastF1
- 🧠 **Semantic Search**: Query race data using natural language powered by sentence embeddings
- ⚡ **Efficient Caching**: Built-in caching system to avoid redundant API calls
- 🎯 **Smart Queries**: Specialized handling for fastest lap queries with instant results
- 📊 **Lap Analytics**: Access comprehensive lap metrics including times, positions, and driver information

## Installation

### Requirements

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-rag-cli.git
cd f1-rag-cli
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dependencies

- **fastf1**: Official F1 data API client
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **sentence-transformers**: Semantic embedding model (all-MiniLM-L6-v2)

## Project Structure

```
f1-rag-cli/
├── fetch_data.py       # Fetch F1 race data from FastF1
├── embed.py            # Create embeddings for lap data
├── search.py           # Semantic search interface
├── requirements.txt    # Project dependencies
├── cache/              # FastF1 cache directory
│   └── 2026/          # Season data
└── data/              # Generated embeddings and lap data
    ├── f1_laps.csv            # Raw lap data
    └── f1_vectors.json        # Embeddings and chunks
```

## Usage

### 1. Fetch F1 Data

Fetch race lap data for the 2026 Australian Grand Prix:

```bash
python fetch_data.py
```

This command:
- Downloads lap data for the specified F1 season/round
- Caches the data locally to avoid repeated API calls
- Saves lap data to `data/f1_laps.csv`

### 2. Create Embeddings

Generate semantic embeddings for all laps:

```bash
python embed.py
```

This command:
- Reads lap data from CSV
- Converts lap information to text chunks
- Creates embeddings using sentence-transformers
- Saves chunks and embeddings to `data/f1_vectors.json`

### 3. Search Race Data

Query the race data semantically:

```bash
python search.py
```

Then enter your question at the prompt:

```
Ask about F1 race: Who had the fastest lap?
```

**Example Queries:**

- "What was the fastest lap time?"
- "How did Lewis Hamilton perform?"
- "Show me Verstappen's lap data"
- "Which driver had the best position?"
- "Tell me about the top 3 drivers"

The tool returns the top 3 most relevant results with similarity scores.

## How It Works

1. **Data Fetching**: FastF1 retrieves official F1 telemetry and lap data
2. **Chunking**: Each lap is converted to descriptive text
3. **Embedding**: Text chunks are converted to vector embeddings using `all-MiniLM-L6-v2`
4. **Retrieval**: User queries are embedded and compared to lap embeddings using cosine similarity
5. **Ranking**: Results are ranked by similarity and returned to the user

## Configuration

### Modify Race Data

Edit the race details in `fetch_data.py`:

```python
season = 2026
round_number = 1  # Australian Grand Prix
session = fastf1.get_session(season, round_number, 'R')  # 'R' = Race
```

Available session types:
- `'R'` - Race
- `'FP1'` - Free Practice 1
- `'FP2'` - Free Practice 2
- `'FP3'` - Free Practice 3
- `'Q'` - Qualifying
- `'S'` - Sprint

### Modify Embedding Model

In `embed.py` and `search.py`, change the model:

```python
model = SentenceTransformer('all-MiniLM-L6-v2')  # Current model
# Or try other models:
# model = SentenceTransformer('all-mpnet-base-v2')  # Larger, more accurate
# model = SentenceTransformer('all-distilroberta-v1')  # Lightweight
```

## Example Workflow

```bash
# Step 1: Get F1 data
python fetch_data.py
# Output: Loaded Australian Grand Prix - Australia
#         Saved 1234 laps to data/f1_laps.csv

# Step 2: Create embeddings
python embed.py
# Output: Creating embeddings for 1234 laps...
#         Saved 1234 chunks to data/f1_vectors.json

# Step 3: Search
python search.py
# Ask about F1 race: Who finished first?
# ============================================================
# Question: Who finished first?
# ============================================================
# 
# [Score: 0.8234]
# Driver Max Verstappen (VER) lap 58 time 1:42.456 (102.456 seconds). Finished position 1.
# ...
```

## Troubleshooting

**Issue**: "ModuleNotFoundError: No module named 'fastf1'"

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt
```

**Issue**: "No such file or directory: 'data/f1_laps.csv'"

**Solution**: Run `fetch_data.py` first to generate the data:
```bash
python fetch_data.py
```

**Issue**: Cache issues with old data

**Solution**: Clear the cache directory:
```bash
rm -r cache/  # On macOS/Linux
rmdir /s cache  # On Windows
```

## Performance Tips

- Run `embed.py` once after `fetch_data.py` - embeddings are cached in JSON
- FastF1 caches data locally; subsequent runs are fast
- The embedding model runs on CPU; GPU acceleration can be enabled via sentence-transformers

## Future Enhancements

- [ ] Multi-season search
- [ ] Driver comparison queries
- [ ] Stint analysis
- [ ] Pit stop telemetry integration
- [ ] Web interface
- [ ] API server

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - Official F1 telemetry API
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- Formula 1 official data sources

## Contact

For questions or issues, please open a GitHub issue.
