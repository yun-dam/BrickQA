# 🧱 BrickQA

BrickQA is a friendly tool designed to help you explore and query building data using the Brick schema. Whether you're a building scientist, facility manager, or just curious about smart buildings, this project makes it easy to ask natural language questions about your building's sensors, equipment, and systems.

By leveraging a LLM-based agent, BrickQA bridges the semantic gap in building operations, translating your questions into executable SPARQL queries through dynamic graph exploration and structured query decomposition. It effectively transforms abstract semantic models into actionable facility management insights without requiring you to master complex query languages.

## 🚀 Getting Started

1. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Google Cloud authentication (required for the Gemini API):
   ```bash
   gcloud auth application-default login
   ```

3. Run a query against your Brick TTL file:
   ```bash
   python main.py --ttl your_building.ttl --question "What temperature sensors are in the building?"
   ```

## ⚙️ Usage Options

### 💬 Interactive Mode
Launch an interactive session where you can ask multiple questions:
```bash
python main.py --ttl my_building.ttl
```

### ❓ Single Question Mode
Ask a one-off question and get your answer:
```bash
python main.py --ttl my_building.ttl --question "What VAVs feed HVAC zones?"
python main.py --ttl my_building.ttl -q "List all AHUs with their supply air temperature sensors."
```

### 🎯 Improve Accuracy with Schema Samples
For better results, provide a sample of your TTL schema in the prompt:
```bash
python main.py --ttl my_building.ttl --schema-in-prompt my_building_sample.ttl
```

### 🔧 Simplified Mode (Disable Decomposition)
If you prefer simpler queries without decomposition:
```bash
python main.py --ttl my_building.ttl --no-decomposer
```

### ⏰ Disable Temporal Handling
Skip temporal constraint processing:
```bash
python main.py --ttl my_building.ttl --no-temporal
```

### ☁️ Specify Google Cloud Project
Override the default project:
```bash
python main.py --ttl my_building.ttl --project my-gcp-project
```

### 🤫 Quiet Mode
Reduce output for cleaner results:
```bash
python main.py --ttl my_building.ttl --question "What sensors are in Zone 1?" --quiet
```

## 📊 Temporal Evaluation

Got a question about time ranges, averages, or "what happened when X was true"? Use `main_temporal.py`. It loads your TTL plus a timeseries CSV, runs one question, and saves the whole run to a JSON file so you can see what the agent did.

```bash
python main_temporal.py \
    --building-id my_building \
    --ttl path/to/my_building.ttl \
    --csv path/to/my_building_timeseries.csv \
    --schema-in-prompt path/to/my_building_description.txt \
    --question "Find the maximum outdoor air temperature observed during the last 24 hours." \
    --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

## 💡 Example Questions
Here are some questions you can try:
- 🌡️ "What temperature sensors are in the building?"
- 🎛️ "Which equipment has zone air temperature setpoints?"
- 🌀 "What VAVs feed HVAC zones?"
- 🏭 "List all AHUs with their supply air temperature sensors."
- 📊 "What is the timeseries ID for each zone's occupancy sensor?"

And a few temporal questions (when running with timeseries data loaded, e.g. via `main_temporal.py`):
- ⏱️ "Find periods in the last 24 hours where the room temperature was higher than the discharge air temperature."
- 📅 "Compare the room temperature and the cooling setpoint between 9 AM and 5 PM on September 1st."
- 📈 "Calculate the average difference between discharge air temperature and room temperature on August 25th."
- 🔺 "Find the maximum outdoor air temperature observed during the last 24 hours."
- 🔢 "Count how many times the room temperature exceeded the cooling setpoint on September 10th."

We hope BrickQA makes your building data exploration a little easier and a lot more enjoyable!

## 🧪 Evaluation

BrickQA is benchmarked on the [BuildingQA dataset](https://github.com/INFERLab/BuildingQA), a collection of natural-language questions over Brick-modeled buildings designed to stress-test question-answering systems on building knowledge graphs. The dataset spans multiple buildings and a range of difficulty levels, and it is the reference benchmark used in this repository for measuring generation accuracy.

Per-query results are stored in `evaluation/`:

| File | Setting |
|---|---|
| `evaluation_per_query_100triples.csv` | Sub-sampled graphs with 100 triples per building |
| `evaluation_per_query_5000triples.csv` | Sub-sampled graphs with 5000 triples per building |
| `evaluation_per_query_no-knowledge-graph.csv` | LLM-only baseline with no knowledge graph |

Each row reports `arity_f1`, `entity_set_f1`, `row_matching_f1`, and `exact_match_f1` for a single `(dataset, query, method)` tuple, along with the execution status, letting you compare BrickQA against a ReAct baseline across graph sizes.

## 📚 Citation

If you find BrickQA helpful in your research or work, please consider citing it:

TBD
