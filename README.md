# 🧱 BrickQA

👋 Welcome! We're so glad you're here.

BrickQA is a friendly tool designed to help you explore and query building data using the Brick schema. Whether you're a building scientist, facility manager, or just curious about smart buildings, this project makes it easy to ask natural language questions about your building's sensors, equipment, and systems.

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

## 💡 Example Questions
Here are some questions you can try:
- 🌡️ "What temperature sensors are in the building?"
- 🎛️ "Which equipment has zone air temperature setpoints?"
- 🌀 "What VAVs feed HVAC zones?"
- 🏭 "List all AHUs with their supply air temperature sensors."
- 📊 "What is the timeseries ID for each zone's occupancy sensor?"

We hope BrickQA makes your building data exploration a little easier and a lot more enjoyable. Happy querying! 🎉

## 📚 Citation

