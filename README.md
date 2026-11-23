# DATS 6401: Weeks 11-12 — F1 APIs & Model Serving

**Soumay Patidar**

## What's in here

| File | What it does |
|---|---|
| `app.py` | Streamlit app with 2 tabs — Part A (F1 API client) + Part B (podium prediction client) |
| `server.py` | FastAPI server that serves the F1 podium prediction model |
| `train_model.py` | Trains the model on F1 race results (2000-2024) and saves `model.joblib` |
| `test_endpoint.py` | Test script — 6 tests covering predictions, bad input, and health check |
| `data/` | F1 Ergast CSV files used for model training |
| `requirements.txt` | All dependencies |

## Setup

```bash
pip install -r requirements.txt
```

## Part B — Run the server first

```bash
# 1. Train the model (creates model.joblib)
python train_model.py

# 2. Start the FastAPI server
uvicorn server:app --reload

# 3. Check the auto-generated docs
# http://127.0.0.1:8000/docs

# 4. Run tests (second terminal)
python test_endpoint.py
```

## Part A — Run the Streamlit client

```bash
streamlit run app.py
```

- **Tab 1 (Part A):** Fetches live F1 season results from the Jolpica API
- **Tab 2 (Part B):** Sends predictions to the FastAPI server

## API used (Part A)

**Jolpica F1 API** (Ergast successor)
- Base URL: `https://api.jolpi.ca/ergast/f1/`
- No API key needed
- Used endpoint: `/{season}/results.json`

## Model details (Part B)

- **Task:** Predict whether a driver finishes on the podium (top 3)
- **Data:** F1 race results 2000-2024 from the Ergast database
- **Features:** grid position, constructor podium rate, constructor, circuit, laps, round
- **Model:** Random Forest (150 trees, max depth 10)
- **Accuracy:** ~91%
