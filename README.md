# Real Estate Price Prediction System

End-to-end machine learning project that predicts the price of a property from its features (`Area`, `Bedrooms`, `Bathrooms`, `Age`, `Location`, `Property_Type`). The system covers the full ML lifecycle: **data → EDA → preprocessing → feature engineering → model training → tuning → API → frontend → testing → Docker → documentation**.

---

## 1. Project Overview

| Component | Technology |
|---|---|
| Language | Python 3.14 |
| Data analysis | pandas, matplotlib, seaborn |
| Machine learning | scikit-learn, XGBoost |
| API backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Serialization | joblib |
| Testing | pytest |
| Containerization | Docker / docker-compose |
| Logging | Python `logging` (rotating file + console) |

## 2. Dataset

- File: `data/house_prices.csv`
- **300 property records**, 8 columns.

| Column | Description |
|---|---|
| `Property_ID` | Unique identifier (dropped) |
| `Area` | Property area (sq ft) |
| `Bedrooms` | Number of bedrooms |
| `Bathrooms` | Number of bathrooms |
| `Age` | Property age (years) |
| `Location` | `Rural` / `Suburb` / `City Center` |
| `Property_Type` | `House` / `Villa` / `Apartment` |
| `Price` | **Target variable** (PKR) |

No missing values, no duplicate records.

## 3. Project Structure

```
Task6/
├── data/
│   └── house_prices.csv
├── notebooks/
│   ├── analysis.ipynb          # EDA notebook
│   └── training_pipeline.ipynb # full ML workflow -> saves model
├── scripts/
│   ├── generate_notebook.py
│   └── generate_training_notebook.py
├── src/
│   ├── config.py               # paths & settings
│   ├── data_loader.py          # dataset loading
│   ├── eda.py                  # EDA script + plots
│   ├── preprocessing.py        # pipeline + feature engineering
│   ├── train.py                # train / evaluate / tune / save
│   ├── predict.py              # load model + predict
│   ├── evaluation.py           # MAE / RMSE / R² / MAPE
│   └── logging_config.py       # shared logging
├── models/
│   └── house_price_model.pkl   # saved pipeline
│   └── metrics.json            # best model test metrics
├── app/
│   ├── main.py                 # FastAPI app
│   ├── schemas.py              # pydantic schemas
│   └── app.py                  # Streamlit frontend
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_api.py
├── outputs/eda/                # generated plots
├── logs/api.log                # API logs
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 4. Installation

```bash
# 1. Create a virtual environment
python -m venv task6

# 2. Activate it (Windows PowerShell)
task6\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
```

## 5. How to Train the Model

Two equivalent ways to run the full ML workflow
(**Dataset → Cleaning → EDA → Feature Engineering → Preprocessing Pipeline →
Train Models → Evaluation → Tuning → Select Best → Save to `models/`**):

**Option A - Notebook** (recommended, step-by-step with plots):

```bash
jupyter notebook notebooks/training_pipeline.ipynb
```

Run every cell. The final step saves the tuned pipeline to
`models/house_price_model.pkl` and writes `models/metrics.json`.

**Option B - Script:**

```bash
# Train, evaluate and save the pipeline (no tuning)
python -m src.train

# With hyperparameter tuning (GridSearchCV on the best model)
python -m src.train --tune
```

Run the EDA analysis:

```bash
python -m src.eda
jupyter notebook notebooks/analysis.ipynb   # or open the generated notebook
```

## 6. How to Build the API From the Saved Model

The FastAPI backend (`app/main.py`) simply loads `models/house_price_model.pkl`
on startup, so once the notebook/script has saved the model:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 7. Model Performance

Best model: **XGBoost** (with feature engineering) on the 20% held-out test set.

| Model | Engineering | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|---:|
| Linear Regression | no | 2,188,736 | 2,907,633 | 0.9406 | 13.16% |
| Linear Regression | yes | 2,280,105 | 2,978,351 | 0.9377 | 13.48% |
| Random Forest | no | 1,470,070 | 1,971,481 | 0.9727 | 6.76% |
| Random Forest | yes | 1,387,287 | 1,853,888 | 0.9759 | 6.17% |
| XGBoost | no | 1,258,976 | 1,608,064 | 0.9818 | 5.97% |
| **XGBoost** | **yes** | **1,196,390** | **1,574,489** | **0.9826** | **5.02%** |

Tuned XGBoost parameters: `n_estimators=300`, `learning_rate=0.1`, `max_depth=3`.

Engineered features: `Area_per_Bedroom`, `Total_Rooms`, `Age_Group`, `Has_Garden_Room_Budget`.

## 8. How to Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` → `{"status": "ok", "model_loaded": true, ...}`
- `POST /predict` →

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"area": 3000, "bedrooms": 3, "bathrooms": 2, "age": 10, "location": "City Center", "property_type": "Villa"}'
```

```json
{"predicted_price": 34829620.0}
```

Interactive docs: http://localhost:8000/docs

## 9. How to Run the Frontend

```bash
# Terminal 1 - API
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2 - Streamlit
streamlit run app/app.py
```

Open http://localhost:8501, enter property details and click **Predict**.

## 10. Running Tests

```bash
pytest tests -v
```

Covers: data loading, missing/duplicates, feature engineering, pipeline fitting, model prediction, API health, valid/invalid predictions.

## 11. Docker

```bash
# Build and start API + frontend
docker compose up --build
```

- API: http://localhost:8000
- Frontend: http://localhost:8501

## 12. Example Prediction

Input: `Area=3000, Bedrooms=3, Bathrooms=2, Age=10, City Center, Villa`

Output: `predicted_price = 34,829,620`

## 13. Logging & Monitoring

- API requests, prediction requests, errors and model loading are logged to `logs/api.log` (rotating) and the console.
- `GET /health` reports `total_predictions` and `total_errors` counters.

## 14. Screenshots

- EDA plots: `outputs/eda/*.png`
- API docs: `http://localhost:8000/docs`

---

**Workflow:** Dataset → Cleaning → EDA → Feature Engineering → Preprocessing Pipeline → Train Multiple Models → Evaluation → Hyperparameter Tuning → Select Best Model → Save Pipeline → FastAPI → Streamlit → Testing → Docker → Documentation
