# T2-HYDRO

T2-HYDRO is a production-style AI climate intelligence application for drought prediction, severity forecasting, and explainable climate analytics. It combines a Flask REST backend, Streamlit analytics frontend, PyTorch sequence models, sklearn and XGBoost baselines, Plotly dashboards, and SHAP explainability.

## Architecture

```text
NASA POWER API -> preprocessing -> feature engineering -> ML services -> Flask REST API
                                                                   |
                                                                   v
                                                    Streamlit analytics frontend
```

## Features

- Predicts drought probability, severity class, severity score, and confidence.
- Forecasts future drought trend and climate risk over configurable horizons.
- Engineers temporal, rolling, anomaly, and climate stress features.
- Includes sklearn, XGBoost, PyTorch LSTM, and transfer LSTM model modules.
- Provides SHAP-style local feature explanations for prediction drivers.
- Ships with Plotly dashboards for risk gauges, trend lines, heatmaps, model comparison, and confusion matrix views.

## Project Structure

```text
T2-HYDRO/
├── backend/                 # Flask API, services, models, utilities
├── frontend/                # Streamlit app, pages, reusable components
├── data/                    # Raw and processed climate data
├── notebooks/               # Experiments and model training notebooks
├── outputs/                 # Generated plots, metrics, reports
├── tests/                   # API smoke tests
├── README.md
├── requirements.txt
└── .gitignore
```

## API Documentation

### `GET /health`

Returns service status.

```json
{
  "status": "ok",
  "service": "T2-HYDRO",
  "environment": "development"
}
```

### `POST /predict`

Request:

```json
{
  "rainfall": 2.8,
  "temperature": 33.5,
  "humidity": 42,
  "wind_speed": 3.1,
  "solar_radiation": 21.5,
  "pressure": 100.7
}
```

Response includes drought probability, severity class, severity score, confidence, and top risk factors.

### `POST /forecast`

Request:

```json
{
  "rainfall": 2.8,
  "temperature": 33.5,
  "humidity": 42,
  "wind_speed": 3.1,
  "solar_radiation": 21.5,
  "pressure": 100.7,
  "horizon_days": 45
}
```

Response includes a daily forecast series, predicted climate risk, and trend label.

### `POST /explain`

Returns SHAP-style feature contributions and a local explanation narrative.

## Data Pipeline

The backend utility layer includes NASA POWER ingestion for:

- Rainfall: `PRECTOTCORR`
- Temperature: `T2M`
- Humidity: `RH2M`
- Wind speed: `WS2M`
- Solar radiation: `ALLSKY_SFC_SW_DWN`
- Pressure: `PS`

The preprocessing pipeline handles schema normalization, missing value markers, interpolation, median imputation, scaling, temporal encodings, rolling averages, moving trends, anomaly features, drought severity scoring, and drought class creation.

## Model Layer

Implemented model families:

- Linear Regression
- Random Forest
- XGBoost
- PyTorch LSTM
- Transfer-learning LSTM

Evaluation helpers support RMSE, MAE, R2, accuracy, precision, recall, and F1-score. The Streamlit model page includes production-style comparison and confusion matrix views.

## Local Setup

```bash
cd T2-HYDRO
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the backend:

```bash
cd backend
python app.py
```

Run the frontend in another terminal:

```bash
cd frontend
streamlit run streamlit_app.py
```

Set `T2_HYDRO_API_URL` in Streamlit Community Cloud or your local shell when the backend is not running on `http://localhost:5000`.

## Deployment

### Render Backend

1. Create a Render Web Service.
2. Set root directory to `backend`.
3. Use build command `pip install -r requirements.txt`.
4. Use start command `gunicorn app:app`.
5. Configure environment variables:
   - `APP_ENV=production`
   - `LOG_LEVEL=INFO`
   - `CORS_ORIGIN=<your Streamlit app URL>`
   - `MODEL_DIR=saved_models`

### Streamlit Community Cloud

1. Set app path to `frontend/streamlit_app.py`.
2. Set requirements file to `frontend/requirements.txt`.
3. Add secret or environment variable:
   - `T2_HYDRO_API_URL=https://<your-render-service>.onrender.com`

## Screenshots

Add screenshots after first deployment:

- `outputs/home_dashboard.png`
- `outputs/prediction_page.png`
- `outputs/forecasting_page.png`
- `outputs/explainability_page.png`

## Testing

```bash
pytest
```

## Future Improvements

- Persist NASA POWER datasets in a scheduled ETL job.
- Add MLflow experiment tracking and model registry integration.
- Train calibrated probabilistic classifiers on region-specific drought labels.
- Add authentication and per-region saved dashboards.
- Add CI/CD workflows for tests, linting, and deployment.
