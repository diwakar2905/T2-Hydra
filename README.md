# T2-HYDRO

T2-HYDRO is a modular climate intelligence project that predicts drought risk, produces short-term forecasts, and explains model outputs. It is designed as a scaffold for research and lightweight production demos — combining a Flask backend, Streamlit frontend, and an LSTM-based transfer-learning model with explainability support (SHAP-like).

**Repository layout**

- `backend/` — Flask API and services
	- `app.py` — Flask application factory and `/health` endpoint
	- `wsgi.py` — WSGI entry for `gunicorn`
	- `routes/` — API blueprints: `/predict`, `/forecast`, `/explain`
	- `services/` — prediction, forecasting and explainability logic (synthetic outputs in scaffold)
	- `Dockerfile`, `requirements.txt`
- `frontend/` — Streamlit app and pages
	- `streamlit_app.py` — Streamlit router and app entry
	- `pages/` — `dashboard.py`, `prediction.py`, `forecasting.py`, `explainability.py`, `model_analysis.py`
	- `Dockerfile`, `requirements.txt`
- `T2_Hydro_f.ipynb` — pipeline notebook with data, feature engineering, model training, SHAP, and dashboard plotting
- `render.yaml` — Render service definitions (optional)

High-level architecture
-----------------------

 (User / Frontend) ---> Streamlit (frontend/) ---> HTTP ---> Flask API (backend/) ---> Model services

Key components:
- Frontend: interactive UI to request predictions, forecasts and explanations; displays charts and metrics.
- Backend: exposes three main endpoints:
	- `POST /predict` — accepts `{"features": {...}}`, returns drought probability, severity and confidence.
	- `POST /forecast` — accepts `{"horizon": N, "base": float}` and returns a probability series.
	- `POST /explain` — accepts `{"features": {...}}`, returns local feature contributions.
- Notebook: complete pipeline for data download (NASA POWER fallback), feature engineering (`spi`, NDVI/NDWI proxies), model training (LSTM + fusion), SHAP explainability, and saved dashboard image.

Models & explainability
------------------------
- Model: `T2HydroModel` — spatial encoder + temporal LSTM encoder + attention fusion + sigmoid head producing a drought risk score (0-1). Transfer learning strategy: train on source region (California) then fine-tune on target (India).
- Training metrics: RMSE, MAE, R²; classification metrics (precision/recall/F1) can be computed after thresholding.
