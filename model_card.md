# Model Card — Car Price Analysis (Demo)

Model: RandomForestRegressor pipeline (preprocessing + model)

Purpose
- Provide a compact, reproducible demonstration model for interactive exploration and recruiter-facing demos.

Data
- Source: `car_price_prediction_.csv` (repository root).
- Size: varies with loaded rows; demo trains on a reproducible sample (default 25%).

Metrics
- Evaluation metrics demonstrated in `car_price_prediction.ipynb` (RMSE, MAE, R²). The demo model is intended for interactivity and not final production performance.

Limitations
- Trained on a sample for speed; not production-calibrated.
- Sensitive to dataset schema: requires a numeric target column named `price` and at least one numeric feature.

Usage
- Demo app: `streamlit run app.py`
- Programmatic training: `python scripts/train.py --data car_price_prediction_.csv --out models/demo_model.joblib`

Contact
- For questions: `abhirajgautam28@gmail.com`
