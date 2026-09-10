# Predictive Forecasting of Care Load & Placement Demand

A Streamlit forecasting dashboard for the HHS Unaccompanied Alien Children (UAC) dataset.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Final model
The Naive forecasting baseline was selected because it achieved the lowest held-out test error:
- MAE: 10.24
- RMSE: 14.05
- MAPE: 0.454%

## Important data note
The source data contains irregular reporting dates. Missing calendar dates were not filled by artificial interpolation.
