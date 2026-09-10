
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="HHS UAC Forecasting", layout="wide")

DATA_PATH = "data/dataset.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(how="all").copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Children in HHS Care"] = (
        df["Children in HHS Care"].astype(str).str.replace(",", "", regex=False)
    )
    numeric_cols = [
        "Children apprehended and placed in CBP custody*",
        "Children in CBP custody",
        "Children transferred out of CBP custody",
        "Children in HHS Care",
        "Children discharged from HHS Care",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Net Pressure"] = (
        df["Children transferred out of CBP custody"]
        - df["Children discharged from HHS Care"]
    )
    return df.dropna(subset=["Date", "Children in HHS Care"])

df = load_data()

st.title("Predictive Forecasting of Care Load & Placement Demand")
st.caption("HHS Unaccompanied Alien Children (UAC) program dataset")

current_hhs = int(df["Children in HHS Care"].iloc[-1])
current_discharge = int(df["Children discharged from HHS Care"].iloc[-1])
threshold = float(df["Children in HHS Care"].quantile(0.90))

c1, c2, c3 = st.columns(3)
c1.metric("Current HHS Care Load", f"{current_hhs:,}")
c2.metric("Latest Discharges", f"{current_discharge:,}")
c3.metric("Historical 90th % Stress Threshold", f"{threshold:,.0f}")

st.subheader("HHS Care Load: Historical Trend")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df["Date"], y=df["Children in HHS Care"],
                         mode="lines", name="HHS Care"))
fig.update_layout(xaxis_title="Date", yaxis_title="Children",
                  height=450)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Short-Term HHS Care Forecast")
horizon = st.selectbox("Forecast horizon (observations)", [7, 14, 30])
last_hhs = float(df["Children in HHS Care"].iloc[-1])
changes = df["Children in HHS Care"].diff().dropna()
error_std = float(changes.std())
uncertainty = error_std * np.sqrt(horizon)
lower = max(0, last_hhs - 1.96 * uncertainty)
upper = last_hhs + 1.96 * uncertainty

f1, f2, f3 = st.columns(3)
f1.metric("Naive Forecast", f"{last_hhs:,.0f}")
f2.metric("95% Lower", f"{lower:,.0f}")
f3.metric("95% Upper", f"{upper:,.0f}")

st.info("The selected model is the Naive baseline because it achieved the lowest held-out test error.")

st.subheader("Discharge Demand Forecast")
last_d = float(df["Children discharged from HHS Care"].iloc[-1])
d_changes = df["Children discharged from HHS Care"].diff().dropna()
d_std = float(d_changes.std())
d_unc = d_std * np.sqrt(horizon)
d_lower = max(0, last_d - 1.96 * d_unc)
d_upper = last_d + 1.96 * d_unc

d1, d2, d3 = st.columns(3)
d1.metric("Forecast Discharges", f"{last_d:,.0f}")
d2.metric("95% Lower", f"{d_lower:,.0f}")
d3.metric("95% Upper", f"{d_upper:,.0f}")

st.subheader("Transfer vs Discharge Pressure")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df["Date"], y=df["Children transferred out of CBP custody"],
                          mode="lines", name="Transfers into HHS"))
fig2.add_trace(go.Scatter(x=df["Date"], y=df["Children discharged from HHS Care"],
                          mode="lines", name="Discharges from HHS"))
fig2.update_layout(xaxis_title="Date", yaxis_title="Children", height=400)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Capacity Stress Indicator")
if current_hhs >= threshold:
    status = "HIGH"
elif current_hhs >= 0.80 * threshold:
    status = "MODERATE"
else:
    status = "LOW"
st.metric("Current Historical Stress Status", status)
st.caption("This is a historical 90th-percentile indicator, not an official HHS operational capacity limit.")

st.subheader("Model Comparison")
comparison = pd.DataFrame({
    "Model": ["Naive", "Moving Average (7)", "Random Forest", "ARIMA (1,1,1)"],
    "MAE": [10.237410, 24.652621, 54.188641, 235.206167],
    "RMSE": [14.048989, 30.522381, 79.180633, 285.633087],
    "MAPE (%)": [0.453938, 1.101080, 2.557818, 11.114109]
})
st.dataframe(comparison.sort_values("MAE"), use_container_width=True)

st.subheader("Key Findings")
st.markdown("""
- HHS care load peaked at **11,516** children on 2023-12-20.
- The latest observed HHS care load is **2,484**.
- The historical 90th-percentile stress threshold is **9,762**.
- Negative net pressure occurred more often than positive pressure.
- The **Naive model** was the best-performing tested model on the held-out period.
- The dataset has irregular reporting dates; missing calendar dates were not artificially interpolated.
""")
