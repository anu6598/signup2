import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import matplotlib.pyplot as plt

# -----------------------
# Streamlit UI setup
# -----------------------
st.set_page_config(page_title="🚨 Attack Prediction Dashboard", layout="wide")
st.title("🚨 OTP/Brute-force Attack Detection & Forecasting")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Logs (CSV)", type=["csv"])

# -----------------------
# Data Loading
# -----------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure datetime column
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    elif 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        st.error("❌ No timestamp/date column found in dataset.")
        st.stop()

    # Sort by time
    df = df.sort_values('timestamp')

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -----------------------
    # Feature Engineering
    # -----------------------
    st.subheader("Feature Engineering")
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day'] = df['timestamp'].dt.date

    # Request frequency per minute
    freq = df.set_index('timestamp').resample('1min').size().reset_index(name='requests')

    st.line_chart(freq.set_index('timestamp'))

    # -----------------------
    # Anomaly Detection (Isolation Forest)
    # -----------------------
    st.subheader("🔍 Anomaly Detection")
    X = freq[['requests']]

    iso = IsolationForest(contamination=0.02, random_state=42)
    freq['anomaly'] = iso.fit_predict(X)

    anomalies = freq[freq['anomaly'] == -1]

    st.write(f"Detected {len(anomalies)} anomalies (suspicious spikes). Example:")
    st.dataframe(anomalies.head())

    # -----------------------
    # Forecasting Next Attack (Prophet)
    # -----------------------
    st.subheader("📈 Forecast Next Attack Time")

    df_prophet = freq[['timestamp','requests']].rename(columns={'timestamp':'ds','requests':'y'})

    model = Prophet(interval_width=0.95, daily_seasonality=True)
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=60, freq='min')  # Forecast next 60 minutes
    forecast = model.predict(future)

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10,5))
    model.plot(forecast, ax=ax)
    st.pyplot(fig)

    # Highlight next potential attack
    next_spike = forecast[forecast['yhat'] > forecast['yhat'].quantile(0.95)].head(1)
    if not next_spike.empty:
        st.success(f"⚠️ Next suspicious activity window likely around: **{next_spike['ds'].values[0]}**")
    else:
        st.info("No major spikes predicted in the next 60 minutes.")

else:
    st.info("⬅️ Upload a CSV log file to begin analysis.")
