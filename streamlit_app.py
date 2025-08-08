import streamlit as st
from data_loader import load_signup_data
from anomaly_detection import detect_anomalies
from utils import daily_summary, plot_time_series
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Signup Anomaly Detection", layout="wide")

st.title("📊 Signup Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("Upload Signup Log CSV", type=["csv"])

if uploaded_file:
    df = load_signup_data(uploaded_file)
    st.write("Preview of Uploaded Data", df.head())

    threshold = st.slider("Anomaly threshold (signups per IP/day)", 1, 50, 5)

    df_analyzed, model = detect_anomalies(
        df,
        time_col="start_time",
        ip_col="true_client_ip",
        threshold=threshold
    )

    st.subheader("Daily Summary")
    summary_df = daily_summary(df_analyzed, date_col="start_time", ip_col="true_client_ip")
    st.dataframe(summary_df)

    st.subheader("📈 Time Series Chart")
    chart = plot_time_series(summary_df)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Anomalous Records")
    st.dataframe(df_analyzed[df_analyzed["is_anomaly"]])


# ---- Adaptive explanation generator ----
def generate_explanation(df, df_anomalous, ip_col="true_client_ip", duration_col="duration", code_col="response_code"):
    total_logs = len(df)
    total_anomalies = len(df_anomalous)

    explanations = []

    # 1. High request frequency from certain IPs
    if ip_col in df.columns:
        freq_threshold = df[ip_col].value_counts().quantile(0.95)  # top 5%
        anomaly_ips = df_anomalous[ip_col].value_counts()
        high_freq_ips = anomaly_ips[anomaly_ips > freq_threshold]
        if not high_freq_ips.empty:
            explanations.append(f"Some IPs made unusually high numbers of requests (above {freq_threshold:.0f} requests in a day).")

    # 2. Rare HTTP response codes
    if code_col in df.columns:
        common_codes = df[code_col].value_counts(normalize=True)
        rare_codes = common_codes[common_codes < 0.02].index  # less than 2%
        if any(code in rare_codes for code in df_anomalous[code_col]):
            explanations.append("Several anomalies have rare HTTP response codes that occur in less than 2% of requests.")

    # 3. Unusually long durations
    if duration_col in df.columns:
        duration_threshold = df[duration_col].quantile(0.95)
        if any(df_anomalous[duration_col] > duration_threshold):
            explanations.append(f"Some requests had durations above the 95th percentile ({duration_threshold:.2f} seconds).")

    if not explanations:
        explanations.append("The Isolation Forest model flagged these logs as statistical outliers based on request patterns.")

    explanation_text = f"""
    **Anomaly Detection Summary**
    - Total logs analyzed: **{total_logs}**
    - Total anomalies detected: **{total_anomalies}**
    
    **Why were these flagged?**
    {''.join([f'- {reason}\n' for reason in explanations])}
    """
    return explanation_text


# ---- Histogram comparison ----
def plot_comparison(df, df_anomalous, duration_col="duration"):
    st.subheader("Visual Comparison: Normal vs Anomalous Logs (Duration)")
    fig, ax = plt.subplots(figsize=(8, 4))
    if duration_col in df.columns:
        ax.hist(df[duration_col], bins=30, alpha=0.5, label="Normal")
        ax.hist(df_anomalous[duration_col], bins=30, alpha=0.5, label="Anomalous", color="red")
        ax.set_xlabel(duration_col)
        ax.set_ylabel("Count")
        ax.legend()
    st.pyplot(fig)


# ---- Time-series with anomalies highlighted ----
def plot_time_series(df, df_anomalous, time_col="start_time", duration_col="duration"):
    st.subheader("Time-Series View with Anomalies Highlighted")
    fig, ax = plt.subplots(figsize=(10, 4))

    if time_col in df.columns and duration_col in df.columns:
        # Sort by time
        df_sorted = df.sort_values(time_col)
        df_anomalous_sorted = df_anomalous.sort_values(time_col)

        # Plot normal logs
        ax.plot(df_sorted[time_col], df_sorted[duration_col], label="Normal", alpha=0.5)

        # Highlight anomalies
        ax.scatter(df_anomalous_sorted[time_col], df_anomalous_sorted[duration_col],
                   color="red", label="Anomalous", zorder=5)

        ax.set_xlabel("Time")
        ax.set_ylabel(duration_col)
        ax.legend()

    st.pyplot(fig)


# ---- Example usage after loading your data ----
# df = load_signup_data()  # full dataset
# df_anomalous = anomalies_detected_df  # just anomalies

st.markdown(generate_explanation(df, df_analyzed, ip_col="true_client_ip", duration_col="duration", code_col="response_code"))
plot_comparison(df, df_analyzed, duration_col="duration")
plot_time_series(df, df_analyzed, time_col="start_time", duration_col="duration")
