import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_signup_data
from anomaly_detection import detect_anomalies
from utils import daily_summary, plot_time_series  # daily summary chart from utils

st.set_page_config(page_title="Signup Anomaly Detection", layout="wide")
st.title("📊 Signup Anomaly Detection Dashboard")

# ---------------- HOW IT WORKS INFO BOX ----------------
with st.expander("ℹ️ How Anomaly Detection Works"):
    st.markdown("""
    This dashboard detects unusual signup patterns using **two methods**:

    1. **Rule-based checks**:
       - Flags IPs with unusually high requests in a single day.
       - Flags rare HTTP response codes (appear in < 2% of logs).
       - Flags unusually long request durations (above 95th percentile).

    2. **Isolation Forest model**:
       - A machine learning model that learns normal traffic patterns.
       - Flags requests that are statistically unusual (outliers).

    Together, these methods help catch:
    - Bots or automated scripts.
    - Unusual client behavior.
    - Potential attacks or abuse of the signup system.
    """)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Signup Log CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = load_signup_data(uploaded_file)
    st.write("Preview of Uploaded Data", df.head())

    # ---------------- ANOMALY DETECTION ----------------
    threshold = st.slider("Anomaly threshold (signups per IP/day)", 1, 50, 5)

    df_analyzed, model = detect_anomalies(
        df,
        time_col="start_time",       # using start_time instead of timestamp
        ip_col="true_client_ip",
        threshold=threshold
    )

    # ---------------- DAILY SUMMARY ----------------
    st.subheader("Daily Summary")
    summary_df = daily_summary(df_analyzed, date_col="start_time", ip_col="true_client_ip")
    st.dataframe(summary_df)

    # Daily summary chart (Altair)
    st.subheader("📈 Signups per Day")
    st.altair_chart(plot_time_series(summary_df), use_container_width=True)

    # ---------------- ANOMALY EXPLANATION ----------------
    def generate_explanation(df, df_anomalous, ip_col="true_client_ip", duration_col="duration", code_col="response_code"):
        total_logs = len(df)
        total_anomalies = len(df_anomalous)

        explanations = []

        # 1. High request frequency
        if ip_col in df.columns:
            freq_threshold = df[ip_col].value_counts().quantile(0.95)  # top 5%
            anomaly_ips = df_anomalous[ip_col].value_counts()
            high_freq_ips = anomaly_ips[anomaly_ips > freq_threshold]
            if not high_freq_ips.empty:
                explanations.append(f"Some IPs made unusually high numbers of requests (above {freq_threshold:.0f} in a day).")

        # 2. Rare HTTP response codes
        if code_col in df.columns:
            common_codes = df[code_col].value_counts(normalize=True)
            rare_codes = common_codes[common_codes < 0.02].index
            if any(code in rare_codes for code in df_anomalous[code_col]):
                explanations.append("Several anomalies have rare HTTP response codes (< 2% frequency).")

        # 3. Long durations
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

    st.subheader("📝 Explanation of Detected Anomalies")
    st.markdown(generate_explanation(
        df,
        df_analyzed[df_analyzed["is_anomaly"]],
        ip_col="true_client_ip",
        duration_col="duration",
        code_col="response_code"
    ))

    # ---------------- HISTOGRAM COMPARISON ----------------
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

    plot_comparison(
        df,
        df_analyzed[df_analyzed["is_anomaly"]],
        duration_col="duration"
    )

    # ---------------- TIME-SERIES WITH ANOMALIES ----------------
    def plot_anomaly_timeseries(df, df_anomalous, time_col="start_time", duration_col="duration"):
        st.subheader("Time-Series View with Anomalies Highlighted")
        fig, ax = plt.subplots(figsize=(10, 4))

        if time_col in df.columns and duration_col in df.columns:
            df_sorted = df.sort_values(time_col)
            df_anomalous_sorted = df_anomalous.sort_values(time_col)

            ax.plot(df_sorted[time_col], df_sorted[duration_col], label="Normal", alpha=0.5)
            ax.scatter(df_anomalous_sorted[time_col], df_anomalous_sorted[duration_col],
                       color="red", label="Anomalous", zorder=5)

            ax.set_xlabel("Time")
            ax.set_ylabel(duration_col)
            ax.legend()

        st.pyplot(fig)

    plot_anomaly_timeseries(
        df,
        df_analyzed[df_analyzed["is_anomaly"]],
        time_col="start_time",
        duration_col="duration"
    )
