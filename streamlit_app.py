import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_signup_data
from anomaly_detection import detect_anomalies
from utils import daily_summary, plot_time_series  # daily summary chart from utils

# ---------------- STREAMLIT PAGE CONFIG ----------------
st.set_page_config(page_title="Signup Anomaly Detection Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1E3D59;'>🚨 Signup Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- EXPLANATION BOX (STATIC) ----------------
with st.container():
    col_explain, col_empty = st.columns([1.2, 2])
    with col_explain:
        st.markdown(
            """
            <div style='background-color: #d9fdd3; padding: 15px; border-radius: 8px;'>
            <h3>📖 How Anomaly Detection Works</h3>
            <p>We use a hybrid approach combining:</p>
            <ul>
                <li><b>Isolation Forest Model</b>: Flags statistical outliers in signup activity patterns.</li>
                <li><b>Rule-Based Logic</b>: Detects IPs with unusually high signups in short time frames.</li>
                <ul>
                    <li>IPs with more than <b>10 signups in 15 minutes</b> are flagged.</li>
                    <li>IPs with more than <b>5 signups in 10 minutes</b> are highlighted in the table below.</li>
                </ul>
            </ul>
            <p>Detected anomalies are then displayed with IP and User Agent for cross-verification.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📂 Upload Signup Log CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = load_signup_data(uploaded_file)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df = df.sort_values(["true_client_ip", "start_time"])
    st.write("Preview of Uploaded Data", df.head())

    # ---------------- ANOMALY DETECTION ----------------
    threshold = st.slider("Anomaly threshold (signups per IP/day)", 1, 50, 5)

    df_analyzed, model = detect_anomalies(
        df,
        time_col="start_time",       # using start_time instead of timestamp
        ip_col="true_client_ip",
        threshold=threshold
    )

    # ---------------- RULE-BASED LOGIC ----------------
    # >10 signups in 15 minutes
    df["window_15min"] = (
        df.groupby("true_client_ip")
        .rolling("15min", on="start_time")
        .true_client_ip.transform("count")
    )
    high_15min_ips = df[df["window_15min"] > 10]["true_client_ip"].unique()

    # >5 signups in 10 minutes
    df["window_10min"] = (
        df.groupby("true_client_ip")
        .rolling("10min", on="start_time")
        .true_client_ip.transform("count")
    )
    table_10min = df[df["window_10min"] > 5][["true_client_ip", "user_agent", "start_time"]]

    # ---------------- DAILY SUMMARY ----------------
    st.subheader("📆 Daily Summary")
    summary_df = daily_summary(df_analyzed, date_col="start_time", ip_col="true_client_ip")
    st.dataframe(summary_df)

    # Daily summary chart
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

    st.subheader("📝 Anomaly Explanation")
    st.markdown(generate_explanation(
        df,
        df_analyzed[df_analyzed["is_anomaly"]],
        ip_col="true_client_ip",
        duration_col="duration",
        code_col="response_code"
    ))

    # ---------------- VISUAL COMPARISON ----------------
    def plot_comparison(df, df_anomalous, duration_col="duration"):
        st.subheader("📊 Normal vs Anomalous Logs (Duration)")
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

    # ---------------- TIME SERIES WITH ANOMALIES ----------------
    def plot_anomaly_timeseries(df, df_anomalous, time_col="start_time", duration_col="duration"):
        st.subheader("⏳ Time-Series with Anomalies Highlighted")
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

    # ---------------- RULE-BASED ANOMALY TABLE ----------------
    st.subheader("📋 IPs with >5 Signups in 10 Minutes (Rule-Based)")
    st.dataframe(table_10min)

    # ---------------- CROSS-VERIFICATION: IP + USER AGENT ----------------
    st.subheader("🔍 Cross-Verification: Anomalous IPs and User Agents")
    anomaly_cross = df_analyzed[df_analyzed["is_anomaly"]][["true_client_ip", "user_agent", "start_time"]]
    st.dataframe(anomaly_cross)

