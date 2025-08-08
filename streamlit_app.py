import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_signup_data
from anomaly_detection import detect_anomalies
from utils import daily_summary, plot_time_series

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Signup Anomaly Detection", layout="wide")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        🚨 Signup Anomaly Detection Dashboard
    </h1>
    <p style='text-align: center; font-size: 16px; color: gray;'>
        Detect unusual signup patterns and investigate suspicious activity.
    </p>
    """,
    unsafe_allow_html=True
)

# ---------------- TOP ROW ----------------
col1, col2 = st.columns([3, 1])
with col2:
    with st.expander("ℹ️ How Anomaly Detection Works", expanded=False):
        st.markdown("""
        This dashboard detects unusual signup patterns using:

        **Rule-based checks:**
        - IPs with >10 signups in 15 minutes.
        - IPs with unusually high requests/day.
        - Rare HTTP response codes (< 2% frequency).
        - Long request durations (> 95th percentile).

        **ML model:**
        - Isolation Forest detects statistical outliers in traffic patterns.
        """)

# ---------------- FILE UPLOAD ----------------
with col1:
    uploaded_file = st.file_uploader("📂 Upload Signup Log CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = load_signup_data(uploaded_file)
    st.write("**📋 Preview of Uploaded Data:**", df.head())

    # ---------------- ANOMALY DETECTION ----------------
    threshold = st.slider("🔍 Anomaly threshold (signups per IP/day)", 1, 50, 5)

    df_analyzed, model = detect_anomalies(
        df,
        time_col="start_time",
        ip_col="true_client_ip",
        threshold=threshold
    )

    # ---------------- ADDITIONAL RULE: >10 SIGNUPS IN 15 MIN ----------------
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["window_15min"] = df.groupby("true_client_ip")["start_time"].transform(
        lambda x: x.rolling("15min", on=x).count()
    )
    high_15min_ips = df[df["window_15min"] > 10]["true_client_ip"].unique()

    # ---------------- DAILY SUMMARY ----------------
    st.markdown("---")
    st.subheader("📆 Daily Summary")
    summary_df = daily_summary(df_analyzed, date_col="start_time", ip_col="true_client_ip")
    st.dataframe(summary_df)

    # Chart
    st.subheader("📈 Signups per Day")
    st.altair_chart(plot_time_series(summary_df), use_container_width=True)

    # ---------------- ANOMALY EXPLANATION ----------------
    def generate_explanation(df, df_anomalous, ip_col="true_client_ip", duration_col="duration", code_col="response_code"):
        total_logs = len(df)
        total_anomalies = len(df_anomalous)

        explanations = []

        # Rule 1: High request frequency
        freq_threshold = df[ip_col].value_counts().quantile(0.95)
        high_freq_ips = df_anomalous[ip_col].value_counts()
        if any(high_freq_ips > freq_threshold):
            explanations.append(f"Some IPs made unusually high numbers of requests (above {freq_threshold:.0f} in a day).")

        # Rule 2: Rare HTTP codes
        common_codes = df[code_col].value_counts(normalize=True)
        rare_codes = common_codes[common_codes < 0.02].index
        if any(code in rare_codes for code in df_anomalous[code_col]):
            explanations.append("Several anomalies have rare HTTP response codes (< 2% frequency).")

        # Rule 3: Long durations
        duration_threshold = df[duration_col].quantile(0.95)
        if any(df_anomalous[duration_col] > duration_threshold):
            explanations.append(f"Some requests had durations above the 95th percentile ({duration_threshold:.2f} seconds).")

        # Rule 4: >10 signups in 15 mins
        if len(high_15min_ips) > 0:
            explanations.append("Some IPs had more than 10 signups in a 15-minute window.")

        if not explanations:
            explanations.append("The Isolation Forest model flagged these logs as statistical outliers.")

        return f"""
        **Anomaly Detection Summary**
        - Total logs analyzed: **{total_logs}**
        - Total anomalies detected: **{total_anomalies}**

        **Why were these flagged?**
        {''.join([f'- {reason}\n' for reason in explanations])}
        """

    st.markdown(generate_explanation(
        df,
        df_analyzed[df_analyzed["is_anomaly"]],
        ip_col="true_client_ip",
        duration_col="duration",
        code_col="response_code"
    ))

    # ---------------- VISUALS ----------------
    # Histogram
    st.markdown("---")
    st.subheader("📊 Duration Distribution: Normal vs Anomalous")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["duration"], bins=30, alpha=0.5, label="Normal")
    ax.hist(df_analyzed[df_analyzed["is_anomaly"]]["duration"], bins=30, alpha=0.5, label="Anomalous", color="red")
    ax.set_xlabel("Duration")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    # Time Series
    st.subheader("🕒 Time-Series View with Anomalies Highlighted")
    fig, ax = plt.subplots(figsize=(10, 4))
    df_sorted = df.sort_values("start_time")
    df_anomalous_sorted = df_analyzed[df_analyzed["is_anomaly"]].sort_values("start_time")
    ax.plot(df_sorted["start_time"], df_sorted["duration"], label="Normal", alpha=0.5)
    ax.scatter(df_anomalous_sorted["start_time"], df_anomalous_sorted["duration"], color="red", label="Anomalous", zorder=5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Duration")
    ax.legend()
    st.pyplot(fig)

    # ---------------- TABLE: IPs >5 SIGNUPS IN 10 MINS ----------------
    st.markdown("---")
    st.subheader("📋 IPs with >5 Signups in 10 Minutes")
    df["window_10min"] = df.groupby("true_client_ip")["start_time"].transform(
        lambda x: x.rolling("10min", on=x).count()
    )
    table_10min = df[df["window_10min"] > 5][["true_client_ip", "user_agent", "start_time"]]
    st.dataframe(table_10min.drop_duplicates())

    # ---------------- TABLE: Anomalous IP + User Agent ----------------
    st.subheader("🚩 Anomalous IPs and User Agents")
    anomaly_table = df_analyzed[df_analyzed["is_anomaly"]][["true_client_ip", "user_agent", "start_time"]]
    st.dataframe(anomaly_table.drop_duplicates())
