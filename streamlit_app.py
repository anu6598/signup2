import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import load_signup_data
from anomaly_detection import detect_anomalies
from utils import daily_summary, plot_time_series

st.set_page_config(page_title="Signup Anomaly Detection", layout="wide")

# --------- HEADER ---------
st.markdown(
    "<h1 style='text-align:center;'>📊 Signup Anomaly Detection Dashboard</h1>",
    unsafe_allow_html=True
)

# --------- EXPLANATION BOX (STATIC) ---------
st.markdown(
    """
    <div style="background-color:#d9fdd3; padding:15px; border-radius:10px;">
    <h3>🔍 How Anomaly Detection Works</h3>
    <p>
    This dashboard uses a combination of <b>Machine Learning</b> and <b>rule-based logic</b> to spot unusual signup patterns:
    </p>
    <ul>
        <li><b>Isolation Forest (ML):</b> Finds patterns that look statistically different from the rest.</li>
        <li><b>Rules:</b> Flags IPs with more than <b>9 signups in 15 minutes</b> or <b>5 signups in 10 minutes</b>.</li>
        <li>Also checks for rare HTTP codes and unusually long request durations.</li>
    </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# --------- FILE UPLOAD ---------
uploaded_file = st.file_uploader("📂 Upload Signup Log CSV", type=["csv"])

if uploaded_file:
    # Load data
    df = load_signup_data(uploaded_file)
    st.write("### Preview of Uploaded Data")
    st.dataframe(df.head())

    # --------- ANOMALY DETECTION (ML) ---------
    threshold = st.slider("ML Anomaly Threshold (signups per IP/day)", 1, 50, 5)

    df_analyzed, model = detect_anomalies(
        df,
        time_col="start_time",
        ip_col="true_client_ip",
        threshold=threshold
    )

    # --------- RULE-BASED DETECTION ---------
    df = df.sort_values(["true_client_ip", "start_time"])
    
    # Count signups per IP in rolling 15-min window
    df["count_15min"] = (
        df.groupby("true_client_ip")["start_time"]
          .transform(lambda x: x.rolling("15min").count())
    )
    rule_anomalies_15 = df[df["count_15min"] > 9]

    # Count signups per IP in rolling 10-min window
    df["count_10min"] = (
        df.groupby("true_client_ip")["start_time"]
          .transform(lambda x: x.rolling("10min").count())
    )
    table_10min = df[df["count_10min"] > 5]

    # --------- DAILY SUMMARY ---------
    st.subheader("📅 Daily Summary")
    summary_df = daily_summary(df_analyzed, date_col="start_time", ip_col="true_client_ip")
    st.dataframe(summary_df)

    st.subheader("📈 Signups per Day")
    st.altair_chart(plot_time_series(summary_df), use_container_width=True)

    # --------- ANOMALY EXPLANATION ---------
    def generate_explanation(df, df_anomalous):
        total_logs = len(df)
        total_anomalies = len(df_anomalous)

        explanations = []
        freq_threshold = df["true_client_ip"].value_counts().quantile(0.95)
        high_freq_ips = df_anomalous["true_client_ip"].value_counts()
        if not high_freq_ips.empty:
            explanations.append(f"Some IPs made unusually high numbers of requests (above {freq_threshold:.0f} in a day).")

        if "response_code" in df.columns:
            rare_codes = df["response_code"].value_counts(normalize=True)
            rare_codes = rare_codes[rare_codes < 0.02].index
            if any(code in rare_codes for code in df_anomalous["response_code"]):
                explanations.append("Several anomalies have rare HTTP response codes (< 2% frequency).")

        if "duration" in df.columns:
            duration_threshold = df["duration"].quantile(0.95)
            if any(df_anomalous["duration"] > duration_threshold):
                explanations.append(f"Some requests had durations above the 95th percentile ({duration_threshold:.2f} seconds).")

        if not explanations:
            explanations.append("The Isolation Forest model flagged these logs as statistical outliers.")

        return f"""
        **Anomaly Detection Summary**
        - Total logs analyzed: **{total_logs}**
        - Total anomalies detected: **{total_anomalies}**

        **Why were these flagged?**
        {''.join([f'- {reason}\n' for reason in explanations])}
        """

    st.subheader("📝 ML-Based Anomalies Explanation")
    st.markdown(generate_explanation(
        df,
        df_analyzed[df_analyzed["is_anomaly"]]
    ))

    # --------- TABLE: RULE-BASED ANOMALIES (15-min > 9) ---------
    st.subheader("🚨 Rule-Based Anomalies (15-min > 9 signups)")
    st.dataframe(rule_anomalies_15[["start_time", "true_client_ip", "user_agent"]])

    # --------- TABLE: IPs with 5+ signups in 10 minutes ---------
    st.subheader("📋 IPs with 5+ Signups in 10 Minutes")
    st.dataframe(table_10min[["start_time", "true_client_ip", "user_agent"]])

    # --------- HISTOGRAM COMPARISON ---------
    st.subheader("Histogram: Duration Comparison")
    fig, ax = plt.subplots(figsize=(8, 4))
    if "duration" in df.columns:
        ax.hist(df["duration"], bins=30, alpha=0.5, label="All")
        ax.hist(df_analyzed[df_analyzed["is_anomaly"]]["duration"], bins=30, alpha=0.5, label="Anomalous", color="red")
        ax.set_xlabel("Duration")
        ax.set_ylabel("Count")
        ax.legend()
    st.pyplot(fig)

    # --------- TIME-SERIES ANOMALIES ---------
    st.subheader("Time-Series with Anomalies Highlighted")
    fig, ax = plt.subplots(figsize=(10, 4))
    if "duration" in df.columns:
        df_sorted = df.sort_values("start_time")
        anomalies_sorted = df_analyzed[df_analyzed["is_anomaly"]].sort_values("start_time")
        ax.plot(df_sorted["start_time"], df_sorted["duration"], alpha=0.5, label="Normal")
        ax.scatter(anomalies_sorted["start_time"], anomalies_sorted["duration"], color="red", label="Anomaly", zorder=5)
        ax.legend()
    st.pyplot(fig)
