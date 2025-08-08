import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_loader import load_signup_data
from anomaly_detection import detect_anomalies
from utils import daily_summary

st.set_page_config(page_title="Signup Anomaly Detector", layout="wide")

st.title("Signup Anomaly Detector")

uploaded_file = st.file_uploader("Upload signup log CSV", type=["csv"])

if uploaded_file:
    df = load_signup_data(uploaded_file)
    st.write("Preview of Uploaded Data", df.head())

    threshold = st.slider("Anomaly threshold (signups per IP/day)", 1, 50, 5)

    # ✅ Updated call with correct timestamp & IP column names
    df_analyzed = detect_anomalies(
    df,
    time_col="timestamp",
    ip_col="true_client_ip",
    threshold=threshold
)

summary_df = daily_summary(df_analyzed, date_col="timestamp", ip_col="true_client_ip")

    

    st.subheader("Daily Summary")
    summary_df = daily_summary(df_analyzed)
    st.dataframe(summary_df)

    st.subheader("Signup Trend Over Time")
    fig, ax = plt.subplots()
    ax.plot(summary_df["date"], summary_df["total_signups"], label="Total Signups", marker="o")
    ax.plot(summary_df["date"], summary_df["anomalies"], label="Anomalies", marker="x", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Detailed Anomalies")
    anomalies_df = df_analyzed[df_analyzed["is_anomaly"]]
    st.dataframe(anomalies_df)
