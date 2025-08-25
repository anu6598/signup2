import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="OTP Attack Tracker (Web)", layout="wide")

st.title("🔐 OTP Attack Tracking & Anomaly Detection — Web")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload OTP Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Convert date column to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Show raw data
    st.subheader("📊 Raw Data")
    st.dataframe(df.head())

    # OTP per user distribution
    st.subheader("📈 OTP Requests per User")
    fig = px.histogram(df, x="otp_count", nbins=30, title="Distribution of OTP Requests")
    st.plotly_chart(fig, use_container_width=True)

    # Time series: OTP requests over time
    st.subheader("📅 OTP Requests Over Time")
    time_series = df.groupby("date")["otp_count"].sum().reset_index()
    fig = px.line(time_series, x="date", y="otp_count", title="OTP Requests Trend")
    st.plotly_chart(fig, use_container_width=True)

    # Threshold check
    st.subheader("🚨 Threshold Breach Detection")
    threshold = st.number_input("Set OTP Threshold (per 3 mins)", min_value=1, value=2)
    suspicious = df[df["otp_count"] > threshold]

    st.write(f"Found **{len(suspicious)} suspicious records** above threshold.")
    st.dataframe(suspicious)

    # User-level aggregation
    st.subheader("👤 Suspicious Users Summary")
    user_summary = df.groupby("user_count")["otp_count"].sum().reset_index()
    fig = px.bar(user_summary, x="user_count", y="otp_count",
                 title="Total OTP Requests per User")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Upload a CSV file to begin analysis.")
