import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta

st.set_page_config(page_title="Android OTP Dashboard", layout="wide")
st.title("📱 Android OTP Dashboard")

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Upload OTP CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure datetime if exists
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        except:
            pass

    st.subheader("Daily Stats Summary")
    # -------------------------
    # Daily Stats (Example)
    # -------------------------
    if "date" in df.columns:
        daily_summary = df.groupby("date").agg(
            total_otps=("request_id", "count"),
            unique_ips=("true_client_ip", "nunique"),
            unique_devices=("dr_dv", "nunique"),
        ).reset_index()
        st.dataframe(daily_summary)

    # -------------------------
    # Category detection logic
    # -------------------------
    st.subheader("Category Detection (1-Day Analysis)")

    if "date" in df.columns:
        # pick the first date from dataset
        day = df["date"].iloc[0]
        df_day = df[df["date"] == day]

        # metrics
        total_otps = len(df_day)
        ip_counts = df_day["true_client_ip"].value_counts()
        device_counts = df_day["dr_dv"].value_counts()

        high_ip = any(ip_counts > 25)
        high_device = any(device_counts > 15)
        high_total = total_otps > 1000

        # proxy condition (dummy, since no proxy column in dataset — replace if available)
        if "proxy_status" in df_day.columns:
            proxy_rate = (df_day["proxy_status"] == "proxy").mean()
        else:
            proxy_rate = 0  # default to 0%

        high_proxy = proxy_rate > 0.7

        # determine categories
        categories = []
        if high_total or high_ip or high_device or high_proxy:
            if high_total or high_ip or high_device:
                categories.append("OTP Abuse/Attack detected")
                categories.append("HIGH OTP request detected")
            if high_proxy:
                categories.append("HIGH proxy status detected")
        else:
            categories.append("No suspicious activity detected")

        result_df = pd.DataFrame({
            "date": [day],
            "category": [", ".join(categories)]
        })

        st.table(result_df)

    # -------------------------
    # Raw data preview
    # -------------------------
    st.subheader("Raw data preview (first 10 rows)")
    st.dataframe(df.head(10))

else:
    st.info("Please upload a CSV file to get started.")
