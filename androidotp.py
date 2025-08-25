import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="OTP Tracking System", layout="wide")
st.title("🔐 OTP Tracking & Attack Detection Dashboard")

# Upload CSV
uploaded_file = st.file_uploader("Upload OTP logs CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head())

    # Ensure datetime
    if "date" in df.columns and "start_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["start_time"], errors="coerce")
    elif "start_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["start_time"], errors="coerce")
    else:
        st.error("No valid datetime columns found")
        st.stop()

    df = df.dropna(subset=["timestamp"])
    df["minute_bucket"] = df["timestamp"].dt.floor("T")

    # Focus only on OTP requests
    otp_df = df[df["request_path"].str.contains("otp", case=False, na=False)]

    # --- Replicating SQL Logic ---
    grouped = otp_df.groupby(["true_client_ip", "minute_bucket"]).agg(
        signup_attempts=("request_path", "count"),
        akamai_epd=("akamai_epd", "max")
    ).reset_index()

    # Rolling 10-min window per IP
    grouped = grouped.sort_values(["true_client_ip", "minute_bucket"])
    grouped["attempts_in_10_min"] = grouped.groupby("true_client_ip")["signup_attempts"].transform(
        lambda x: x.rolling("10min", on=grouped["minute_bucket"]).sum()
    )

    grouped["is_proxy_ip"] = grouped["akamai_epd"].notnull().astype(int)

    attack_candidates = grouped[grouped["attempts_in_10_min"] > 10]

    st.subheader("🚨 Attack Candidates (10+ OTPs in 10 min)")
    st.dataframe(attack_candidates)

    # --- IP Age ---
    ip_age = otp_df.groupby("true_client_ip")["date"].nunique().reset_index(name="unique_days_seen")
    st.subheader("📅 IP Age (unique days seen)")
    st.dataframe(ip_age)

    # --- Proxy Spikes ---
    proxy_counts = otp_df.groupby(["true_client_ip", "date"]).akamai_epd.apply(lambda x: x.notnull().sum()).reset_index(name="proxy_hits")
    sudden_spikes = proxy_counts[proxy_counts["proxy_hits"] > proxy_counts["proxy_hits"].mean() + 3*proxy_counts["proxy_hits"].std()]
    st.subheader("⚡ Sudden Proxy Spikes")
    st.dataframe(sudden_spikes)

    # --- Requests/minute, 10min, 1h, 1d ---
    time_windows = {
        "1min": "1T",
        "10min": "10T",
        "1h": "1H",
        "1d": "1D"
    }

    st.subheader("📊 Requests per Time Window")
    for label, window in time_windows.items():
        counts = otp_df.set_index("timestamp").groupby("true_client_ip").resample(window).size().reset_index(name="otp_count")
        st.write(f"**{label} window**")
        st.dataframe(counts.head())

    # --- Device ID Suspicion ---
    if "dr_dv" in df.columns:
        device_counts = otp_df.groupby(["dr_dv", "minute_bucket"]).size().reset_index(name="otp_count")
        suspicious_devices = device_counts[device_counts["otp_count"] > 10]
        st.subheader("📱 Suspicious Device IDs (10+ OTPs/min)")
        st.dataframe(suspicious_devices)

    # --- BMP Score Rule ---
    if "bmp_score" in df.columns:
        bmp_flags = df[df["bmp_score"] > 90].groupby(["true_client_ip", "date"]).size().reset_index(name="high_score_count")
        bmp_flags = bmp_flags[bmp_flags["high_score_count"] >= 5]
        st.subheader("🏴 BMP Score Violations")
        st.dataframe(bmp_flags)

    # --- IP Repetition Benchmark ---
    ip_reps = otp_df.groupby(["date", "true_client_ip"]).size().reset_index(name="daily_requests")
    threshold = ip_reps["daily_requests"].mean() + 3*ip_reps["daily_requests"].std()
    abnormal_ips = ip_reps[ip_reps["daily_requests"] > threshold]
    st.subheader("📈 Abnormal IP Repetitions")
    st.dataframe(abnormal_ips)

    # --- Avg OTPs requested in 1, 5, 10 mins ---
    avg_windows = {
        "1min": "1T",
        "5min": "5T",
        "10min": "10T"
    }

    st.subheader("📐 Average OTPs per Window")
    for label, window in avg_windows.items():
        avg_otps = otp_df.set_index("timestamp").resample(window).size().mean()
        st.write(f"**{label}:** {avg_otps:.2f} OTPs on average")

    # --- Visualization ---
    st.subheader("📉 OTP Request Timeline")
    otp_counts = otp_df.set_index("timestamp").resample("1T").size().reset_index(name="otp_count")
    fig = px.line(otp_counts, x="timestamp", y="otp_count", title="OTP Requests per Minute")
    st.plotly_chart(fig, use_container_width=True)
