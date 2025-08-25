import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="OTP Tracking & Anomaly Detection", layout="wide")
st.title("🔐 OTP Tracking & Anomaly Detection Dashboard")

# Upload file
uploaded_file = st.file_uploader("Upload OTP logs CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📂 Raw Data Preview")
    st.dataframe(df.head())

    # Ensure timestamp
    if "date" in df.columns and "start_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["start_time"], errors="coerce")
    elif "start_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["start_time"], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")

    # Minute bucket
    df["minute_bucket"] = df["timestamp"].dt.floor("T")

    # --- Rule 1: OTP attempts per IP in 10-min window ---
    grouped = df.groupby(["true_client_ip", "minute_bucket"]).size().reset_index(name="signup_attempts")
    grouped = grouped.sort_values(["true_client_ip", "minute_bucket"])

    grouped["attempts_in_10_min"] = grouped.groupby("true_client_ip")["signup_attempts"].transform(
        lambda x: x.rolling("10min", on=grouped["minute_bucket"]).sum()
    )

    # Flag proxy
    if "akamai_epd" in df.columns:
        proxy_map = df.groupby("true_client_ip")["akamai_epd"].apply(lambda x: x.notna().any()).to_dict()
        grouped["is_proxy_ip"] = grouped["true_client_ip"].map(proxy_map).astype(int)
    else:
        grouped["is_proxy_ip"] = 0

    st.subheader("🚨 Attack Candidates (more than 10 OTPs in 10 mins or Proxy)")
    attack_candidates = grouped[(grouped["attempts_in_10_min"] > 10) | (grouped["is_proxy_ip"] == 1)]
    st.dataframe(attack_candidates)

    # --- Rule 2: Age of IP (unique dates seen) ---
    df["day"] = df["timestamp"].dt.date
    ip_age = df.groupby("true_client_ip")["day"].nunique().reset_index(name="unique_days_seen")
    st.subheader("📅 Age of IPs (Unique Days Seen)")
    st.dataframe(ip_age)

    # --- Rule 3: Sudden spike in akamai_epd ---
    if "akamai_epd" in df.columns:
        proxy_spikes = df.groupby(["true_client_ip", "day"])["akamai_epd"].apply(lambda x: x.notna().sum()).reset_index(name="proxy_count")
        proxy_spikes = proxy_spikes[proxy_spikes["proxy_count"] > proxy_spikes["proxy_count"].mean() + 3*proxy_spikes["proxy_count"].std()]
        st.subheader("⚡ Sudden Spikes in Proxy Usage (akamai_epd)")
        st.dataframe(proxy_spikes)

    # --- Rule 4: Requests per minute/hour/day (IP & device) ---
    if "dr_dv" in df.columns:
        device_counts = df.groupby(["dr_dv", df["timestamp"].dt.floor("T")]).size().reset_index(name="req_per_min")
        st.subheader("📱 Requests per Minute by Device ID")
        st.dataframe(device_counts.head())

    ip_counts = df.groupby(["true_client_ip", df["timestamp"].dt.floor("T")]).size().reset_index(name="req_per_min")
    st.subheader("🌐 Requests per Minute by IP")
    st.dataframe(ip_counts.head())

    # --- Rule 5: BMP Score >90 more than 5 times/day ---
    if "bmp_score" in df.columns:
        high_bmp = df[df["bmp_score"] > 90]
        suspicious_bmp = high_bmp.groupby(["true_client_ip", "day"]).size().reset_index(name="count90plus")
        suspicious_bmp = suspicious_bmp[suspicious_bmp["count90plus"] >= 5]
        st.subheader("🎯 Suspicious IPs with BMP score >90 at least 5 times/day")
        st.dataframe(suspicious_bmp)

    # --- Rule 6: IP repetitions baseline ---
    ip_reps = df.groupby(["true_client_ip", "day"]).size().reset_index(name="daily_count")
    mean_rep = ip_reps["daily_count"].mean()
    st.subheader("📊 IP Repetition Benchmarking")
    st.write(f"Normal daily average: {mean_rep:.2f} requests/IP")
    abnormal = ip_reps[ip_reps["daily_count"] > mean_rep*2]
    st.write("IPs exceeding benchmark (2x average):")
    st.dataframe(abnormal)

    # --- Rule 7: Averages over windows (1m, 5m, 10m) ---
    st.subheader("📈 Average OTP Requests per Window")
    df.set_index("timestamp", inplace=True)
    avg_1m = df.resample("1min").size().mean()
    avg_5m = df.resample("5min").size().mean()
    avg_10m = df.resample("10min").size().mean()
    st.write({"1min": avg_1m, "5min": avg_5m, "10min": avg_10m})

    # --- Visualization: OTP requests timeline ---
    st.subheader("📉 OTP Requests Timeline")
    otp_counts = df.resample("1min").size().reset_index(name="otp_count")
    fig = px.line(otp_counts, x="timestamp", y="otp_count", title="OTP Requests Over Time")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please upload a CSV file to start analysis.")
