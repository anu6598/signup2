# pages/Daily Stats.py
import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="Daily Stats", layout="wide")
st.title("📊 Daily Stats — OTP Abuse Rules")

# -------------------------
# Helper: Proxy detection API
# -------------------------
def check_proxy(ip):
    """
    Replace this with your actual Proxy Detection API.
    For demo, returns random True/False.
    """
    try:
        # Example API: https://ipinfo.io or https://proxycheck.io
        # resp = requests.get(f"https://proxycheck.io/v2/{ip}?key=YOUR_KEY&vpn=1")
        # data = resp.json()
        # return data.get(ip, {}).get("proxy") == "yes"
        return np.random.choice([True, False], p=[0.3, 0.7])
    except Exception:
        return False

# -------------------------
# Load uploaded data from session_state
# -------------------------
if "uploaded_file" not in st.session_state:
    st.error("Please upload a CSV in the main dashboard first.")
    st.stop()

uploaded_file = st.session_state["uploaded_file"]
df_raw = pd.read_csv(uploaded_file)

# normalize column names
cols_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_raw.columns}
df_raw.rename(columns=cols_map, inplace=True)
df = df_raw.copy()

# identify useful columns
col_date = "date" if "date" in df.columns else None
col_start_time = "start_time" if "start_time" in df.columns else None
col_ip = "true_client_ip" if "true_client_ip" in df.columns else "ip"
col_device = "dr_dv" if "dr_dv" in df.columns else None
col_request_path = "request_path" if "request_path" in df.columns else "path"

# build timestamp if possible
if col_date and col_start_time:
    df["timestamp"] = pd.to_datetime(df[col_date].astype(str) + " " + df[col_start_time].astype(str), errors="coerce")
elif col_date:
    df["timestamp"] = pd.to_datetime(df[col_date], errors="coerce")
elif col_start_time:
    df["timestamp"] = pd.to_datetime(df[col_start_time], errors="coerce")

df["date"] = df["timestamp"].dt.date
df["request_path"] = df[col_request_path].astype(str)
df["true_client_ip"] = df[col_ip].astype(str)
if col_device:
    df["dr_dv"] = df[col_device].astype(str)
else:
    df["dr_dv"] = np.nan

# OTP-only
df["is_otp"] = df["request_path"].str.contains("otp", case=False, na=False)
otp_df = df[df["is_otp"]].copy()

if len(otp_df) == 0:
    st.warning("No OTP-related requests detected in this dataset.")
    st.stop()

# -------------------------
# 1. More than 1000 (or 800+) OTPs per day per IP
# -------------------------
st.subheader("1️⃣ IPs with >800 OTP requests per day")
daily_counts = otp_df.groupby(["date", "true_client_ip"]).size().reset_index(name="daily_requests")
high_daily = daily_counts[daily_counts["daily_requests"] > 800]
st.dataframe(high_daily)

# -------------------------
# 2. Any IP with >25 requests per day
# -------------------------
st.subheader("2️⃣ IPs with >25 requests per day")
ip_over_25 = daily_counts[daily_counts["daily_requests"] > 25]
st.dataframe(ip_over_25)

# -------------------------
# 3. More than 70% of IPs are proxies
# -------------------------
st.subheader("3️⃣ % of IPs flagged as proxy (via API)")
unique_ips = otp_df["true_client_ip"].unique()
proxy_flags = {ip: check_proxy(ip) for ip in unique_ips}
proxy_df = pd.DataFrame(list(proxy_flags.items()), columns=["ip", "is_proxy"])
proxy_ratio = proxy_df["is_proxy"].mean()

st.write(f"Proxy ratio: **{proxy_ratio:.1%}** ({proxy_df['is_proxy'].sum()} of {len(proxy_df)} IPs)")

if proxy_ratio > 0.7:
    st.warning("⚠️ More than 70% of IPs are flagged as proxy")
else:
    st.success("✅ Less than 70% of IPs are proxies")

st.dataframe(proxy_df)

# -------------------------
# 4. Device IDs with >15 requests per day
# -------------------------
st.subheader("4️⃣ Device IDs with >15 requests per day")
if col_device:
    device_daily = otp_df.groupby(["date", "dr_dv"]).size().reset_index(name="daily_requests")
    suspicious_devices = device_daily[device_daily["daily_requests"] > 15]
    st.dataframe(suspicious_devices)
else:
    st.info("No device ID column available in this dataset.")
