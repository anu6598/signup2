import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from PIL import Image

# ------------------ CONFIG ------------------
RANDOM_SEED = 42
ISO_CONTAMINATION = 0.02
BURST_10MIN_THRESHOLD = 100
BURST_1MIN_THRESHOLD = 50
HIGH_FAILURE_RATE = 0.9
MIN_ATTEMPTS_FOR_FAILURE_CHECK = 10
BMP_HIGH_THRESHOLD = 70
# --------------------------------------------

# ---------- helpers ----------
def pick_col(df, choices):
    for c in choices:
        if c in df.columns:
            return c
    return None

def extract_digits(s):
    if pd.isna(s): return np.nan
    m = re.search(r"(\d+)", str(s))
    return int(m.group(1)) if m else np.nan

def normalize_colnames(df):
    cols_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns}
    df = df.rename(columns=cols_map)
    return df

def is_failure(x):
    try:
        xi = int(x)
        return xi >= 400
    except Exception:
        s = str(x).lower()
        return ("fail" in s) or ("error" in s) or ("invalid" in s) or ("unauthorized" in s) or ("403" in s) or ("401" in s)

def row_is_proxy(x):
    if pd.isna(x): return False
    s = str(x).strip().lower()
    return (s != "") and (s not in ["-","rp","nan","none","null"])

def rule_reasons(row):
    reasons = []
    if row["max_attempts_10m"] >= BURST_10MIN_THRESHOLD or row["max_attempts_1m"] >= BURST_1MIN_THRESHOLD:
        reasons.append(f"burst(max1m={row['max_attempts_1m']},max10m={row['max_attempts_10m']})")
    if row["total_requests"] >= MIN_ATTEMPTS_FOR_FAILURE_CHECK:
        if row["failure_rate"] >= HIGH_FAILURE_RATE:
            reasons.append(f"high_failure_rate({row['failure_rate']:.2f})")
    if row["unique_usernames"] >= 10 and row["total_requests"] > 20:
        reasons.append(f"many_usernames({row['unique_usernames']})")
    if row["is_proxy_ip"]:
        reasons.append("proxy_present")
    if not np.isnan(row["bmp_max"]) and row["bmp_max"] >= BMP_HIGH_THRESHOLD:
        reasons.append(f"bmp_high({int(row['bmp_max'])})")
    return reasons

def final_label_and_reasons(row):
    reasons = list(row["rule_reasons"]) if isinstance(row["rule_reasons"], list) else []
    reasons_set = set(reasons)
    if row["iso_anomaly"] == 1:
        reasons_set.add("isolationforest_anomaly")
    label = "suspicious" if (row["rule_flag"] or row["iso_anomaly"]==1) else "benign"
    return pd.Series({"final_label": label, "all_reasons": ";".join(sorted(reasons_set))})

# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title="🔐 Suspicious IP Dashboard", layout="wide")

# ---------- SIDEBAR ----------
st.sidebar.image("cybersecurity.png", use_column_width=True)  # replace with your own image
st.sidebar.title("Suspicious IP Dashboard")
st.sidebar.markdown("""
Upload your attack-day logs CSV to analyze suspicious IPs based on:
- Burst attempts  
- High failure rate  
- Proxy presence  
- BMP score  
- IsolationForest anomaly
- Forecasting suspicious login attempts (Linear Regression)  
""")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

filter_option = st.sidebar.selectbox(
    "Filter suspicious IPs by feature",
    ["All","rule_flag", "iso_anomaly", "is_proxy_ip", "high_failure_rate", "many_usernames", "bmp_high"]
)

# ---------- LOAD CSV ----------
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df = normalize_colnames(df)

    # Detect columns
    col_ts = pick_col(df, ["timestamp","time","created_on","ts","start_time","event_time"])
    col_ip = pick_col(df, ["x_real_ip","true_client_ip","client_ip","ip","remote_addr","remote_ip"])
    col_user = pick_col(df, ["user_id","user","username","dr_uid"])
    col_status = pick_col(df, ["response_code","status","result","response"])
    col_device = pick_col(df, ["dr_dv","device_id","device"])
    col_epd = pick_col(df, ["akamai_epd","epd","akamai_proxy"])
    col_bot = pick_col(df, ["akamai_bot","akamai_bmp","bmp","bot_info"])

    # normalize timestamp
    if col_ts:
        df["ts"] = pd.to_datetime(df[col_ts], errors="coerce")
    else:
        df["ts"] = pd.NaT

    if col_ip:
        df["ip_addr"] = df[col_ip].astype(str)
    else:
        st.error("No IP column found")
    
    df = df.dropna(subset=["ts"])
    df["username"] = df[col_user].astype(str) if col_user else ""
    df["device_id"] = df[col_device].astype(str) if col_device else ""
    df["status_raw"] = df[col_status].astype(str) if col_status else ""
    df["akamai_epd"] = df[col_epd].astype(str) if col_epd else ""
    df["akamai_bot"] = df[col_bot].astype(str) if col_bot else ""
    df["is_fail"] = df["status_raw"].apply(is_failure)
    df["is_proxy_row"] = df["akamai_epd"].apply(row_is_proxy)
    df["bmp_score"] = df["akamai_bot"].apply(extract_digits)

    df = df.sort_values("ts")
    df["minute_floor"] = df["ts"].dt.floor("T")
    df["min10_floor"] = df["ts"].dt.floor("10T")

    # Per-IP aggregation
    ip_agg = (
        df.groupby("ip_addr")
        .agg(
            total_requests=("ts","count"),
            failed_requests=("is_fail","sum"),
            unique_usernames=("username", lambda s: s.nunique(dropna=True)),
            unique_devices=("device_id", lambda s: s.nunique(dropna=True)),
            proxy_row_count=("is_proxy_row","sum"),
            bmp_max=("bmp_score", lambda s: pd.Series(s).dropna().astype(float).max() if s.dropna().shape[0]>0 else np.nan)
        )
        .reset_index()
    )

    # Short-window features
    max_1m = df.groupby(["ip_addr","minute_floor"]).size().groupby("ip_addr").max().reset_index(name="max_attempts_1m")
    max_10m = df.groupby(["ip_addr","min10_floor"]).size().groupby("ip_addr").max().reset_index(name="max_attempts_10m")
    ip_agg = ip_agg.merge(max_1m, on="ip_addr", how="left")
    ip_agg = ip_agg.merge(max_10m, on="ip_addr", how="left")

    # fill NaNs
    ip_agg.fillna(0, inplace=True)
    ip_agg["failure_rate"] = ip_agg["failed_requests"]/ip_agg["total_requests"].replace(0,np.nan)
    ip_agg["is_proxy_ip"] = (ip_agg["proxy_row_count"]>0).astype(int)

    # Rule reasons
    ip_agg["rule_reasons"] = ip_agg.apply(rule_reasons, axis=1)
    ip_agg["rule_flag"] = ip_agg["rule_reasons"].apply(lambda x: len(x)>0)

    # IsolationForest
    numeric_features = ["total_requests","failed_requests","failure_rate",
                        "unique_usernames","unique_devices","max_attempts_1m","max_attempts_10m","is_proxy_ip","bmp_max"]
    ip_agg[numeric_features] = ip_agg[numeric_features].fillna(0)
    clf = IsolationForest(random_state=RANDOM_SEED, contamination=ISO_CONTAMINATION)
    clf.fit(ip_agg[numeric_features])
    ip_agg["iso_anomaly"] = (clf.predict(ip_agg[numeric_features])==-1).astype(int)

    # Final label
    res = ip_agg.apply(final_label_and_reasons, axis=1)
    ip_agg = pd.concat([ip_agg, res], axis=1)

    # ---------- DISPLAY ----------
    st.title("🔐 Suspicious IP Dashboard")
    
    st.subheader("Top Suspicious IPs")
    st.dataframe(ip_agg.sort_values(["rule_flag","iso_anomaly","total_requests"], ascending=[False,False,False]).head(20))

    st.subheader("Filtered IPs by Feature")
    if filter_option != "All":
        if filter_option in ["rule_flag","iso_anomaly","is_proxy_ip"]:
            filtered_ips = ip_agg[ip_agg[filter_option]==1]
        elif filter_option=="high_failure_rate":
            filtered_ips = ip_agg[ip_agg["failure_rate"]>=0.9]
        elif filter_option=="many_usernames":
            filtered_ips = ip_agg[ip_agg["unique_usernames"]>=10]
        elif filter_option=="bmp_high":
            filtered_ips = ip_agg[ip_agg["bmp_max"]>=70]
        st.dataframe(filtered_ips[["ip_addr","total_requests","failed_requests","failure_rate",
                                   "unique_usernames","unique_devices","proxy_row_count","bmp_max","all_reasons"]])

    st.subheader("Daily Login Attempt Summary")
    daily_summary = df.groupby(df["ts"].dt.date).agg(
        total_requests=("ts","count"),
        failed_requests=("is_fail","sum"),
        unique_ips=("ip_addr","nunique")
    ).reset_index().rename(columns={"ts":"date"})
    st.dataframe(daily_summary)

    # ---------- DAILY LOGINS + 7-DAY FORECAST USING LINEAR REGRESSION ----------
    st.subheader("📊 Daily Login Count & 7-Day Forecast (Linear Regression)")
    daily_logins = df.groupby(df["ts"].dt.date).size().rename("login_count")
    daily_logins = daily_logins.sort_index()
    st.line_chart(daily_logins)

    if len(daily_logins) >= 3:
        # Prepare data for Linear Regression
        X = np.arange(len(daily_logins)).reshape(-1,1)
        y = daily_logins.values
        lr_model = LinearRegression()
        lr_model.fit(X, y)

        # Forecast next 7 days
        future_X = np.arange(len(daily_logins), len(daily_logins)+7).reshape(-1,1)
        forecast_7d = lr_model.predict(future_X)
        future_dates = pd.date_range(daily_logins.index[-1] + pd.Timedelta(days=1), periods=7)
        forecast_series = pd.Series(forecast_7d, index=future_dates)

        # Plot
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(daily_logins.index, daily_logins.values, label="Historical", marker='o')
        ax.plot(forecast_series.index, forecast_series.values, label="7-Day Forecast", linestyle='--', marker='o')
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Logins")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Forecasted Logins (Next 7 Days)")
        st.dataframe(forecast_series.rename("forecasted_logins"))
    else:
        st.info("Need at least 3 days of data for forecast demo.")

else:
    st.info("Upload a CSV file to see suspicious IP analysis.")
