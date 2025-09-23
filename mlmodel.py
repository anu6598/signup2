# ip_suspicion_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="IP Suspicion Detector", layout="wide")
st.title("🔐 IP Suspicion Detector (ML + Rule-based)")

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
    if row["max_attempts_10m"] >= 100 or row["max_attempts_1m"] >= 50:
        reasons.append(f"burst(max1m={row['max_attempts_1m']},max10m={row['max_attempts_10m']})")
    if row["total_requests"] >= 10:
        if row["failure_rate"] >= 0.9:
            reasons.append(f"high_failure_rate({row['failure_rate']:.2f})")
    if row["unique_usernames"] >= 10 and row["total_requests"] > 20:
        reasons.append(f"many_usernames({row['unique_usernames']})")
    if row["is_proxy_ip"]:
        reasons.append("proxy_present")
    if not np.isnan(row["bmp_max"]) and row["bmp_max"] >= 70:
        reasons.append(f"bmp_high({int(row['bmp_max'])})")
    return reasons

# ---------- upload CSV ----------
uploaded_file = st.file_uploader("Upload attack-day CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
    df = normalize_colnames(df)
    st.write(f"Loaded {len(df)} rows with columns: {list(df.columns)[:10]} ...")

    # ---------- detect key columns ----------
    col_ts = pick_col(df, ["timestamp","time","created_on","ts","start_time","event_time"])
    col_ip = pick_col(df, ["x_real_ip","true_client_ip","client_ip","ip","remote_addr","remote_ip"])
    col_user = pick_col(df, ["user_id","user","username","dr_uid"])
    col_path = pick_col(df, ["request_path","path","url","endpoint","request"])
    col_status = pick_col(df, ["response_code","status","result","response"])
    col_device = pick_col(df, ["dr_dv","device_id","device"])
    col_epd = pick_col(df, ["akamai_epd","epd","akamai_proxy"])
    col_bot = pick_col(df, ["akamai_bot","akamai_bmp","bmp","bot_info"])

    # ---------- normalize timestamp ----------
    if col_ts:
        df["ts"] = pd.to_datetime(df[col_ts], errors="coerce")
    else:
        df["ts"] = pd.NaT

    if col_ip:
        df["ip_addr"] = df[col_ip].astype(str)
    else:
        st.error("No IP column found.")
        st.stop()

    df = df.dropna(subset=["ts"])

    # ---------- normalize fields ----------
    df["username"] = df[col_user].astype(str) if col_user else ""
    df["device_id"] = df[col_device].astype(str) if col_device else ""
    df["status_raw"] = df[col_status].astype(str) if col_status else ""
    df["akamai_epd"] = df[col_epd].astype(str) if col_epd else ""
    df["akamai_bot"] = df[col_bot].astype(str) if col_bot else ""
    df["is_fail"] = df["status_raw"].apply(is_failure)
    df["is_proxy_row"] = df["akamai_epd"].apply(row_is_proxy)
    df["bmp_score"] = df["akamai_bot"].apply(extract_digits)

    # ---------- short-window features ----------
    df = df.sort_values("ts")
    df["minute_floor"] = df["ts"].dt.floor("T")
    df["min10_floor"] = df["ts"].dt.floor("10T")
    df["date"] = df["ts"].dt.date

    cnt_1m = df.groupby(["ip_addr","minute_floor"]).size().reset_index(name="cnt_1m")
    max_1m = cnt_1m.groupby("ip_addr")["cnt_1m"].max().reset_index(name="max_attempts_1m")
    cnt_10m = df.groupby(["ip_addr","min10_floor"]).size().reset_index(name="cnt_10m")
    max_10m = cnt_10m.groupby("ip_addr")["cnt_10m"].max().reset_index(name="max_attempts_10m")
    sum_10m = cnt_10m.groupby("ip_addr")["cnt_10m"].sum().reset_index(name="sum_10m")

    # ---------- per-IP aggregates ----------
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

    ip_agg = ip_agg.merge(max_1m, on="ip_addr", how="left")
    ip_agg = ip_agg.merge(max_10m, on="ip_addr", how="left")
    ip_agg = ip_agg.merge(sum_10m, on="ip_addr", how="left")

    for c in ["max_attempts_1m","max_attempts_10m","sum_10m"]:
        ip_agg[c] = ip_agg[c].fillna(0).astype(int)
    ip_agg["failed_requests"] = ip_agg["failed_requests"].fillna(0).astype(int)
    ip_agg["total_requests"] = ip_agg["total_requests"].fillna(0).astype(int)
    ip_agg["proxy_row_count"] = ip_agg["proxy_row_count"].fillna(0).astype(int)
    ip_agg["unique_usernames"] = ip_agg["unique_usernames"].fillna(0).astype(int)
    ip_agg["unique_devices"] = ip_agg["unique_devices"].fillna(0).astype(int)
    ip_agg["failure_rate"] = ip_agg["failed_requests"] / ip_agg["total_requests"].replace(0, np.nan)
    ip_agg["is_proxy_ip"] = (ip_agg["proxy_row_count"] > 0).astype(int)
    ip_agg["bmp_max"] = ip_agg["bmp_max"].fillna(0).astype(float)

    # ---------- rule reasons ----------
    ip_agg["rule_reasons"] = ip_agg.apply(rule_reasons, axis=1)
    ip_agg["rule_flag"] = ip_agg["rule_reasons"].apply(lambda x: len(x)>0)

    # ---------- IsolationForest anomaly ----------
    numeric_features = ["total_requests","failed_requests","failure_rate","unique_usernames",
                        "unique_devices","max_attempts_1m","max_attempts_10m","sum_10m","is_proxy_ip","bmp_max"]
    ip_agg[numeric_features] = ip_agg[numeric_features].fillna(0)

    clf_iso = IsolationForest(random_state=42, contamination=0.02)
    clf_iso.fit(ip_agg[numeric_features])
    ip_agg["iso_score_raw"] = clf_iso.decision_function(ip_agg[numeric_features])
    ip_agg["iso_anomaly"] = (clf_iso.predict(ip_agg[numeric_features])==-1).astype(int)

    # ---------- ML classifier ----------
    # load pretrained RF model (trained on prior rules)
    try:
        rf_model = joblib.load("rf_ip_model.pkl")
        ip_agg["ml_pred"] = rf_model.predict(ip_agg[numeric_features])
        ip_agg["ml_prob"] = rf_model.predict_proba(ip_agg[numeric_features])[:,1]
    except:
        st.warning("No pretrained RF model found. Skipping ML predictions.")
        ip_agg["ml_pred"] = np.nan
        ip_agg["ml_prob"] = np.nan

    # ---------- final label ----------
    ip_agg["final_label"] = np.where(ip_agg["rule_flag"] | (ip_agg["iso_anomaly"]==1), "suspicious","benign")
    ip_agg["all_reasons"] = ip_agg["rule_reasons"].apply(lambda x: ";".join(x)) + np.where(ip_agg["iso_anomaly"]==1,";isolationforest_anomaly","")

    # ---------- display ----------
    st.subheader("Top Suspicious IPs")
    st.dataframe(ip_agg.sort_values(["rule_flag","iso_score_raw"], ascending=[False,True]).head(30))

    st.subheader("Feature Distribution")
    feature_to_plot = st.selectbox("Select feature to plot", numeric_features)
    fig, ax = plt.subplots()
    ip_agg[feature_to_plot].hist(bins=30, ax=ax)
    ax.set_title(f"Distribution of {feature_to_plot}")
    st.pyplot(fig)

    st.subheader("Summary Counts")
    st.write(ip_agg["final_label"].value_counts())
