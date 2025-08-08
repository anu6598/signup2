import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(page_title="Signup Anomaly Detection", layout="wide")

# ---------- HEADER ----------
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🚨 Signup Anomaly Detection Dashboard 🚨</h1>", unsafe_allow_html=True)

# ---------- EXPLANATION BOX ----------
st.markdown("""
<div style='position: absolute; top: 80px; right: 30px; background-color: #d4edda; border: 1px solid #c3e6cb; padding: 15px; border-radius: 8px; width: 300px;'>
<h4>How Anomaly Detection Works</h4>
<p>
We use two layers of anomaly detection:
<ul>
<li><b>Rule-Based:</b> Flags any IP with >9 signups in 15 minutes, or >5 in 10 minutes.</li>
<li><b>Machine Learning:</b> Isolation Forest looks for unusual patterns in signup timestamps &amp; frequencies.</li>
</ul>
</p>
</div>
""", unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("📂 Upload your signup CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Ensure datetime
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

    # ---------- RULE-BASED DETECTION (robust version) ----------
    df = df.sort_values(["true_client_ip", "start_time"]).reset_index(drop=True)
    df['__ts'] = (df['start_time'].astype('int64') // 10**9).astype(np.int64)

    grouped_times = df.groupby('true_client_ip')['__ts'].apply(list).to_dict()

    rows = []
    for ip, times in grouped_times.items():
        times_arr = np.array(times, dtype=np.int64)
        c15 = np.empty(len(times_arr), dtype=np.int32)
        c10 = np.empty(len(times_arr), dtype=np.int32)
        for i, t in enumerate(times_arr):
            left15 = t - 900
            left10 = t - 600
            l15 = np.searchsorted(times_arr, left15, side='left')
            l10 = np.searchsorted(times_arr, left10, side='left')
            r = np.searchsorted(times_arr, t, side='right')
            c15[i] = int(r - l15)
            c10[i] = int(r - l10)
        for t, a, b in zip(times_arr, c15, c10):
            rows.append({'true_client_ip': ip, '__ts': int(t), 'count_15min': int(a), 'count_10min': int(b)})

    counts_df = pd.DataFrame(rows)
    df = df.merge(counts_df, on=['true_client_ip', '__ts'], how='left')
    df['count_15min'] = df['count_15min'].fillna(0).astype(int)
    df['count_10min'] = df['count_10min'].fillna(0).astype(int)
    # df.drop(columns=['__ts'], inplace=True)  # optional

    # Rule-based anomalies
    anomalies_rule_15 = df[df['count_15min'] > 9]
    anomalies_rule_10 = df[df['count_10min'] > 5]

    # ---------- MACHINE LEARNING DETECTION ----------
    feature_df = df[['count_15min', 'count_10min']].copy()
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['ml_anomaly'] = iso.fit_predict(feature_df)
    anomalies_ml = df[df['ml_anomaly'] == -1]

    # ---------- DISPLAY ----------
    st.subheader("🚨 Rule-Based Anomalies (15-min > 9)")
    st.dataframe(anomalies_rule_15[['true_client_ip', 'user_agent', 'start_time', 'count_15min']])

    st.subheader("📋 IPs with 5+ Signups in 10 Minutes")
    st.dataframe(anomalies_rule_10[['true_client_ip', 'user_agent', 'start_time', 'count_10min']])

    st.subheader("🤖 Machine Learning Detected Anomalies")
    st.dataframe(anomalies_ml[['true_client_ip', 'user_agent', 'start_time', 'count_15min', 'count_10min']])
