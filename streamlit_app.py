import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(layout="wide")

# ============================
# SECTION 1: Heading + Info Box
# ============================
st.markdown(
    """
    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 20px;">
        <div style="flex: 1;">
            <h1 style="margin-bottom: 0;">🚨 Signup Anomaly Detection Dashboard</h1>
            <p style="margin-top: 5px; font-size: 16px; color: #555;">
                Detect unusual signup patterns using rule-based and machine learning methods
            </p>
        </div>
        <div style="flex: 1; background-color: #e6ffe6; padding: 15px; border-radius: 10px; border: 1px solid #b3ffb3;">
            <h3 style="margin-top: 0;">ℹ️ How Anomaly Detection Works</h3>
            <p style="font-size: 14px; color: #333; line-height: 1.5;">
                We use two approaches:<br><br>
                <b>1. Rule-Based:</b> Flags IPs with unusually high signup counts in short time windows
                (e.g., more than 9 in 15 minutes or more than 5 in 10 minutes).<br>
                <b>2. Machine Learning:</b> An Isolation Forest model learns the normal signup pattern
                (signups in last 10 and 15 minutes) and flags IPs whose patterns deviate strongly.
                Each anomaly comes with an explanation so you know why it was flagged.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================
# File Upload
# ============================
uploaded_file = st.file_uploader("Upload signup CSV", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df = df.sort_values(['true_client_ip', 'start_time']).reset_index(drop=True)

    # Helper numeric timestamp
    df['__ts'] = (df['start_time'].astype('int64') // 10**9).astype(np.int64)

    # ============================
    # SECTION 2: Time Series Graph
    # ============================
    st.subheader("📈 Signup Activity Over Time by IP")

    agg_counts = df.groupby(['start_time', 'true_client_ip']).size().reset_index(name='count')

    fig, ax = plt.subplots(figsize=(12, 5))
    scatter = ax.scatter(
        agg_counts['start_time'],
        agg_counts['true_client_ip'],
        s=agg_counts['count'] * 20,
        alpha=0.6,
        c=agg_counts['count'],
        cmap='viridis'
    )

    ax.set_xlabel("Time")
    ax.set_ylabel("IP Address")
    ax.set_title("Signup Counts per IP Over Time", fontsize=14)
    fig.colorbar(scatter, ax=ax, label="Signup Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # ============================
    # Rule-Based Detection
    # ============================
    st.subheader("🛠 Rule-Based Anomaly Detection")

    # Precompute counts in rolling windows
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
    df[['count_15min', 'count_10min']] = df[['count_15min', 'count_10min']].fillna(0).astype(int)

    # 15-min > 9
    rb_15 = df[df['count_15min'] > 9].copy()
    rb_15['reason'] = rb_15.apply(
        lambda x: f"IP {x['true_client_ip']} made {x['count_15min']} signups in 15 minutes — above the threshold of 9.",
        axis=1
    )

    # 10-min >= 5
    rb_10 = df[df['count_10min'] >= 5].copy()
    rb_10['reason'] = rb_10.apply(
        lambda x: f"IP {x['true_client_ip']} made {x['count_10min']} signups in 10 minutes — above the threshold of 5.",
        axis=1
    )

    st.markdown("**Rule: More than 9 signups in 15 minutes**")
    st.dataframe(rb_15[['start_time', 'true_client_ip', 'count_15min', 'reason']])

    st.markdown("**Rule: More than or equal to 5 signups in 10 minutes**")
    st.dataframe(rb_10[['start_time', 'true_client_ip', 'count_10min', 'reason']])

    # ============================
    # Machine Learning Detection
    # ============================
    st.subheader("🤖 Machine Learning Detected Anomalies")

    features = df[['count_15min', 'count_10min']].copy()
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['ml_anomaly'] = iso.fit_predict(features)

    anomalies_ml = df[df['ml_anomaly'] == -1].copy()

    median_15 = df['count_15min'].median()
    median_10 = df['count_10min'].median()

    def explain_row(row):
        reasons = []
        if row['count_15min'] > 2 * median_15:
            reasons.append(f"15-min count ({row['count_15min']}) is more than twice the normal median ({median_15}).")
        if row['count_10min'] > 2 * median_10:
            reasons.append(f"10-min count ({row['count_10min']}) is more than twice the normal median ({median_10}).")
        if not reasons:
            reasons.append("Signup counts are statistically far from typical patterns detected by the model.")
        return " ".join(reasons)

    anomalies_ml['reason'] = anomalies_ml.apply(explain_row, axis=1)

    st.dataframe(anomalies_ml[['start_time', 'true_client_ip', 'count_15min', 'count_10min', 'reason']])
