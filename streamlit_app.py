import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(page_title="Signup Anomaly Detector", layout="wide")

# =====================
# HEADER & INFO BOX
# =====================
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("<h1 style='color:#2C3E50;'>🚨 Signup Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)

with col2:
    st.markdown(
        """
        <div style='background-color:#D5F5E3; padding:10px; border-radius:8px;'>
        <b>How This Works:</b><br>
        This dashboard uses:
        <ul>
            <li><b>Rule-based checks</b> — Detect IPs with unusually high signups in short time.</li>
            <li><b>Machine Learning</b> — Isolation Forest to flag unusual signup patterns.</li>
        </ul>
        Anomalies may indicate suspicious activity or bot signups.
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================
# FILE UPLOAD
# =====================
uploaded_file = st.file_uploader("Upload signup data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # =====================
    # DATA PREP
    # =====================
    if "start_time" not in df.columns or "true_client_ip" not in df.columns:
        st.error("CSV must have 'start_time' and 'true_client_ip' columns.")
        st.stop()

    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df = df.dropna(subset=["start_time"])

    # =====================
    # RULE 1: >9 signups in 15 minutes
    # =====================
    df_15 = (
        df.set_index("start_time")
          .groupby("true_client_ip")
          .rolling("15min")
          .size()
          .reset_index(name="count_15min")
    )
    df = df.merge(df_15, on=["true_client_ip", "start_time"], how="left")
    rule_anomalies_15 = df[df["count_15min"] > 9]

    # =====================
    # RULE 2: >5 signups in 10 minutes (for table)
    # =====================
    df_10 = (
        df.set_index("start_time")
          .groupby("true_client_ip")
          .rolling("10min")
          .size()
          .reset_index(name="count_10min")
    )
    df = df.merge(df_10, on=["true_client_ip", "start_time"], how="left")
    table_10min = df[df["count_10min"] > 5]

    # =====================
    # MACHINE LEARNING ANOMALY DETECTION
    # =====================
    feature_df = (
        df.groupby("true_client_ip")
          .agg(signup_count=("true_client_ip", "size"))
          .reset_index()
    )

    iso = IsolationForest(contamination=0.02, random_state=42)
    feature_df["anomaly"] = iso.fit_predict(feature_df[["signup_count"]])

    ml_anomalies = feature_df[feature_df["anomaly"] == -1]
    df_ml_anomalies = df[df["true_client_ip"].isin(ml_anomalies["true_client_ip"])]

    # =====================
    # VISUALS
    # =====================
    st.subheader("📊 Anomaly Counts by IP")
    fig = px.histogram(feature_df, x="signup_count", nbins=30, title="Signup Count Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ Rule-based anomalies (>9 signups in 15 mins)")
    if not rule_anomalies_15.empty:
        st.dataframe(rule_anomalies_15[["start_time", "true_client_ip", "user_agent", "count_15min"]])
    else:
        st.success("No anomalies found for 15-min rule.")

    st.subheader("📋 IPs with >5 signups in 10 mins (Table)")
    if not table_10min.empty:
        st.dataframe(table_10min[["start_time", "true_client_ip", "user_agent", "count_10min"]])
    else:
        st.info("No IPs with >5 signups in 10 minutes.")

    st.subheader("🤖 ML-detected anomalies")
    if not df_ml_anomalies.empty:
        st.dataframe(df_ml_anomalies[["start_time", "true_client_ip", "user_agent"]])
    else:
        st.success("No anomalies detected by ML.")

else:
    st.info("Please upload a CSV file to get started.")
