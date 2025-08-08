import streamlit as st
from data_loader import load_signup_data
from anomaly_detection import detect_anomalies
from utils import daily_summary, plot_time_series

st.set_page_config(page_title="Signup Anomaly Detection", layout="wide")

st.title("📊 Signup Anomaly Detection Dashboard")

uploaded_file = st.file_uploader("Upload Signup Log CSV", type=["csv"])

if uploaded_file:
    df = load_signup_data(uploaded_file)
    st.write("Preview of Uploaded Data", df.head())

    threshold = st.slider("Anomaly threshold (signups per IP/day)", 1, 50, 5)

    df_analyzed, model = detect_anomalies(
        df,
        time_col="start_time",
        ip_col="true_client_ip",
        threshold=threshold
    )

    st.subheader("Daily Summary")
    summary_df = daily_summary(df_analyzed, date_col="start_time", ip_col="true_client_ip")
    st.dataframe(summary_df)

    st.subheader("📈 Time Series Chart")
    chart = plot_time_series(summary_df)
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Anomalous Records")
    st.dataframe(df_analyzed[df_analyzed["is_anomaly"]])
