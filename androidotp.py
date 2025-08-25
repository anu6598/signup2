import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import timedelta

# -------------------------------
# Streamlit App Layout
# -------------------------------
st.set_page_config(page_title="OTP Abuse Detection Dashboard", layout="wide")

st.title("🔐 OTP Abuse Detection Dashboard")

uploaded_file = st.file_uploader("Upload OTP dataset (CSV)", type=["csv"])

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Ensure timestamp
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Extract BMP score from akamai_bot (digits only)
    df['bmp_score'] = df['akamai_bot'].astype(str).str.extract(r'(\d+)').astype(float)

    # -------------------------------
    # Normal OTP Behavior Benchmark
    # -------------------------------
    hourly_counts = df.set_index('start_time').resample('1H')['true_client_ip'].count()
    baseline = hourly_counts.median()
    df['hour_bucket'] = df['start_time'].dt.floor('H')
    df['otp_count_hour'] = df.groupby('hour_bucket')['true_client_ip'].transform('count')
    df['otp_label'] = np.where(df['otp_count_hour'] > baseline, 'Suspicious', 'Normal')

    # -------------------------------
    # 10-Minute Burst Detection
    # -------------------------------
    df['minute_bucket'] = df['start_time'].dt.floor('min')
    burst = (
        df.groupby(['true_client_ip', pd.Grouper(key='start_time', freq='10min')])
        .size()
        .reset_index(name='otp_count_10min')
    )
    burst['burst_flag'] = burst['otp_count_10min'] > 10

    # -------------------------------
    # Proxy flagging
    # -------------------------------
    proxy_flags = df[df['akamai_epd'].notna()].groupby('true_client_ip').size().reset_index(name='proxy_hits')
    proxy_flags['is_proxy'] = proxy_flags['proxy_hits'] > 1

    # -------------------------------
    # BMP Score Rules
    # -------------------------------
    bmp_flags = (
        df[df['bmp_score'] > 90]
        .groupby(['true_client_ip', 'date'])
        .size()
        .reset_index(name='bmp_high_count')
    )
    bmp_flags['bmp_flag'] = bmp_flags['bmp_high_count'] >= 5

    # -------------------------------
    # IP Age & Repetition Benchmark
    # -------------------------------
    ip_appearance = df.groupby('true_client_ip')['date'].nunique().reset_index(name='days_seen')
    ip_age = df.groupby('true_client_ip')['date'].agg(['min', 'max']).reset_index()
    ip_stats = ip_appearance.merge(ip_age, on='true_client_ip')
    repetition_threshold = ip_appearance['days_seen'].median()
    ip_stats['repetition_flag'] = ip_stats['days_seen'] > repetition_threshold

    # -------------------------------
    # Spikes in akamai_epd
    # -------------------------------
    epd_spikes = df.groupby(['date', 'true_client_ip'])['akamai_epd'].count().reset_index()
    daily_mean = epd_spikes.groupby('date')['akamai_epd'].mean().reset_index(name='mean_epd')
    epd_spikes = epd_spikes.merge(daily_mean, on='date')
    epd_spikes['spike_flag'] = epd_spikes['akamai_epd'] > (epd_spikes['mean_epd'] * 2)

    # -------------------------------
    # Request Rate Analysis
    # -------------------------------
    rates = {}
    for freq, label in [('1min', 'Per Minute'), ('10min', 'Per 10 Minutes'), ('1H', 'Per Hour'), ('1D', 'Per Day')]:
        rates[label] = df.set_index('start_time').resample(freq)['true_client_ip'].count().reset_index(name='requests')

    # -------------------------------
    # Suspicious Behavior (Device/IP)
    # -------------------------------
    device_flags = df.groupby('dr_dv').size().reset_index(name='device_requests')
    ip_flags = df.groupby('true_client_ip').size().reset_index(name='ip_requests')

    # -------------------------------
    # Merge all anomaly flags
    # -------------------------------
    anomalies = burst.merge(proxy_flags, on='true_client_ip', how='left') \
                     .merge(bmp_flags, on='true_client_ip', how='left') \
                     .merge(ip_stats, on='true_client_ip', how='left')

    anomalies = anomalies.fillna(False)

    # -------------------------------
    # Dashboard Tabs
    # -------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚠️ Anomalies", "📈 Request Rates", "⬇️ Downloads"])

    with tab1:
        st.subheader("Normal Behavior Benchmark")
        st.metric("Median OTP requests per hour", f"{baseline:.2f}")
        st.line_chart(hourly_counts, height=300)

        st.subheader("BMP Score Distribution")
        st.bar_chart(df['bmp_score'].dropna(), height=300)

    with tab2:
        st.subheader("Flagged Anomalies")
        st.dataframe(anomalies.head(50))

        st.subheader("Proxy Spikes")
        st.dataframe(epd_spikes[epd_spikes['spike_flag']])

    with tab3:
        st.subheader("Request Rates by Time Window")
        for label, data in rates.items():
            st.write(f"**{label}**")
            st.line_chart(data.set_index('start_time')['requests'], height=200)

    with tab4:
        st.download_button(
            "Download Anomalies as CSV",
            anomalies.to_csv(index=False).encode('utf-8'),
            "anomalies.csv",
            "text/csv"
        )
        st.download_button(
            "Download Proxy Spikes as CSV",
            epd_spikes.to_csv(index=False).encode('utf-8'),
            "proxy_spikes.csv",
            "text/csv"
        )

else:
    st.info("👆 Upload a dataset to begin analysis")
