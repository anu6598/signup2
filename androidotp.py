import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="OTP Attack Monitoring System", layout="wide")

st.title("🔐 OTP Attack Monitoring & Detection System")

# ======================================
# File Upload
# ======================================
uploaded_file = st.file_uploader("Upload OTP logs CSV", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Normalize
    df['start_time'] = pd.to_datetime(df['start_time'], errors="coerce")
    df['date'] = df['start_time'].dt.date
    if 'bmp_score' not in df.columns:
        df['bmp_score'] = np.random.randint(50,100, size=len(df))  # fallback if not present

    st.subheader("📂 Data Preview")
    st.dataframe(df.head(20))

    # ======================================
    # RULE 1: OTPs per IP in 10 min window
    # ======================================
    df['minute_bucket'] = df['start_time'].dt.floor("T")

    grouped = (
        df.groupby(['true_client_ip','minute_bucket'])
        .agg(signup_attempts=('request_path','count'),
             akamai_epd=('akamai_epd','max'),
             dr_dv=('dr_dv','max'))
        .reset_index()
    )

    # Rolling 10 min sum
    grouped['attempts_in_10_min'] = (
        grouped.groupby('true_client_ip')['signup_attempts']
        .rolling('10T', on='minute_bucket').sum().reset_index(drop=True)
    )

    grouped['is_proxy_ip'] = grouped['akamai_epd'].notna().astype(int)

    suspicious_10min = grouped[grouped['attempts_in_10_min'] > 10]

    st.subheader("🚨 Suspicious Activity: >10 OTPs in 10 mins")
    st.dataframe(suspicious_10min)

    # ======================================
    # RULE 2: Repeated IPs across days (IP Age)
    # ======================================
    ip_age = df.groupby('true_client_ip')['date'].nunique().reset_index()
    ip_age.columns = ['true_client_ip','days_seen']
    st.subheader("📌 IP Age (Repetition across days)")
    st.dataframe(ip_age.sort_values('days_seen', ascending=False))

    # ======================================
    # RULE 3: Spike in proxy usage
    # ======================================
    proxy_trend = df.groupby(['date'])['akamai_epd'].apply(lambda x: x.notna().sum()).reset_index(name='proxy_count')
    st.subheader("📈 Proxy Usage Trend (akamai_epd populated)")
    fig_proxy = px.line(proxy_trend, x='date', y='proxy_count', title="Daily Proxy Usage")
    st.plotly_chart(fig_proxy, use_container_width=True)

    # ======================================
    # RULE 4: Requests per device & IP (per min, hour, day)
    # ======================================
    df['hour_bucket'] = df['start_time'].dt.floor("H")
    df['day_bucket'] = df['start_time'].dt.floor("D")

    per_ip_min = df.groupby(['true_client_ip','minute_bucket']).size().reset_index(name='req_per_min')
    per_ip_hour = df.groupby(['true_client_ip','hour_bucket']).size().reset_index(name='req_per_hour')
    per_ip_day = df.groupby(['true_client_ip','day_bucket']).size().reset_index(name='req_per_day')

    per_device_min = df.groupby(['dr_dv','minute_bucket']).size().reset_index(name='req_per_min')
    per_device_hour = df.groupby(['dr_dv','hour_bucket']).size().reset_index(name='req_per_hour')

    st.subheader("📊 Requests per IP per Minute")
    st.dataframe(per_ip_min.head(20))

    # ======================================
    # RULE 5: BMP Score anomalies
    # ======================================
    bmp_flags = (
        df[df['bmp_score']>=90]
        .groupby(['true_client_ip','date'])
        .size().reset_index(name='high_score_count')
    )
    flagged_ips = bmp_flags[bmp_flags['high_score_count']>=5]

    st.subheader("🚨 High BMP Score IPs (>=90 at least 5 times/day)")
    st.dataframe(flagged_ips)

    # ======================================
    # RULE 6: Benchmark IP activity (Normal vs Spike)
    # ======================================
    ip_daily = df.groupby(['true_client_ip','date']).size().reset_index(name='daily_requests')
    benchmark = ip_daily['daily_requests'].median()
    ip_daily['is_above_benchmark'] = ip_daily['daily_requests'] > benchmark

    st.subheader("📌 Benchmark Analysis (IP daily requests)")
    st.write(f"Median requests per IP per day: **{benchmark}**")
    st.dataframe(ip_daily[ip_daily['is_above_benchmark']])

    # ======================================
    # RULE 7: Avg OTPs in 1m, 5m, 10m windows
    # ======================================
    df['minute_bucket'] = df['start_time'].dt.floor("T")

    avg_1m = df.groupby('minute_bucket').size().mean()
    avg_5m = df.set_index('start_time').resample('5T').size().mean()
    avg_10m = df.set_index('start_time').resample('10T').size().mean()

    st.subheader("📊 Avg OTPs per window")
    st.write(f"Average OTPs per 1 min: {avg_1m:.2f}")
    st.write(f"Average OTPs per 5 min: {avg_5m:.2f}")
    st.write(f"Average OTPs per 10 min: {avg_10m:.2f}")

    # ======================================
    # VISUALS
    # ======================================
    st.subheader("📈 OTP Requests Over Time (IP-level)")
    ip_trend = df.groupby(['minute_bucket','true_client_ip']).size().reset_index(name='requests')
    fig = px.line(ip_trend, x='minute_bucket', y='requests', color='true_client_ip', title="Requests per IP per Minute")
    st.plotly_chart(fig, use_container_width=True)

    # Download anomaly tables
    st.subheader("📥 Download Suspicious Activity Tables")
    st.download_button("Download 10-min Anomalies", suspicious_10min.to_csv(index=False), "anomalies_10min.csv", "text/csv")
    st.download_button("Download High BMP Score IPs", flagged_ips.to_csv(index=False), "bmp_flags.csv", "text/csv")
    st.download_button("Download IP Daily Benchmark Flags", ip_daily[ip_daily['is_above_benchmark']].to_csv(index=False), "ip_benchmark_flags.csv", "text/csv")

else:
    st.info("👆 Upload your OTP logs CSV to begin analysis.")
