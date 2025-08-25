import streamlit as st
import pandas as pd
import plotly.express as px

st.title("OTP Request Tracking Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your OTP logs CSV", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)

    # Ensure 'date' is parsed as datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Check required columns exist
    if 'date' in df.columns and 'true_client_ip' in df.columns:
        # Aggregate OTP requests per minute
        df['minute_bucket'] = df['date'].dt.floor('T')
        otp_counts = df.groupby(['minute_bucket', 'true_client_ip']).size().reset_index(name='otp_count')

        st.subheader("OTP Request Counts per Minute per IP")
        st.dataframe(otp_counts.head())

        # Plot OTP requests
        fig = px.line(
            otp_counts,
            x='minute_bucket',
            y='otp_count',
            color='true_client_ip',
            title="OTP Requests Over Time by IP"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Identify suspicious IPs with high request volume
        threshold = st.slider("Set suspicious activity threshold (requests/minute)", 5, 100, 20)
        suspicious_ips = otp_counts[otp_counts['otp_count'] > threshold]

        st.subheader("Suspicious IPs")
        st.dataframe(suspicious_ips)
    else:
        st.error("CSV must contain at least 'date' and 'true_client_ip' columns.")
