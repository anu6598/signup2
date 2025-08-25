import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from sqlite3 import Connection
import re

# Set up the page
st.set_page_config(
    page_title="OTP Request Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🔍 OTP Request Anomaly Detection System")
st.markdown("""
This system analyzes OTP request patterns to detect suspicious activities based on multiple criteria:
- High frequency requests from same IP
- Requests through proxy servers
- Device ID patterns
- IP reputation and behavior history
- Temporal patterns
""")

# Create a database connection
def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(':memory:', check_same_thread=False)
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
    return conn

# Parse the CSV data
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Extract IP from x_forwarded_for
def extract_ip(x_forwarded_for):
    if pd.isna(x_forwarded_for):
        return None
    ips = x_forwarded_for.split(',')
    return ips[0].strip() if ips else None

# Extract device ID from user_agent
def extract_device_id(user_agent):
    if pd.isna(user_agent):
        return None
    match = re.search(r'Device\)', user_agent)
    if match:
        # Extract a unique identifier from user agent
        patterns = [
            r'm\.[a-f0-9]{16}',  # m. followed by 16 hex chars
            r't\.[a-f0-9]{16}'   # t. followed by 16 hex chars
        ]
        for pattern in patterns:
            match = re.search(pattern, user_agent)
            if match:
                return match.group(0)
    return None

# Process the data
def process_data(df):
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Extract IP address
    df_processed['ip_address'] = df_processed['x_forwarded_for'].apply(extract_ip)
    
    # Extract device ID
    df_processed['device_id'] = df_processed['user_agent'].apply(extract_device_id)
    
    # Convert start_time to datetime
    df_processed['timestamp'] = pd.to_datetime(df_processed['start_time'])
    
    # Filter only OTP-related requests
    otp_patterns = ['login', 'signup', 'otp', 'verify']
    df_processed['is_otp_request'] = df_processed['request_path'].apply(
        lambda x: any(pattern in x.lower() for pattern in otp_patterns)
    )
    
    return df_processed

# Analysis functions
def analyze_otp_requests(df):
    # Filter only OTP requests
    otp_df = df[df['is_otp_request']].copy()
    
    if otp_df.empty:
        st.warning("No OTP requests found in the data")
        return None
    
    # Set timestamp as index for time-based analysis
    otp_df.set_index('timestamp', inplace=True)
    
    # 1. Analyze requests by IP address
    ip_analysis = otp_df.groupby('ip_address').agg({
        'response_code': 'count',
        'akamai_epd': lambda x: (x != '-').sum(),
        'device_id': 'nunique'
    }).rename(columns={
        'response_code': 'request_count',
        'akamai_epd': 'proxy_requests',
        'device_id': 'unique_devices'
    })
    
    ip_analysis['proxy_ratio'] = ip_analysis['proxy_requests'] / ip_analysis['request_count']
    
    # 2. Time-based analysis (10-min intervals)
    time_analysis = otp_df.resample('10T').agg({
        'response_code': 'count',
        'ip_address': 'nunique',
        'device_id': 'nunique',
        'akamai_epd': lambda x: (x != '-').sum()
    }).rename(columns={
        'response_code': 'total_requests',
        'ip_address': 'unique_ips',
        'device_id': 'unique_devices',
        'akamai_epd': 'proxy_requests'
    })
    
    time_analysis['requests_per_ip'] = time_analysis['total_requests'] / time_analysis['unique_ips']
    time_analysis['requests_per_device'] = time_analysis['total_requests'] / time_analysis['unique_devices']
    
    # 3. Device analysis
    device_analysis = otp_df.groupby('device_id').agg({
        'response_code': 'count',
        'ip_address': 'nunique',
        'akamai_epd': lambda x: (x != '-').sum()
    }).rename(columns={
        'response_code': 'request_count',
        'ip_address': 'unique_ips',
        'akamai_epd': 'proxy_requests'
    })
    
    return {
        'ip_analysis': ip_analysis,
        'time_analysis': time_analysis,
        'device_analysis': device_analysis,
        'raw_otp_data': otp_df
    }

# Detection functions
def detect_anomalies(analysis_results, threshold=10):
    ip_analysis = analysis_results['ip_analysis']
    time_analysis = analysis_results['time_analysis']
    device_analysis = analysis_results['device_analysis']
    
    anomalies = {
        'high_frequency_ips': ip_analysis[ip_analysis['request_count'] > threshold],
        'proxy_ips': ip_analysis[ip_analysis['proxy_ratio'] > 0.5],
        'high_frequency_devices': device_analysis[device_analysis['request_count'] > threshold],
        'time_based_anomalies': time_analysis[time_analysis['total_requests'] > threshold],
        'ip_device_mismatch': ip_analysis[ip_analysis['unique_devices'] > 3],
        'device_ip_mismatch': device_analysis[device_analysis['unique_ips'] > 3]
    }
    
    return anomalies

# Visualization functions
def plot_time_analysis(time_analysis):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Requests', 'Unique IPs', 'Unique Devices', 'Proxy Requests')
    )
    
    fig.add_trace(
        go.Scatter(x=time_analysis.index, y=time_analysis['total_requests'], name='Total Requests'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=time_analysis.index, y=time_analysis['unique_ips'], name='Unique IPs'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=time_analysis.index, y=time_analysis['unique_devices'], name='Unique Devices'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=time_analysis.index, y=time_analysis['proxy_requests'], name='Proxy Requests'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="OTP Requests Time Analysis (10-min intervals)")
    st.plotly_chart(fig, use_container_width=True)

def plot_ip_analysis(ip_analysis):
    top_ips = ip_analysis.nlargest(20, 'request_count')
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Top IPs by Request Count', 'Proxy Usage by IP', 
                       'Devices per IP', 'Request Distribution')
    )
    
    fig.add_trace(
        go.Bar(x=top_ips.index, y=top_ips['request_count'], name='Request Count'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=top_ips.index, y=top_ips['proxy_requests'], name='Proxy Requests'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=top_ips.index, y=top_ips['unique_devices'], name='Unique Devices'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=ip_analysis['request_count'], name='Request Distribution', nbinsx=50),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="IP Analysis")
    st.plotly_chart(fig, use_container_width=True)

# Main application
def main():
    # Sidebar for upload and configuration
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        st.header("Detection Parameters")
        threshold = st.slider("Request threshold for anomalies", min_value=5, max_value=50, value=10)
        proxy_ratio_threshold = st.slider("Proxy ratio threshold", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    if uploaded_file is not None:
        # Load and process data
        df = load_data(uploaded_file)
        if df is not None:
            df_processed = process_data(df)
            
            # Perform analysis
            analysis_results = analyze_otp_requests(df_processed)
            
            if analysis_results:
                # Detect anomalies
                anomalies = detect_anomalies(analysis_results, threshold)
                
                # Display overview
                st.header("Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                total_requests = len(analysis_results['raw_otp_data'])
                unique_ips = analysis_results['ip_analysis'].shape[0]
                unique_devices = analysis_results['device_analysis'].shape[0]
                proxy_requests = analysis_results['raw_otp_data']['akamai_epd'].apply(lambda x: x != '-').sum()
                
                col1.metric("Total OTP Requests", total_requests)
                col2.metric("Unique IPs", unique_ips)
                col3.metric("Unique Devices", unique_devices)
                col4.metric("Proxy Requests", proxy_requests)
                
                # Display time-based analysis
                st.header("Time-based Analysis")
                plot_time_analysis(analysis_results['time_analysis'])
                
                # Display IP analysis
                st.header("IP Analysis")
                plot_ip_analysis(analysis_results['ip_analysis'])
                
                # Display anomalies
                st.header("Detected Anomalies")
                
                if not anomalies['high_frequency_ips'].empty:
                    st.subheader("High Frequency IPs")
                    st.dataframe(anomalies['high_frequency_ips'])
                
                if not anomalies['proxy_ips'].empty:
                    st.subheader("IPs with High Proxy Usage")
                    st.dataframe(anomalies['proxy_ips'])
                
                if not anomalies['high_frequency_devices'].empty:
                    st.subheader("High Frequency Devices")
                    st.dataframe(anomalies['high_frequency_devices'])
                
                if not anomalies['time_based_anomalies'].empty:
                    st.subheader("Time-based Anomalies")
                    st.dataframe(anomalies['time_based_anomalies'])
                
                if not anomalies['ip_device_mismatch'].empty:
                    st.subheader("IPs with Multiple Devices")
                    st.dataframe(anomalies['ip_device_mismatch'])
                
                if not anomalies['device_ip_mismatch'].empty:
                    st.subheader("Devices with Multiple IPs")
                    st.dataframe(anomalies['device_ip_mismatch'])
                
                # Export results
                st.header("Export Results")
                if st.button("Generate Report"):
                    # Create a comprehensive report
                    report_data = {
                        "summary": {
                            "total_requests": total_requests,
                            "unique_ips": unique_ips,
                            "unique_devices": unique_devices,
                            "proxy_requests": proxy_requests,
                            "analysis_period": f"{df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}"
                        },
                        "anomalies": {k: v.to_dict() for k, v in anomalies.items() if not v.empty}
                    }
                    
                    # Convert to DataFrame for download
                    report_df = pd.DataFrame.from_dict(report_data, orient='index')
                    csv = report_df.to_csv(index=True)
                    
                    st.download_button(
                        label="Download Report as CSV",
                        data=csv,
                        file_name="otp_anomaly_report.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No OTP requests found in the data")
    else:
        st.info("Please upload a CSV file to begin analysis")

if __name__ == "__main__":
    main()
