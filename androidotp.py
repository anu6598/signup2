import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
from sqlite3 import Connection
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="OTP Request Anomaly Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .anomaly-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 1rem;
    }
    .normal-alert {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None

# Load and preprocess data
def load_data(uploaded_file):
    """Load and preprocess the CSV data"""
    try:
        df = pd.read_csv(uploaded_file)
        # Convert date and time columns to datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['start_time'].str.split('T').str[1].str.split('Z').str[0])
        
        # Extract IP from x_forwarded_for (first IP in the list)
        df['client_ip'] = df['x_forwarded_for'].str.split(',').str[0].str.strip()
        
        # Check if request is an OTP request
        df['is_otp_request'] = df['request_path'].str.contains('login_email|request_login_otp', case=False)
        
        # Extract device ID if available
        df['device_id'] = df['dr_dv'].where(df['dr_dv'] != '-', None)
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Database functions
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect(':memory:', check_same_thread=False)
    return conn

def setup_db(conn, df):
    """Set up database with the data"""
    df.to_sql('login_logs', conn, if_exists='replace', index=False)
    
    # Create indices for better performance
    conn.execute('CREATE INDEX idx_ip ON login_logs(client_ip)')
    conn.execute('CREATE INDEX idx_datetime ON login_logs(datetime)')
    conn.execute('CREATE INDEX idx_otp ON login_logs(is_otp_request)')
    conn.execute('CREATE INDEX idx_device ON login_logs(device_id)')
    
    return conn

# Analysis functions
def detect_otp_anomalies(conn, time_window_minutes=10, threshold=10):
    """Detect OTP anomalies based on the provided rules"""
    query = f"""
    WITH otp_requests AS (
        SELECT 
            client_ip,
            datetime,
            akamai_epd,
            device_id,
            COUNT(*) OVER (
                PARTITION BY client_ip 
                ORDER BY datetime 
                RANGE BETWEEN INTERVAL '{time_window_minutes}' MINUTE PRECEDING AND CURRENT ROW
            ) as requests_in_window,
            CASE WHEN akamai_epd IS NOT NULL AND akamai_epd != '-' THEN 1 ELSE 0 END as is_proxy
        FROM login_logs 
        WHERE is_otp_request = 1
    ),
    ip_stats AS (
        SELECT 
            client_ip,
            MIN(datetime) as first_seen,
            MAX(datetime) as last_seen,
            COUNT(DISTINCT DATE(datetime)) as unique_days,
            COUNT(*) as total_requests,
            SUM(CASE WHEN is_proxy = 1 THEN 1 ELSE 0 END) as proxy_requests,
            MAX(is_proxy) as ever_used_proxy
        FROM otp_requests
        GROUP BY client_ip
    ),
    time_based_anomalies AS (
        SELECT 
            client_ip,
            datetime,
            requests_in_window,
            is_proxy,
            CASE 
                WHEN requests_in_window > {threshold} THEN 1
                ELSE 0
            END as is_anomaly
        FROM otp_requests
        WHERE requests_in_window > {threshold}
    )
    SELECT 
        t.client_ip,
        t.datetime,
        t.requests_in_window,
        t.is_proxy,
        t.is_anomaly,
        i.unique_days,
        i.total_requests,
        i.ever_used_proxy,
        i.proxy_requests
    FROM time_based_anomalies t
    JOIN ip_stats i ON t.client_ip = i.client_ip
    ORDER BY t.requests_in_window DESC, t.datetime
    """
    
    return pd.read_sql_query(query, conn)

def get_minute_bucket_analysis(conn, time_window_minutes=10):
    """Analyze OTP requests by minute buckets"""
    query = f"""
    WITH minute_buckets AS (
        SELECT 
            client_ip,
            datetime,
            strftime('%Y-%m-%d %H:%M:00', datetime) as minute_bucket,
            CASE WHEN akamai_epd IS NOT NULL AND akamai_epd != '-' THEN 1 ELSE 0 END as is_proxy
        FROM login_logs 
        WHERE is_otp_request = 1
    ),
    bucket_counts AS (
        SELECT 
            client_ip,
            minute_bucket,
            COUNT(*) as requests_in_minute,
            MAX(is_proxy) as is_proxy
        FROM minute_buckets
        GROUP BY client_ip, minute_bucket
    ),
    window_analysis AS (
        SELECT 
            client_ip,
            minute_bucket,
            requests_in_minute,
            is_proxy,
            SUM(requests_in_minute) OVER (
                PARTITION BY client_ip 
                ORDER BY minute_bucket 
                ROWS BETWEEN {time_window_minutes-1} PRECEDING AND CURRENT ROW
            ) as requests_in_window
        FROM bucket_counts
    )
    SELECT * FROM window_analysis 
    WHERE requests_in_window > 0
    ORDER BY client_ip, minute_bucket
    """
    
    return pd.read_sql_query(query, conn)

def get_device_analysis(conn):
    """Analyze OTP requests by device"""
    query = """
    SELECT 
        device_id,
        client_ip,
        COUNT(*) as request_count,
        COUNT(DISTINCT client_ip) as unique_ips,
        MIN(datetime) as first_request,
        MAX(datetime) as last_request
    FROM login_logs 
    WHERE is_otp_request = 1 AND device_id IS NOT NULL
    GROUP BY device_id, client_ip
    HAVING request_count > 1
    ORDER BY request_count DESC
    """
    
    return pd.read_sql_query(query, conn)

def get_ip_frequency_analysis(conn):
    """Analyze IP frequency patterns"""
    query = """
    WITH ip_daily_counts AS (
        SELECT 
            client_ip,
            DATE(datetime) as date,
            COUNT(*) as daily_requests
        FROM login_logs 
        WHERE is_otp_request = 1
        GROUP BY client_ip, DATE(datetime)
    ),
    ip_stats AS (
        SELECT 
            client_ip,
            COUNT(DISTINCT date) as days_active,
            AVG(daily_requests) as avg_daily_requests,
            MAX(daily_requests) as max_daily_requests,
            SUM(daily_requests) as total_requests
        FROM ip_daily_counts
        GROUP BY client_ip
    )
    SELECT * FROM ip_stats ORDER BY total_requests DESC
    """
    
    return pd.read_sql_query(query, conn)

def get_proxy_analysis(conn):
    """Analyze proxy usage patterns"""
    query = """
    SELECT 
        client_ip,
        COUNT(*) as total_requests,
        SUM(CASE WHEN akamai_epd IS NOT NULL AND akamai_epd != '-' THEN 1 ELSE 0 END) as proxy_requests,
        MIN(datetime) as first_seen,
        MAX(datetime) as last_seen,
        COUNT(DISTINCT DATE(datetime)) as unique_days
    FROM login_logs 
    WHERE is_otp_request = 1
    GROUP BY client_ip
    HAVING proxy_requests > 0
    ORDER BY proxy_requests DESC
    """
    
    return pd.read_sql_query(query, conn)

# Visualization functions
def plot_requests_over_time(df, time_col='datetime', value_col='requests_in_window', title='OTP Requests Over Time'):
    """Plot requests over time"""
    fig = px.line(df, x=time_col, y=value_col, color='client_ip', 
                  title=title, labels={value_col: 'Number of Requests', time_col: 'Time'})
    fig.update_layout(height=400)
    return fig

def plot_anomalies(anomalies_df):
    """Plot anomalies"""
    if anomalies_df.empty:
        return None
        
    fig = px.scatter(anomalies_df, x='datetime', y='requests_in_window', 
                     color='client_ip', size='requests_in_window',
                     title='OTP Request Anomalies', 
                     labels={'requests_in_window': 'Requests in 10-min Window', 'datetime': 'Time'})
    fig.update_layout(height=500)
    return fig

def plot_ip_heatmap(ip_analysis_df):
    """Create a heatmap of IP activity"""
    if ip_analysis_df.empty:
        return None
        
    fig = px.density_heatmap(ip_analysis_df, x='days_active', y='total_requests', 
                             title='IP Activity Heatmap',
                             labels={'days_active': 'Days Active', 'total_requests': 'Total Requests'})
    fig.update_layout(height=400)
    return fig

# Main application
def main():
    st.title("🔒 OTP Request Anomaly Detection System")
    st.markdown("""
    This system detects suspicious OTP request patterns based on multiple criteria:
    - More than 10 OTP requests from the same IP in 10 minutes
    - Requests through proxy servers (akamai_epd column populated)
    - Multiple requests from the same device ID
    - Unusual IP frequency patterns
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file with login data", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state.data is None or not st.session_state.processed:
            with st.spinner("Loading and processing data..."):
                df = load_data(uploaded_file)
                if df is not None:
                    st.session_state.data = df
                    
                    # Initialize database
                    conn = init_db()
                    conn = setup_db(conn, df)
                    st.session_state.conn = conn
                    
                    # Detect anomalies
                    st.session_state.anomalies = detect_otp_anomalies(conn)
                    st.session_state.processed = True
    
    if st.session_state.processed and st.session_state.data is not None:
        df = st.session_state.data
        anomalies = st.session_state.anomalies
        conn = st.session_state.conn
        
        # Display basic stats
        col1, col2, col3, col4 = st.columns(4)
        total_requests = len(df)
        otp_requests = len(df[df['is_otp_request']])
        unique_ips = df['client_ip'].nunique()
        anomaly_count = len(anomalies) if not anomalies.empty else 0
        
        with col1:
            st.metric("Total Requests", total_requests)
        with col2:
            st.metric("OTP Requests", otp_requests)
        with col3:
            st.metric("Unique IPs", unique_ips)
        with col4:
            st.metric("Anomalies Detected", anomaly_count)
        
        # Display tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Anomaly Overview", 
            "Time Analysis", 
            "IP Analysis", 
            "Device Analysis", 
            "Proxy Analysis"
        ])
        
        with tab1:
            st.header("Anomaly Overview")
            
            if not anomalies.empty:
                st.dataframe(anomalies, use_container_width=True)
                
                # Plot anomalies
                fig = plot_anomalies(anomalies)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Show top anomalous IPs
                st.subheader("Top Anomalous IPs")
                top_ips = anomalies['client_ip'].value_counts().head(10)
                fig = px.bar(x=top_ips.index, y=top_ips.values, 
                            title="Top IPs with Anomalies",
                            labels={'x': 'IP Address', 'y': 'Number of Anomalies'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No anomalies detected!")
                
        with tab2:
            st.header("Time Analysis")
            
            # Minute bucket analysis
            minute_analysis = get_minute_bucket_analysis(conn)
            if not minute_analysis.empty:
                st.subheader("Requests by Minute Buckets")
                st.dataframe(minute_analysis, use_container_width=True)
                
                # Plot requests over time
                time_agg = minute_analysis.groupby('minute_bucket').agg({
                    'requests_in_minute': 'sum',
                    'requests_in_window': 'mean'
                }).reset_index()
                time_agg['minute_bucket'] = pd.to_datetime(time_agg['minute_bucket'])
                
                fig = plot_requests_over_time(time_agg, 'minute_bucket', 'requests_in_minute', 
                                             'Total OTP Requests per Minute')
                st.plotly_chart(fig, use_container_width=True)
                
                fig2 = plot_requests_over_time(time_agg, 'minute_bucket', 'requests_in_window', 
                                              'Average Requests in 10-min Window per Minute')
                st.plotly_chart(fig2, use_container_width=True)
                
        with tab3:
            st.header("IP Analysis")
            
            ip_analysis = get_ip_frequency_analysis(conn)
            if not ip_analysis.empty:
                st.dataframe(ip_analysis, use_container_width=True)
                
                # Plot IP activity heatmap
                fig = plot_ip_heatmap(ip_analysis)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Show IPs with high daily averages
                high_avg_ips = ip_analysis[ip_analysis['avg_daily_requests'] > ip_analysis['avg_daily_requests'].median() * 2]
                if not high_avg_ips.empty:
                    st.subheader("IPs with High Daily Request Averages")
                    st.dataframe(high_avg_ips, use_container_width=True)
                    
        with tab4:
            st.header("Device Analysis")
            
            device_analysis = get_device_analysis(conn)
            if not device_analysis.empty:
                st.dataframe(device_analysis, use_container_width=True)
                
                # Show devices with multiple IPs
                multi_ip_devices = device_analysis[device_analysis['unique_ips'] > 1]
                if not multi_ip_devices.empty:
                    st.subheader("Devices Associated with Multiple IPs")
                    st.dataframe(multi_ip_devices, use_container_width=True)
                    
        with tab5:
            st.header("Proxy Analysis")
            
            proxy_analysis = get_proxy_analysis(conn)
            if not proxy_analysis.empty:
                st.dataframe(proxy_analysis, use_container_width=True)
                
                # Show IPs with high proxy usage
                high_proxy_ips = proxy_analysis[proxy_analysis['proxy_requests'] > 0]
                if not high_proxy_ips.empty:
                    st.subheader("IPs Using Proxy Servers")
                    
                    for _, row in high_proxy_ips.iterrows():
                        st.markdown(f"""
                        <div class="anomaly-alert">
                            <strong>IP: {row['client_ip']}</strong><br>
                            Total Requests: {row['total_requests']}<br>
                            Proxy Requests: {row['proxy_requests']}<br>
                            First Seen: {row['first_seen']}<br>
                            Last Seen: {row['last_seen']}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Download results
        st.sidebar.header("Download Results")
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        if not anomalies.empty:
            csv = convert_df(anomalies)
            st.sidebar.download_button(
                label="Download Anomalies as CSV",
                data=csv,
                file_name="otp_anomalies.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
