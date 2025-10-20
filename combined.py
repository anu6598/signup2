# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from datetime import datetime, timedelta
import re
import json

st.set_page_config(page_title="Brute Force Detection Dashboard", layout="wide")

# ------------------------------
# Improved Brute Force Detection Functions
# ------------------------------
def normalize_dataframe(df_raw):
    """Normalize column names and ensure consistent formatting"""
    cols_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_raw.columns}
    df_raw.rename(columns=cols_map, inplace=True)
    df = df_raw.copy()

    # Pick columns dynamically
    def pick_col(possible):
        for p in possible:
            if p in df.columns:
                return p
        return None

    # Find time column
    col_time = pick_col(["start_time", "timestamp", "time", "created_on", "date"])
    col_ip = pick_col(["x_real_ip", "client_ip", "ip", "remote_addr", "true_client_ip"])
    col_device = pick_col(["dr_dv", "device_id", "device"])
    col_akamai_epd = pick_col(["akamai_epd", "epd", "akamai_proxy"])
    col_user = pick_col(["username", "user", "user_id", "email"])

    # Set time column - use the first available time column
    if col_time:
        df["timestamp"] = pd.to_datetime(df[col_time], errors="coerce")
    else:
        # If no time column found, create a dummy one
        df["timestamp"] = pd.to_datetime("today")
    
    df["ip_address"] = df[col_ip].astype(str) if col_ip else "unknown"
    df["device_id"] = df[col_device].astype(str) if col_device else np.nan
    df["username"] = df[col_user].astype(str) if col_user else np.nan
    df["akamai_epd"] = df[col_akamai_epd] if col_akamai_epd else np.nan
    df["is_proxy"] = df["akamai_epd"].notna() & (df["akamai_epd"] != "")

    return df

def analyze_brute_force_patterns(df):
    """
    Comprehensive brute force detection with hourly analysis
    Returns aggregated patterns and suspicious IPs
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Ensure timestamp is parsed
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Create hourly time buckets
    df['hour_bucket'] = df['timestamp'].dt.floor('H')
    df['date'] = df['timestamp'].dt.date
    
    # 🎯 HOURLY IP ANALYSIS
    hourly_ip_analysis = df.groupby(['date', 'hour_bucket', 'ip_address']).agg({
        'timestamp': 'count',
        'username': 'nunique',
        'device_id': 'nunique',
        'is_proxy': 'mean'
    }).reset_index()
    
    hourly_ip_analysis.columns = ['date', 'hour_bucket', 'ip_address', 'request_count', 
                                 'unique_users', 'unique_devices', 'proxy_ratio']
    
    # 🎯 BRUTE FORCE DETECTION RULES
    suspicious_ips = []
    
    for _, row in hourly_ip_analysis.iterrows():
        ip = row['ip_address']
        hour_data = hourly_ip_analysis[hourly_ip_analysis['ip_address'] == ip]
        
        # Rule 1: High request volume per hour
        volume_score = 0
        if row['request_count'] > 50:
            volume_score = 3
        elif row['request_count'] > 25:
            volume_score = 2
        elif row['request_count'] > 10:
            volume_score = 1
        
        # Rule 2: Multiple users from same IP (credential stuffing)
        user_score = 0
        if row['unique_users'] > 5:
            user_score = 3
        elif row['unique_users'] > 3:
            user_score = 2
        elif row['unique_users'] > 1:
            user_score = 1
        
        # Rule 3: Multiple devices from same IP
        device_score = 0
        if row['unique_devices'] > 3:
            device_score = 2
        elif row['unique_devices'] > 1:
            device_score = 1
        
        # Rule 4: Proxy usage
        proxy_score = 1 if row['proxy_ratio'] > 0.5 else 0
        
        # Rule 5: Sustained activity across multiple hours
        hour_count = len(hour_data)
        sustained_score = min(2, hour_count // 3)  # 2 points for 6+ hours
        
        total_score = volume_score + user_score + device_score + proxy_score + sustained_score
        
        if total_score >= 4:  # Threshold for suspicion
            suspicious_ips.append({
                'ip_address': ip,
                'date': row['date'],
                'hour_bucket': row['hour_bucket'],
                'request_count': row['request_count'],
                'unique_users': row['unique_users'],
                'unique_devices': row['unique_devices'],
                'proxy_ratio': f"{row['proxy_ratio']:.1%}",
                'volume_score': volume_score,
                'user_score': user_score,
                'device_score': device_score,
                'proxy_score': proxy_score,
                'sustained_score': sustained_score,
                'total_risk_score': total_score,
                'risk_level': 'CRITICAL' if total_score >= 6 else 'HIGH' if total_score >= 4 else 'MEDIUM'
            })
    
    suspicious_df = pd.DataFrame(suspicious_ips)
    
    return hourly_ip_analysis, suspicious_df

def detect_advanced_patterns(df, hourly_analysis):
    """Detect advanced brute force patterns"""
    patterns = []
    
    # Pattern 1: Rapid succession attempts (same IP, different users)
    user_switching = df.groupby(['ip_address', 'hour_bucket']).agg({
        'username': ['count', 'nunique']
    }).reset_index()
    user_switching.columns = ['ip_address', 'hour_bucket', 'total_attempts', 'unique_users']
    user_switching['user_switch_ratio'] = user_switching['unique_users'] / user_switching['total_attempts']
    
    high_switching = user_switching[
        (user_switching['user_switch_ratio'] > 0.7) & 
        (user_switching['total_attempts'] > 5)
    ]
    
    for _, row in high_switching.iterrows():
        patterns.append({
            'pattern_type': 'CREDENTIAL_STUFFING',
            'ip_address': row['ip_address'],
            'hour_bucket': row['hour_bucket'],
            'metric': f"High user switching ({row['user_switch_ratio']:.1%})",
            'attempts': row['total_attempts'],
            'unique_users': row['unique_users']
        })
    
    # Pattern 2: Distributed attacks (multiple IPs, same user)
    if 'username' in df.columns:
        user_targeting = df.groupby(['username', 'hour_bucket']).agg({
            'ip_address': 'nunique'
        }).reset_index()
        user_targeting.columns = ['username', 'hour_bucket', 'unique_ips']
        
        high_targeting = user_targeting[user_targeting['unique_ips'] > 3]
        
        for _, row in high_targeting.iterrows():
            patterns.append({
                'pattern_type': 'DISTRIBUTED_ATTACK',
                'username': row['username'],
                'hour_bucket': row['hour_bucket'],
                'metric': f"Multiple source IPs ({row['unique_ips']})",
                'unique_ips': row['unique_ips']
            })
    
    return pd.DataFrame(patterns)

def create_brute_force_dashboard(hourly_analysis, suspicious_ips, advanced_patterns):
    """Create comprehensive brute force detection dashboard"""
    
    st.header("🚨 Brute Force Attack Detection")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_suspicious = len(suspicious_ips)
        st.metric("Suspicious IPs", total_suspicious)
    
    with col2:
        critical_ips = len(suspicious_ips[suspicious_ips['risk_level'] == 'CRITICAL'])
        st.metric("Critical IPs", critical_ips)
    
    with col3:
        total_patterns = len(advanced_patterns)
        st.metric("Attack Patterns", total_patterns)
    
    with col4:
        if not hourly_analysis.empty:
            avg_requests = hourly_analysis['request_count'].mean()
            st.metric("Avg Requests/IP/Hour", f"{avg_requests:.1f}")
    
    st.markdown("---")
    
    # Suspicious IPs Table
    st.subheader("🔍 Suspicious IPs - Hourly Analysis")
    if not suspicious_ips.empty:
        # Sort by risk score and show details
        display_cols = ['ip_address', 'hour_bucket', 'request_count', 'unique_users', 
                       'unique_devices', 'proxy_ratio', 'total_risk_score', 'risk_level']
        
        suspicious_display = suspicious_ips[display_cols].sort_values(['total_risk_score', 'request_count'], ascending=False)
        st.dataframe(suspicious_display, use_container_width=True)
        
        # Risk explanation
        st.info("""
        **Risk Score Explanation:** 
        - Volume (0-3): Requests per hour [>50=3, >25=2, >10=1]
        - Users (0-3): Unique users per IP [>5=3, >3=2, >1=1]  
        - Devices (0-2): Unique devices [>3=2, >1=1]
        - Proxy (0-1): Proxy usage [>50%=1]
        - Sustained (0-2): Activity across hours [6+ hours=2]
        """)
    else:
        st.success("✅ No suspicious IPs detected in hourly analysis")
    
    # Advanced Patterns
    if not advanced_patterns.empty:
        st.subheader("🎯 Advanced Attack Patterns")
        st.dataframe(advanced_patterns, use_container_width=True)
    
    # Hourly Trends Visualization
    if not hourly_analysis.empty:
        st.subheader("📈 Hourly Request Trends")
        
        # Aggregate by hour
        hourly_trends = hourly_analysis.groupby('hour_bucket').agg({
            'request_count': 'sum',
            'ip_address': 'nunique'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_requests = px.line(hourly_trends, x='hour_bucket', y='request_count',
                                 title='Total Requests by Hour')
            st.plotly_chart(fig_requests, use_container_width=True)
        
        with col2:
            fig_ips = px.line(hourly_trends, x='hour_bucket', y='ip_address',
                            title='Unique IPs by Hour')
            st.plotly_chart(fig_ips, use_container_width=True)

def create_ip_behavior_analysis(df, top_n=20):
    """Analyze IP behavior patterns"""
    if df.empty or 'ip_address' not in df.columns:
        return pd.DataFrame()
    
    # Group by IP and analyze behavior
    ip_behavior = df.groupby('ip_address').agg({
        'timestamp': ['count', lambda x: (x.max() - x.min()).total_seconds() / 3600],  # activity duration in hours
        'username': 'nunique',
        'device_id': 'nunique',
        'is_proxy': 'mean',
        'hour_bucket': 'nunique'
    }).reset_index()
    
    ip_behavior.columns = ['ip_address', 'total_requests', 'activity_hours', 
                          'unique_users', 'unique_devices', 'proxy_ratio', 'unique_hours']
    
    # Calculate requests per hour
    ip_behavior['requests_per_hour'] = ip_behavior['total_requests'] / ip_behavior['activity_hours'].clip(lower=1)
    
    # Score based on multiple factors
    ip_behavior['volume_score'] = np.where(ip_behavior['total_requests'] > 100, 3,
                                         np.where(ip_behavior['total_requests'] > 50, 2,
                                                np.where(ip_behavior['total_requests'] > 20, 1, 0)))
    
    ip_behavior['user_diversity_score'] = np.where(ip_behavior['unique_users'] > 10, 3,
                                                 np.where(ip_behavior['unique_users'] > 5, 2,
                                                        np.where(ip_behavior['unique_users'] > 1, 1, 0)))
    
    ip_behavior['sustained_score'] = np.where(ip_behavior['unique_hours'] > 6, 2,
                                            np.where(ip_behavior['unique_hours'] > 3, 1, 0))
    
    ip_behavior['total_risk_score'] = (ip_behavior['volume_score'] + 
                                     ip_behavior['user_diversity_score'] + 
                                     ip_behavior['sustained_score'])
    
    ip_behavior['risk_level'] = np.where(ip_behavior['total_risk_score'] >= 6, 'CRITICAL',
                                       np.where(ip_behavior['total_risk_score'] >= 4, 'HIGH',
                                              np.where(ip_behavior['total_risk_score'] >= 2, 'MEDIUM', 'LOW')))
    
    return ip_behavior.sort_values('total_risk_score', ascending=False).head(top_n)

# ------------------------------
# Main Application
# ------------------------------
def main():
    st.sidebar.title("🔐 Brute Force Detection Dashboard")
    
    # Initialize session state
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    # File upload
    st.sidebar.header("📁 Upload Security Data")
    uploaded_file = st.sidebar.file_uploader("Upload Authentication Data (CSV)", type=["csv"])
    
    if uploaded_file:
        try:
            df = normalize_dataframe(pd.read_csv(uploaded_file))
            st.session_state.current_df = df
            st.sidebar.success("✅ Data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")
    
    # Analysis controls
    st.sidebar.header("⚙️ Detection Settings")
    min_requests = st.sidebar.number_input("Minimum requests for analysis", value=5, step=1)
    risk_threshold = st.sidebar.select_slider("Risk threshold", options=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], value='MEDIUM')
    
    # Main display
    if st.session_state.current_df is not None:
        display_analysis_dashboard(st.session_state.current_df, min_requests, risk_threshold)
    else:
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen with instructions"""
    st.title("🔐 Brute Force Detection Dashboard")
    
    st.markdown("""
    ## Advanced Brute Force Attack Detection
    
    This dashboard provides comprehensive brute force detection for authentication data:
    - **Hourly analysis** of IP behavior patterns
    - **Multi-factor risk scoring** (volume, users, devices, duration, proxy)
    - **Advanced pattern detection** (credential stuffing, distributed attacks)
    - **Real-time risk assessment** with actionable insights
    
    ### 🚀 Getting Started
    1. **Upload authentication data** (login or signup CSV)
    2. **Adjust detection sensitivity** in sidebar
    3. **View comprehensive analysis** with hourly patterns
    4. **Identify suspicious IPs** and attack patterns
    
    ### 📊 Detection Methodology
    - **Hourly bucketing** for temporal analysis
    - **Risk scoring** across multiple dimensions
    - **Pattern recognition** for advanced attacks
    - **Proxy detection** and behavioral analysis
    """)

def display_analysis_dashboard(df, min_requests, risk_threshold):
    """Display comprehensive brute force analysis"""
    st.title("🔐 Brute Force Attack Analysis")
    
    # Show data overview
    st.subheader("📋 Data Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
        if 'ip_address' in df.columns:
            st.metric("Unique IPs", f"{df['ip_address'].nunique():,}")
    
    with col2:
        if 'username' in df.columns:
            st.metric("Unique Users", f"{df['username'].nunique():,}")
        if 'device_id' in df.columns:
            st.metric("Unique Devices", f"{df['device_id'].nunique():,}")
    
    with col3:
        if 'timestamp' in df.columns:
            time_range = df['timestamp'].max() - df['timestamp'].min()
            st.metric("Time Range", f"{time_range.days} days, {time_range.seconds//3600} hours")
    
    st.markdown("---")
    
    # Run brute force analysis
    with st.spinner("🔍 Analyzing brute force patterns..."):
        hourly_analysis, suspicious_ips = analyze_brute_force_patterns(df)
        advanced_patterns = detect_advanced_patterns(df, hourly_analysis)
        ip_behavior = create_ip_behavior_analysis(df)
    
    # Display main dashboard
    create_brute_force_dashboard(hourly_analysis, suspicious_ips, advanced_patterns)
    
    # IP Behavior Analysis
    if not ip_behavior.empty:
        st.markdown("---")
        st.header("📊 IP Behavior Analysis (Top 20)")
        
        display_cols = ['ip_address', 'total_requests', 'activity_hours', 'requests_per_hour',
                       'unique_users', 'unique_devices', 'proxy_ratio', 'total_risk_score', 'risk_level']
        
        st.dataframe(ip_behavior[display_cols], use_container_width=True)
        
        # Risk distribution
        risk_dist = ip_behavior['risk_level'].value_counts()
        st.plotly_chart(px.pie(values=risk_dist.values, names=risk_dist.index, 
                             title="Risk Level Distribution"), use_container_width=True)
    
    # Raw data preview
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
