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
    col_platform = pick_col(["platform", "app_version", "source", "client_type"])
    col_extra_data = pick_col(["extra_data_str", "extra_data", "additional_data"])

    # Set time column - use the first available time column
    if col_time:
        df["timestamp"] = pd.to_datetime(df[col_time], errors="coerce")
    else:
        # If no time column found, create a dummy one
        df["timestamp"] = pd.to_datetime("today")
    
    df["ip_address"] = df[col_ip].astype(str) if col_ip else "unknown"
    df["true_client_ip"] = df["ip_address"]  # Ensure true_client_ip exists
    df["device_id"] = df[col_device].astype(str) if col_device else np.nan
    df["username"] = df[col_user].astype(str) if col_user else np.nan
    df["akamai_epd"] = df[col_akamai_epd] if col_akamai_epd else np.nan
    df["platform"] = df[col_platform].astype(str) if col_platform else "unknown"
    df["is_proxy"] = df["akamai_epd"].notna() & (df["akamai_epd"] != "")
    
    # Parse extra_data_str if it exists
    if col_extra_data:
        df = parse_extra_data(df, col_extra_data)
    else:
        # Initialize recaptcha columns with default values if extra_data_str doesn't exist
        df = initialize_recaptcha_columns(df)

    return df

def parse_extra_data(df, extra_data_col):
    """Parse JSON data from extra_data_str column"""
    try:
        # Parse JSON strings
        df[extra_data_col] = df[extra_data_col].fillna('{}')
        extra_data = df[extra_data_col].apply(lambda x: json.loads(x) if isinstance(x, str) and x.strip() else {})
        
        # Extract key fields
        df['recaptcha_valid'] = extra_data.apply(lambda x: x.get('valid', True))
        df['recaptcha_actions'] = extra_data.apply(lambda x: x.get('recaptcha_actions', []))
        df['response_action'] = extra_data.apply(lambda x: x.get('response_action', ''))
        df['risk_analysis_score'] = extra_data.apply(lambda x: x.get('risk_analysis_score', 0.0))
        df['reasons'] = extra_data.apply(lambda x: x.get('reasons', ''))
        df['invalid_reason'] = extra_data.apply(lambda x: x.get('invalid_reason', ''))
        df['country_code'] = extra_data.apply(lambda x: x.get('country_code', ''))
        
        # Create risk flags based on recaptcha data
        df['high_risk_recaptcha'] = df['risk_analysis_score'] > 0.7
        df['suspicious_environment'] = df['reasons'].str.contains('UNEXPECTED_ENVIRONMENT', na=False)
        df['failed_recaptcha'] = df['recaptcha_valid'] == False
        
    except Exception as e:
        st.warning(f"Could not parse extra_data_str: {str(e)}")
        # Initialize with default values if parsing fails
        df = initialize_recaptcha_columns(df)
    
    return df

def initialize_recaptcha_columns(df):
    """Initialize recaptcha columns with default values"""
    df['recaptcha_valid'] = True
    df['recaptcha_actions'] = [[] for _ in range(len(df))]
    df['response_action'] = ''
    df['risk_analysis_score'] = 0.0
    df['reasons'] = ''
    df['invalid_reason'] = ''
    df['country_code'] = ''
    df['high_risk_recaptcha'] = False
    df['suspicious_environment'] = False
    df['failed_recaptcha'] = False
    return df

def prepare_hourly_data(df):
    """Prepare hourly data for analysis"""
    if df is None or df.empty:
        return df
    
    # Ensure timestamp is parsed
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    
    # Create hourly time buckets
    df['hour_bucket'] = df['timestamp'].dt.floor('H')
    df['date'] = df['timestamp'].dt.date
    
    return df

def analyze_brute_force_patterns(df):
    """
    Comprehensive brute force detection with hourly analysis
    Returns aggregated patterns and suspicious IPs
    """
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Prepare data with hourly buckets
    df = prepare_hourly_data(df)
    
    # Define aggregation columns safely
    agg_columns = {
        'timestamp': 'count',
        'username': 'nunique',
        'device_id': 'nunique',
        'is_proxy': 'mean'
    }
    
    # Add recaptcha columns only if they exist
    if 'risk_analysis_score' in df.columns:
        agg_columns['risk_analysis_score'] = 'mean'
    if 'high_risk_recaptcha' in df.columns:
        agg_columns['high_risk_recaptcha'] = 'sum'
    if 'suspicious_environment' in df.columns:
        agg_columns['suspicious_environment'] = 'sum'
    if 'failed_recaptcha' in df.columns:
        agg_columns['failed_recaptcha'] = 'sum'
    
    # 🎯 HOURLY IP ANALYSIS
    hourly_ip_analysis = df.groupby(['date', 'hour_bucket', 'ip_address', 'platform']).agg(agg_columns).reset_index()
    
    # Rename columns safely
    base_columns = ['date', 'hour_bucket', 'ip_address', 'platform', 'request_count', 
                   'unique_users', 'unique_devices', 'proxy_ratio']
    
    additional_columns = []
    if 'risk_analysis_score' in agg_columns:
        additional_columns.append('avg_risk_score')
    if 'high_risk_recaptcha' in agg_columns:
        additional_columns.append('high_risk_count')
    if 'suspicious_environment' in agg_columns:
        additional_columns.append('suspicious_env_count')
    if 'failed_recaptcha' in agg_columns:
        additional_columns.append('failed_recaptcha_count')
    
    hourly_ip_analysis.columns = base_columns + additional_columns
    
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
        
        # Rule 6: Recaptcha risk scoring (only if available)
        recaptcha_score = 0
        failed_recaptcha_score = 0
        
        if 'avg_risk_score' in row:
            if row['avg_risk_score'] > 0.7:
                recaptcha_score = 3
            elif row['avg_risk_score'] > 0.4:
                recaptcha_score = 2
            elif row['avg_risk_score'] > 0.2:
                recaptcha_score = 1
        
        if 'failed_recaptcha_count' in row:
            failed_recaptcha_score = min(2, row['failed_recaptcha_count'])
        
        total_score = (volume_score + user_score + device_score + proxy_score + 
                      sustained_score + recaptcha_score + failed_recaptcha_score)
        
        if total_score >= 4:  # Threshold for suspicion
            ip_data = {
                'ip_address': ip,
                'platform': row['platform'],
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
                'recaptcha_score': recaptcha_score,
                'failed_recaptcha_score': failed_recaptcha_score,
                'total_risk_score': total_score,
                'risk_level': 'CRITICAL' if total_score >= 8 else 'HIGH' if total_score >= 5 else 'MEDIUM'
            }
            
            # Add recaptcha data only if available
            if 'avg_risk_score' in row:
                ip_data['avg_risk_score'] = f"{row['avg_risk_score']:.3f}"
            if 'high_risk_count' in row:
                ip_data['high_risk_count'] = row['high_risk_count']
            if 'suspicious_env_count' in row:
                ip_data['suspicious_env_count'] = row['suspicious_env_count']
            if 'failed_recaptcha_count' in row:
                ip_data['failed_recaptcha_count'] = row['failed_recaptcha_count']
            
            suspicious_ips.append(ip_data)
    
    suspicious_df = pd.DataFrame(suspicious_ips)
    
    return hourly_ip_analysis, suspicious_df

def detect_advanced_patterns(df):
    """Detect advanced brute force patterns using hourly data"""
    patterns = []
    
    # Prepare data first
    df = prepare_hourly_data(df)
    
    # Pattern 1: Rapid succession attempts (same IP, different users)
    if 'username' in df.columns:
        agg_columns = {
            'username': ['count', 'nunique'],
            'device_id': 'nunique'
        }
        if 'risk_analysis_score' in df.columns:
            agg_columns['risk_analysis_score'] = 'mean'
            
        user_switching = df.groupby(['ip_address', 'hour_bucket', 'platform']).agg(agg_columns).reset_index()
        
        # Handle column names safely
        if 'risk_analysis_score' in agg_columns:
            user_switching.columns = ['ip_address', 'hour_bucket', 'platform', 'total_attempts', 'unique_users', 'unique_devices', 'avg_risk_score']
        else:
            user_switching.columns = ['ip_address', 'hour_bucket', 'platform', 'total_attempts', 'unique_users', 'unique_devices']
            user_switching['avg_risk_score'] = 0.0
            
        user_switching['user_switch_ratio'] = user_switching['unique_users'] / user_switching['total_attempts']
        
        high_switching = user_switching[
            (user_switching['user_switch_ratio'] > 0.7) & 
            (user_switching['total_attempts'] > 5)
        ]
        
        for _, row in high_switching.iterrows():
            pattern_data = {
                'pattern_type': 'CREDENTIAL_STUFFING',
                'ip_address': row['ip_address'],
                'platform': row['platform'],
                'hour_bucket': row['hour_bucket'],
                'metric': f"High user switching ({row['user_switch_ratio']:.1%})",
                'attempts': row['total_attempts'],
                'unique_users': row['unique_users'],
                'unique_devices': row['unique_devices']
            }
            if 'avg_risk_score' in row:
                pattern_data['avg_risk_score'] = f"{row['avg_risk_score']:.3f}"
            patterns.append(pattern_data)
    
    # Pattern 2: Distributed attacks (multiple IPs, same user)
    if 'username' in df.columns:
        agg_columns = {
            'ip_address': 'nunique',
            'device_id': 'nunique'
        }
        if 'risk_analysis_score' in df.columns:
            agg_columns['risk_analysis_score'] = 'mean'
            
        user_targeting = df.groupby(['username', 'hour_bucket', 'platform']).agg(agg_columns).reset_index()
        
        if 'risk_analysis_score' in agg_columns:
            user_targeting.columns = ['username', 'hour_bucket', 'platform', 'unique_ips', 'unique_devices', 'avg_risk_score']
        else:
            user_targeting.columns = ['username', 'hour_bucket', 'platform', 'unique_ips', 'unique_devices']
            user_targeting['avg_risk_score'] = 0.0
        
        high_targeting = user_targeting[user_targeting['unique_ips'] > 3]
        
        for _, row in high_targeting.iterrows():
            pattern_data = {
                'pattern_type': 'DISTRIBUTED_ATTACK',
                'username': row['username'],
                'platform': row['platform'],
                'hour_bucket': row['hour_bucket'],
                'metric': f"Multiple source IPs ({row['unique_ips']})",
                'unique_ips': row['unique_ips'],
                'unique_devices': row['unique_devices']
            }
            if 'avg_risk_score' in row:
                pattern_data['avg_risk_score'] = f"{row['avg_risk_score']:.3f}"
            patterns.append(pattern_data)
    
    # Pattern 3: High frequency attempts from single IP
    agg_columns = {
        'timestamp': 'count',
        'device_id': 'nunique'
    }
    if 'risk_analysis_score' in df.columns:
        agg_columns['risk_analysis_score'] = 'mean'
    if 'failed_recaptcha' in df.columns:
        agg_columns['failed_recaptcha'] = 'sum'
        
    ip_frequency = df.groupby(['ip_address', 'hour_bucket', 'platform']).agg(agg_columns).reset_index()
    
    # Handle column names based on available columns
    column_names = ['ip_address', 'hour_bucket', 'platform', 'attempt_count', 'unique_devices']
    if 'risk_analysis_score' in agg_columns:
        column_names.append('avg_risk_score')
    if 'failed_recaptcha' in agg_columns:
        column_names.append('failed_recaptcha_count')
        
    ip_frequency.columns = column_names
    
    # Add missing columns with default values
    if 'avg_risk_score' not in ip_frequency.columns:
        ip_frequency['avg_risk_score'] = 0.0
    if 'failed_recaptcha_count' not in ip_frequency.columns:
        ip_frequency['failed_recaptcha_count'] = 0
    
    high_frequency = ip_frequency[ip_frequency['attempt_count'] > 20]
    
    for _, row in high_frequency.iterrows():
        patterns.append({
            'pattern_type': 'HIGH_FREQUENCY',
            'ip_address': row['ip_address'],
            'platform': row['platform'],
            'hour_bucket': row['hour_bucket'],
            'metric': f"High attempt frequency ({row['attempt_count']} requests)",
            'attempt_count': row['attempt_count'],
            'unique_devices': row['unique_devices'],
            'avg_risk_score': f"{row['avg_risk_score']:.3f}",
            'failed_recaptcha_count': row['failed_recaptcha_count']
        })
    
    # Pattern 4: High risk recaptcha patterns (only if columns exist)
    if all(col in df.columns for col in ['high_risk_recaptcha', 'suspicious_environment', 'risk_analysis_score']):
        agg_columns = {
            'high_risk_recaptcha': 'sum',
            'suspicious_environment': 'sum',
            'risk_analysis_score': 'max',
            'device_id': 'nunique'
        }
        
        high_risk_patterns = df.groupby(['ip_address', 'hour_bucket', 'platform']).agg(agg_columns).reset_index()
        high_risk_patterns.columns = ['ip_address', 'hour_bucket', 'platform', 'high_risk_count', 'suspicious_env_count', 'max_risk_score', 'unique_devices']
        
        suspicious_recaptcha = high_risk_patterns[
            (high_risk_patterns['high_risk_count'] > 0) | 
            (high_risk_patterns['suspicious_env_count'] > 0)
        ]
        
        for _, row in suspicious_recaptcha.iterrows():
            patterns.append({
                'pattern_type': 'SUSPICIOUS_RECAPTCHA',
                'ip_address': row['ip_address'],
                'platform': row['platform'],
                'hour_bucket': row['hour_bucket'],
                'metric': f"Suspicious recaptcha activity (risk: {row['max_risk_score']:.3f})",
                'high_risk_count': row['high_risk_count'],
                'suspicious_env_count': row['suspicious_env_count'],
                'unique_devices': row['unique_devices'],
                'max_risk_score': f"{row['max_risk_score']:.3f}"
            })
    
    return pd.DataFrame(patterns)

def create_rule_categorization_table(df):
    """Create daily categorization table with OTP-specific rules"""
    if df is None or df.empty:
        return []
    
    # Prepare data with dates
    df = prepare_hourly_data(df)
    
    final_categories = []
    
    # Group by date for daily analysis
    date_groups = df.groupby('date')
    
    for day, group in date_groups:
        total_otps = len(group)
        
        # Calculate metrics for rule categorization
        max_requests_ip = group['ip_address'].value_counts().max() if 'ip_address' in group.columns else 0
        max_requests_device = group['device_id'].value_counts().max() if 'device_id' in group.columns else 0
        
        # Initialize recaptcha metrics with defaults
        avg_risk_score = 0.0
        high_risk_count = 0
        suspicious_env_count = 0
        
        # Only calculate if columns exist
        if 'risk_analysis_score' in group.columns:
            avg_risk_score = group['risk_analysis_score'].mean()
        if 'high_risk_recaptcha' in group.columns:
            high_risk_count = group['high_risk_recaptcha'].sum()
        if 'suspicious_environment' in group.columns:
            suspicious_env_count = group['suspicious_environment'].sum()
        
        # Proxy ratio calculation
        if "akamai_epd" in group.columns:
            epd_norm = group["akamai_epd"].astype(str).str.strip().str.lower()
            proxy_ratio = (~epd_norm.isin(["-", "rp", "", "nan"])).mean() * 100
        else:
            proxy_ratio = 0

        # 🎯 OTP RULE CATEGORIZATION
        if (total_otps > 1000) and (max_requests_ip > 25) and (proxy_ratio > 20) and (max_requests_device > 15):
            category = "OTP Abuse/Attack detected"
            risk_level = "CRITICAL"
        elif (max_requests_ip > 25) and (total_otps > 1000) and (max_requests_device > 15):
            category = "HIGH OTP request detected"
            risk_level = "HIGH"
        elif proxy_ratio > 20:
            category = "HIGH proxy status detected"
            risk_level = "HIGH"
        elif high_risk_count > 10 or avg_risk_score > 0.7:
            category = "HIGH recaptcha risk detected"
            risk_level = "HIGH"
        elif suspicious_env_count > 5:
            category = "Suspicious environment detected"
            risk_level = "MEDIUM"
        else:
            category = "No suspicious activity detected"
            risk_level = "NORMAL"

        category_data = {
            "date": day,
            "category": category,
            "risk_level": risk_level,
            "total_otps": total_otps,
            "max_requests_ip": max_requests_ip,
            "max_requests_device": max_requests_device,
            "proxy_ratio": f"{proxy_ratio:.1f}%",
            "avg_risk_score": f"{avg_risk_score:.3f}",
            "high_risk_recaptcha": high_risk_count,
            "suspicious_environment": suspicious_env_count
        }

        final_categories.append(category_data)

    return final_categories

def create_brute_force_dashboard(hourly_analysis, suspicious_ips, advanced_patterns, categorization):
    """Create comprehensive brute force detection dashboard"""
    
    st.header("🚨 Brute Force Attack Detection")
    
    # Key Metrics with safe handling for empty dataframes
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_suspicious = len(suspicious_ips) if not suspicious_ips.empty else 0
        st.metric("Suspicious IPs", total_suspicious)
    
    with col2:
        if not suspicious_ips.empty and 'risk_level' in suspicious_ips.columns:
            critical_ips = len(suspicious_ips[suspicious_ips['risk_level'] == 'CRITICAL'])
        else:
            critical_ips = 0
        st.metric("Critical IPs", critical_ips)
    
    with col3:
        total_patterns = len(advanced_patterns) if not advanced_patterns.empty else 0
        st.metric("Attack Patterns", total_patterns)
    
    with col4:
        if not hourly_analysis.empty and 'avg_risk_score' in hourly_analysis.columns:
            avg_risk = hourly_analysis['avg_risk_score'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
        else:
            st.metric("Avg Risk Score", "0.000")
    
    st.markdown("---")
    
    # Rule Categorization Table
    st.subheader("📊 Daily OTP Rule Categorization")
    if categorization:
        # Display with color coding
        for category in categorization:
            risk_colors = {
                "CRITICAL": "red",
                "HIGH": "orange", 
                "MEDIUM": "yellow",
                "NORMAL": "green"
            }
            
            color = risk_colors.get(category['risk_level'], "gray")
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; margin: 10px 0;">
                <h4 style="color: {color}; margin: 0;">{category['category']}</h4>
                <p style="margin: 5px 0;"><strong>Date:</strong> {category['date']}</p>
                <p style="margin: 5px 0;"><strong>Total OTPs:</strong> {category['total_otps']:,}</p>
                <p style="margin: 5px 0;"><strong>Max Requests per IP:</strong> {category['max_requests_ip']}</p>
                <p style="margin: 5px 0;"><strong>Max Requests per Device:</strong> {category['max_requests_device']}</p>
                <p style="margin: 5px 0;"><strong>Proxy Ratio:</strong> {category['proxy_ratio']}</p>
                <p style="margin: 5px 0;"><strong>Avg Risk Score:</strong> {category['avg_risk_score']}</p>
                <p style="margin: 5px 0;"><strong>High Risk Recaptcha:</strong> {category['high_risk_recaptcha']}</p>
                <p style="margin: 5px 0;"><strong>Suspicious Environment:</strong> {category['suspicious_environment']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Also show as dataframe
        st.subheader("📋 Categorization Summary Table")
        cat_df = pd.DataFrame(categorization)
        st.dataframe(cat_df, use_container_width=True)
    else:
        st.info("No categorization data available.")
    
    st.markdown("---")
    
    # Suspicious IPs Table
    st.subheader("🔍 Suspicious IPs - Hourly Analysis")
    if not suspicious_ips.empty and 'risk_level' in suspicious_ips.columns:
        # Define base columns that should always exist
        base_cols = ['ip_address', 'platform', 'hour_bucket', 'request_count', 'unique_users', 
                    'unique_devices', 'proxy_ratio', 'total_risk_score', 'risk_level']
        
        # Add optional columns only if they exist
        optional_cols = ['avg_risk_score', 'high_risk_count', 'suspicious_env_count', 'failed_recaptcha_count']
        available_cols = base_cols + [col for col in optional_cols if col in suspicious_ips.columns]
        
        suspicious_display = suspicious_ips[available_cols].sort_values(['total_risk_score', 'request_count'], ascending=False)
        st.dataframe(suspicious_display, use_container_width=True)
        
        # Risk explanation
        st.info("""
        **Enhanced Risk Score Explanation:** 
        - Volume (0-3): Requests per hour [>50=3, >25=2, >10=1]
        - Users (0-3): Unique users per IP [>5=3, >3=2, >1=1]  
        - Devices (0-2): Unique devices [>3=2, >1=1]
        - Proxy (0-1): Proxy usage [>50%=1]
        - Sustained (0-2): Activity across hours [6+ hours=2]
        - Recaptcha (0-3): Risk score [>0.7=3, >0.4=2, >0.2=1]
        - Failed Recaptcha (0-2): Failed attempts count
        """)
    else:
        st.success("✅ No suspicious IPs detected in hourly analysis")
    
    # Advanced Patterns with Filter
    if not advanced_patterns.empty and 'pattern_type' in advanced_patterns.columns:
        st.subheader("🎯 Advanced Attack Patterns")
        
        # Pattern type filter
        pattern_types = advanced_patterns['pattern_type'].unique()
        selected_patterns = st.multiselect(
            "Filter by Pattern Type:",
            options=pattern_types,
            default=pattern_types,
            key="pattern_filter"
        )
        
        filtered_patterns = advanced_patterns[advanced_patterns['pattern_type'].isin(selected_patterns)]
        st.dataframe(filtered_patterns, use_container_width=True)
    else:
        st.info("No advanced attack patterns detected")
    
    # Hourly Trends Visualization
    if not hourly_analysis.empty:
        st.subheader("📈 Hourly Trends")
        
        # Aggregate by hour
        hourly_trends = hourly_analysis.groupby('hour_bucket').agg({
            'request_count': 'sum',
            'ip_address': 'nunique'
        }).reset_index()
        
        # Add risk score if available
        if 'avg_risk_score' in hourly_analysis.columns:
            risk_trends = hourly_analysis.groupby('hour_bucket')['avg_risk_score'].mean().reset_index()
            hourly_trends = hourly_trends.merge(risk_trends, on='hour_bucket')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_requests = px.line(hourly_trends, x='hour_bucket', y='request_count',
                                 title='Total Requests by Hour')
            st.plotly_chart(fig_requests, use_container_width=True)
        
        with col2:
            if 'avg_risk_score' in hourly_trends.columns:
                fig_risk = px.line(hourly_trends, x='hour_bucket', y='avg_risk_score',
                                 title='Average Risk Score by Hour')
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                fig_ips = px.line(hourly_trends, x='hour_bucket', y='ip_address',
                                title='Unique IPs by Hour')
                st.plotly_chart(fig_ips, use_container_width=True)

def create_ip_behavior_analysis(df):
    """Analyze IP behavior patterns including recaptcha data"""
    if df.empty or 'ip_address' not in df.columns:
        return pd.DataFrame()
    
    # Prepare data first
    df = prepare_hourly_data(df)
    
    # Define aggregation columns safely
    agg_columns = {
        'timestamp': ['count', lambda x: (x.max() - x.min()).total_seconds() / 3600],
        'username': 'nunique',
        'device_id': 'nunique',
        'is_proxy': 'mean',
        'hour_bucket': 'nunique'
    }
    
    # Add recaptcha columns only if they exist
    if 'risk_analysis_score' in df.columns:
        agg_columns['risk_analysis_score'] = 'mean'
    if 'high_risk_recaptcha' in df.columns:
        agg_columns['high_risk_recaptcha'] = 'sum'
    if 'suspicious_environment' in df.columns:
        agg_columns['suspicious_environment'] = 'sum'
    if 'failed_recaptcha' in df.columns:
        agg_columns['failed_recaptcha'] = 'sum'
    
    # Group by IP and analyze behavior
    ip_behavior = df.groupby(['ip_address', 'platform']).agg(agg_columns).reset_index()
    
    # Handle column names safely
    base_columns = ['ip_address', 'platform', 'total_requests', 'activity_hours', 
                   'unique_users', 'unique_devices', 'proxy_ratio', 'unique_hours']
    
    additional_columns = []
    if 'risk_analysis_score' in agg_columns:
        additional_columns.append('avg_risk_score')
    if 'high_risk_recaptcha' in agg_columns:
        additional_columns.append('high_risk_count')
    if 'suspicious_environment' in agg_columns:
        additional_columns.append('suspicious_env_count')
    if 'failed_recaptcha' in agg_columns:
        additional_columns.append('failed_recaptcha_count')
    
    ip_behavior.columns = base_columns + additional_columns
    
    # Calculate requests per hour
    ip_behavior['requests_per_hour'] = ip_behavior['total_requests'] / ip_behavior['activity_hours'].clip(lower=1)
    
    # Enhanced scoring with recaptcha factors
    ip_behavior['volume_score'] = np.where(ip_behavior['total_requests'] > 100, 3,
                                         np.where(ip_behavior['total_requests'] > 50, 2,
                                                np.where(ip_behavior['total_requests'] > 20, 1, 0)))
    
    ip_behavior['user_diversity_score'] = np.where(ip_behavior['unique_users'] > 10, 3,
                                                 np.where(ip_behavior['unique_users'] > 5, 2,
                                                        np.where(ip_behavior['unique_users'] > 1, 1, 0)))
    
    ip_behavior['sustained_score'] = np.where(ip_behavior['unique_hours'] > 6, 2,
                                            np.where(ip_behavior['unique_hours'] > 3, 1, 0))
    
    # Recaptcha scoring (only if available)
    ip_behavior['recaptcha_score'] = 0
    if 'avg_risk_score' in ip_behavior.columns:
        ip_behavior['recaptcha_score'] = np.where(ip_behavior['avg_risk_score'] > 0.7, 3,
                                                np.where(ip_behavior['avg_risk_score'] > 0.4, 2,
                                                       np.where(ip_behavior['avg_risk_score'] > 0.2, 1, 0)))
    
    ip_behavior['total_risk_score'] = (ip_behavior['volume_score'] + 
                                     ip_behavior['user_diversity_score'] + 
                                     ip_behavior['sustained_score'] +
                                     ip_behavior['recaptcha_score'])
    
    ip_behavior['risk_level'] = np.where(ip_behavior['total_risk_score'] >= 8, 'CRITICAL',
                                       np.where(ip_behavior['total_risk_score'] >= 5, 'HIGH',
                                              np.where(ip_behavior['total_risk_score'] >= 3, 'MEDIUM', 'LOW')))
    
    return ip_behavior.sort_values('total_risk_score', ascending=False).head(20)

def create_device_analysis_table(df):
    """Create a simple table of device_id, total requests, and proxy status"""
    if df is None or df.empty or 'device_id' not in df.columns:
        return pd.DataFrame()
    
    # Prepare data
    df = df.copy()
    
    # Calculate proxy status
    if 'akamai_epd' in df.columns:
        # Consider as proxy if akamai_epd is not blank, not '-', and not 'hp'
        df['is_proxy'] = df['akamai_epd'].astype(str).apply(
            lambda x: 'Yes' if x not in ['', '-', 'hp', 'nan', 'None'] else 'No'
        )
    else:
        df['is_proxy'] = 'No'
    
    # Group by device_id and count total requests
    device_analysis = df.groupby('device_id').agg({
        'timestamp': 'count',  # Total number of requests
        'is_proxy': 'first'    # Take the first proxy status for each device
    }).reset_index()
    
    device_analysis.columns = ['device_id', 'total_requests', 'is_proxy']
    
    # Sort by total_requests descending
    device_analysis = device_analysis.sort_values('total_requests', ascending=False)
    
    return device_analysis

def display_suspicious_activity_log(suspicious_df):
    """Display suspicious activity log data"""
    st.header("📋 Suspicious Activity Log")
    
    if suspicious_df is not None and not suspicious_df.empty:
        # Normalize column names for suspicious activity log
        suspicious_df = normalize_dataframe(suspicious_df)
        
        # Ensure required columns exist
        if 'true_client_ip' in suspicious_df.columns:
            # Select relevant columns including recaptcha data
            display_cols = []
            possible_cols = ['true_client_ip', 'risk_analysis_score', 'reasons', 'country_code', 
                           'timestamp', 'platform', 'ip_address', 'recaptcha_valid', 
                           'response_action', 'invalid_reason', 'recaptcha_actions']
            
            for col in possible_cols:
                if col in suspicious_df.columns:
                    display_cols.append(col)
            
            if display_cols:
                st.dataframe(suspicious_df[display_cols], use_container_width=True)
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Suspicious Entries", len(suspicious_df))
                with col2:
                    if 'country_code' in suspicious_df.columns:
                        st.metric("Unique Countries", suspicious_df['country_code'].nunique())
                with col3:
                    st.metric("Unique IPs", suspicious_df['true_client_ip'].nunique())
                with col4:
                    if 'risk_analysis_score' in suspicious_df.columns:
                        high_risk = len(suspicious_df[suspicious_df['risk_analysis_score'] > 0.7])
                        st.metric("High Risk Entries", high_risk)
            else:
                st.warning("No relevant columns found in suspicious activity log")
        else:
            st.warning("Suspicious activity log missing 'true_client_ip' column")
    else:
        st.info("No suspicious activity log data uploaded")

def main():
    st.sidebar.title("🔐 Brute Force Detection Dashboard")
    
    # Initialize session state
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
    if 'suspicious_df' not in st.session_state:
        st.session_state.suspicious_df = None
    
    # File uploads
    st.sidebar.header("📁 Upload Security Data")
    
    # Main authentication data
    uploaded_file = st.sidebar.file_uploader("Upload Authentication Data (CSV)", type=["csv"])
    
    if uploaded_file:
        try:
            df = normalize_dataframe(pd.read_csv(uploaded_file))
            st.session_state.current_df = df
            st.sidebar.success("✅ Authentication data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading authentication data: {str(e)}")
    
    # Suspicious activity log
    suspicious_file = st.sidebar.file_uploader("Upload Suspicious Activity Log (CSV)", type=["csv"], key="suspicious")
    if suspicious_file:
        try:
            suspicious_df = normalize_dataframe(pd.read_csv(suspicious_file))
            st.session_state.suspicious_df = suspicious_df
            st.sidebar.success("✅ Suspicious activity log loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading suspicious activity log: {str(e)}")
    
    # Analysis controls
    st.sidebar.header("⚙️ Detection Settings")
    min_requests = st.sidebar.number_input("Minimum requests for analysis", value=5, step=1)
    risk_threshold = st.sidebar.select_slider("Risk threshold", options=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], value='MEDIUM')
    
    # Main display
    if st.session_state.current_df is not None:
        display_analysis_dashboard(st.session_state.current_df, st.session_state.suspicious_df, min_requests, risk_threshold)
    else:
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen with instructions"""
    st.title("🔐 Brute Force Detection Dashboard")
    
    st.markdown("""
    ## Advanced Brute Force Attack Detection
    
    This dashboard provides comprehensive brute force detection for authentication data:
    - **Hourly analysis** of IP behavior patterns
    - **Multi-factor risk scoring** (volume, users, devices, duration, proxy, recaptcha)
    - **Advanced pattern detection** (credential stuffing, distributed attacks)
    - **Recaptcha risk analysis** with real-time scoring
    - **Real-time risk assessment** with actionable insights
    
    ### 🚀 Getting Started
    1. **Upload authentication data** (login or signup CSV)
    2. **Upload suspicious activity log** (optional)
    3. **Adjust detection sensitivity** in sidebar
    4. **View comprehensive analysis** with hourly patterns
    5. **Identify suspicious IPs** and attack patterns
    
    ### 📊 Enhanced Detection Methodology
    - **Hourly bucketing** for temporal analysis
    - **Risk scoring** across multiple dimensions including recaptcha
    - **Pattern recognition** for advanced attacks
    - **Proxy detection** and behavioral analysis
    - **Recaptcha risk integration** for improved accuracy
    """)

def display_analysis_dashboard(df, suspicious_df, min_requests, risk_threshold):
    """Display comprehensive brute force analysis"""
    st.title("🔐 Brute Force Attack Analysis")
    
    # Global Filters
    st.sidebar.header("🔍 Global Filters")
    
    # IP Filter
    if 'true_client_ip' in df.columns:
        unique_ips = df['true_client_ip'].unique()
        selected_ips = st.sidebar.multiselect(
            "Filter by IP Address:",
            options=unique_ips,
            default=[],
            key="ip_filter"
        )
    else:
        selected_ips = []
    
    # Platform Filter
    if 'platform' in df.columns:
        unique_platforms = df['platform'].unique()
        selected_platforms = st.sidebar.multiselect(
            "Filter by Platform:",
            options=unique_platforms,
            default=unique_platforms,
            key="platform_filter"
        )
    else:
        selected_platforms = []
    
    # Risk Score Filter (only if column exists)
    if 'risk_analysis_score' in df.columns:
        min_risk = st.sidebar.slider(
            "Minimum Risk Score:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            key="risk_filter"
        )
    else:
        min_risk = 0.0
    
    # Apply filters
    filtered_df = df.copy()
    if selected_ips:
        filtered_df = filtered_df[filtered_df['true_client_ip'].isin(selected_ips)]
    if selected_platforms:
        filtered_df = filtered_df[filtered_df['platform'].isin(selected_platforms)]
    if 'risk_analysis_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['risk_analysis_score'] >= min_risk]
    
    # Show data overview
    st.subheader("📋 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")
        if 'ip_address' in filtered_df.columns:
            st.metric("Unique IPs", f"{filtered_df['ip_address'].nunique():,}")
    
    with col2:
        if 'username' in filtered_df.columns:
            st.metric("Unique Users", f"{filtered_df['username'].nunique():,}")
        if 'device_id' in filtered_df.columns:
            st.metric("Unique Devices", f"{filtered_df['device_id'].nunique():,}")
    
    with col3:
        if 'platform' in filtered_df.columns:
            st.metric("Platforms", f"{filtered_df['platform'].nunique():,}")
        if 'timestamp' in filtered_df.columns:
            time_range = filtered_df['timestamp'].max() - filtered_df['timestamp'].min()
            st.metric("Time Range", f"{time_range.days} days, {time_range.seconds//3600} hours")
    
    with col4:
        if 'risk_analysis_score' in filtered_df.columns:
            avg_risk = filtered_df['risk_analysis_score'].mean()
            high_risk = len(filtered_df[filtered_df['risk_analysis_score'] > 0.7])
            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
            st.metric("High Risk Entries", high_risk)
        else:
            st.metric("Recaptcha Data", "Not Available")
    
    st.markdown("---")
    
    # Run brute force analysis on filtered data
    with st.spinner("🔍 Analyzing brute force patterns..."):
        hourly_analysis, suspicious_ips = analyze_brute_force_patterns(filtered_df)
        advanced_patterns = detect_advanced_patterns(filtered_df)
        ip_behavior = create_ip_behavior_analysis(filtered_df)
        categorization = create_rule_categorization_table(filtered_df)
        device_analysis = create_device_analysis_table(filtered_df)  # NEW: Add device analysis
    
    # Display main dashboard
    create_brute_force_dashboard(hourly_analysis, suspicious_ips, advanced_patterns, categorization)
    
    # IP Behavior Analysis
    if not ip_behavior.empty:
        st.markdown("---")
        st.header("📊 IP Behavior Analysis (Top 20)")
        
        display_cols = ['ip_address', 'platform', 'total_requests', 'activity_hours', 'requests_per_hour',
                       'unique_users', 'unique_devices', 'proxy_ratio', 'total_risk_score', 'risk_level']
        
        # Add optional columns only if they exist
        optional_cols = ['avg_risk_score', 'high_risk_count', 'suspicious_env_count']
        available_cols = display_cols + [col for col in optional_cols if col in ip_behavior.columns]
        
        st.dataframe(ip_behavior[available_cols], use_container_width=True)
        
        # Risk distribution
        if 'risk_level' in ip_behavior.columns:
            risk_dist = ip_behavior['risk_level'].value_counts()
            if not risk_dist.empty:
                st.plotly_chart(px.pie(values=risk_dist.values, names=risk_dist.index, 
                                     title="Risk Level Distribution"), use_container_width=True)
    
    # NEW: Device Analysis Table
    if not device_analysis.empty:
        st.markdown("---")
        st.header("📱 Device Analysis")
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_devices = len(device_analysis)
            st.metric("Total Devices", total_devices)
        with col2:
            proxy_devices = len(device_analysis[device_analysis['is_proxy'] == 'Yes'])
            st.metric("Proxy Devices", proxy_devices)
        with col3:
            avg_requests_per_device = device_analysis['total_requests'].mean()
            st.metric("Avg Requests/Device", f"{avg_requests_per_device:.1f}")
        
        # Show the device table
        st.subheader("Device Request Summary")
        st.dataframe(device_analysis, use_container_width=True)
        
        # Optional: Show top devices with most requests
        st.subheader("Top 10 Devices by Request Count")
        top_devices = device_analysis.head(10)
        st.dataframe(top_devices, use_container_width=True)
    
    # Suspicious Activity Log
    if suspicious_df is not None:
        st.markdown("---")
        display_suspicious_activity_log(suspicious_df)
    
    # Raw data preview
    with st.expander("📋 Raw Data Preview"):
        st.dataframe(filtered_df.head(10), use_container_width=True)

if __name__ == "__main__":
    main()
