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

st.set_page_config(page_title="Security Intelligence Dashboard", layout="wide")

# ------------------------------
# Helper functions
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

    # Set time column - use the first available time column
    if col_time:
        df["timestamp"] = pd.to_datetime(df[col_time], errors="coerce")
    else:
        # If no time column found, create a dummy one
        df["timestamp"] = pd.to_datetime("today")
    
    df["x_real_ip"] = df[col_ip].astype(str) if col_ip else "unknown"
    df["true_client_ip"] = df["x_real_ip"]  # Ensure true_client_ip exists
    df["dr_dv"] = df[col_device].astype(str) if col_device else np.nan
    df["akamai_epd"] = df[col_akamai_epd] if col_akamai_epd else np.nan
    df["is_proxy"] = df["akamai_epd"].notna() & (df["akamai_epd"] != "")

    return df

def parse_time_columns(df, time_col="timestamp"):
    """Parse time columns and create time labels"""
    df = df.copy()
    
    # Use timestamp column that we created in normalize_dataframe
    if time_col not in df.columns:
        st.error(f"Time column '{time_col}' not found in data. Available columns: {list(df.columns)}")
        return df
        
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    
    # Create time labels for aggregation
    df['hour_label'] = df[time_col].dt.strftime('%H')
    df['minute_label'] = df[time_col].dt.strftime('%H:%M')
    df['second_label'] = df[time_col].dt.strftime('%H:%M:%S')
    df['date'] = df[time_col].dt.date
    
    return df

def collapse_ip(ip, mask_octets=1):
    """Collapse IP into ranges by masking last N octets."""
    if not isinstance(ip, str) or "." not in ip:
        return ip
    parts = ip.split(".")
    for i in range(1, mask_octets+1):
        if len(parts) >= i:
            parts[-i] = "*"
    return ".".join(parts)

def explain_rule_row(row, rule_name):
    if rule_name == '15min_>9':
        return f"IP {row['true_client_ip']} had {row['count_15min']} requests within 15 minutes — exceeds threshold 9."
    if rule_name == '10min_>=5':
        return f"IP {row['true_client_ip']} had {row['count_10min']} requests within 10 minutes — meets/exceeds threshold 5."
    return "Rule triggered."

def explain_ml_row(row, median_15, median_10):
    reasons = []
    if row['count_15min'] > median_15 * 2:
        reasons.append(f"High 15-min count ({row['count_15min']}) >> median {median_15:.1f}")
    if row['count_10min'] > median_10 * 2:
        reasons.append(f"High 10-min count ({row['count_10min']}) >> median {median_10:.1f}")
    if (row.get('akamai_bot') is not None) and (str(row.get('akamai_bot')).lower() != '-' and 'bot' in str(row.get('akamai_bot')).lower()):
        reasons.append("Akamai bot indicator present")
    if not reasons:
        reasons.append("Statistical outlier detected by IsolationForest on (count_15min, count_10min).")
    return "; ".join(reasons)

def apply_rule_categorization(df):
    """Apply the rule categorization table"""
    if df is None or df.empty:
        return []
    
    final_categories = []
    
    # Group by date for daily analysis
    if 'date' in df.columns:
        date_groups = df.groupby('date')
    else:
        date_groups = [("All Data", df)]
    
    for day, group in date_groups:
        total_requests = len(group)
        
        # Calculate metrics for rule categorization
        max_requests_ip = group['true_client_ip'].value_counts().max() if 'true_client_ip' in group.columns else 0
        max_requests_device = group['dr_dv'].value_counts().max() if 'dr_dv' in group.columns else 0
        
        # Proxy ratio calculation
        if "akamai_epd" in group.columns:
            epd_norm = group["akamai_epd"].astype(str).str.strip().str.lower()
            proxy_ratio = (~epd_norm.isin(["-", "rp", "", "nan"])).mean() * 100
        else:
            proxy_ratio = 0

        # 🎯 RULE CATEGORIZATION TABLE
        if (total_requests > 1000) and (max_requests_ip > 25) and (proxy_ratio > 20) and (max_requests_device > 15):
            category = "🚨 CRITICAL: Attack Detected"
            risk_level = "CRITICAL"
            explanation = f"High volume activity ({total_requests}) with suspicious patterns: single IP made {max_requests_ip} requests, {proxy_ratio:.1f}% proxy usage, device made {max_requests_device} requests"
        elif (max_requests_ip > 25) and (total_requests > 1000) and (max_requests_device > 15):
            category = "🔴 HIGH: High Request Volume"
            risk_level = "HIGH"
            explanation = f"Elevated activity ({total_requests}) with concentrated requests: IP={max_requests_ip}, device={max_requests_device}"
        elif proxy_ratio > 20:
            category = "🔴 HIGH: Elevated Proxy Usage"
            risk_level = "HIGH"
            explanation = f"Suspicious proxy usage detected: {proxy_ratio:.1f}% of requests used proxies"
        else:
            category = "✅ NORMAL: No Suspicious Activity"
            risk_level = "NORMAL"
            explanation = "All metrics within normal thresholds"

        final_categories.append({
            "date": day,
            "category": category,
            "risk_level": risk_level,
            "explanation": explanation,
            "total_requests": total_requests,
            "max_requests_ip": max_requests_ip,
            "max_requests_device": max_requests_device,
            "proxy_ratio": f"{proxy_ratio:.1f}%"
        })

    return final_categories

def compute_rolling_counts(df, burst_window_mins):
    """Compute rolling counts for anomaly detection"""
    if df.empty:
        return df
        
    df = df.sort_values(['true_client_ip', 'timestamp']).reset_index(drop=True)
    df['__ts'] = (df['timestamp'].astype('int64') // 10**9).astype(np.int64)

    grouped_times = df.groupby('true_client_ip')['__ts'].apply(list).to_dict()

    rows = []
    for ip, times in grouped_times.items():
        if not times:  # Skip empty lists
            continue
        times_arr = np.array(times, dtype=np.int64)
        n = len(times_arr)
        c15 = np.empty(n, dtype=np.int32)
        c10 = np.empty(n, dtype=np.int32)
        for i, t in enumerate(times_arr):
            left15 = t - 900  # 15min
            left10 = t - 600  # 10min
            l15 = np.searchsorted(times_arr, left15, side='left')
            l10 = np.searchsorted(times_arr, left10, side='left')
            r = np.searchsorted(times_arr, t, side='right')
            c15[i] = int(r - l15)
            c10[i] = int(r - l10)
        for t, a, b in zip(times_arr, c15, c10):
            rows.append({'true_client_ip': ip, '__ts': int(t), 'count_15min': int(a), 'count_10min': int(b)})

    if rows:  # Only merge if we have data
        counts_df = pd.DataFrame(rows)
        df = df.merge(counts_df, on=['true_client_ip', '__ts'], how='left')
        df['count_15min'] = df['count_15min'].fillna(0).astype(int)
        df['count_10min'] = df['count_10min'].fillna(0).astype(int)
    
    return df

# ------------------------------
# Main Application
# ------------------------------
def main():
    st.sidebar.title("🔐 Security Intelligence Dashboard")
    
    # Initialize session state
    if 'login_df' not in st.session_state:
        st.session_state.login_df = None
    if 'signup_df' not in st.session_state:
        st.session_state.signup_df = None  
    if 'suspicious_df' not in st.session_state:
        st.session_state.suspicious_df = None
    if 'current_analysis_type' not in st.session_state:
        st.session_state.current_analysis_type = None
    
    # File uploads
    st.sidebar.header("📁 Upload Security Data")
    
    analysis_type = st.sidebar.radio("Select Analysis Type", ["Login Analysis", "Signup Analysis"])
    
    if analysis_type == "Login Analysis":
        uploaded_file = st.sidebar.file_uploader("Upload Login Data (CSV)", type=["csv"], key="login")
        if uploaded_file:
            try:
                st.session_state.login_df = normalize_dataframe(pd.read_csv(uploaded_file))
                st.session_state.current_analysis_type = "login"
                st.sidebar.success("✅ Login data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading login data: {str(e)}")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Signup Data (CSV)", type=["csv"], key="signup")
        if uploaded_file:
            try:
                st.session_state.signup_df = normalize_dataframe(pd.read_csv(uploaded_file))
                st.session_state.current_analysis_type = "signup"
                st.sidebar.success("✅ Signup data loaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error loading signup data: {str(e)}")
    
    suspicious_file = st.sidebar.file_uploader("Upload Suspicious Activity (CSV)", type=["csv"], key="suspicious")
    if suspicious_file:
        try:
            st.session_state.suspicious_df = normalize_dataframe(pd.read_csv(suspicious_file))
            st.sidebar.success("✅ Suspicious activity data loaded!")
        except Exception as e:
            st.sidebar.error(f"Error loading suspicious data: {str(e)}")
    
    # Threshold controls
    st.sidebar.header("⚙️ Detection Thresholds")
    burst_threshold = st.sidebar.number_input("Burst threshold (requests within 10 min)", value=10, step=1)
    burst_window_mins = st.sidebar.number_input("Burst window (minutes)", value=10, step=1)
    
    # Main display
    if st.session_state.current_analysis_type == "login" and st.session_state.login_df is not None:
        display_login_analysis(st.session_state.login_df, burst_threshold, burst_window_mins)
    elif st.session_state.current_analysis_type == "signup" and st.session_state.signup_df is not None:
        display_signup_analysis(st.session_state.signup_df, burst_threshold, burst_window_mins)
    else:
        display_welcome_screen()

def display_welcome_screen():
    """Display welcome screen with instructions"""
    st.title("🔐 Security Intelligence Dashboard")
    
    st.markdown("""
    ## Welcome to Your Security Command Center
    
    This dashboard provides comprehensive security analysis for:
    - **Login Activity** - Detect brute force attacks
    - **Signup Activity** - Identify fraudulent registrations  
    - **Suspicious Behavior** - Analyze security alerts and bot activity
    
    ### 🚀 Getting Started
    1. **Select Analysis Type** in sidebar (Login or Signup)
    2. **Upload corresponding CSV file**
    3. **Upload Suspicious Activity data** (optional)
    4. **Adjust detection thresholds** as needed
    5. **View comprehensive analysis** with rule categorization
    
    ### 📊 What You'll Get
    - Rule categorization table with risk assessment
    - Top suspicious IP analysis
    - Machine learning anomaly detection
    - Time-series patterns and trends
    - Platform-wise breakdown
    """)

def display_login_analysis(df, burst_threshold, burst_window_mins):
    """Display comprehensive login analysis"""
    st.title("🔐 Login Security Analysis")
    
    # Show raw data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Precompute rolling counts
    df = parse_time_columns(df, time_col='timestamp')
    if df.empty:
        st.error("No valid time data found after parsing. Please check your CSV file.")
        return
        
    df = compute_rolling_counts(df, burst_window_mins)
    
    # 🎯 RULE CATEGORIZATION TABLE
    st.header("🎯 Rule Categorization Analysis")
    categorization = apply_rule_categorization(df)
    
    if categorization:
        for category in categorization:
            risk_colors = {
                "CRITICAL": "red",
                "HIGH": "orange", 
                "MEDIUM": "yellow",
                "LOW": "blue",
                "NORMAL": "green"
            }
            
            color = risk_colors.get(category['risk_level'], "gray")
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; margin: 10px 0;">
                <h4 style="color: {color}; margin: 0;">{category['category']}</h4>
                <p style="margin: 5px 0;"><strong>Date:</strong> {category['date']}</p>
                <p style="margin: 5px 0;"><strong>Explanation:</strong> {category['explanation']}</p>
                <p style="margin: 5px 0;"><strong>Metrics:</strong> Total Logins: {category['total_requests']:,} | Max IP Requests: {category['max_requests_ip']} | Max Device Requests: {category['max_requests_device']} | Proxy Ratio: {category['proxy_ratio']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No categorization data available.")
    
    st.markdown("---")
    
    # Top IP Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔐 Top 10 Login IPs")
        if 'true_client_ip' in df.columns:
            top_ips = df['true_client_ip'].value_counts().head(10).reset_index()
            top_ips.columns = ['IP Address', 'Login Count']
            st.dataframe(top_ips, use_container_width=True)
        else:
            st.info("No IP data available")
    
    with col2:
        st.subheader("📊 Login Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Total Logins", f"{len(df):,}")
            if 'true_client_ip' in df.columns:
                st.metric("Unique IPs", f"{df['true_client_ip'].nunique():,}")
        with metrics_col2:
            if 'response_code' in df.columns:
                success_rate = (df['response_code'] == 200).mean() * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    # Rule-based anomalies
    st.header("🚨 Rule-Based Anomalies")
    
    # Check if rolling counts were computed
    if 'count_15min' in df.columns:
        # Rule: >9 in 15 minutes
        rb_15 = df[df['count_15min'] > 9].copy()
        if not rb_15.empty:
            rb_15['explanation'] = rb_15.apply(lambda r: explain_rule_row(r, '15min_>9'), axis=1)
            st.subheader("Rule: more than 9 logins in 15 minutes")
            st.dataframe(rb_15[['timestamp', 'true_client_ip', 'count_15min', 'explanation']].sort_values('count_15min', ascending=False).head(20))
        else:
            st.success("No IPs exceed 9 logins in 15 minutes.")

        # Rule: >=5 in 10 minutes
        rb_10 = df[df['count_10min'] >= 5].copy()
        if not rb_10.empty:
            rb_10['explanation'] = rb_10.apply(lambda r: explain_rule_row(r, '10min_>=5'), axis=1)
            st.subheader("Rule: 5 or more logins in 10 minutes")
            st.dataframe(rb_10[['timestamp', 'true_client_ip', 'count_10min', 'explanation']].sort_values('count_10min', ascending=False).head(20))
        else:
            st.info("No IPs with 5 or more logins in 10 minutes.")
    else:
        st.info("Rolling counts not available for rule-based analysis")
    
    # ML Anomaly Detection
    st.header("🤖 Machine Learning Anomaly Detection")
    if 'count_15min' in df.columns and 'count_10min' in df.columns:
        features = df[['count_15min', 'count_10min']].fillna(0)
        iso = IsolationForest(contamination=0.01, random_state=42)
        df['ml_flag'] = iso.fit_predict(features)

        anomalies_ml = df[df['ml_flag'] == -1].copy()
        median_15 = max(1.0, df['count_15min'].median())
        median_10 = max(1.0, df['count_10min'].median())

        if not anomalies_ml.empty:
            anomalies_ml['reason'] = anomalies_ml.apply(lambda r: explain_ml_row(r, median_15, median_10), axis=1)
            st.dataframe(anomalies_ml[['timestamp', 'true_client_ip', 'count_15min', 'count_10min', 'reason']].sort_values(['count_15min','count_10min'], ascending=False).head(20))
        else:
            st.success("Isolation Forest did not detect anomalies in this dataset.")
    else:
        st.info("ML analysis requires rolling count data")

def display_signup_analysis(df, burst_threshold, burst_window_mins):
    """Display comprehensive signup analysis"""
    st.title("📝 Signup Security Analysis")
    
    # Show raw data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Precompute rolling counts
    df = parse_time_columns(df, time_col='timestamp')
    if df.empty:
        st.error("No valid time data found after parsing. Please check your CSV file.")
        return
        
    df = compute_rolling_counts(df, burst_window_mins)
    
    # 🎯 RULE CATEGORIZATION TABLE
    st.header("🎯 Rule Categorization Analysis")
    categorization = apply_rule_categorization(df)
    
    if categorization:
        for category in categorization:
            risk_colors = {
                "CRITICAL": "red",
                "HIGH": "orange", 
                "MEDIUM": "yellow",
                "LOW": "blue",
                "NORMAL": "green"
            }
            
            color = risk_colors.get(category['risk_level'], "gray")
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; margin: 10px 0;">
                <h4 style="color: {color}; margin: 0;">{category['category']}</h4>
                <p style="margin: 5px 0;"><strong>Date:</strong> {category['date']}</p>
                <p style="margin: 5px 0;"><strong>Explanation:</strong> {category['explanation']}</p>
                <p style="margin: 5px 0;"><strong>Metrics:</strong> Total Signups: {category['total_requests']:,} | Max IP Requests: {category['max_requests_ip']} | Max Device Requests: {category['max_requests_device']} | Proxy Ratio: {category['proxy_ratio']}</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No categorization data available.")
    
    st.markdown("---")
    
    # Top IP Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Top 10 Signup IPs")
        if 'true_client_ip' in df.columns:
            top_ips = df['true_client_ip'].value_counts().head(10).reset_index()
            top_ips.columns = ['IP Address', 'Signup Count']
            st.dataframe(top_ips, use_container_width=True)
        else:
            st.info("No IP data available")
    
    with col2:
        st.subheader("📊 Signup Metrics")
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("Total Signups", f"{len(df):,}")
            if 'true_client_ip' in df.columns:
                st.metric("Unique IPs", f"{df['true_client_ip'].nunique():,}")
        with metrics_col2:
            if 'response_code' in df.columns:
                success_rate = (df['response_code'] == 200).mean() * 100
                st.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.markdown("---")
    
    # Rule-based anomalies
    st.header("🚨 Rule-Based Anomalies")
    
    # Check if rolling counts were computed
    if 'count_15min' in df.columns:
        # Rule: >9 in 15 minutes
        rb_15 = df[df['count_15min'] > 9].copy()
        if not rb_15.empty:
            rb_15['explanation'] = rb_15.apply(lambda r: explain_rule_row(r, '15min_>9'), axis=1)
            st.subheader("Rule: more than 9 signups in 15 minutes")
            st.dataframe(rb_15[['timestamp', 'true_client_ip', 'count_15min', 'explanation']].sort_values('count_15min', ascending=False).head(20))
        else:
            st.success("No IPs exceed 9 signups in 15 minutes.")

        # Rule: >=5 in 10 minutes
        rb_10 = df[df['count_10min'] >= 5].copy()
        if not rb_10.empty:
            rb_10['explanation'] = rb_10.apply(lambda r: explain_rule_row(r, '10min_>=5'), axis=1)
            st.subheader("Rule: 5 or more signups in 10 minutes")
            st.dataframe(rb_10[['timestamp', 'true_client_ip', 'count_10min', 'explanation']].sort_values('count_10min', ascending=False).head(20))
        else:
            st.info("No IPs with 5 or more signups in 10 minutes.")
    else:
        st.info("Rolling counts not available for rule-based analysis")
    
    # ML Anomaly Detection
    st.header("🤖 Machine Learning Anomaly Detection")
    if 'count_15min' in df.columns and 'count_10min' in df.columns:
        features = df[['count_15min', 'count_10min']].fillna(0)
        iso = IsolationForest(contamination=0.01, random_state=42)
        df['ml_flag'] = iso.fit_predict(features)

        anomalies_ml = df[df['ml_flag'] == -1].copy()
        median_15 = max(1.0, df['count_15min'].median())
        median_10 = max(1.0, df['count_10min'].median())

        if not anomalies_ml.empty:
            anomalies_ml['reason'] = anomalies_ml.apply(lambda r: explain_ml_row(r, median_15, median_10), axis=1)
            st.dataframe(anomalies_ml[['timestamp', 'true_client_ip', 'count_15min', 'count_10min', 'reason']].sort_values(['count_15min','count_10min'], ascending=False).head(20))
        else:
            st.success("Isolation Forest did not detect anomalies in this dataset.")
    else:
        st.info("ML analysis requires rolling count data")

if __name__ == "__main__":
    main()
