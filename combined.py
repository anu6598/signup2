# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import re

st.set_page_config(layout="wide", page_title="Security Intelligence Dashboard")

# ------------------------------
# Enhanced Security Analysis Functions
# ------------------------------
def normalize_dataframe(df_raw):
    """Normalize column names and ensure consistent formatting"""
    cols_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_raw.columns}
    df_raw.rename(columns=cols_map, inplace=True)
    df = df_raw.copy()

    # Ensure critical columns exist
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['timestamp'] = df['start_time']
        df['date'] = df['start_time'].dt.date
    
    if 'timestamp' in df.columns and 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Platform detection
    if 'dr_platform' in df.columns:
        df['platform'] = df['dr_platform'].fillna('web')
    elif 'user_agent' in df.columns:
        df['platform'] = df['user_agent'].apply(detect_platform_from_ua)
    else:
        df['platform'] = 'unknown'
    
    # Ensure true_client_ip column exists
    if 'true_client_ip' not in df.columns:
        # Try to find IP column
        ip_col = None
        for col in ['x_real_ip', 'client_ip', 'ip', 'remote_addr']:
            if col in df.columns:
                ip_col = col
                break
        if ip_col:
            df['true_client_ip'] = df[ip_col].astype(str)
        else:
            df['true_client_ip'] = 'unknown'
    
    return df

def detect_platform_from_ua(user_agent):
    """Detect platform from user agent string"""
    if not isinstance(user_agent, str):
        return 'unknown'
    ua = user_agent.lower()
    if 'android' in ua:
        return 'android'
    elif 'ios' in ua or 'iphone' in ua:
        return 'ios'
    elif 'mozilla' in ua or 'chrome' in ua or 'safari' in ua:
        return 'web'
    else:
        return 'other'

def analyze_request_success(df, request_type):
    """Analyze successful vs unsuccessful requests"""
    if 'response_code' not in df.columns:
        return {}
    
    if request_type == 'login':
        login_df = df[df['request_path'].str.contains('/login', na=False)]
        success_codes = [200, 201]
        failed_codes = [400, 401, 403, 404, 422, 429, 500]
    elif request_type == 'signup':
        login_df = df[df['request_path'].str.contains('/signup', na=False)]
        success_codes = [200, 201]
        failed_codes = [400, 409, 422, 429, 500]
    else:
        return {}
    
    successful = login_df[login_df['response_code'].isin(success_codes)]
    failed = login_df[login_df['response_code'].isin(failed_codes)]
    
    analysis = {
        'request_type': request_type,
        'total_requests': len(login_df),
        'successful_requests': len(successful),
        'failed_requests': len(failed),
        'success_rate': len(successful) / len(login_df) * 100 if len(login_df) > 0 else 0,
        'unique_ips': login_df['true_client_ip'].nunique() if 'true_client_ip' in login_df.columns else 0,
        'unique_successful_ips': successful['true_client_ip'].nunique() if 'true_client_ip' in successful.columns else 0,
        'unique_failed_ips': failed['true_client_ip'].nunique() if 'true_client_ip' in failed.columns else 0
    }
    
    return analysis

def enhanced_security_assessment(login_df, signup_df, suspicious_df=None):
    """Comprehensive security assessment with clear risk categorization"""
    
    # Initialize results
    assessment = {
        'date': datetime.now().date(),
        'overall_risk': 'LOW',
        'critical_findings': [],
        'suspicious_ips': [],
        'platform_analysis': {},
        'attack_indicators': {},
        'rule_categorization': [],
        'top_login_ips': [],
        'top_signup_ips': []
    }
    
    # Analyze login data
    if login_df is not None and not login_df.empty:
        login_analysis = analyze_login_patterns(login_df)
        assessment['login_analysis'] = login_analysis
        assessment['top_login_ips'] = login_analysis.get('top_ips', [])
    
    # Analyze signup data  
    if signup_df is not None and not signup_df.empty:
        signup_analysis = analyze_signup_patterns(signup_df)
        assessment['signup_analysis'] = signup_analysis
        assessment['top_signup_ips'] = signup_analysis.get('top_ips', [])
    
    # Analyze suspicious activity
    if suspicious_df is not None and not suspicious_df.empty:
        suspicious_analysis = analyze_suspicious_activity(suspicious_df)
        assessment['suspicious_analysis'] = suspicious_analysis
        assessment['suspicious_ips'].extend(suspicious_analysis.get('high_risk_ips', []))
    
    # Apply Rule Categorization (from your original code)
    assessment = apply_rule_categorization(assessment, login_df, signup_df)
    
    # Determine overall risk
    assessment = calculate_overall_risk(assessment)
    
    return assessment

def apply_rule_categorization(assessment, login_df, signup_df):
    """Apply the rule categorization table from your original code"""
    
    # Combine login and signup data for OTP analysis
    combined_df = pd.concat([login_df, signup_df]) if login_df is not None and signup_df is not None else login_df if login_df is not None else signup_df
    
    if combined_df is None or combined_df.empty:
        return assessment
    
    # Group by date for daily analysis
    if 'date' in combined_df.columns:
        date_groups = combined_df.groupby('date')
    else:
        # If no date column, treat as single day
        date_groups = [("All Data", combined_df)]
    
    final_categories = []
    
    for day, group in date_groups:
        total_otps = len(group)
        
        # Calculate metrics for rule categorization
        max_requests_ip = group['true_client_ip'].value_counts().max() if 'true_client_ip' in group.columns else 0
        max_requests_device = group['dr_dv'].value_counts().max() if 'dr_dv' in group.columns else 0
        
        # Proxy ratio calculation
        if "akamai_epd" in group.columns:
            epd_norm = group["akamai_epd"].astype(str).str.strip().str.lower()
            proxy_ratio = (~epd_norm.isin(["-", "rp", "", "nan"])).mean() * 100
        else:
            proxy_ratio = 0

        # 🎯 RULE CATEGORIZATION TABLE (From your original code)
        if (total_otps > 1000) and (max_requests_ip > 25) and (proxy_ratio > 20) and (max_requests_device > 15):
            category = "🚨 CRITICAL: OTP Abuse/Attack Detected"
            risk_level = "CRITICAL"
            explanation = f"High volume OTP activity ({total_otps}) with suspicious patterns: single IP made {max_requests_ip} requests, {proxy_ratio:.1f}% proxy usage, device made {max_requests_device} requests"
        elif (max_requests_ip > 25) and (total_otps > 1000) and (max_requests_device > 15):
            category = "🔴 HIGH: High OTP Request Volume"
            risk_level = "HIGH"
            explanation = f"Elevated OTP activity ({total_otps}) with concentrated requests: IP={max_requests_ip}, device={max_requests_device}"
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
            "total_otps": total_otps,
            "max_requests_ip": max_requests_ip,
            "max_requests_device": max_requests_device,
            "proxy_ratio": f"{proxy_ratio:.1f}%"
        })

    assessment['rule_categorization'] = final_categories
    return assessment

def analyze_login_patterns(df):
    """Enhanced login pattern analysis with clear explanations"""
    analysis = {
        'total_logins': len(df),
        'unique_ips': df['true_client_ip'].nunique() if 'true_client_ip' in df.columns else 0,
        'platform_breakdown': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        'top_ips': []
    }
    
    # Success rate analysis
    success_analysis = analyze_request_success(df, 'login')
    analysis.update(success_analysis)
    
    # Get top 10 IPs by request count
    if 'true_client_ip' in df.columns:
        ip_counts = df['true_client_ip'].value_counts().head(10)
        for ip, count in ip_counts.items():
            analysis['top_ips'].append({
                'ip': ip,
                'count': count,
                'type': 'LOGIN_ATTEMPTS'
            })
    
    return analysis

def analyze_signup_patterns(df):
    """Enhanced signup pattern analysis with clear explanations"""
    analysis = {
        'total_signups': len(df),
        'unique_ips': df['true_client_ip'].nunique() if 'true_client_ip' in df.columns else 0,
        'platform_breakdown': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        'top_ips': []
    }
    
    # Success rate analysis
    success_analysis = analyze_request_success(df, 'signup')
    analysis.update(success_analysis)
    
    # Get top 10 IPs by request count
    if 'true_client_ip' in df.columns:
        ip_counts = df['true_client_ip'].value_counts().head(10)
        for ip, count in ip_counts.items():
            analysis['top_ips'].append({
                'ip': ip,
                'count': count,
                'type': 'SIGNUP_ATTEMPTS'
            })
    
    return analysis

def analyze_suspicious_activity(df):
    """Analyze suspicious activity logs with clear correlation"""
    analysis = {
        'total_alerts': len(df),
        'high_risk_ips': [],
        'bot_detections': [],
        'recaptcha_failures': []
    }
    
    # Extract Akamai bot scores and recaptcha data
    for _, row in df.iterrows():
        ip = row.get('true_client_ip')
        if not ip:
            continue
            
        # Parse request_data_str for Akamai-Bot
        request_data = row.get('request_data_str', '{}')
        try:
            if isinstance(request_data, str):
                request_json = json.loads(request_data)
                akamai_bot = request_json.get('Akamai-Bot', '')
                if 'bot' in akamai_bot.lower():
                    analysis['bot_detections'].append(ip)
        except:
            pass
        
        # Parse extra_data_str for recaptcha scores
        extra_data = row.get('extra_data_str', '{}')
        try:
            if isinstance(extra_data, str):
                extra_json = json.loads(extra_data)
                risk_score = extra_json.get('risk_analysis_score', 1.0)
                if risk_score > 0.7:  # High risk recaptcha
                    analysis['recaptcha_failures'].append(ip)
        except:
            pass
    
    # Compile high risk IPs
    high_risk_ips = set()
    high_risk_ips.update(analysis['bot_detections'])
    high_risk_ips.update(analysis['recaptcha_failures'])
    
    analysis['high_risk_ips'] = list(high_risk_ips)
    
    return analysis

def calculate_overall_risk(assessment):
    """Calculate overall risk level based on all indicators"""
    risk_score = 0
    critical_findings = []
    
    # Rule categorization risk
    for category in assessment.get('rule_categorization', []):
        if category['risk_level'] == 'CRITICAL':
            risk_score += 3
            critical_findings.append(f"CRITICAL: {category['explanation']}")
        elif category['risk_level'] == 'HIGH':
            risk_score += 2
            critical_findings.append(f"HIGH: {category['explanation']}")
    
    # Login risk factors
    if 'login_analysis' in assessment:
        login = assessment['login_analysis']
        if login.get('failed_requests', 0) > 1000:
            risk_score += 3
            critical_findings.append("High volume of failed logins detected")
        if login.get('success_rate', 100) < 30:
            risk_score += 2
            critical_findings.append("Very low login success rate")
    
    # Signup risk factors  
    if 'signup_analysis' in assessment:
        signup = assessment['signup_analysis']
        if signup.get('failed_requests', 0) > 500:
            risk_score += 2
            critical_findings.append("High volume of failed signups")
    
    # Suspicious activity risk factors
    if 'suspicious_analysis' in assessment:
        suspicious = assessment['suspicious_analysis']
        if len(suspicious.get('high_risk_ips', [])) > 10:
            risk_score += 3
            critical_findings.append("Multiple high-risk IPs detected")
        if len(suspicious.get('bot_detections', [])) > 5:
            risk_score += 2
            critical_findings.append("Multiple bot detections")
    
    # Determine overall risk
    if risk_score >= 5:
        assessment['overall_risk'] = 'HIGH'
    elif risk_score >= 3:
        assessment['overall_risk'] = 'MEDIUM'
    else:
        assessment['overall_risk'] = 'LOW'
    
    assessment['critical_findings'] = critical_findings
    assessment['risk_score'] = risk_score
    
    return assessment

def create_executive_summary(assessment):
    """Create a clear executive summary for team review"""
    
    risk_colors = {
        'HIGH': '🔴',
        'MEDIUM': '🟡', 
        'LOW': '🟢'
    }
    
    summary = f"""
# {risk_colors[assessment['overall_risk']]} Security Intelligence Report
**Date:** {assessment['date']} | **Overall Risk:** {assessment['overall_risk']}

## 📊 Quick Overview
"""
    
    # Add key metrics
    if 'login_analysis' in assessment:
        login = assessment['login_analysis']
        summary += f"""
- **Login Activity:** {login.get('total_logins', 0):,} total | {login.get('success_rate', 0):.1f}% success rate
- **Unique IPs (Login):** {login.get('unique_ips', 0):,}
"""
    
    if 'signup_analysis' in assessment:
        signup = assessment['signup_analysis']
        summary += f"""
- **Signup Activity:** {signup.get('total_signups', 0):,} total | {signup.get('success_rate', 0):.1f}% success rate  
- **Unique IPs (Signup):** {signup.get('unique_ips', 0):,}
"""
    
    if 'suspicious_analysis' in assessment:
        suspicious = assessment['suspicious_analysis']
        summary += f"""
- **Security Alerts:** {suspicious.get('total_alerts', 0):,} total
- **High-Risk IPs:** {len(suspicious.get('high_risk_ips', [])):,}
"""
    
    return summary

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
    if 'current_assessment' not in st.session_state:
        st.session_state.current_assessment = None
    
    # File uploads
    st.sidebar.header("📁 Upload Security Data")
    
    login_file = st.sidebar.file_uploader("Login Data (CSV)", type=["csv"], key="login")
    signup_file = st.sidebar.file_uploader("Signup Data (CSV)", type=["csv"], key="signup")
    suspicious_file = st.sidebar.file_uploader("Suspicious Activity (CSV)", type=["csv"], key="suspicious")
    
    # Platform filter
    st.sidebar.header("🔧 Filters")
    platform_filter = st.sidebar.multiselect(
        "Platform Filter",
        ['android', 'ios', 'web', 'other', 'unknown'],
        default=['android', 'ios', 'web']
    )
    
    # Process uploaded files
    if login_file:
        st.session_state.login_df = normalize_dataframe(pd.read_csv(login_file))
        if platform_filter:
            st.session_state.login_df = st.session_state.login_df[st.session_state.login_df['platform'].isin(platform_filter)]
    
    if signup_file:
        st.session_state.signup_df = normalize_dataframe(pd.read_csv(signup_file))
        if platform_filter:
            st.session_state.signup_df = st.session_state.signup_df[st.session_state.signup_df['platform'].isin(platform_filter)]
    
    if suspicious_file:
        st.session_state.suspicious_df = normalize_dataframe(pd.read_csv(suspicious_file))
    
    # Run analysis when data is available
    if st.sidebar.button("🚀 Run Security Analysis") or st.session_state.current_assessment is not None:
        if st.session_state.login_df is not None or st.session_state.signup_df is not None:
            with st.spinner("🔍 Analyzing security patterns..."):
                assessment = enhanced_security_assessment(
                    st.session_state.login_df, 
                    st.session_state.signup_df,
                    st.session_state.suspicious_df
                )
                st.session_state.current_assessment = assessment
    
    # Main display
    if st.session_state.current_assessment:
        display_security_dashboard(st.session_state.current_assessment)
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
    1. **Upload Login Data** - CSV with login attempts (request_path containing '/login')
    2. **Upload Signup Data** - CSV with signup attempts (request_path containing '/signup') 
    3. **Upload Suspicious Activity** - CSV with security alerts (optional)
    4. **Apply platform filters** if needed
    5. **Click 'Run Security Analysis'** to generate report
    
    ### 📊 What You'll Get
    - Executive summary with risk assessment
    - **Rule Categorization Table** with clear explanations
    - **Top 10 Suspicious IPs** for login and signup
    - Platform-wise breakdown
    - Success/failure rate analysis
    """)

def display_security_dashboard(assessment):
    """Display the main security dashboard"""
    
    # Executive Summary
    st.markdown(create_executive_summary(assessment))
    
    st.markdown("---")
    
    # 🎯 RULE CATEGORIZATION TABLE (Prominently Displayed)
    st.header("🎯 Rule Categorization Analysis")
    
    if assessment.get('rule_categorization'):
        categorization_df = pd.DataFrame(assessment['rule_categorization'])
        
        # Display with color coding
        for _, row in categorization_df.iterrows():
            risk_colors = {
                "CRITICAL": "red",
                "HIGH": "orange", 
                "MEDIUM": "yellow",
                "LOW": "blue",
                "NORMAL": "green"
            }
            
            color = risk_colors.get(row['risk_level'], "gray")
            
            st.markdown(f"""
            <div style="background-color: {color}20; padding: 15px; border-radius: 10px; border-left: 5px solid {color}; margin: 10px 0;">
                <h4 style="color: {color}; margin: 0;">{row['category']}</h4>
                <p style="margin: 5px 0;"><strong>Date:</strong> {row['date']}</p>
                <p style="margin: 5px 0;"><strong>Explanation:</strong> {row['explanation']}</p>
                <p style="margin: 5px 0;"><strong>Metrics:</strong> Total OTPs: {row['total_otps']:,} | Max IP Requests: {row['max_requests_ip']} | Max Device Requests: {row['max_requests_device']} | Proxy Ratio: {row['proxy_ratio']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Also show as dataframe for detailed view
        with st.expander("📋 Detailed Categorization Table"):
            display_df = categorization_df[['date', 'category', 'risk_level', 'total_otps', 'max_requests_ip', 'max_requests_device', 'proxy_ratio']]
            st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No rule categorization available.")
    
    st.markdown("---")
    
    # SIMPLIFIED IP ANALYSIS - Just Top 10 Tables
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 Login IPs
        if assessment.get('top_login_ips'):
            st.subheader("🔐 Top 10 Login IPs")
            login_ips_df = pd.DataFrame(assessment['top_login_ips'])
            st.dataframe(login_ips_df, use_container_width=True)
        else:
            st.info("No login IP data available")
    
    with col2:
        # Top 10 Signup IPs
        if assessment.get('top_signup_ips'):
            st.subheader("📝 Top 10 Signup IPs")
            signup_ips_df = pd.DataFrame(assessment['top_signup_ips'])
            st.dataframe(signup_ips_df, use_container_width=True)
        else:
            st.info("No signup IP data available")
    
    st.markdown("---")
    
    # Detailed Analysis Sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Login Analysis
        if 'login_analysis' in assessment:
            display_login_analysis(assessment['login_analysis'])
    
    with col2:
        # Signup Analysis  
        if 'signup_analysis' in assessment:
            display_signup_analysis(assessment['signup_analysis'])
        
        # Platform Breakdown
        display_platform_analysis(assessment)

def display_login_analysis(analysis):
    """Display login analysis details with clear explanations"""
    st.subheader("🔐 Login Activity Analysis")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Total Logins", f"{analysis.get('total_logins', 0):,}")
        st.metric("Unique IPs", f"{analysis.get('unique_ips', 0):,}")
    
    with metrics_col2:
        success_rate = analysis.get('success_rate', 0)
        color = "red" if success_rate < 50 else "green"
        st.metric("Success Rate", f"{success_rate:.1f}%", delta=None, delta_color="normal")
    
    with metrics_col3:
        st.metric("Failed Logins", f"{analysis.get('failed_requests', 0):,}")

def display_signup_analysis(analysis):
    """Display signup analysis details with clear explanations"""
    st.subheader("📝 Signup Activity Analysis")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Total Signups", f"{analysis.get('total_signups', 0):,}")
        st.metric("Unique IPs", f"{analysis.get('unique_ips', 0):,}")
    
    with metrics_col2:
        success_rate = analysis.get('success_rate', 0)
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with metrics_col3:
        st.metric("Failed Signups", f"{analysis.get('failed_requests', 0):,}")

def display_platform_analysis(assessment):
    """Display platform-wise analysis"""
    st.subheader("📱 Platform Analysis")
    
    platform_data = {}
    
    # Collect platform data from all sources
    for source in ['login_analysis', 'signup_analysis']:
        if source in assessment:
            analysis = assessment[source]
            if 'platform_breakdown' in analysis:
                for platform, count in analysis['platform_breakdown'].items():
                    if platform not in platform_data:
                        platform_data[platform] = {'logins': 0, 'signups': 0}
                    if 'login' in source:
                        platform_data[platform]['logins'] += count
                    else:
                        platform_data[platform]['signups'] += count
    
    if platform_data:
        platform_df = pd.DataFrame(platform_data).T.fillna(0)
        platform_df['total'] = platform_df.sum(axis=1)
        platform_df = platform_df.sort_values('total', ascending=False)
        
        st.dataframe(platform_df)
        
        # Platform distribution chart
        fig = px.pie(
            values=platform_df['total'], 
            names=platform_df.index,
            title="Request Distribution by Platform"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
