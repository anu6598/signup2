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
        return pd.DataFrame()
    
    if request_type == 'login':
        login_df = df[df['request_path'].str.contains('/login', na=False)]
        success_codes = [200, 201]
        failed_codes = [400, 401, 403, 404, 422, 429, 500]
    elif request_type == 'signup':
        login_df = df[df['request_path'].str.contains('/signup', na=False)]
        success_codes = [200, 201]
        failed_codes = [400, 409, 422, 429, 500]
    else:
        return pd.DataFrame()
    
    successful = login_df[login_df['response_code'].isin(success_codes)]
    failed = login_df[login_df['response_code'].isin(failed_codes)]
    
    analysis = {
        'request_type': request_type,
        'total_requests': len(login_df),
        'successful_requests': len(successful),
        'failed_requests': len(failed),
        'success_rate': len(successful) / len(login_df) * 100 if len(login_df) > 0 else 0,
        'unique_ips': login_df['x_real_ip'].nunique() if 'x_real_ip' in login_df.columns else 0,
        'unique_successful_ips': successful['x_real_ip'].nunique() if 'x_real_ip' in successful.columns else 0,
        'unique_failed_ips': failed['x_real_ip'].nunique() if 'x_real_ip' in failed.columns else 0
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
        'attack_indicators': {}
    }
    
    # Analyze login data
    if login_df is not None and not login_df.empty:
        login_analysis = analyze_login_patterns(login_df)
        assessment['login_analysis'] = login_analysis
        assessment['attack_indicators']['login'] = login_analysis.get('attack_indicators', [])
    
    # Analyze signup data  
    if signup_df is not None and not signup_df.empty:
        signup_analysis = analyze_signup_patterns(signup_df)
        assessment['signup_analysis'] = signup_analysis
        assessment['attack_indicators']['signup'] = signup_analysis.get('attack_indicators', [])
    
    # Analyze suspicious activity
    if suspicious_df is not None and not suspicious_df.empty:
        suspicious_analysis = analyze_suspicious_activity(suspicious_df)
        assessment['suspicious_analysis'] = suspicious_analysis
        assessment['suspicious_ips'].extend(suspicious_analysis.get('high_risk_ips', []))
    
    # Determine overall risk
    assessment = calculate_overall_risk(assessment)
    
    return assessment

def analyze_login_patterns(df):
    """Enhanced login pattern analysis"""
    analysis = {
        'total_logins': len(df),
        'unique_ips': df['x_real_ip'].nunique() if 'x_real_ip' in df.columns else 0,
        'platform_breakdown': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        'attack_indicators': []
    }
    
    # Success rate analysis
    success_analysis = analyze_request_success(df, 'login')
    analysis.update(success_analysis)
    
    # IP frequency analysis
    ip_counts = df['x_real_ip'].value_counts()
    suspicious_ips = ip_counts[ip_counts > 10]  # More than 10 login attempts
    
    for ip, count in suspicious_ips.items():
        analysis['attack_indicators'].append({
            'ip': ip,
            'type': 'HIGH_LOGIN_ATTEMPTS',
            'count': count,
            'risk': 'HIGH' if count > 50 else 'MEDIUM'
        })
    
    # Platform-specific anomalies
    for platform in analysis['platform_breakdown']:
        platform_df = df[df['platform'] == platform]
        platform_ip_counts = platform_df['x_real_ip'].value_counts()
        if len(platform_ip_counts) > 0 and platform_ip_counts.max() > 20:
            analysis['attack_indicators'].append({
                'ip': platform_ip_counts.idxmax(),
                'type': f'HIGH_{platform.upper()}_LOGINS',
                'count': platform_ip_counts.max(),
                'risk': 'HIGH'
            })
    
    return analysis

def analyze_signup_patterns(df):
    """Enhanced signup pattern analysis"""
    analysis = {
        'total_signups': len(df),
        'unique_ips': df['x_real_ip'].nunique() if 'x_real_ip' in df.columns else 0,
        'platform_breakdown': df['platform'].value_counts().to_dict() if 'platform' in df.columns else {},
        'attack_indicators': []
    }
    
    # Success rate analysis
    success_analysis = analyze_request_success(df, 'signup')
    analysis.update(success_analysis)
    
    # Rapid signup detection (multiple signups from same IP)
    ip_counts = df['x_real_ip'].value_counts()
    rapid_signups = ip_counts[ip_counts > 5]  # More than 5 signups
    
    for ip, count in rapid_signups.items():
        analysis['attack_indicators'].append({
            'ip': ip,
            'type': 'RAPID_SIGNUPS',
            'count': count,
            'risk': 'HIGH' if count > 10 else 'MEDIUM'
        })
    
    return analysis

def analyze_suspicious_activity(df):
    """Analyze suspicious activity logs"""
    analysis = {
        'total_alerts': len(df),
        'high_risk_ips': [],
        'bot_detections': [],
        'recaptcha_failures': []
    }
    
    # Extract Akamai bot scores and recaptcha data
    for _, row in df.iterrows():
        ip = row.get('x_real_ip')
        if not ip:
            continue
            
        # Parse request_data_str for Akamai-Bot
        request_data = row.get('request_data_str', '{}')
        try:
            if isinstance(request_data, str):
                request_json = json.loads(request_data)
                akamai_bot = request_json.get('Akamai-Bot', '')
                if 'bot' in akamai_bot.lower():
                    analysis['bot_detections'].append({
                        'ip': ip,
                        'bot_info': akamai_bot,
                        'risk': 'HIGH'
                    })
        except:
            pass
        
        # Parse extra_data_str for recaptcha scores
        extra_data = row.get('extra_data_str', '{}')
        try:
            if isinstance(extra_data, str):
                extra_json = json.loads(extra_data)
                risk_score = extra_json.get('risk_analysis_score', 1.0)
                if risk_score > 0.7:  # High risk recaptcha
                    analysis['recaptcha_failures'].append({
                        'ip': ip,
                        'risk_score': risk_score,
                        'risk': 'HIGH'
                    })
        except:
            pass
    
    # Compile high risk IPs
    high_risk_ips = set()
    for detection in analysis['bot_detections'] + analysis['recaptcha_failures']:
        high_risk_ips.add(detection['ip'])
    
    analysis['high_risk_ips'] = list(high_risk_ips)
    
    return analysis

def calculate_overall_risk(assessment):
    """Calculate overall risk level based on all indicators"""
    risk_score = 0
    critical_findings = []
    
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
        if any(indicator['risk'] == 'HIGH' for indicator in signup.get('attack_indicators', [])):
            risk_score += 2
    
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
    
    # Add critical findings
    if assessment['critical_findings']:
        summary += "\n## 🚨 Critical Findings\n"
        for finding in assessment['critical_findings']:
            summary += f"- {finding}\n"
    else:
        summary += "\n## ✅ No Critical Issues Detected\n"
    
    # Add top suspicious IPs
    suspicious_ips = set()
    for source in ['login_analysis', 'signup_analysis', 'suspicious_analysis']:
        if source in assessment:
            data = assessment[source]
            if 'attack_indicators' in data:
                for indicator in data['attack_indicators']:
                    if indicator['risk'] in ['HIGH', 'MEDIUM']:
                        suspicious_ips.add(indicator['ip'])
            if 'high_risk_ips' in data:
                suspicious_ips.update(data['high_risk_ips'])
    
    if suspicious_ips:
        summary += f"\n## 🎯 Top Suspicious IPs to Investigate ({len(suspicious_ips)} total)\n"
        for ip in list(suspicious_ips)[:10]:  # Show top 10
            summary += f"- `{ip}`\n"
    
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
    - Platform-wise breakdown
    - Top suspicious IPs to investigate
    - Attack pattern detection
    - Success/failure rate analysis
    """)
    
    # Sample data structure guidance
    with st.expander("📋 Expected Data Structure"):
        st.markdown("""
        **Login/Signup CSV should contain:**
        - `request_path` (to identify login/signup endpoints)
        - `response_code` (to determine success/failure)
        - `x_real_ip` (client IP address)
        - `start_time` or `timestamp` (for time analysis)
        - `dr_platform` or `user_agent` (for platform detection)
        
        **Suspicious Activity CSV should contain:**
        - `x_real_ip` (client IP address)
        - `request_data_str` (JSON with Akamai-Bot info)
        - `extra_data_str` (JSON with recaptcha scores)
        - `category`, `alert_level` (alert metadata)
        """)

def display_security_dashboard(assessment):
    """Display the main security dashboard"""
    
    # Executive Summary
    st.markdown(create_executive_summary(assessment))
    
    st.markdown("---")
    
    # Detailed Analysis Sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Login Analysis
        if 'login_analysis' in assessment:
            display_login_analysis(assessment['login_analysis'])
        
        # Signup Analysis  
        if 'signup_analysis' in assessment:
            display_signup_analysis(assessment['signup_analysis'])
    
    with col2:
        # Suspicious Activity Analysis
        if 'suspicious_analysis' in assessment:
            display_suspicious_analysis(assessment['suspicious_analysis'])
        
        # Platform Breakdown
        display_platform_analysis(assessment)
    
    # Detailed IP Analysis
    st.markdown("---")
    display_detailed_ip_analysis(assessment)

def display_login_analysis(analysis):
    """Display login analysis details"""
    st.subheader("🔐 Login Activity Analysis")
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.metric("Total Logins", f"{analysis.get('total_logins', 0):,}")
        st.metric("Unique IPs", f"{analysis.get('unique_ips', 0):,}")
    
    with metrics_col2:
        success_rate = analysis.get('success_rate', 0)
        color = "red" if success_rate < 50 else "green"
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with metrics_col3:
        st.metric("Failed Logins", f"{analysis.get('failed_requests', 0):,}")
    
    # Attack indicators
    if analysis.get('attack_indicators'):
        with st.expander("🚨 Login Attack Indicators"):
            for indicator in analysis['attack_indicators']:
                st.write(f"**{indicator['ip']}** - {indicator['type']} (Count: {indicator['count']}) - **{indicator['risk']}**")

def display_signup_analysis(analysis):
    """Display signup analysis details"""
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
    
    # Attack indicators
    if analysis.get('attack_indicators'):
        with st.expander("🚨 Signup Attack Indicators"):
            for indicator in analysis['attack_indicators']:
                st.write(f"**{indicator['ip']}** - {indicator['type']} (Count: {indicator['count']}) - **{indicator['risk']}**")

def display_suspicious_analysis(analysis):
    """Display suspicious activity analysis"""
    st.subheader("🕵️ Suspicious Activity Analysis")
    
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.metric("Total Alerts", f"{analysis.get('total_alerts', 0):,}")
        st.metric("High-Risk IPs", f"{len(analysis.get('high_risk_ips', [])):,}")
    
    with metrics_col2:
        st.metric("Bot Detections", f"{len(analysis.get('bot_detections', [])):,}")
        st.metric("Recaptcha Failures", f"{len(analysis.get('recaptcha_failures', [])):,}")
    
    # Show sample of high-risk IPs
    if analysis.get('high_risk_ips'):
        with st.expander("🔍 High-Risk IP Details"):
            for ip in analysis['high_risk_ips'][:10]:  # Show first 10
                st.write(f"`{ip}`")

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

def display_detailed_ip_analysis(assessment):
    """Display detailed IP analysis for investigation"""
    st.subheader("🎯 Detailed IP Analysis for Investigation")
    
    # Compile all suspicious IPs with their reasons
    ip_details = {}
    
    for source_name, source_data in [('Login', assessment.get('login_analysis', {})),
                                   ('Signup', assessment.get('signup_analysis', {})),
                                   ('Suspicious', assessment.get('suspicious_analysis', {}))]:
        
        if 'attack_indicators' in source_data:
            for indicator in source_data['attack_indicators']:
                ip = indicator['ip']
                if ip not in ip_details:
                    ip_details[ip] = []
                ip_details[ip].append(f"{source_name}: {indicator['type']} (Risk: {indicator['risk']})")
        
        if 'high_risk_ips' in source_data:
            for ip in source_data['high_risk_ips']:
                if ip not in ip_details:
                    ip_details[ip] = []
                ip_details[ip].append(f"{source_name}: High-risk IP")
    
    # Display IPs sorted by risk level
    if ip_details:
        for ip, reasons in sorted(ip_details.items(), key=lambda x: len(x[1]), reverse=True)[:20]:  # Top 20
            with st.expander(f"`{ip}` - {len(reasons)} security events"):
                for reason in reasons:
                    st.write(f"- {reason}")
    else:
        st.info("No suspicious IPs detected in today's data.")

if __name__ == "__main__":
    main()
