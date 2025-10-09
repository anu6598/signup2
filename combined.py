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

st.set_page_config(layout="wide", page_title="360° API Security Dashboard")

# ------------------------------
# Helper functions
# ------------------------------
def parse_time_columns(df, time_col="start_time"):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    # Formats used for aggregation labels
    df['hour_label'] = df[time_col].dt.strftime('%H')
    df['minute_label'] = df[time_col].dt.strftime('%H:%M')
    df['second_label'] = df[time_col].dt.strftime('%H:%M:%S')
    return df

def agg_time_counts(df, time_unit):
    """Return aggregated counts grouped by time unit label"""
    if time_unit == 'hour':
        g = df.groupby('hour_label').size().reset_index(name='signup_count')
        g = g.sort_values('hour_label')
    elif time_unit == 'minute':
        g = df.groupby('minute_label').size().reset_index(name='signup_count')
        g = g.sort_values('minute_label')
    else:
        g = df.groupby('second_label').size().reset_index(name='signup_count')
        g = g.sort_values('second_label')
    return g

def make_hour_minute_second_plot(df, title_prefix, filt_mask=None, ip_col='true_client_ip'):
    """
    Build a Plotly figure with 6 traces: Hour bar + trend, Minute bar + trend, Second bar + trend.
    filt_mask: boolean mask on df (if not None, use df[filt_mask])
    """
    if filt_mask is not None:
        d = df[filt_mask].copy()
    else:
        d = df.copy()

    hourly = agg_time_counts(d, 'hour')
    minute = agg_time_counts(d, 'minute')
    second = agg_time_counts(d, 'second')

    fig = go.Figure()

    # Hour traces
    fig.add_trace(go.Bar(x=hourly['hour_label'], y=hourly['signup_count'],
                         name='Hourly Signups', visible=True))
    fig.add_trace(go.Scatter(x=hourly['hour_label'], y=hourly['signup_count'],
                             name='Hourly Trend', mode='lines+markers',
                             line=dict(color='red', dash='dash'), visible=True))

    # Minute traces
    fig.add_trace(go.Bar(x=minute['minute_label'], y=minute['signup_count'],
                         name='Minute Signups', visible=False))
    fig.add_trace(go.Scatter(x=minute['minute_label'], y=minute['signup_count'],
                             name='Minute Trend', mode='lines+markers',
                             line=dict(color='red', dash='dash'), visible=False))

    # Second traces
    fig.add_trace(go.Bar(x=second['second_label'], y=second['signup_count'],
                         name='Second Signups', visible=False))
    fig.add_trace(go.Scatter(x=second['second_label'], y=second['signup_count'],
                             name='Second Trend', mode='lines+markers',
                             line=dict(color='red', dash='dash'), visible=False))

    buttons = [
        dict(label="Hour",
             method="update",
             args=[{"visible": [True, True, False, False, False, False]},
                   {"title": f"{title_prefix} — per Hour"}]),

        dict(label="Minute",
             method="update",
             args=[{"visible": [False, False, True, True, False, False]},
                   {"title": f"{title_prefix} — per Minute"}]),

        dict(label="Second",
             method="update",
             args=[{"visible": [False, False, False, False, True, True]},
                   {"title": f"{title_prefix} — per Second"}])
    ]

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            x=1.02,
            y=1.12,
            showactive=True
        )],
        title=f"{title_prefix} — per Hour",
        xaxis_title="Time",
        yaxis_title="Signup Count",
        height=400,
        margin=dict(l=30, r=30, t=60, b=40)
    )
    return fig

def explain_rule_row(row, rule_name):
    if rule_name == '15min_>9':
        return f"IP {row['true_client_ip']} had {row['count_15min']} /user/signup calls within 15 minutes — exceeds threshold 9."
    if rule_name == '10min_>=5':
        return f"IP {row['true_client_ip']} had {row['count_10min']} /user/signup calls within 10 minutes — meets/exceeds threshold 5."
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

def normalize_dataframe(df_raw):
    cols_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_raw.columns}
    df_raw.rename(columns=cols_map, inplace=True)
    df = df_raw.copy()

    def pick_col(possible):
        for p in possible:
            if p in df.columns:
                return p
        return None

    col_ip = pick_col(["x_real_ip", "client_ip", "ip", "remote_addr", "true_client_ip"])
    col_device = pick_col(["dr_dv", "device_id", "device"])
    col_akamai_epd = pick_col(["akamai_epd", "epd", "akamai_proxy"])

    df["x_real_ip"] = df[col_ip].astype(str) if col_ip else "unknown"
    df["dr_dv"] = df[col_device].astype(str) if col_device else np.nan
    df["akamai_epd"] = df[col_akamai_epd] if col_akamai_epd else np.nan
    df["is_proxy"] = df["akamai_epd"].notna() & (df["akamai_epd"] != "")

    return df

def collapse_ip(ip, mask_octets=1):
    if not isinstance(ip, str) or "." not in ip:
        return ip
    parts = ip.split(".")
    for i in range(1, mask_octets+1):
        if len(parts) >= i:
            parts[-i] = "*"
    return ".".join(parts)

def ip_quality(df):
    stats = (
        df.groupby("x_real_ip")
        .agg(
            days_seen=("date", "nunique"),
            first_seen=("timestamp", "min"),
            last_seen=("timestamp", "max"),
            total_requests=("timestamp", "count"),
        )
        .reset_index()
    )
    stats["ip_age_days"] = (stats["last_seen"] - stats["first_seen"]).dt.days + 1
    stats["avg_reqs_per_day"] = stats["total_requests"] / stats["days_seen"].replace(0, 1)
    return stats

def detect_brute_force(df):
    brute_df = df[df['request_path'].str.contains("login", case=False, na=False)]
    if 'minute' not in df.columns and 'start_time' in df.columns:
        df['minute'] = df['start_time'].dt.floor('min')
    grouped = brute_df.groupby(['x_real_ip', 'minute']).size().reset_index(name='count')
    brute_ips = grouped[grouped['count'] > 5]['x_real_ip'].unique()
    return df[df['x_real_ip'].isin(brute_ips)]

def detect_vpn_geo(df):
    if 'dr_uid' in df.columns:
        geo = df.groupby('dr_uid')['x_country_code'].nunique().reset_index()
        flagged = geo[geo['x_country_code'] > 2]['dr_uid']
        return df[df['dr_uid'].isin(flagged)]
    return pd.DataFrame()

def detect_bots(df):
    suspicious_ua = df['user_agent'].str.contains("bot|curl|python|scrapy|wget", case=False, na=False)
    low_duration = False
    if 'duration' in df.columns:
        low_duration = df['duration'].astype(float) < 0.3
    return df[suspicious_ua | low_duration]

def detect_ddos(df):
    if 'minute' not in df.columns and 'start_time' in df.columns:
        df['minute'] = df['start_time'].dt.floor('min')
    volume = df.groupby(['x_real_ip', 'minute']).size().reset_index(name='count')
    high_vol_ips = volume[volume['count'] > 10]['x_real_ip'].unique()
    return df[df['x_real_ip'].isin(high_vol_ips)]

def summarize_detection(name, df, platform_col='platform'):
    if platform_col not in df.columns:
        df[platform_col] = 'web'
    summary = df.groupby(platform_col).size().reset_index(name='Suspicious Requests')
    summary['attack_type'] = name
    return summary

def user_ip_summary(df, name):
    if 'dr_uid' not in df.columns:
        df['dr_uid'] = "unknown"
    return (
        df.groupby('x_real_ip')['dr_uid']
        .nunique()
        .reset_index(name='Unique Users')
        .sort_values('Unique Users', ascending=False)
        .assign(attack_type=name)
    )

def generate_future_predictions(df):
    """Generate mock future predictions for the next 10 days"""
    if 'date' not in df.columns or df.empty:
        # Create mock data if no date column
        last_date = datetime.now().date()
        dates = [last_date + timedelta(days=i) for i in range(10)]
        # Simulate some pattern with noise
        base_signups = len(df) if not df.empty else 1000
        predictions = [base_signups * (0.8 + 0.4 * np.sin(i/3) + 0.1 * np.random.random()) for i in range(10)]
    else:
        # Use actual data patterns for prediction
        daily_counts = df.groupby('date').size()
        if len(daily_counts) > 1:
            # Simple trend projection
            last_count = daily_counts.iloc[-1]
            trend = np.mean(np.diff(daily_counts.tail(3))) if len(daily_counts) > 1 else 0
            last_date = pd.to_datetime(daily_counts.index[-1]).date()
            dates = [last_date + timedelta(days=i+1) for i in range(10)]
            predictions = [max(0, last_count + trend * (i+1) * (0.9 + 0.2 * np.random.random())) for i in range(10)]
        else:
            last_date = datetime.now().date()
            dates = [last_date + timedelta(days=i) for i in range(10)]
            base_signups = len(df) if not df.empty else 1000
            predictions = [base_signups * (0.8 + 0.4 * np.sin(i/3) + 0.1 * np.random.random()) for i in range(10)]
    
    return pd.DataFrame({'date': dates, 'predicted_signups': predictions})

def show_daily_stats(df):
    st.title("📊 Daily Stats")

    if df is None or df.empty:
        st.warning("No data available — please upload CSV from Main Dashboard.")
        return

    # ✅ Preview
    st.subheader("Daily OTP Abuse Statistics")
    st.write("Preview of Data", df.head())

    # ✅ Daily counts chart (if 'date' column exists)
    if "date" in df.columns:
        daily_counts = df.groupby("date").size().reset_index(name="count")
        st.bar_chart(daily_counts.set_index("date")["count"])
    else:
        st.warning("⚠️ No 'date' column found in dataset, skipping daily counts chart.")

    # ✅ Categorization
    st.subheader("🚨 Final Daily Categorization")
    final_categories = []

    # Precompute is_proxy once for the whole df
    if "akamai_epd" in df.columns:
        epd_norm = df["akamai_epd"].astype(str).str.strip().str.lower()
        df["is_proxy"] = ~epd_norm.isin(["-", "rp", ""])
    else:
        df["is_proxy"] = False

    for day, group in df.groupby("date") if "date" in df.columns else []:
        total_otps = len(group)

        # Safe extraction with defaults
        max_requests_ip = group["x_real_ip"].value_counts().max() if "x_real_ip" in group else 0
        max_requests_device = group["dr_dv"].value_counts().max() if "dr_dv" in group else 0

        # proxy ratio per group
        proxy_ratio = group["is_proxy"].mean() * 100 if "is_proxy" in group else 0.0

        # Rule categorization
        if (total_otps > 1000) and (max_requests_ip > 25) and (proxy_ratio > 20) and (max_requests_device > 15):
            category = "OTP Abuse/Attack detected"
        elif (max_requests_ip > 25) and (total_otps > 1000) and (max_requests_device > 15):
            category = "HIGH OTP request detected"
        elif proxy_ratio > 20:
            category = "HIGH proxy status detected"
        else:
            category = "No suspicious activity detected"

        final_categories.append({
            "date": day,
            "category": category,
            "total_otps": total_otps,
            "max_requests_ip": max_requests_ip,
            "max_requests_device": max_requests_device,
            "proxy_ratio": proxy_ratio
        })

    # ✅ Display results
    if final_categories:
        st.dataframe(pd.DataFrame(final_categories), use_container_width=True)
    else:
        st.info("No daily categorization results available.")

# ------------------------------
# Main Application
# ------------------------------
def main():
    st.sidebar.title("🔐 API Security Dashboard")
    page = st.sidebar.radio("Navigation", [
        "ML Predictions & Overview", 
        "Signup Anomaly Detection", 
        "OTP Abuse Detection", 
        "360° Attack Detection",
        "Daily Stats"
    ])

    # Initialize session state
    if "df" not in st.session_state:
        st.session_state.df = None

    # File upload in sidebar
    uploaded_file = st.sidebar.file_uploader("📁 Upload API Logs CSV", type=["csv"])
    
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state.df = normalize_dataframe(df_raw)
        
        # Ensure timestamp parsing
        if 'start_time' in st.session_state.df.columns:
            st.session_state.df['start_time'] = pd.to_datetime(st.session_state.df['start_time'], errors='coerce')
        if 'timestamp' not in st.session_state.df.columns and 'start_time' in st.session_state.df.columns:
            st.session_state.df['timestamp'] = st.session_state.df['start_time']
        if 'date' not in st.session_state.df.columns and 'timestamp' in st.session_state.df.columns:
            st.session_state.df['date'] = st.session_state.df['timestamp'].dt.date

    # Page routing
    if page == "ML Predictions & Overview":
        show_ml_predictions()
    elif page == "Signup Anomaly Detection":
        show_signup_anomalies()
    elif page == "OTP Abuse Detection":
        show_otp_abuse()
    elif page == "360° Attack Detection":
        show_360_attack()
    elif page == "Daily Stats":
        show_daily_stats(st.session_state.df)

def show_ml_predictions():
    st.title("🤖 ML Predictions & Daily Overview")
    
    if st.session_state.df is None:
        st.info("👆 Please upload a CSV file to begin analysis.")
        return
        
    df = st.session_state.df
    
    # Generate predictions
    st.header("📈 Signup Predictions for Next 10 Days")
    predictions_df = generate_future_predictions(df)
    
    # Interactive prediction chart
    fig = px.line(predictions_df, x='date', y='predicted_signups', 
                  title='Predicted Signups for Next 10 Days',
                  markers=True)
    fig.update_traces(line=dict(color='#FF4B4B', width=3))
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Predicted Signups",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Common Stats
    st.header("📊 Today's Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_signups = len(df[df['request_path'].str.contains('/user/signup', na=False)]) if 'request_path' in df.columns else len(df)
        st.metric("Total Signups", f"{total_signups:,}")
    
    with col2:
        unique_devices = df['dr_dv'].nunique() if 'dr_dv' in df.columns else 0
        st.metric("Unique Devices", f"{unique_devices:,}")
    
    with col3:
        unique_ips = df['x_real_ip'].nunique() if 'x_real_ip' in df.columns else 0
        st.metric("Unique IPs", f"{unique_ips:,}")
    
    with col4:
        if 'date' in df.columns:
            today = datetime.now().date()
            today_data = df[df['date'] == today]
            today_signups = len(today_data[today_data['request_path'].str.contains('/user/signup', na=False)]) if 'request_path' in today_data.columns else len(today_data)
            st.metric("Today's Signups", f"{today_signups:,}")
        else:
            st.metric("Today's Data", "N/A")
    
    # Attack Status Table
    st.header("🚨 Today's Attack Assessment")
    
    # Calculate metrics for assessment
    if 'request_path' in df.columns:
        otp_requests = len(df[df['request_path'].str.contains('otp', case=False, na=False)])
        login_requests = len(df[df['request_path'].str.contains('login', case=False, na=False)])
        signup_requests = len(df[df['request_path'].str.contains('signup', case=False, na=False)])
    else:
        otp_requests = login_requests = signup_requests = 0
    
    # Proxy ratio
    proxy_ratio = df['is_proxy'].mean() * 100 if 'is_proxy' in df.columns else 0
    
    # Max requests per IP
    max_requests_ip = df['x_real_ip'].value_counts().max() if 'x_real_ip' in df.columns else 0
    
    # Assessment logic
    attack_risk = "LOW"
    reasons = []
    
    if max_requests_ip > 100:
        attack_risk = "HIGH"
        reasons.append(f"Single IP made {max_requests_ip} requests")
    elif proxy_ratio > 30:
        attack_risk = "HIGH" 
        reasons.append(f"High proxy usage ({proxy_ratio:.1f}%)")
    elif max_requests_ip > 50:
        attack_risk = "MEDIUM"
        reasons.append(f"Single IP made {max_requests_ip} requests")
    elif proxy_ratio > 15:
        attack_risk = "MEDIUM"
        reasons.append(f"Moderate proxy usage ({proxy_ratio:.1f}%)")
    
    if not reasons:
        reasons.append("Normal traffic patterns detected")
    
    # Display assessment
    risk_color = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}
    
    st.markdown(f"""
    <div style="background-color: {risk_color[attack_risk]}20; padding: 20px; border-radius: 10px; border-left: 5px solid {risk_color[attack_risk]};">
        <h3 style="color: {risk_color[attack_risk]}; margin-top: 0;">Attack Risk: {attack_risk}</h3>
        <p><strong>Reasons:</strong> {', '.join(reasons)}</p>
        <p><strong>Total Requests:</strong> {len(df):,} | <strong>OTP Requests:</strong> {otp_requests:,} | <strong>Login Requests:</strong> {login_requests:,}</p>
    </div>
    """, unsafe_allow_html=True)

def show_signup_anomalies():
    st.title("🚨 Signup Anomaly Detection Dashboard")
    
    if st.session_state.df is None:
        st.info("👆 Please upload a CSV file to begin analysis.")
        return
        
    df = st.session_state.df
    
    # Header with info
    left_col, right_col = st.columns([3, 1])
    
    with left_col:
        st.markdown(
            "<h1 style='margin:0; color:#0B486B;'>🚨 Signup Anomaly Detection Dashboard</h1>",
            unsafe_allow_html=True
        )
        st.markdown(
            "<p style='margin-top:6px; color:#555;'>Upload signup logs and inspect spikes & anomalies across multiple indicators, focusing on <b>start_time vs signup count</b>.</p>",
            unsafe_allow_html=True
        )

    with right_col:
        st.markdown(
            """
            <div style="background:#e6ffea; padding:12px; border-radius:8px; border:1px solid #ccefd9;">
                <h3 style="margin:0 0 6px 0;">ℹ️ How This Works</h3>
                <div style="font-size:13px; color:#222; line-height:1.35;">
                    <b>Rule-based</b>: Fast deterministic checks based on <b>IP address</b>. 
                    Flags IPs with more than <b>9 signups in 15 minutes</b> or <b>5 or more signups in 10 minutes</b>.<br>
                    <b>ML-based</b>: Uses <b>IsolationForest</b> to analyze signup activity from each IP in short windows and detect statistical outliers based on historical patterns.<br>
                    Click indicator cards below to explore hour/minute/second views.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Parse times and add labels
    df = parse_time_columns(df, time_col='start_time')

    # Make sure necessary columns exist; create fallbacks if missing
    for col in ['true_client_ip', 'request_path', 'response_code', 'user_agent', 'dr_dv', 'dr_app_version', 'x_country_code', 'akamai_bot', 'dr_platform']:
        if col not in df.columns:
            df[col] = np.nan

    # Precompute rolling counts
    df = df.sort_values(['true_client_ip', 'start_time']).reset_index(drop=True)
    df['__ts'] = (df['start_time'].astype('int64') // 10**9).astype(np.int64)

    grouped_times = df.groupby('true_client_ip')['__ts'].apply(list).to_dict()

    rows = []
    for ip, times in grouped_times.items():
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

    counts_df = pd.DataFrame(rows)
    df = df.merge(counts_df, on=['true_client_ip', '__ts'], how='left')
    df['count_15min'] = df['count_15min'].fillna(0).astype(int)
    df['count_10min'] = df['count_10min'].fillna(0).astype(int)

    # Show Daily Signup Counts
    st.subheader("📅 Daily Total Signups")

    # Make sure date column exists
    if 'date' not in df.columns and 'start_time' in df.columns:
        df['date'] = pd.to_datetime(df['start_time']).dt.date

    if 'date' in df.columns:
        daily_signups = df.groupby(df['date'])['request_path'].count().reset_index(name='total_signups')
        st.dataframe(daily_signups, use_container_width=True)
    else:
        st.warning("No date column found for daily analysis")

    # IP Summary
    st.subheader("🗻 Total Signups Per IP")
    ip_table = (
        df.groupby("true_client_ip")["request_path"]
        .count()
        .reset_index(name="request_count")
        .sort_values("request_count", ascending=False)
    )
    st.dataframe(ip_table)

    # Sidebar IP filter
    ip_input = st.sidebar.text_input("Enter IP to filter")
    if ip_input:
        df = df[df['true_client_ip'] == ip_input]

    # Adaptive Time Series
    st.header("1 Adaptive Time Series")
    
    if 'start_time' in df.columns and 'true_client_ip' in df.columns:
        df_grouped = (
            df.groupby([pd.Grouper(key='start_time', freq='1min'), 'true_client_ip'])
            .size()
            .reset_index(name='signup_count')
        )
        
        df_grouped = df_grouped.dropna(subset=['true_client_ip', 'start_time'])

        if not df_grouped.empty:
            fig = px.scatter(
                df_grouped,
                x="start_time",
                y="signup_count",
                size="signup_count",
                color="true_client_ip",
                hover_data={
                    "start_time": True,
                    "signup_count": True,
                    "true_client_ip": True
                },
                title="Adaptive Time Series: Time vs Signup Count"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the Adaptive Time Series chart.")
    else:
        st.error("Missing required columns for time series analysis")

    # Rule-based anomalies
    st.header("2) Rule-Based Anomalies")
    
    # Rule: >9 in 15 minutes
    rb_15 = df[df['count_15min'] > 9].copy()
    if not rb_15.empty:
        rb_15['explanation'] = rb_15.apply(lambda r: explain_rule_row(r, '15min_>9'), axis=1)
        st.subheader("Rule: more than 9 signups in 15 minutes")
        st.dataframe(rb_15[['start_time', 'true_client_ip', 'request_path', 'count_15min', 'user_agent', 'explanation']].sort_values('count_15min', ascending=False))
    else:
        st.success("No IPs exceed 9 signups in 15 minutes.")

    # Rule: >=5 in 10 minutes
    rb_10 = df[df['count_10min'] >= 5].copy()
    if not rb_10.empty:
        rb_10['explanation'] = rb_10.apply(lambda r: explain_rule_row(r, '10min_>=5'), axis=1)
        st.subheader("Rule: 5 or more signups in 10 minutes")
        st.dataframe(rb_10[['start_time', 'true_client_ip', 'request_path', 'count_10min', 'user_agent', 'explanation']].sort_values('count_10min', ascending=False))
    else:
        st.info("No IPs with 5 or more signups in 10 minutes.")

    # ML anomalies
    st.header("3) Machine Learning Detected Anomalies")
    
    features = df[['count_15min', 'count_10min']].fillna(0)
    iso = IsolationForest(contamination=0.01, random_state=42)
    df['ml_flag'] = iso.fit_predict(features)

    anomalies_ml = df[df['ml_flag'] == -1].copy()
    median_15 = max(1.0, df['count_15min'].median())
    median_10 = max(1.0, df['count_10min'].median())

    if not anomalies_ml.empty:
        anomalies_ml['reason'] = anomalies_ml.apply(lambda r: explain_ml_row(r, median_15, median_10), axis=1)
        st.dataframe(anomalies_ml[['start_time', 'true_client_ip', 'request_path', 'count_15min', 'count_10min', 'user_agent', 'akamai_bot', 'reason']].sort_values(['count_15min','count_10min'], ascending=False))
    else:
        st.success("Isolation Forest did not detect anomalies in this dataset.")

    # ML scatter plot
    df['anomaly_score'] = iso.decision_function(features)
    fig_ml_scatter = go.Figure()
    fig_ml_scatter.add_trace(go.Scatter(
        x=df['count_15min'],
        y=df['count_10min'],
        mode='markers',
        marker=dict(
            size=8,
            color=df['anomaly_score'],
            colorscale='RdBu',
            colorbar=dict(title="Anomaly Score"),
            line=dict(width=0.5, color='black')
        ),
        text=df['true_client_ip'],
        name='All Points',
        hovertemplate='IP: %{text}<br>15-min Count: %{x}<br>10-min Count: %{y}<br>Score: %{marker.color:.3f}<extra></extra>'
    ))

    anom_points = df[df['ml_flag'] == -1]
    fig_ml_scatter.add_trace(go.Scatter(
        x=anom_points['count_15min'],
        y=anom_points['count_10min'],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Anomalies'
    ))

    fig_ml_scatter.update_layout(
        title="Isolation Forest Feature Space",
        xaxis_title="Count in 15 min",
        yaxis_title="Count in 10 min",
        height=450
    )
    st.plotly_chart(fig_ml_scatter, use_container_width=True)

def show_otp_abuse():
    st.title("🔐 OTP Abuse Detection Dashboard")
    
    if st.session_state.df is None:
        st.info("👆 Please upload a CSV file to begin analysis.")
        return
        
    df = st.session_state.df
    
    # OTP-specific processing
    if 'request_path' in df.columns:
        df["is_otp_or_login"] = df["request_path"].str.contains("otp|login", case=False, na=False)
        otp_login_df = df[df["is_otp_or_login"]].copy()
    else:
        otp_login_df = df.copy()
    
    st.write(f"Detected {len(otp_login_df)} OTP/Login-related rows out of {len(df)} total rows.")
    
    if len(otp_login_df) == 0:
        st.warning("No OTP/login requests found in the data.")
        return

    # Extract BMP scores
    def extract_digits(x):
        m = re.search(r"(\d+)", str(x))
        return float(m.group(1)) if m else np.nan

    if 'akamai_bot' in otp_login_df.columns:
        otp_login_df["bmp_score"] = otp_login_df["akamai_bot"].apply(extract_digits)

    # Minute bucket analysis
    if 'timestamp' in otp_login_df.columns:
        otp_login_df["minute_bucket"] = otp_login_df["timestamp"].dt.floor("T")
        
        grouped = (
            otp_login_df.groupby(["x_real_ip", "minute_bucket"], as_index=False)
            .agg(
                login_attempts=("request_path", "count"),
                akamai_epd=("akamai_epd", lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan)
            )
            .sort_values(["x_real_ip", "minute_bucket"])
        )

        # Threshold controls in sidebar
        burst_threshold = st.sidebar.number_input("Burst threshold (OTPs within 10 min)", value=10, step=1)
        burst_window_mins = st.sidebar.number_input("Burst window (minutes)", value=10, step=1)

        # Compute rolling attempts
        def compute_rolling_attempts(g, window_minutes=10):
            g = g.set_index("minute_bucket").sort_index()
            g["attempts_in_10_min"] = g["login_attempts"].rolling(f"{window_minutes}T").sum()
            g = g.reset_index()
            return g

        grouped_rolled = grouped.groupby("x_real_ip", group_keys=False).apply(compute_rolling_attempts).reset_index(drop=True)

        # Proxy logic
        def is_proxy_ip(series):
            epd_norm = series.astype(str).str.strip().str.lower()
            return (~epd_norm.isin(["-", "rp", ""])).any()

        proxy_by_ip = otp_login_df.groupby("x_real_ip")["akamai_epd"].apply(is_proxy_ip).rename("is_proxy_ip")
        grouped_rolled = grouped_rolled.join(proxy_by_ip, on="x_real_ip")

        proxy_repeat_threshold = st.sidebar.number_input("Proxy hits threshold", value=1, step=1)
        
        proxy_hits = (
            otp_login_df.assign(
                is_proxy_row=~otp_login_df["akamai_epd"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["-", "rp", ""])
            )
            .
