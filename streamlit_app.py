# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from datetime import datetime

st.set_page_config(layout="wide", page_title="Signup Anomaly Dashboard")

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

def agg_time_counts_by_ip(df, time_unit, ip_col='true_client_ip'):
    """Return aggregated counts grouped by IP and time label (then aggregated per IP)"""
    label = 'hour_label' if time_unit == 'hour' else ('minute_label' if time_unit == 'minute' else 'second_label')
    g = df.groupby([ip_col, label]).size().reset_index(name='signup_count')
    # For the per-IP summary chart we want sum per IP across the selected unit
    ip_sum = g.groupby(ip_col)['signup_count'].sum().reset_index()
    ip_sum = ip_sum.sort_values('signup_count', ascending=False)
    return ip_sum, g  # ip_sum used for bar/line, g is detailed per-ip/time if needed

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
    # rule_name: '15min_>9' or '10min_>=5'
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
    # other heuristics
    if (row.get('akamai_bot') is not None) and (str(row.get('akamai_bot')).lower() != '-' and 'bot' in str(row.get('akamai_bot')).lower()):
        reasons.append("Akamai bot indicator present")
    if not reasons:
        reasons.append("Statistical outlier detected by IsolationForest on (count_15min, count_10min).")
    return "; ".join(reasons)

# ------------------------------
# Layout: header + static green box on right (no overlap)
# ------------------------------
left_col, right_col = st.columns([3, 1])
with left_col:
    st.markdown("<h1 style='margin:0; color:#0B486B;'>🚨 Signup Anomaly Detection Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin-top:6px; color:#555;'>Upload signup logs and inspect spikes & anomalies across multiple indicators.</p>", unsafe_allow_html=True)
with right_col:
    st.markdown(
        """
        <div style="background:#e6ffea; padding:12px; border-radius:8px; border:1px solid #ccefd9;">
            <h3 style="margin:0 0 6px 0;">ℹ️ How This Works</h3>
            <div style="font-size:13px; color:#222; line-height:1.35;">
                <b>Rule-based</b>: fast deterministic checks (e.g., >9 signups in 15min).<br>
                <b>ML-based</b>: IsolationForest on short-window counts to find statistical outliers.<br>
                Click indicator cards below to explore hour/minute/second views.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# ------------------------------
# File upload
# ------------------------------
uploaded_file = st.file_uploader("Upload signup CSV (columns: start_time, true_client_ip, request_path, response_code, user_agent, dr_dv, dr_app_version, x_country_code, akamai_bot, dr_platform)", type=["csv"])
if not uploaded_file:
    st.info("Upload a CSV to enable charts and anomaly detection (example columns listed above).")
    st.stop()

# read CSV
df_raw = pd.read_csv(uploaded_file)
# normalize column names to lower-case stripped
df_raw.columns = [c.strip() for c in df_raw.columns]

# parse times and add labels
df = parse_time_columns(df_raw, time_col='start_time')

# make sure necessary columns exist; create fallbacks if missing
for col in ['true_client_ip', 'request_path', 'response_code', 'user_agent', 'dr_dv', 'dr_app_version', 'x_country_code', 'akamai_bot', 'dr_platform']:
    if col not in df.columns:
        df[col] = np.nan

# ------------------------------
# Precompute rolling counts robustly (searchsorted approach)
# ------------------------------
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

# ------------------------------
# Section 2: Big adaptive time-series scatter (IP on Y, time X, bubble size = count)
# ------------------------------
st.header("1) Adaptive Time Series (IP vs Time, bubble size = signup count)")

agg_counts = df.groupby(['start_time', 'true_client_ip']).size().reset_index(name='count')

if agg_counts.empty:
    st.info("No signup records found to plot.")
else:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=agg_counts['start_time'],
        y=agg_counts['true_client_ip'],
        mode='markers',
        marker=dict(size=np.clip(agg_counts['count'] * 8, 6, 60),
                    color=agg_counts['count'],
                    colorscale='Viridis',
                    showscale=True),
        hovertemplate='IP: %{y}<br>Time: %{x}<br>Count: %{marker.color}<extra></extra>'
    ))
    fig_scatter.update_layout(
        height=420,
        margin=dict(l=40, r=20, t=40, b=80),
        xaxis=dict(title='Time'),
        yaxis=dict(title='IP (true_client_ip)'),
        title='Signup counts per IP over time (bubble size = count)'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ------------------------------
# Section 3: Rule-based anomalies (two tables with explanation)
# ------------------------------
st.header("2) Rule-Based Anomalies (auto-generated explanations)")

# Rule: >9 in 15 minutes
rb_15 = df[df['count_15min'] > 9].copy()
if not rb_15.empty:
    rb_15['explanation'] = rb_15.apply(lambda r: explain_rule_row(r, '15min_>9'), axis=1)
    st.subheader("Rule: more than 9 signups in 15 minutes")
    st.write("Meaning: These rows show IPs that crossed 9 signups within a 15-minute window — likely a brute-force/signup flood.")
    st.dataframe(rb_15[['start_time', 'true_client_ip', 'request_path', 'count_15min', 'user_agent', 'explanation']].sort_values('count_15min', ascending=False))
else:
    st.success("No IPs exceed 9 signups in 15 minutes.")

# Rule: >=5 in 10 minutes
rb_10 = df[df['count_10min'] >= 5].copy()
if not rb_10.empty:
    rb_10['explanation'] = rb_10.apply(lambda r: explain_rule_row(r, '10min_>=5'), axis=1)
    st.subheader("Rule: 5 or more signups in 10 minutes")
    st.write("Meaning: These IPs performed at least 5 signups within 10 minutes — suspicious, worth manual verification.")
    st.dataframe(rb_10[['start_time', 'true_client_ip', 'request_path', 'count_10min', 'user_agent', 'explanation']].sort_values('count_10min', ascending=False))
else:
    st.info("No IPs with 5 or more signups in 10 minutes.")

# ------------------------------
# Section 4: ML anomalies + explanation column
# ------------------------------
st.header("3) Machine Learning Detected Anomalies (Isolation Forest)")

# Build features and run IsolationForest on unique (ip,event) rows or per-row features
features = df[['count_15min', 'count_10min']].fillna(0)
iso = IsolationForest(contamination=0.01, random_state=42)
df['ml_flag'] = iso.fit_predict(features)  # -1 => anomaly

anomalies_ml = df[df['ml_flag'] == -1].copy()

median_15 = max(1.0, df['count_15min'].median())
median_10 = max(1.0, df['count_10min'].median())

if not anomalies_ml.empty:
    anomalies_ml['reason'] = anomalies_ml.apply(lambda r: explain_ml_row(r, median_15, median_10), axis=1)
    st.write("The ML model (Isolation Forest) looks at short-window signup counts per IP and flags statistical outliers. Below are rows flagged by the model along with a short human-readable reason.")
    st.dataframe(anomalies_ml[['start_time', 'true_client_ip', 'request_path', 'count_15min', 'count_10min', 'user_agent', 'akamai_bot', 'reason']].sort_values(['count_15min','count_10min'], ascending=False))
else:
    st.success("Isolation Forest did not detect anomalies in this dataset.")

# --- Additional ML model diagnostics ---
# Calculate anomaly scores
df['anomaly_score'] = iso.decision_function(features)
df['anomaly_score_norm'] = (df['anomaly_score'] - df['anomaly_score'].min()) / (df['anomaly_score'].max() - df['anomaly_score'].min())

# Scatter plot: counts vs counts, colored by anomaly status
fig_ml_scatter = go.Figure()
fig_ml_scatter.add_trace(go.Scatter(
    x=df['count_15min'],
    y=df['count_10min'],
    mode='markers',
    marker=dict(
        size=8,
        color=df['anomaly_score'], # continuous color scale
        colorscale='RdBu',
        colorbar=dict(title="Anomaly Score"),
        line=dict(width=0.5, color='black')
    ),
    text=df['true_client_ip'],
    name='All Points',
    hovertemplate='IP: %{text}<br>15-min Count: %{x}<br>10-min Count: %{y}<br>Score: %{marker.color:.3f}<extra></extra>'
))

# Highlight anomalies
anom_points = df[df['ml_flag'] == -1]
fig_ml_scatter.add_trace(go.Scatter(
    x=anom_points['count_15min'],
    y=anom_points['count_10min'],
    mode='markers',
    marker=dict(size=10, color='red', symbol='x'),
    name='Anomalies',
    text=anom_points['true_client_ip'],
    hovertemplate='IP: %{text}<br>15-min Count: %{x}<br>10-min Count: %{y}<extra></extra>'
))

fig_ml_scatter.update_layout(
    title="Isolation Forest Feature Space (Anomalies Highlighted)",
    xaxis_title="Count in 15 min",
    yaxis_title="Count in 10 min",
    height=450
)

st.plotly_chart(fig_ml_scatter, use_container_width=True)

# Show model parameters & contamination info
st.caption(f"IsolationForest trained with contamination={iso.contamination}, random_state=42. "
           "Anomalies are points with ML flag = -1, usually having negative decision_function scores.")

# ------------------------------
# Step 5: Label generation for supervised training
# ------------------------------

# Start with a clean label column: 0 = normal
df['label'] = 0

# Rule-based flags
rule15_ips = set(rb_15['true_client_ip'].unique())
rule10_ips = set(rb_10['true_client_ip'].unique())

# ML-based flags
ml_ips = set(anomalies_ml['true_client_ip'].unique())

# Mark IPs as malicious if they appear in any detection set
df.loc[df['true_client_ip'].isin(rule15_ips.union(rule10_ips).union(ml_ips)), 'label'] = 1

# Optional: add reason text combining all sources
def combined_reason(row):
    reasons = []
    if row['true_client_ip'] in rule15_ips:
        reasons.append("Rule15 (>9 signups in 15 min)")
    if row['true_client_ip'] in rule10_ips:
        reasons.append("Rule10 (>=5 signups in 10 min)")
    if row['true_client_ip'] in ml_ips:
        reasons.append(f"ML anomaly (score={row['anomaly_score']:.3f})")
    return "; ".join(reasons) if reasons else "Normal"

df['label_reason'] = df.apply(combined_reason, axis=1)

# Show label distribution
st.header("5) Final Labelled Dataset")
label_counts = df['label'].value_counts().rename({0: 'Normal', 1: 'Malicious'})
st.bar_chart(label_counts)

st.dataframe(df[['start_time', 'true_client_ip', 'count_15min', 'count_10min',
                 'user_agent', 'akamai_bot', 'label', 'label_reason']].sort_values('label', ascending=False))

# ------------------------------
# Section 5: 12 Indicator interactive plots (3 per row)
# ------------------------------
st.header("4) Indicator Explorer (12 indicators — interactive)")

# We will create 12 indicators with a brief description and a Plotly interactive chart (Hour/Minute/Second toggle).
# Define indicator filters and descriptions
indicators = [
    ("High Volume: frequent /user/signup", df['request_path'].astype(str).str.contains('/user/signup', na=False),
     "Counts of request_path = /user/signup; large spikes indicate times with heavy signup traffic."),
    ("Same IP sending multiple signups", df['true_client_ip'].notna(),
     "Counts grouped by IP show if single IP is responsible for many signups."),
    ("Tightly clustered timestamps (10+ in a minute)", df['count_10min'] >= 10,
     "Rows where 10-minute window count >= 10; indicates tightly clustered requests."),
    ("Off-peak (midnight-4am) activity", df['start_time'].dt.hour.isin([0,1,2,3,4]),
     "Signups during late-night hours which may be unusual for your product."),
    ("Same IP, multiple user_agents", df['true_client_ip'].notna(),
     "Detect IPs that have multiple distinct user_agent values — could be distributed/bot activity."),
    ("IPs from unexpected geolocations", df['x_country_code'].notna(),
     "Check country codes that are not part of your normal user base."),
    ("Missing/malformed headers", df['user_agent'].isna() | df['user_agent'].astype(str).str.strip().eq(''),
     "Missing or blank user_agent or other header fields."),
    ("Known bot indicators (Akamai)", df['akamai_bot'].notna(),
     "Akamai bot field present (string contains 'bot' or non-empty) — potentially automated traffic."),
    ("Unusual app versions/platforms", df['dr_app_version'].isna() | df['dr_platform'].isna(),
     "Null or uncommon app version / platform values."),
    ("Rapid switching of platforms", df['true_client_ip'].notna(),
     "Same IP showing multiple dr_platform values within a short time window."),
    ("High OTP request frequency (/user/send-otp)", df['request_path'].astype(str).str.contains('/user/send-otp', na=False),
     "Multiple /user/send-otp hits could indicate OTP abuse or credential stuffing."),
    ("Failed signups (response_code != 200)", df['response_code'].notna() & (df['response_code'].astype(str) != '200'),
     "Repeated failed signup attempts (HTTP status not 200) may indicate someone attempting to brute force.")
]

# We'll arrange 4 rows * 3 columns
cols_per_row = 3
for i in range(0, len(indicators), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, indicator in enumerate(indicators[i:i+cols_per_row]):
        title, mask_expr, desc = indicator
        with cols[j]:
            st.markdown(f"### {title}")
            st.caption(desc)
            # Build figure: for some indicators mask_expr is a boolean Series; if it's 'broad' we keep df
            try:
                mask = mask_expr if isinstance(mask_expr, pd.Series) else (mask_expr)
                # If the mask is all True (like 'notna') we still want to show relevant aggregation — but we can also keep filter None
                if mask is None:
                    mask = pd.Series([True]*len(df), index=df.index)
            except Exception:
                mask = pd.Series([True]*len(df), index=df.index)

            fig = make_hour_minute_second_plot(df, title, filt_mask=mask)
            st.plotly_chart(fig, use_container_width=True)


