import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
import plotly.express as px

st.set_page_config(page_title='IP Suspicion Dashboard', layout='wide')
st.title('🔐 IP Suspicion & Analysis Dashboard')

# ---------- Upload CSV ----------
uploaded_file = st.file_uploader('Upload attack-day CSV', type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, low_memory=False)
    st.success(f'Loaded {len(df)} rows and {len(df.columns)} columns.')

    # ---------- Normalize Columns ----------
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]

    # Detect key columns (simplified)
    col_ts = next((c for c in ['timestamp','time','created_on','start_time','event_time'] if c in df.columns), None)
    col_ip = next((c for c in ['x_real_ip','true_client_ip','client_ip','ip','remote_addr'] if c in df.columns), None)
    col_user = next((c for c in ['user_id','user','username','dr_uid'] if c in df.columns), None)
    col_status = next((c for c in ['response_code','status','result','response'] if c in df.columns), None)
    col_bot = next((c for c in ['akamai_bot','akamai_bmp','bmp','bot_info'] if c in df.columns), None)

    # ---------- Basic Normalization ----------
    df['ts'] = pd.to_datetime(df[col_ts], errors='coerce') if col_ts else pd.NaT
    df = df.dropna(subset=['ts'])
    df['ip_addr'] = df[col_ip].astype(str) if col_ip else 'unknown'
    df['username'] = df[col_user].astype(str) if col_user else 'unknown'
    df['status_raw'] = df[col_status].astype(str) if col_status else ''
    df['akamai_bot'] = df[col_bot].astype(str) if col_bot else ''

    # ---------- Feature Engineering (Point 1) ----------
    def is_failure(x):
        try:
            return int(x) >= 400
        except:
            return any(k in str(x).lower() for k in ['fail','error','unauthorized','invalid'])
    df['is_fail'] = df['status_raw'].apply(is_failure)

    def extract_digits(s):
        m = re.search(r'(\d+)', str(s))
        return int(m.group(1)) if m else np.nan
    df['bmp_score'] = df['akamai_bot'].apply(extract_digits)

    # ---------- Temporal Features (Points 2 & 3) ----------
    df = df.sort_values('ts')
    df['minute_floor'] = df['ts'].dt.floor('T')
    df['min10_floor'] = df['ts'].dt.floor('10T')

    # Short-window burst counts
    cnt_1m = df.groupby(['ip_addr','minute_floor']).size().reset_index(name='cnt_1m')
    max_1m = cnt_1m.groupby('ip_addr')['cnt_1m'].max().reset_index(name='max_attempts_1m')
    cnt_10m = df.groupby(['ip_addr','min10_floor']).size().reset_index(name='cnt_10m')
    max_10m = cnt_10m.groupby('ip_addr')['cnt_10m'].max().reset_index(name='max_attempts_10m')
    sum_10m = cnt_10m.groupby('ip_addr')['cnt_10m'].sum().reset_index(name='sum_10m')

    # Per-IP aggregates
    ip_agg = df.groupby('ip_addr').agg(
        total_requests=('ts','count'),
        failed_requests=('is_fail','sum'),
        unique_usernames=('username',lambda s: s.nunique(dropna=True)),
        unique_devices=('username',lambda s: s.nunique(dropna=True)),
        bmp_max=('bmp_score',lambda s: pd.Series(s).dropna().max() if s.dropna().shape[0]>0 else 0)
    ).reset_index()

    # Merge burst aggregates
    ip_agg = ip_agg.merge(max_1m, on='ip_addr', how='left').merge(max_10m, on='ip_addr', how='left').merge(sum_10m, on='ip_addr', how='left')
    ip_agg.fillna(0, inplace=True)
    ip_agg['failure_rate'] = ip_agg['failed_requests']/ip_agg['total_requests'].replace(0,np.nan)

    st.subheader('1️⃣ Feature Engineering Overview')
    st.dataframe(ip_agg.head(10))

    # ---------- Rule-based Reasons (Point 4) ----------
    BURST_10MIN_THRESHOLD = 100
    BURST_1MIN_THRESHOLD = 50
    HIGH_FAILURE_RATE = 0.9
    MIN_ATTEMPTS = 10
    BMP_HIGH_THRESHOLD = 70

    def rule_reasons(row):
        reasons=[]
        if row['max_attempts_10m']>=BURST_10MIN_THRESHOLD or row['max_attempts_1m']>=BURST_1MIN_THRESHOLD:
            reasons.append(f'burst(max1m={row["max_attempts_1m"]},max10m={row["max_attempts_10m"]})')
        if row['total_requests']>=MIN_ATTEMPTS and row['failure_rate']>=HIGH_FAILURE_RATE:
            reasons.append(f'high_failure_rate({row["failure_rate"]:.2f})')
        if row['unique_usernames']>=10 and row['total_requests']>20:
            reasons.append(f'many_usernames({row["unique_usernames"]})')
        if row['bmp_max']>=BMP_HIGH_THRESHOLD:
            reasons.append(f'bmp_high({int(row["bmp_max"])})')
        return reasons

    ip_agg['rule_reasons'] = ip_agg.apply(rule_reasons, axis=1)
    ip_agg['rule_flag'] = ip_agg['rule_reasons'].apply(lambda x: len(x)>0)
    st.subheader('2️⃣ Rule-based Reason Flags')
    st.dataframe(ip_agg[['ip_addr','rule_flag','rule_reasons']].head(10))

    # ---------- IsolationForest (Point 5) ----------
    num_features = ['total_requests','failed_requests','failure_rate','unique_usernames','unique_devices','max_attempts_1m','max_attempts_10m','sum_10m','bmp_max']
    clf = IsolationForest(contamination=0.02, random_state=42)
    clf.fit(ip_agg[num_features])
    ip_agg['iso_anomaly'] = (clf.predict(ip_agg[num_features])==-1).astype(int)
    st.subheader('3️⃣ IsolationForest Anomalies')
    st.dataframe(ip_agg[['ip_addr','iso_anomaly']].head(10))

    # ---------- Combined Label & Scoring (Point 6) ----------
    ip_agg['final_label'] = ip_agg.apply(lambda r: 'suspicious' if r['rule_flag'] or r['iso_anomaly']==1 else 'benign', axis=1)
    ip_agg['suspicion_score'] = ip_agg['rule_flag'].astype(int)*10 + (1/(1+np.arange(len(ip_agg))))
    st.subheader('4️⃣ Suspicion Score & Final Label')
    st.dataframe(ip_agg[['ip_addr','final_label','suspicion_score']].sort_values('suspicion_score',ascending=False).head(10))

    # ---------- Temporal Burst Visualization (Point 2 & 3) ----------
    st.subheader('5️⃣ Temporal Burst Analysis')
    fig = px.line(cnt_1m.groupby('minute_floor').cnt_1m.sum().reset_index(), x='minute_floor', y='cnt_1m', title='Requests per Minute (All IPs)')
    st.plotly_chart(fig, use_container_width=True)

    # ---------- Reporting Dashboard Summary (Point 7) ----------
    st.subheader('6️⃣ Summary Counts & Distribution')
    st.write(ip_agg['final_label'].value_counts())
    st.write('Top IPs by Total Requests:')
    st.dataframe(ip_agg.sort_values('total_requests',ascending=False)[['ip_addr','total_requests','rule_flag','iso_anomaly']].head(10))

    st.subheader('7️⃣ Interactive Scatter: Failure Rate vs Max 1-min Burst')
    fig2 = px.scatter(ip_agg, x='max_attempts_1m', y='failure_rate', color='final_label', hover_data=['ip_addr','unique_usernames','bmp_max'])
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info('Upload a CSV file to start analysis.')
