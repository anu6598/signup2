
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import StringIO
from pathlib import Path

st.set_page_config(page_title='Signup Anomaly Detector', layout='wide')

st.title('Signup Logs Anomaly Detector (Rules + IsolationForest)')

uploaded_file = st.file_uploader('Upload signup CSV', type=['csv'], accept_multiple_files=False)

MODEL_PATH = Path('isolation_forest_model.pkl')

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

def run_pipeline(df_raw, model_data):
    df = df_raw.copy()
    df.columns = [c.strip() for c in df.columns]
    if 'event_time' not in df.columns:
        if 'date' in df.columns and 'start_time' in df.columns:
            df['event_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['start_time'].astype(str), errors='coerce')
        else:
            df['event_time'] = pd.to_datetime(df.iloc[:,0], errors='coerce')
    df = df.sort_values('event_time').reset_index(drop=True)
    ip_col = None
    for c in ['true_client_ip','x_real_ip','x_forwarded_for','client_ip','ip','remote_addr']:
        if c in df.columns:
            ip_col = c
            break
    if ip_col is None:
        df['ip'] = 'unknown'
    else:
        df['ip'] = df[ip_col].astype(str).str.split(',').str[0].str.strip()
    for col in ['duration','bytes_sent']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    ak = None
    for c in ['akamai_bot','akamai_epd']:
        if c in df.columns:
            ak = c
            break
    if ak is None:
        df['akamai_bot'] = '-'
    else:
        df['akamai_bot'] = df[ak].astype(str)
    df['is_known_bot'] = df['akamai_bot'].str.contains('bot', case=False, na=False).astype(int)
    df['event_time_ts'] = df['event_time'].astype('int64') // 10**9
    grouped = df.groupby('ip')['event_time_ts'].apply(list).to_dict()
    rows = []
    for ip, times in grouped.items():
        times_arr = np.array(times)
        counts_10m = []
        counts_1h = []
        for t in times_arr:
            left10 = t-600
            lidx = np.searchsorted(times_arr, left10, side='left')
            ridx = np.searchsorted(times_arr, t, side='right')
            counts_10m.append(int(ridx-lidx))
            left1 = t-3600
            lidx1 = np.searchsorted(times_arr, left1, side='left')
            counts_1h.append(int(ridx - lidx1))
        for t,c10,c1 in zip(times, counts_10m, counts_1h):
            rows.append({'ip': ip, 'event_time_ts': t, 'ip_count_10m': c10, 'ip_count_1h': c1})
    count_df = pd.DataFrame(rows)
    df = df.merge(count_df, on=['ip','event_time_ts'], how='left')
    df['ip_count_10m'] = df['ip_count_10m'].fillna(0).astype(int)
    df['ip_count_1h'] = df['ip_count_1h'].fillna(0).astype(int)
    if 'user_agent' not in df.columns:
        df['user_agent'] = '-'
    df['distinct_user_agents_total'] = df.groupby('ip')['user_agent'].transform('nunique')
    df['distinct_user_agents_1h'] = 0
    for idx, row in df.iterrows():
        ip = row['ip']; t = row['event_time_ts']
        mask = (df['ip']==ip) & (df['event_time_ts'] >= t-3600) & (df['event_time_ts'] <= t)
        df.at[idx, 'distinct_user_agents_1h'] = df.loc[mask, 'user_agent'].nunique()
    df['hour'] = df['event_time'].dt.hour.fillna(0).astype(int)
    df['seconds_since_first_seen_for_ip'] = df['event_time_ts'] - df.groupby('ip')['event_time_ts'].transform('min') + 1
    df['signups_per_min_since_first'] = df['ip_count_1h'] / (df['seconds_since_first_seen_for_ip']/60.0).replace(0,1)
    rules = model_data.get('rules', {'high_10min_threshold':10,'high_1h_threshold':20,'many_ua_1h':5})
    reasons = []
    flags = []
    for _, r in df.iterrows():
        fr = []
        if r['ip_count_10m'] >= rules['high_10min_threshold']:
            fr.append(f"high_10min={r['ip_count_10m']}")
        if r['ip_count_1h'] >= rules['high_1h_threshold']:
            fr.append(f"high_1h={r['ip_count_1h']}")
        if r['distinct_user_agents_1h'] >= rules['many_ua_1h']:
            fr.append(f"many_ua_1h={r['distinct_user_agents_1h']}")
        if r['is_known_bot']:
            fr.append('akamai_bot_detected')
        flags.append(1 if fr else 0)
        reasons.append(';'.join(fr) if fr else '')
    df['rule_flag'] = flags
    df['rule_reasons'] = reasons
    feat_cols = model_data['feature_cols']
    X = df[feat_cols].fillna(0).astype(float)
    model = model_data['model']
    df['if_score'] = model.decision_function(X)
    df['if_anomaly'] = (model.predict(X) == -1).astype(int)
    return df

if uploaded_file is not None:
    df_in = pd.read_csv(uploaded_file)
    model_data = load_model()
    out = run_pipeline(df_in, model_data)
    st.write('## Summary')
    st.write('Total rows: {}'.format(len(out)))
    st.write('Rule anomalies: {}'.format(int(out['rule_flag'].sum())))
    st.write('IsolationForest anomalies: {}'.format(int(out['if_anomaly'].sum())))
    st.write('Overlap: {}'.format(int(((out['rule_flag']==1)&(out['if_anomaly']==1)).sum())))
    st.write('---')
    st.write('### Top offending IPs (by total rows)')
    st.dataframe(out['ip'].value_counts().rename_axis('ip').reset_index(name='count').head(20))
    st.write('---')
    st.write('### Rule anomalies sample')
    st.dataframe(out[out['rule_flag']==1].sort_values('event_time').head(200))
    st.write('### IF anomalies sample')
    st.dataframe(out[out['if_anomaly']==1].sort_values('event_time').head(200))
    st.write('### Rows where both flagged')
    st.dataframe(out[(out['if_anomaly']==1)&(out['rule_flag']==1)].sort_values('event_time').head(200))
else:
    st.info('Upload a signup logs CSV to run analysis.')

