import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Signup Path Request Analyzer")

uploaded_file = st.file_uploader("Upload log CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Check if necessary columns exist
        required_cols = {'start_time', 'request_path'}
        if not required_cols.issubset(df.columns):
            st.error("Missing required columns in CSV.")
        else:
            # Filter for '/user/signup' in request_path
            signup_logs = df[df['request_path'].astype(str).str.contains('/user/signup', case=False, na=False)]

            if signup_logs.empty:
                st.warning("No /user/signup entries found in the uploaded log.")
            else:
                # Convert start_time to datetime
                signup_logs['start_time'] = pd.to_datetime(signup_logs['start_time'], errors='coerce')
                signup_logs.dropna(subset=['start_time'], inplace=True)

                # Group by minute
                signup_logs['minute'] = signup_logs['start_time'].dt.floor('T')
                per_minute_counts = signup_logs.groupby('minute').size().reset_index(name='signup_requests')

                # Plot
                st.subheader("Signup Requests Per Minute")
                st.line_chart(per_minute_counts.set_index('minute'))

                # Show raw counts
                with st.expander("Raw Count Table"):
                    st.dataframe(per_minute_counts)

    except Exception as e:
        st.error(f"Error while processing file: {e}")



import pandas as pd

# Convert time
df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

# Filter signup and OTP events
signup_df = df[df['request_path'].str.contains("/user/signup", na=False)]
otp_df = df[df['request_path'].str.contains("/user/send-otp", na=False)]

anomalies = {}

# 1. High Volume - Unusual spike in signups per minute
signup_per_minute = signup_df.groupby(pd.Grouper(key='start_time', freq='1min')).size()
threshold = signup_per_minute.mean() + 3 * signup_per_minute.std()
anomalies['spike_signups'] = signup_per_minute[signup_per_minute > threshold]

# 2. Same IP sending many signups
ip_counts = signup_df.groupby('true_client_ip').size()
anomalies['high_signup_ips'] = ip_counts[ip_counts > ip_counts.quantile(0.99)]

# 3. Tightly clustered signups in short time
signup_df['minute'] = signup_df['start_time'].dt.floor('min')
burst_candidates = signup_df.groupby(['true_client_ip', 'minute']).size()
anomalies['bursts'] = burst_candidates[burst_candidates > 10]

# 4. Signups during off-peak hours (midnight to 5 AM)
signup_df['hour'] = signup_df['start_time'].dt.hour
anomalies['night_activity'] = signup_df[(signup_df['hour'] >= 0) & (signup_df['hour'] <= 5)]

# 5. Same IP but multiple user agents or device IDs
diverse_clients = signup_df.groupby('true_client_ip').agg({
    'user_agent': pd.Series.nunique,
    'dr_dv': pd.Series.nunique,
    'dr_app_version': pd.Series.nunique
})
anomalies['diverse_headers_same_ip'] = diverse_clients[
    (diverse_clients['user_agent'] > 1) |
    (diverse_clients['dr_dv'] > 1) |
    (diverse_clients['dr_app_version'] > 1)
]

# 6. IPs from unusual countries
unexpected_countries = ['RU', 'VN', 'CN', 'IR', 'KP']  # example list
anomalies['suspicious_geos'] = signup_df[signup_df['x_country_code'].isin(unexpected_countries)]

# 7. Missing or malformed headers
anomalies['missing_headers'] = signup_df[
    (signup_df['user_agent'].isnull()) |
    (signup_df['x_forwarded_for'].isnull()) |
    (signup_df['dr_dv'].isnull())
]

# 8. Known bot indicators: High Akamai bot score
anomalies['bot_scores'] = signup_df[signup_df.get('akamai_bot_score', 0) > 60]

# 9. Unusual app versions or platforms
anomalies['unusual_versions'] = signup_df[
    (signup_df['dr_app_version'].isnull()) |
    (signup_df['dr_platform'].isnull()) |
    (~signup_df['dr_platform'].isin(['android', 'ios', 'web']))
]

# 10. Rapid switching of platforms by IP
platform_switch = signup_df.groupby(['true_client_ip', pd.Grouper(key='start_time', freq='5min')])['dr_platform'].nunique()
anomalies['platform_switch'] = platform_switch[platform_switch > 1]

# 11. High OTP request frequency
otp_ip_counts = otp_df.groupby('true_client_ip').size()
anomalies['otp_spam'] = otp_ip_counts[otp_ip_counts > otp_ip_counts.quantile(0.95)]

# 12. Failed signups with repeated attempts
failed_signups = signup_df[signup_df['response_code'] != 200]
failed_per_ip = failed_signups.groupby('true_client_ip').size()
anomalies['repeated_failures'] = failed_per_ip[failed_per_ip > 5]

