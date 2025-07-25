import streamlit as st
import pandas as pd
import plotly.express as px

# Load preprocessed data
df = pd.read_csv('signupweb.csv')

st.title("Pre-Signup Anomaly Monitor")

# Date Filter
dates = df['date'].unique()
selected_date = st.selectbox("Select Date", dates)
filtered_df = df[df['date'] == selected_date]

# OTPs per x_forwarded_for
x_forwarded_for_counts = filtered_df.groupby('x_forwarded_for').size().reset_index(name='count')
fig1 = px.bar(x_forwarded_for_counts[x_forwarded_for_counts['count'] > 20], x='x_forwarded_for', y='count', title="Suspicious x_forwarded_fors")
st.plotly_chart(fig1)

# OTPs per Device
device_counts = filtered_df.groupby('device_id').size().reset_index(name='count')
fig2 = px.bar(device_counts[device_counts['count'] > 20], x='device_id', y='count', title="Suspicious Devices")
st.plotly_chart(fig2)

# Emails per x_forwarded_for
email_counts = filtered_df.groupby('x_forwarded_for')['email'].nunique().reset_index(name='unique_emails')
fig3 = px.bar(email_counts[email_counts['unique_emails'] > 10], x='x_forwarded_for', y='unique_emails', title="Emails per x_forwarded_for")
st.plotly_chart(fig3)

# Anomalies Table
st.markdown("### Raw Anomalous Entries")
filtered_df = filtered_df[filtered_df['is_anomaly'] == True]
filtered_df['reviewed'] = False  # Checkbox per row
edited_df = st.data_editor(filtered_df, use_container_width=True)

# Save reviewed status back to parquet if needed
# edited_df.to_parquet('updated_anomalies.parquet', index=False)
