import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Signup Request Analyzer")

uploaded_file = st.file_uploader("Upload your log CSV file", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Convert start_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

        # Filter for signup-related logs (case-insensitive)
        signup_logs = df[df['request_path'].str.lower().str.contains('/user/signup', na=False)]

        if signup_logs.empty:
            st.warning("No /user/signup logs found.")
        else:
            # Truncate to minute
            signup_logs['minute'] = signup_logs['start_time'].dt.floor('T')

            # Count per minute
            per_minute_counts = signup_logs.groupby('minute').size().reset_index(name='signup_requests')

            # Show data
            st.subheader("Signup Requests Per Minute")
            st.dataframe(per_minute_counts)

            # Plot
            st.line_chart(data=per_minute_counts.set_index('minute'))

    except Exception as e:
        st.error(f"Error while processing file: {e}")
