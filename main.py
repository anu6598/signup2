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
