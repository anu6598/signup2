import streamlit as st
import pandas as pd
import io
from datetime import datetime

st.set_page_config(page_title="Signup Request Analyzer", layout="wide")
st.title("📈 Signup Request Per Minute Analyzer")

uploaded_file = st.file_uploader("Upload your AES log CSV", type=['csv'])

if uploaded_file:
    try:
        # Load the CSV
        df = pd.read_csv(uploaded_file)

        # Convert start_time to datetime
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')

        # Filter only signup requests
        signup_df = df[df['request_path'].str.contains('/user/signup', na=False)]

        # Round time to nearest minute
        signup_df['minute'] = signup_df['start_time'].dt.floor('min')

        # Count signups per minute
        counts = signup_df.groupby('minute').size().reset_index(name='signup_requests')

        st.success(f"Total signup requests found: {len(signup_df)}")
        st.line_chart(counts.set_index('minute'))

        # Optionally show table
        with st.expander("Show raw per-minute counts"):
            st.dataframe(counts)

    except Exception as e:
        st.error(f"Error while processing file: {e}")
else:
    st.info("Please upload a CSV log file to begin.")
