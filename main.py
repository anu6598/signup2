import streamlit as st
import pandas as pd

st.set_page_config(page_title="Signup Bot Detection Dashboard", layout="wide")

st.title("📊 Signup Bot Detection Dashboard")

uploaded_file = st.file_uploader("Upload the Signup CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['signup_time'])

    st.subheader("🔍 Raw Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("📈 Recaptcha Score Distribution")
    st.bar_chart(df['recaptcha_score'])

    st.subheader("🕵️ Suspicious Users Detected")
    suspicious = df[(df['recaptcha_score'] < 0.5) | 
                    (df['otp_attempts'] > 5) | 
                    (df['email'].str.contains("mail.ru|tempmail|guerrillamail")) | 
                    (df['isp'].str.contains("Stark|Crea Nova|Hosting Solution"))]

    st.write(f"Total suspicious users: {len(suspicious)}")
    st.dataframe(suspicious[['user_id', 'email', 'signup_time', 'ip_address', 'country', 'recaptcha_score', 'otp_attempts']], use_container_width=True)

    st.subheader("🌍 Country-wise Signups")
    st.bar_chart(df['country'].value_counts())

    st.subheader("📅 Signups Over Time")
    signups_by_day = df['signup_time'].dt.date.value_counts().sort_index()
    st.line_chart(signups_by_day)

else:
    st.info("Please upload a CSV file to view analysis.")
