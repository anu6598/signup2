import streamlit as st
import pandas as pd

def show():
    st.set_page_config(page_title="OTP Abuse Detection Dashboard", layout="wide")
    st.title("🔐 OTP Abuse Detection Dashboard")

    st.sidebar.header("Upload CSV")
    uploaded_file = st.sidebar.file_uploader("Upload your OTP logs CSV", type=["csv"])

    if uploaded_file is not None:
        # Load CSV
        df = pd.read_csv(uploaded_file)

        # --- Basic cleaning ---
        df.columns = [col.strip().lower() for col in df.columns]
        st.subheader("📊 Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # --- Rule 1: More than 1000 OTP in a day ---
        total_otp = len(df)

        # --- Rule 2: Any IP > 25 requests ---
        ip_counts = df["true_client_ip"].value_counts()
        ip_abuse = ip_counts[ip_counts > 25]

        # --- Rule 3: More than 70% IPs are proxies ---
        if "is_proxy" in df.columns:
            proxy_ratio = df["is_proxy"].mean() * 100
        else:
            proxy_ratio = 0

        # --- Rule 4: Any device ID > 15 requests ---
        if "dr_dv" in df.columns:
            device_counts = df["dr_dv"].value_counts()
            device_abuse = device_counts[device_counts > 15]
        else:
            device_abuse = pd.Series()

        # --- Detection logic ---
        detections = []
        if (total_otp > 1000) or (len(ip_abuse) > 0) or (proxy_ratio > 70) or (len(device_abuse) > 0):
            detections.append("OTP Abuse/Attack detected")
        if (len(ip_abuse) > 0) or (total_otp > 1000) or (len(device_abuse) > 0):
            detections.append("HIGH OTP request detected")
        if proxy_ratio > 70:
            detections.append("HIGH proxy status detected")
        if not detections:
            detections.append("No suspicious activity detected")

        # --- Display results ---
        st.subheader("🚨 Detection Results")
        results_df = pd.DataFrame({"Detection Status": detections})
        st.table(results_df)

        # --- Show details ---
        st.subheader("📌 Detailed Breakdown")
        st.write(f"**Total OTP requests:** {total_otp}")
        st.write(f"**IPs with >25 requests:** {len(ip_abuse)}")
        st.write(f"**Devices with >15 requests:** {len(device_abuse)}")
        st.write(f"**Proxy ratio:** {proxy_ratio:.2f}%")

        if len(ip_abuse) > 0:
            st.write("Top suspicious IPs:")
            st.dataframe(ip_abuse.head(20))
        if len(device_abuse) > 0:
            st.write("Top suspicious Devices:")
            st.dataframe(device_abuse.head(20))

    else:
        st.info("👆 Upload a CSV file from the sidebar to start analysis.")
