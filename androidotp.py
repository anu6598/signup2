import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="OTP Attack Analyzer", layout="wide")

st.title("🔐 OTP Attack Analyzer")

# File uploader
uploaded_file = st.file_uploader("Upload OTP logs CSV", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)

    st.subheader("📂 Raw Data Preview")
    st.dataframe(df.head(20))

    # Convert start_time to datetime if present
    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    # Define OTP-related request filter
    otp_keywords = ["/otp", "/send-otp", "/resend-otp", "/verify-otp"]
    df["is_otp"] = df["request_path"].astype(str).str.contains("|".join(otp_keywords), case=False, na=False)

    otp_df = df[df["is_otp"]].copy()

    st.subheader("📊 OTP Requests Detected")
    st.write(f"Total OTP requests: {len(otp_df)}")

    if not otp_df.empty:
        # Count OTPs per IP in time windows
        otp_df = otp_df.sort_values("start_time")
        otp_counts = otp_df.groupby("true_client_ip").resample("3min", on="start_time").size().reset_index(name="otp_count")

        # Flag anomalies
        otp_counts["anomaly_3min"] = otp_counts["otp_count"] > 2
        otp_counts["anomaly_6h"] = otp_counts["otp_count"] > 16

        st.subheader("🚨 Anomaly Detection")
        st.write("Flagging IPs with more than **2 OTPs in 3 minutes** or **16 OTPs in 6 hours**")

        anomalies = otp_counts[(otp_counts["anomaly_3min"]) | (otp_counts["anomaly_6h"])]
        st.dataframe(anomalies)

        # Visualization
        st.subheader("📈 OTP Request Timeline per IP")
        fig = px.line(
            otp_counts,
            x="start_time",
            y="otp_count",
            color="true_client_ip",
            title="OTP Requests Over Time by IP",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No OTP-related requests found in the uploaded file.")
else:
    st.info("👆 Please upload a CSV file to start the analysis.")
