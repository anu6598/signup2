import streamlit as st
import pandas as pd

def show(df):
    st.title("📊 Daily Stats")

    if df is None or df.empty:
        st.warning("No data available — please upload CSV from Main Dashboard.")
        return

    st.subheader("🚨 Final Daily Categorization")
    final_categories = []
    for day, group in df.groupby("date"):
        total_otps = len(group)
        max_requests_ip = group["true_client_ip"].value_counts().max()
        max_requests_device = group["dr_dv"].value_counts().max()
        proxy_ratio = group["is_proxy"].mean() * 100

        if (total_otps > 1000) or (max_requests_ip > 25) or (proxy_ratio > 70) or (max_requests_device > 15):
            category = "OTP Abuse/Attack detected"
        elif (max_requests_ip > 25) or (total_otps > 1000) or (max_requests_device > 15):
            category = "HIGH OTP request detected"
        elif proxy_ratio > 70:
            category = "HIGH proxy status detected"
        else:
            category = "No suspicious activity detected"

        final_categories.append({
            "date": day,
            "category": category,
            "total_otps": total_otps,
            "max_requests_ip": max_requests_ip,
            "max_requests_device": max_requests_device,
            "proxy_ratio": f"{proxy_ratio:.2f}%"
        })

    st.dataframe(pd.DataFrame(final_categories), use_container_width=True)
