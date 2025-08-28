import streamlit as st
import pandas as pd


def show(df):
    st.title("📊 Daily Stats")

    if df is None or df.empty:
        st.warning("No data available — please upload CSV from Main Dashboard.")
        return

    # ✅ Preview
    st.subheader("Daily OTP Abuse Statistics")
    st.write("Preview of Data", df.head())

    # ✅ Daily counts chart (if 'date' column exists)
    if "date" in df.columns:
        daily_counts = df.groupby("date").size().reset_index(name="count")
        st.bar_chart(daily_counts.set_index("date")["count"])
    else:
        st.warning("⚠️ No 'date' column found in dataset, skipping daily counts chart.")

    # ✅ Categorization
    st.subheader("🚨 Final Daily Categorization")
    final_categories = []

    for day, group in df.groupby("date") if "date" in df.columns else []:
        total_otps = len(group)

        # Safe extraction with defaults
        max_requests_ip = group["true_client_ip"].value_counts().max() if "true_client_ip" in group else 0
        max_requests_device = group["dr_dv"].value_counts().max() if "dr_dv" in group else 0

        # if "is_proxy" in group:
        #     proxy_ratio = group["is_proxy"].mean() * 100
        # else:
        #     proxy_ratio = 0  # default when column missing

        # Rule categorization
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

    # ✅ Display results
    if final_categories:
        st.dataframe(pd.DataFrame(final_categories), use_container_width=True)
    else:
        st.info("No daily categorization results available.")
