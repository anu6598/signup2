# Create a Streamlit app tailored to the user's dataset schema to detect OTP attacks and visualize anomalies.
# Columns expected by default (configurable in sidebar):
# - date, start_time, request_path, true_client_ip, dr_dv, akamai_epd, akamai_bot
# The app filters OTP endpoints and applies the full set of rules requested.

import textwrap

app_code = r"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="OTP Attack Tracker (Web)", layout="wide")

st.title("🔐 OTP Attack Tracking & Anomaly Detection — Web")
st.caption("Monitors endpoints: `/api/user/request_login_otp`, `/api/user/otp_login`")

with st.sidebar:
    st.header("⚙️ Settings / Column Mappings")
    st.write("Map columns from your CSV (defaults match your dataset).")
    date_col = st.text_input("Date column", value="date")
    time_col = st.text_input("Timestamp column", value="start_time")
    ip_col = st.text_input("IP column", value="true_client_ip")
    device_col = st.text_input("Device ID column", value="dr_dv")
    epd_col = st.text_input("Akamai EPD column (proxy signal)", value="akamai_epd")
    endpoint_col = st.text_input("Endpoint/Path column", value="request_path")
    bot_col = st.text_input("Akamai Bot/Score column (optional)", value="akamai_bot")
    # Endpoint filters
    st.divider()
    st.subheader("Endpoint Filters")
    otp_paths = st.multiselect(
        "Substring match (case-insensitive) to keep",
        ["/api/user/request_login_otp", "/api/user/otp_login"],
        default=["/api/user/request_login_otp", "/api/user/otp_login"]
    )
    # Thresholds
    st.divider()
    st.subheader("Detection Thresholds")
    thr_velocity_same_ip_10m = st.number_input("> OTPs in 10 minutes (same IP)", min_value=1, value=10, step=1)
    thr_velocity_proxy_pool_10m = st.number_input("> OTPs in 10 minutes (proxy IPs aggregated)", min_value=1, value=10, step=1)
    thr_proxy_repeat = st.number_input("Always-flag if proxy occurrences/IP ≥", min_value=1, value=2, step=1)
    thr_bmp_score = st.number_input("Flag if BMP score >", min_value=0, value=90, step=5)
    thr_bmp_hits_per_day = st.number_input("Min # windows with BMP >", min_value=1, value=5, step=1)
    z_spike = st.number_input("Proxy spike sensitivity (z-score)", min_value=1.0, value=2.5, step=0.5)
    st.divider()
    st.subheader("Time Zone & Parsing")
    tz = st.text_input("Assume timestamps are in this TZ (IANA)", value="UTC")
    st.caption("If your column is naive string (e.g., '2025-08-05 14:03:11'), we will parse, localize to tz, then convert to UTC internally.")

uploaded = st.file_uploader("📥 Upload logs CSV", type=["csv"])

def parse_timestamp(series, tz_name):
    # Try robust parsing and localize if naive; convert to UTC
    ts = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    # If tz-naive, localize to tz_name; if tz-aware, convert to UTC
    if ts.dt.tz is None:
        try:
            ts = ts.dt.tz_localize(tz_name)
        except Exception:
            # Fallback: assume already UTC
            ts = ts.dt.tz_localize("UTC")
    try:
        ts = ts.dt.tz_convert("UTC")
    except Exception:
        pass
    return ts

if uploaded is None:
    st.info("Upload a CSV with at least: date, start_time, request_path, true_client_ip, dr_dv, akamai_epd.")
else:
    # Load
    df = pd.read_csv(uploaded)
    df.columns = [c.strip() for c in df.columns]

    # Validate columns
    required = [date_col, time_col, endpoint_col, ip_col, epd_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # Filter OTP endpoints
    if otp_paths:
        pat = "|".join([px.utils.strings.regex_escape(p) for p in otp_paths])
        mask = df[endpoint_col].astype(str).str.contains(pat, case=False, regex=True)
        df = df[mask].copy()
    if df.empty:
        st.warning("No rows matched the OTP endpoints filter.")
        st.stop()

    # Parse timestamp
    # Combine date + start_time if needed; else use start_time only
    if date_col in df.columns and df[date_col].notna().any():
        # If start_time already has full datetime with date, keep it
        # Otherwise, try to combine date + start_time
        dt_series = df[time_col].astype(str)
        # Heuristic: if start_time lacks date part (length < 14), combine
        needs_combine = dt_series.str.len().lt(14).fillna(False)
        combined = df[date_col].astype(str) + " " + df[time_col].astype(str)
        df["_raw_ts"] = np.where(needs_combine, combined, df[time_col].astype(str))
    else:
        df["_raw_ts"] = df[time_col].astype(str)

    df["timestamp"] = parse_timestamp(df["_raw_ts"], tz)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Time buckets
    df["minute"] = df["timestamp"].dt.floor("T")
    df["hour"] = df["timestamp"].dt.floor("H")
    df["ten_min"] = df["timestamp"].dt.floor("10T")
    df["date_only"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date  # display-friendly date in IST

    # Proxy boolean
    df["_is_proxy"] = df[epd_col].astype(str).str.strip().ne("").fillna(False) & df[epd_col].notna() & (df[epd_col].astype(str).str.strip() != "-")

    # -----------------------------
    # 1) Minute bucket table (per IP)
    # -----------------------------
    st.subheader("🧱 Per-minute Buckets by IP")
    minute_bucket = df.groupby(["minute", ip_col]).size().reset_index(name="requests")
    st.dataframe(minute_bucket.head(500))

    # -----------------------------
    # 2) Baseline: Normal OTPs per hour (median across IP-hours)
    # -----------------------------
    st.subheader("📏 Baseline: Normal OTPs per Hour")
    hour_counts = df.groupby([ip_col, "hour"]).size().reset_index(name="otps_per_hour")
    global_median = hour_counts["otps_per_hour"].median() if not hour_counts.empty else 0
    st.metric("Global median OTPs/hour per IP", int(global_median))
    fig_hour = px.box(hour_counts, x=ip_col, y="otps_per_hour", points="outliers",
                      title="Distribution of OTPs/hour per IP")
    fig_hour.update_layout(xaxis_title="IP", yaxis_title="OTPs per Hour")
    st.plotly_chart(fig_hour, use_container_width=True)

    # -----------------------------
    # 3) Velocity rules: >N OTPs in 10 minutes
    # -----------------------------
    st.subheader("🚨 Velocity Rules — 10-minute windows")
    # Same IP
    ip_10m = df.groupby([ip_col, "ten_min"]).size().reset_index(name="reqs_10m")
    ip_10m["flag_velocity_ip"] = ip_10m["reqs_10m"] > thr_velocity_same_ip_10m

    # Proxy pool (across different IPs but proxy only)
    proxy_10m = df[df["_is_proxy"]].groupby("ten_min").size().reset_index(name="proxy_reqs_10m")
    proxy_10m["flag_velocity_proxy_pool"] = proxy_10m["proxy_reqs_10m"] > thr_velocity_proxy_pool_10m

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Same IP — 10-minute windows**")
        st.dataframe(ip_10m.sort_values(["flag_velocity_ip","reqs_10m"], ascending=[False, False]).head(500))
    with col2:
        st.markdown("**Across proxies only — 10-minute windows**")
        st.dataframe(proxy_10m.sort_values(["flag_velocity_proxy_pool","proxy_reqs_10m"], ascending=[False, False]).head(500))

    # Visualize velocity for top flagged IPs
    top_ips = ip_10m[ip_10m["flag_velocity_ip"]].groupby(ip_col)["reqs_10m"].sum().nlargest(10).index.tolist()
    if len(top_ips) > 0:
        fig_vel = px.line(ip_10m[ip_10m[ip_col].isin(top_ips)],
                          x="ten_min", y="reqs_10m", color=ip_col,
                          title="10-min OTP Velocity — Top Flagged IPs")
        fig_vel.update_layout(xaxis_title="Time (10-min buckets)", yaxis_title="Requests")
        st.plotly_chart(fig_vel, use_container_width=True)

    # -----------------------------
    # 4) Always flag proxy IPs with >1 occurrence
    # -----------------------------
    st.subheader("🕵️ Proxy IPs with ≥ N occurrences (always flag)")
    proxy_counts = df.groupby(ip_col)["_is_proxy"].sum().reset_index(name="proxy_occurrences")
    flagged_proxies = proxy_counts[proxy_counts["proxy_occurrences"] >= thr_proxy_repeat]
    st.dataframe(flagged_proxies.sort_values("proxy_occurrences", ascending=False))

    # -----------------------------
    # 5) Age of IP — unique days seen
    # -----------------------------
    st.subheader("📅 Age of IP (unique days seen)")
    ip_age = df.groupby(ip_col)["date_only"].nunique().reset_index(name="unique_days_seen")
    st.dataframe(ip_age.sort_values("unique_days_seen", ascending=False))

    # -----------------------------
    # 6) Sudden spike in Akamai EPD for certain IPs
    # -----------------------------
    st.subheader("📈 Spike Detection — Akamai EPD usage")
    daily = df.groupby([ip_col, "date_only"]).agg(
        total=("date_only", "count"),
        proxy_hits=("_is_proxy", "sum")
    ).reset_index()
    daily["proxy_rate"] = daily["proxy_hits"] / daily["total"]
    ip_stats = daily.groupby(ip_col)["proxy_rate"].agg(["mean", "std"]).reset_index().rename(columns={"mean":"pr_mean","std":"pr_std"})
    daily = daily.merge(ip_stats, on=ip_col, how="left")
    daily["z_proxy_rate"] = (daily["proxy_rate"] - daily["pr_mean"]) / daily["pr_std"].replace(0, np.nan)
    spike_ips = daily[(daily["z_proxy_rate"] >= z_spike) | (daily["proxy_rate"] >= 0.9)]
    st.write("**Flagged IP-days with spike in proxy usage**")
    st.dataframe(spike_ips.sort_values(["z_proxy_rate","proxy_rate"], ascending=[False, False]))

    # -----------------------------
    # 7) Requests per minute (same device / same IP) across windows
    # -----------------------------
    st.subheader("⏱️ Requests per Minute — IP / Device")
    per_min_ip = df.groupby(["minute", ip_col]).size().reset_index(name="reqs_per_min_ip")
    st.write("**Per minute by IP**")
    st.dataframe(per_min_ip.head(500))

    fig_min_ip = px.line(per_min_ip, x="minute", y="reqs_per_min_ip", color=ip_col,
                         title="Per-minute Requests by IP")
    fig_min_ip.update_layout(xaxis_title="Time (minute)", yaxis_title="Requests")
    st.plotly_chart(fig_min_ip, use_container_width=True)

    if device_col in df.columns:
        per_min_device = df.groupby(["minute", device_col]).size().reset_index(name="reqs_per_min_device")
        st.write("**Per minute by Device**")
        st.dataframe(per_min_device.head(500))
        # Show top devices
        top_devices = per_min_device.groupby(device_col)["reqs_per_min_device"].sum().nlargest(8).index
        fig_min_dev = px.line(per_min_device[per_min_device[device_col].isin(top_devices)],
                              x="minute", y="reqs_per_min_device", color=device_col,
                              title="Per-minute Requests by Top Devices")
        fig_min_dev.update_layout(xaxis_title="Time (minute)", yaxis_title="Requests")
        st.plotly_chart(fig_min_dev, use_container_width=True)

    # -----------------------------
    # 8) BMP score windows and daily flags
    #    If you don't have a numeric score, derive a heuristic.
    # -----------------------------
    st.subheader("🧮 BMP Score — 10-min windows → daily flags")
    feats_10m = df.groupby([ip_col, "ten_min"]).agg(
        reqs_10m=("minute", "count"),
        proxy_hits=("_is_proxy", "sum"),
        unique_devices=(device_col, "nunique") if device_col in df.columns else ("minute", "count")
    ).reset_index()

    # Heuristic BMP: base 60 if proxy in window; +25 if > threshold; +15 if unique devices ≥ 3
    feats_10m["bmp_score"] = 0
    feats_10m.loc[feats_10m["proxy_hits"] > 0, "bmp_score"] += 60
    feats_10m.loc[feats_10m["reqs_10m"] > thr_velocity_same_ip_10m, "bmp_score"] += 25
    feats_10m.loc[feats_10m["unique_devices"] >= 3, "bmp_score"] += 15
    feats_10m["bmp_score"] = feats_10m["bmp_score"].clip(upper=100)

    feats_10m["date_only"] = feats_10m["ten_min"].dt.tz_convert("Asia/Kolkata").dt.date

    daily_bmp = feats_10m.groupby([ip_col, "date_only"]).apply(
        lambda g: (g["bmp_score"] > thr_bmp_score).sum()
    ).reset_index(name="bmp_hits_gt_threshold")

    persistently_bad = daily_bmp[daily_bmp["bmp_hits_gt_threshold"] >= thr_bmp_hits_per_day]
    st.write("**IPs with persistent high BMP windows (daily)**")
    st.dataframe(persistently_bad.sort_values("bmp_hits_gt_threshold", ascending=False))

    # -----------------------------
    # 9) IP repetitions benchmark (normal day) — median & p95
    # -----------------------------
    st.subheader("📊 IP Repetition Benchmark")
    ip_daily_reqs = df.groupby([ip_col, "date_only"]).size().reset_index(name="daily_requests")
    bench = int(ip_daily_reqs["daily_requests"].median()) if not ip_daily_reqs.empty else 0
    p95 = int(ip_daily_reqs["daily_requests"].quantile(0.95)) if not ip_daily_reqs.empty else 0
    c1, c2 = st.columns(2)
    c1.metric("Median daily requests/IP", bench)
    c2.metric("95th percentile", p95)
    flagged_repetition = ip_daily_reqs[ip_daily_reqs["daily_requests"] > p95]
    st.write("**IPs above 95th percentile of daily requests**")
    st.dataframe(flagged_repetition.sort_values("daily_requests", ascending=False))

    # -----------------------------
    # 10) Overall rolling averages (1,5,10 minutes)
    # -----------------------------
    st.subheader("📈 Overall Rolling Averages (1, 5, 10 minutes)")
    series = df.set_index("minute").assign(cnt=1)["cnt"].resample("T").sum().fillna(0)
    roll1 = series.rolling(1, min_periods=1).mean()
    roll5 = series.rolling(5, min_periods=1).mean()
    roll10 = series.rolling(10, min_periods=1).mean()

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Bar(x=series.index, y=series.values, name="Requests per minute"))
    fig_roll.add_trace(go.Scatter(x=roll1.index, y=roll1.values, name="1-min avg", mode="lines"))
    fig_roll.add_trace(go.Scatter(x=roll5.index, y=roll5.values, name="5-min avg", mode="lines"))
    fig_roll.add_trace(go.Scatter(x=roll10.index, y=roll10.values, name="10-min avg", mode="lines"))
    fig_roll.update_layout(title="Requests per Minute with Rolling Averages", xaxis_title="Time", yaxis_title="Count")
    st.plotly_chart(fig_roll, use_container_width=True)

    # -----------------------------
    # 11) Exports
    # -----------------------------
    st.subheader("⬇️ Download Key Tables")
    def make_download(df_, name):
        return st.download_button(
            label=f"Download {name} CSV",
            data=df_.to_csv(index=False).encode("utf-8"),
            file_name=f"{name}.csv",
            mime="text/csv"
        )

    make_download(minute_bucket, "minute_bucket_by_ip")
    make_download(ip_10m, "ip_10min_velocity")
    make_download(proxy_10m, "proxy_pool_10min_velocity")
    make_download(flagged_proxies, "flagged_proxies_repeated")
    make_download(ip_age, "ip_age")
    make_download(spike_ips, "spike_ips_proxy_usage")
    make_download(persistently_bad, "persistently_high_bmp_ips")
    make_download(flagged_repetition, "ip_repetition_outliers")
"""

reqs = """
streamlit
pandas
numpy
plotly
"""

# Write files
with open("/mnt/data/otp_tracker_app.py", "w") as f:
    f.write(app_code)

with open("/mnt/data/requirements.txt", "w") as f:
    f.write(reqs.strip())

"/mnt/data/otp_tracker_app.py and /mnt/data/requirements.txt created."
