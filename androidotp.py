import streamlit as st
import pandas as pd
import numpy as np
import dailystats
import re

st.set_page_config(page_title="OTP Abuse Detection Dashboard", layout="wide")

# Sidebar navigation only (no CSV upload here)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Dashboard", "Daily Stats"])

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None

def normalize_dataframe(df_raw):
    cols_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_raw.columns}
    df_raw.rename(columns=cols_map, inplace=True)
    df = df_raw.copy()

    # Pick columns dynamically
    def pick_col(possible):
        for p in possible:
            if p in df.columns:
                return p
        return None

    col_ip = pick_col(["true_client_ip", "client_ip", "ip", "remote_addr"])
    col_device = pick_col(["dr_dv", "device_id", "device"])
    col_akamai_epd = pick_col(["akamai_epd", "epd", "akamai_proxy"])

    df["true_client_ip"] = df[col_ip].astype(str) if col_ip else "unknown"
    df["dr_dv"] = df[col_device].astype(str) if col_device else np.nan
    df["akamai_epd"] = df[col_akamai_epd] if col_akamai_epd else np.nan
    df["is_proxy"] = df["akamai_epd"].notna() & (df["akamai_epd"] != "")

    return df


# -------------------------
# Page routing
# -------------------------
if page == "Main Dashboard":
    st.title("🔐 OTP Abuse Detection Dashboard (Main)")

    uploaded_file = st.file_uploader("Upload OTP logs CSV", type=["csv"])
    if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
        st.session_state.df = normalize_dataframe(df_raw)  # save to session

        st.success("✅ File uploaded and processed!")
        st.subheader("Raw data preview (first 10 rows)")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

        st.subheader("Quick stats")
        st.write(f"Total rows: {len(st.session_state.df)}")
        st.write(f"Unique IPs: {st.session_state.df['true_client_ip'].nunique()}")
        st.write(f"Proxy ratio: {st.session_state.df['is_proxy'].mean()*100:.2f}%")
    else:
        st.info("👆 Upload a CSV file to begin.")

elif page == "Daily Stats":
    dailystats.show(st.session_state.df)  # pass df if available











# -------------------------
# Sidebar: threshold controls
# -------------------------
st.sidebar.header("Detection thresholds / controls")

burst_threshold = st.sidebar.number_input("Burst threshold (OTPs within 10 min)", value=10, step=1)
burst_window_mins = st.sidebar.number_input("Burst window (minutes)", value=10, step=1)
bmp_threshold = st.sidebar.number_input("BMP score threshold", value=90, step=1)
bmp_times_threshold = st.sidebar.number_input("BMP occurrences/day threshold", value=5, step=1)
proxy_repeat_threshold = st.sidebar.number_input("Proxy hits threshold to mark proxy IP", value=1, step=1)
device_min_threshold = st.sidebar.number_input("Device requests threshold per minute", value=10, step=1)
ip_benchmark_multiplier = st.sidebar.number_input("IP daily benchmark multiplier (for flagging)", value=2.0, step=0.1)
date_filter = st.sidebar.date_input("Filter date (optional) — pick single date or range", [])

st.sidebar.markdown("---")

# -------------------------
# File upload
# -------------------------
uploaded_file = st.file_uploader("Upload OTP logs CSV", type=["csv"], help="Must contain (or close variants of): start_time/date, request_path, true_client_ip, dr_dv, akamai_epd, akamai_bot")

if not uploaded_file:
    st.info("Upload a CSV file to begin. The app will try to automatically detect columns and run all rules.")
    st.stop()


# -------------------------
# Load & normalize columns
# -------------------------
df_raw = pd.read_csv(uploaded_file)
st.subheader("Raw data preview (first 10 rows)")
st.dataframe(df_raw.head(10))

# normalize column names to simpler forms
cols_map = {c: c.strip().lower().replace(" ", "_").replace("-", "_") for c in df_raw.columns}
df_raw.rename(columns=cols_map, inplace=True)
df = df_raw.copy()

# helper to get best-matching column
def pick_col(possible):
    for p in possible:
        if p in df.columns:
            return p
    return None

# expected columns (variants)
col_start_time = pick_col(["start_time", "timestamp", "time", "created_on"])
col_date = pick_col(["date"])
col_request_path = pick_col(["request_path", "path", "request"])
col_ip = pick_col(["true_client_ip", "client_ip", "ip", "remote_addr"])
col_device = pick_col(["dr_dv", "device_id", "dr_dv_id", "device"])
col_akamai_epd = pick_col(["akamai_epd", "epd", "akamai_proxy"])
col_akamai_bot = pick_col(["akamai_bot", "akamaibot", "bot_info", "bmp", "akamai_bmp"])

missing = []
if not col_start_time:
    missing.append("start_time/timestamp")
if not col_request_path:
    missing.append("request_path")
if not col_ip:
    missing.append("true_client_ip")
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}. Please upload file with these columns (or similar names).")
    st.stop()

# parse timestamp (try combining date + start_time if date exists and start_time is time-only)
if col_date and col_start_time:
    # check if start_time appears to be a time without date
    sample_start = df[col_start_time].astype(str).iloc[0]
    if re.match(r'^\d{1,2}:\d{2}(:\d{2})?$', sample_start):
        df["timestamp"] = pd.to_datetime(df[col_date].astype(str) + " " + df[col_start_time].astype(str), errors='coerce')
    else:
        df["timestamp"] = pd.to_datetime(df[col_start_time], errors='coerce')
else:
    df["timestamp"] = pd.to_datetime(df[col_start_time], errors='coerce')

# drop rows without timestamp
df = df.dropna(subset=["timestamp"])
df = df.sort_values("timestamp")

# normalize other columns existence
df["request_path"] = df[col_request_path].astype(str) if col_request_path else ""
df["true_client_ip"] = df[col_ip].astype(str)
if col_device:
    df["dr_dv"] = df[col_device].astype(str)
else:
    df["dr_dv"] = np.nan
if col_akamai_epd:
    df["akamai_epd"] = df[col_akamai_epd]
else:
    df["akamai_epd"] = np.nan
if col_akamai_bot:
    df["akamai_bot"] = df[col_akamai_bot].astype(str)
else:
    df["akamai_bot"] = ""

# optional date column for grouping
df["date"] = df["timestamp"].dt.date

# filter to only OTP related requests: anything containing 'otp' (case-insensitive)
df["is_otp"] = df["request_path"].str.contains("otp", case=False, na=False)
otp_df = df[df["is_otp"]].copy()
st.write(f"Detected {len(otp_df)} OTP-related rows out of {len(df)} total rows.")

if len(otp_df) == 0:
    st.warning("No request_path rows matched 'otp'. Inspect `request_path` values and retry.")
    st.subheader("Sample unique request paths")
    st.write(df["request_path"].dropna().unique()[:50])
    st.stop()

# apply optional date filter from sidebar
if isinstance(date_filter, (list, tuple)) and len(date_filter) == 2:
    start_d, end_d = date_filter
    otp_df = otp_df[(otp_df["date"] >= start_d) & (otp_df["date"] <= end_d)]
elif isinstance(date_filter, (pd.Timestamp,)) or (isinstance(date_filter, list) and len(date_filter)==1):
    # if single date chosen
    if isinstance(date_filter, (pd.Timestamp,)):
        otp_df = otp_df[otp_df["date"] == date_filter.date()]
    else:
        try:
            otp_df = otp_df[otp_df["date"] == pd.to_datetime(date_filter).date()]
        except:
            pass


# -------------------------
# Extract BMP score from akamai_bot (digits only)
# -------------------------
def extract_digits(x):
    m = re.search(r"(\d+)", str(x))
    return float(m.group(1)) if m else np.nan

otp_df["bmp_score"] = otp_df["akamai_bot"].apply(extract_digits)

# -------------------------
# Minute bucket table & grouped_windows (SQL replication)
# -------------------------
otp_df["minute_bucket"] = otp_df["timestamp"].dt.floor("T")

grouped = (
    otp_df.groupby(["true_client_ip", "minute_bucket"], as_index=False)
    .agg(signup_attempts=("request_path", "count"),
         akamai_epd=("akamai_epd", lambda s: s.dropna().iloc[0] if s.dropna().shape[0] > 0 else np.nan))
    .sort_values(["true_client_ip", "minute_bucket"])
)

# compute rolling sum attempts_in_10_min per IP using time-based rolling on DatetimeIndex
def compute_rolling_attempts(g, window_minutes=int(burst_window_mins)):
    g = g.set_index("minute_bucket").sort_index()
    # rolling with time window, centers on right by default (includes current row)
    g["attempts_in_10_min"] = g["signup_attempts"].rolling(f"{window_minutes}T").sum()
    g = g.reset_index()
    return g

grouped_rolled = grouped.groupby("true_client_ip", group_keys=False).apply(compute_rolling_attempts).reset_index(drop=True)

# add is_proxy_ip flag (if akamai_epd present for that IP ever)
proxy_by_ip = otp_df.groupby("true_client_ip")["akamai_epd"].apply(lambda s: s.notna().any()).rename("is_proxy_ip")
grouped_rolled = grouped_rolled.join(proxy_by_ip, on="true_client_ip")

# attack candidates: attempts exceed burst_threshold OR is proxy ip (and proxy hits > proxy_repeat_threshold)
proxy_hits = otp_df[otp_df["akamai_epd"].notna()].groupby("true_client_ip").size().rename("proxy_hits")
grouped_rolled = grouped_rolled.join(proxy_hits, on="true_client_ip")
grouped_rolled["proxy_hits"] = grouped_rolled["proxy_hits"].fillna(0).astype(int)
grouped_rolled["is_proxy_ip"] = grouped_rolled["is_proxy_ip"].fillna(False)
# Mark candidate only when attempts_in_10_min > burst_threshold OR (is proxy and proxy_hits>proxy_repeat_threshold)
grouped_rolled["attack_candidate"] = ((grouped_rolled["attempts_in_10_min"] > burst_threshold) | 
                                     ((grouped_rolled["is_proxy_ip"]) & (grouped_rolled["proxy_hits"] > proxy_repeat_threshold)))

# only keep rows with more than 1 instance? The user asked "more than 1 instance - filter these in a new table minute bucket table"
minute_bucket_table = grouped_rolled.copy()

# -------------------------
# IP Age & repetition count
# -------------------------
ip_stats = (
    otp_df.groupby("true_client_ip")
    .agg(days_seen=("date", lambda s: s.nunique()),
         first_seen=("timestamp", "min"),
         last_seen=("timestamp", "max"),
         total_requests=("timestamp", "count"))
    .reset_index()
)
# Benchmark for repetitions on a normal day:
ip_daily = otp_df.groupby(["true_client_ip", "date"]).size().reset_index(name="daily_requests")
# compute per-IP baseline mean daily requests
ip_baseline = ip_daily.groupby("true_client_ip")["daily_requests"].agg(["mean", "median", "std"]).reset_index().rename(columns={"mean":"daily_mean","median":"daily_median","std":"daily_std"})
ip_stats = ip_stats.merge(ip_baseline, on="true_client_ip", how="left")
# flag IPs above baseline*multiplier
ip_stats["above_benchmark"] = ip_stats["total_requests"] > (ip_stats["daily_mean"].fillna(0) * ip_benchmark_multiplier)

# -------------------------
# Spike detection in akamai_epd for certain IPs
# detect when an IP's daily proxy_count is >> its normal (mean+3*std)
# -------------------------
proxy_daily = otp_df.groupby(["true_client_ip", "date"])["akamai_epd"].apply(lambda s: s.notna().sum()).reset_index(name="proxy_count")
# compute per-IP stats and flag spikes
proxy_stats = proxy_daily.groupby("true_client_ip")["proxy_count"].agg(["mean", "std"]).reset_index().rename(columns={"mean":"proxy_mean","std":"proxy_std"})
proxy_daily = proxy_daily.merge(proxy_stats, on="true_client_ip", how="left")
proxy_daily["proxy_spike"] = proxy_daily.apply(lambda r: (r["proxy_count"] > (r["proxy_mean"] + 3*(r["proxy_std"] if not np.isnan(r["proxy_std"]) else 0))) if not np.isnan(r["proxy_mean"]) else False, axis=1)
spike_ips = proxy_daily[proxy_daily["proxy_spike"]]

# -------------------------
# BMP rules: BMP > bmp_threshold and seen >= bmp_times_threshold times in a day
# -------------------------
bmp_daily = otp_df[otp_df["bmp_score"].notna()].groupby(["true_client_ip", "date"])["bmp_score"].apply(lambda s: (s > bmp_threshold).sum()).reset_index(name="bmp_high_count")
bmp_flagged = bmp_daily[bmp_daily["bmp_high_count"] >= bmp_times_threshold]

# -------------------------
# Request rates overall (per-minute, per-10min, per-hour, per-day)
# -------------------------
otp_df.set_index("timestamp", inplace=False)  # don't permanently set
rates = {}
for label, rule in [("Per Minute", "1T"), ("Per 10 Minutes", f"{burst_window_mins}T"), ("Per Hour", "1H"), ("Per Day", "1D")]:
    series = otp_df.set_index("timestamp").resample(rule)["true_client_ip"].count().rename("requests").reset_index()
    rates[label] = series

# -------------------------
# Suspicious device behavior (counts per device)
# -------------------------
if "dr_dv" in otp_df.columns:
    device_per_min = otp_df.groupby(["dr_dv", "minute_bucket"]).size().reset_index(name="requests_per_min")
    suspicious_devices = device_per_min[device_per_min["requests_per_min"] >= device_min_threshold]
else:
    device_per_min = pd.DataFrame(columns=["dr_dv", "minute_bucket", "requests_per_min"])
    suspicious_devices = device_per_min

# -------------------------
# Suspicious IPs (simple summary counts)
# -------------------------
ip_request_counts = otp_df.groupby("true_client_ip").size().reset_index(name="total_requests")
suspicious_ips_by_burst = grouped_rolled[grouped_rolled["attack_candidate"]].groupby("true_client_ip").agg(first_seen=("minute_bucket", "min"), max_attempts=("attempts_in_10_min", "max"), proxy_hits=("proxy_hits", "max")).reset_index().sort_values("max_attempts", ascending=False)

# -------------------------
# Combine flags into anomalies table (one row per ip + reason columns)
# -------------------------
anomalies = ip_stats[["true_client_ip","days_seen","first_seen","last_seen","total_requests","daily_mean","daily_median","daily_std","above_benchmark"]].copy()
anomalies = anomalies.merge(suspicious_ips_by_burst[["true_client_ip","max_attempts","proxy_hits"]], on="true_client_ip", how="left")
anomalies = anomalies.merge(bmp_flagged.groupby("true_client_ip")["bmp_high_count"].sum().reset_index().rename(columns={"bmp_high_count":"bmp_high_days_sum"}), on="true_client_ip", how="left")
anomalies = anomalies.merge(proxy_daily.groupby("true_client_ip")["proxy_count"].sum().reset_index().rename(columns={"proxy_count":"total_proxy_hits"}), on="true_client_ip", how="left")
anomalies["suspicious_by_burst"] = anomalies["max_attempts"].fillna(0) > burst_threshold
anomalies["suspicious_by_bmp"] = anomalies["bmp_high_days_sum"].fillna(0) > 0
anomalies["suspicious_by_proxy_spike"] = anomalies["true_client_ip"].isin(spike_ips["true_client_ip"].unique())
anomalies["suspicious_by_device"] = anomalies["true_client_ip"].isin(suspicious_devices["dr_dv"].unique())  # approximate mapping might be empty
anomalies["reason"] = anomalies.apply(lambda r: ", ".join(
    [x for x, flag in [
        ("burst", r["suspicious_by_burst"]),
        ("bmp", r["suspicious_by_bmp"]),
        ("proxy_spike", r["suspicious_by_proxy_spike"]),
        ("above_benchmark", r["above_benchmark"])
    ] if flag]), axis=1)

# -------------------------
# UI: filters & display
# -------------------------
st.subheader("Controls & Filters")
col1, col2, col3 = st.columns(3)
with col1:
    ip_filter = st.text_input("Filter by IP (partial match)", "")
with col2:
    device_filter = st.text_input("Filter by device id (partial)", "")
with col3:
    min_requests_filter = st.number_input("Minimum total requests (for IP table)", min_value=0, value=0, step=1)

# filtered anomalies for display
display_anomalies = anomalies.copy()
if ip_filter:
    display_anomalies = display_anomalies[display_anomalies["true_client_ip"].str.contains(ip_filter)]
if min_requests_filter > 0:
    display_anomalies = display_anomalies[display_anomalies["total_requests"] >= min_requests_filter]

st.subheader("Anomalies summary (IP-level)")
st.write(f"Total IPs flagged with any reason: {display_anomalies[display_anomalies['reason']!=''].shape[0]}")
st.dataframe(display_anomalies.sort_values(["total_requests"], ascending=False).head(200))

st.subheader("Minute-bucket table (grouped OTP attempts per IP per minute)")
st.dataframe(minute_bucket_table.sort_values(["attempts_in_10_min"], ascending=False).head(200))

st.subheader("Suspicious devices (high requests per minute)")
st.dataframe(suspicious_devices.head(200))

# -------------------------
# Visualizations
# -------------------------
st.subheader("Visualizations")

# time series overall per minute
st.markdown("**OTP requests over time (per minute)**")
st.line_chart(rates["Per Minute"].set_index("timestamp")["requests"])

# bursty IPs over time (top 10 by max_attempts)
top_bursty_ips = suspicious_ips_by_burst["true_client_ip"].head(10).tolist()
if top_bursty_ips:
    st.markdown("**Top bursty IPs (by attempts in 10min) timeline**")
    mask = otp_df["true_client_ip"].isin(top_bursty_ips)
    per_min_top = otp_df[mask].groupby([pd.Grouper(key="timestamp", freq="1T"), "true_client_ip"]).size().reset_index(name="requests")
    fig = None
    try:
        import plotly.express as px
        fig = px.line(per_min_top, x="timestamp", y="requests", color="true_client_ip", title="Top bursty IPs timeline")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.dataframe(per_min_top.head())

# # BMP histogram
# st.markdown("**BMP score distribution (extracted from akamai_bot)**")
# bmp_hist = otp_df["bmp_score"].dropna()
# if len(bmp_hist) > 0:
#     st.bar_chart(bmp_hist.value_counts().sort_index())
# else:
#     st.info("No BMP scores extracted from akamai_bot (column missing or no digits found).")

# proxy spike table
st.subheader("Proxy spikes (per IP per day flagged)")
st.dataframe(spike_ips.sort_values(["proxy_count"], ascending=False).head(200))

# -------------------------
# Requests per time basis (1min, 10min, 1h, 1d) by Device & IP
# -------------------------
st.subheader("📊 Requests per time buckets (Device & IP)")

# Per device ID
device_time_buckets = (
    otp_df.groupby(["dr_dv", pd.Grouper(key="timestamp", freq="1T")])
    .size().reset_index(name="requests_per_min")
)
device_time_buckets["requests_10min"] = (
    otp_df.groupby(["dr_dv", pd.Grouper(key="timestamp", freq="10T")])
    .size().reset_index(name="requests")["requests"].reindex(device_time_buckets.index).fillna(0).astype(int)
)
device_time_buckets["requests_1h"] = (
    otp_df.groupby(["dr_dv", pd.Grouper(key="timestamp", freq="1H")])
    .size().reset_index(name="requests")["requests"].reindex(device_time_buckets.index).fillna(0).astype(int)
)
device_time_buckets["requests_1d"] = (
    otp_df.groupby(["dr_dv", pd.Grouper(key="timestamp", freq="1D")])
    .size().reset_index(name="requests")["requests"].reindex(device_time_buckets.index).fillna(0).astype(int)
)

st.markdown("**Requests per Device ID (with thresholds applied)**")
st.dataframe(device_time_buckets.sort_values("requests_per_min", ascending=False).head(100))

# Per IP address
ip_time_buckets = (
    otp_df.groupby(["true_client_ip", pd.Grouper(key="timestamp", freq="1T")])
    .size().reset_index(name="requests_per_min")
)
ip_time_buckets["requests_10min"] = (
    otp_df.groupby(["true_client_ip", pd.Grouper(key="timestamp", freq="10T")])
    .size().reset_index(name="requests")["requests"].reindex(ip_time_buckets.index).fillna(0).astype(int)
)
ip_time_buckets["requests_1h"] = (
    otp_df.groupby(["true_client_ip", pd.Grouper(key="timestamp", freq="1H")])
    .size().reset_index(name="requests")["requests"].reindex(ip_time_buckets.index).fillna(0).astype(int)
)
ip_time_buckets["requests_1d"] = (
    otp_df.groupby(["true_client_ip", pd.Grouper(key="timestamp", freq="1D")])
    .size().reset_index(name="requests")["requests"].reindex(ip_time_buckets.index).fillna(0).astype(int)
)

st.markdown("**Requests per IP Address (with thresholds applied)**")
st.dataframe(ip_time_buckets.sort_values("requests_per_min", ascending=False).head(100))


# rates summary table downloads
st.subheader("Downloads")
st.download_button("Download anomalies (CSV)", display_anomalies.to_csv(index=False).encode("utf-8"), "anomalies.csv", "text/csv")
st.download_button("Download minute-bucket table (CSV)", minute_bucket_table.to_csv(index=False).encode("utf-8"), "minute_bucket_table.csv", "text/csv")
st.download_button("Download suspicious devices (CSV)", suspicious_devices.to_csv(index=False).encode("utf-8"), "suspicious_devices.csv", "text/csv")
st.download_button("Download bmp-flagged (CSV)", bmp_flagged.to_csv(index=False).encode("utf-8"), "bmp_flagged.csv", "text/csv")
