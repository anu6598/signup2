import pandas as pd
import numpy as np

def detect_anomalies(df, time_col="timestamp", ip_col="ip", threshold=5):
    """
    Detect IPs with unusually high signup counts per day.

    Args:
        df: DataFrame with signup logs
        time_col: column with datetime
        ip_col: column with IP address
        threshold: count threshold for anomaly

    Returns:
        df with anomaly flag
    """
    df[time_col] = pd.to_datetime(df[time_col])
    df["date"] = df[time_col].dt.date

    daily_counts = df.groupby(["date", ip_col]).size().reset_index(name="signup_count")
    daily_counts["is_anomaly"] = daily_counts["signup_count"] > threshold

    return df.merge(daily_counts, on=["date", ip_col])
