import pandas as pd

def daily_summary(df):
    """
    Returns daily summary of signups & anomalies.
    """
    return (
        df.groupby("date")
        .agg(
            total_signups=("ip", "count"),
            anomalies=("is_anomaly", "sum")
        )
        .reset_index()
    )
