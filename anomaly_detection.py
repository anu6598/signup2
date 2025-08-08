import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, time_col="start_time", ip_col="true_client_ip", threshold=10):
    """Rule-based anomaly detection + Isolation Forest."""
    df = df.copy()

    # Rule-based: count IPs per day
    df["date"] = df[time_col].dt.date
    ip_counts = df.groupby([ip_col, "date"]).size().reset_index(name="count")

    # Flag anomalies based on threshold
    ip_counts["is_anomaly_rule"] = ip_counts["count"] > threshold

    # Merge back to main DF
    df = df.merge(ip_counts, on=[ip_col, "date"], how="left")

    # Isolation Forest: detect anomalies
    daily_ip_counts = ip_counts.groupby(ip_col)["count"].sum().reset_index()

    model = IsolationForest(contamination=0.05, random_state=42)
    daily_ip_counts["is_anomaly_ml"] = model.fit_predict(
        daily_ip_counts[["count"]]
    )
    daily_ip_counts["is_anomaly_ml"] = daily_ip_counts["is_anomaly_ml"].apply(lambda x: x == -1)

    # Merge ML results
    df = df.merge(daily_ip_counts[[ip_col, "is_anomaly_ml"]], on=ip_col, how="left")

    # Final anomaly column
    df["is_anomaly"] = df["is_anomaly_rule"] | df["is_anomaly_ml"]

    return df, model
