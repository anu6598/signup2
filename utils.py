import pandas as pd

def daily_summary(df, date_col="timestamp", ip_col="true_client_ip"):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    summary = (
        df.groupby(df[date_col].dt.date)
          .agg(
              total_signups=(ip_col, "count"),
              anomalies=("is_anomaly", "sum")
          )
          .reset_index()
          .rename(columns={date_col: "date"})
    )
    return summary
