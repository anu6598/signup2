import pandas as pd
import altair as alt

def daily_summary(df, date_col="start_time", ip_col="true_client_ip"):
    """Summarize total signups and anomalies per day."""
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

def plot_time_series(summary_df):
    """Plot signups & anomalies over time."""
    base = alt.Chart(summary_df).encode(
        x="date:T"
    )

    signup_line = base.mark_line(color="blue").encode(
        y="total_signups:Q",
        tooltip=["date", "total_signups", "anomalies"]
    )

    anomaly_points = base.mark_circle(color="red", size=60).encode(
        y="anomalies:Q"
    )

    return signup_line + anomaly_points
