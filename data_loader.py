import pandas as pd

def load_signup_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "date" in df.columns and "start_time" in df.columns:
        # Combine date and time, handle multiple formats
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["start_time"].astype(str),
            errors="coerce",
            infer_datetime_format=True,
            utc=True
        )
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["date"].astype(str),
            errors="coerce",
            infer_datetime_format=True,
            utc=True
        )
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"].astype(str),
            errors="coerce",
            infer_datetime_format=True,
            utc=True
        )
    else:
        raise ValueError("No suitable timestamp columns found in dataset.")

    # Drop rows where timestamp couldn't be parsed
    df = df.dropna(subset=["timestamp"])

    return df
