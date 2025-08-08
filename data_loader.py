import pandas as pd

def load_signup_data(file):
    """Load signup logs CSV and ensure start_time & IP are clean."""
    df = pd.read_csv(file)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Ensure start_time column exists
    if "start_time" not in df.columns:
        raise ValueError("No 'start_time' column found in CSV.")

    # Parse timestamps
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)

    # Drop rows without valid start_time
    df = df.dropna(subset=["start_time"])

    # Ensure IP column exists
    if "true_client_ip" not in df.columns:
        raise ValueError("No 'true_client_ip' column found in CSV.")

    return df
