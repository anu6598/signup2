import pandas as pd

def load_signup_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "date" in df.columns and "start_time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"] + " " + df["start_time"])
    elif "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"])
    else:
        raise ValueError("No suitable timestamp columns found in dataset.")
    
    return df
