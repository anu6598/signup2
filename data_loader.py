import pandas as pd

def load_signup_data(filepath):
    df = pd.read_csv(filepath)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df
