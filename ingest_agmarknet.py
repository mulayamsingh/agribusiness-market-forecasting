import requests
import pandas as pd
import numpy as np

def fetch_and_clean_mandi_data(api_key: str, commodity: str = "Soyabean", state: str = "Maharashtra") -> pd.DataFrame:
    """Automated harvester and statistical cleaner for Agmarknet public API (Week 1 Task)."""
    url = f"https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?api-key={api_key}&format=json&limit=5000"
    params = {"filters[commodity]": commodity, "filters[state]": state}
    response = requests.get(url, params=params, timeout=30)
    records = response.json().get("records", [])
    df = pd.DataFrame(records)

    df["arrival_date"] = pd.to_datetime(df["arrival_date"], format="%d/%m/%Y")
    for col in ["modal_price", "min_price", "max_price", "arrivals"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    median = df["modal_price"].median()
    mad = np.median(np.abs(df["modal_price"] - median))
    threshold = 3.0 * mad
    df_clean = df[(df["modal_price"] >= median - threshold) & (df["modal_price"] <= median + threshold)].copy()

    df_clean = df_clean.sort_values("arrival_date")
    df_clean["price_ema_7"] = df_clean["modal_price"].ewm(span=7, adjust=False).mean()
    df_clean["price_lag_7"] = df_clean["modal_price"].shift(7)
    df_clean["price_momentum"] = (df_clean["modal_price"] - df_clean["price_lag_7"]) / df_clean["price_lag_7"]
    return df_clean
