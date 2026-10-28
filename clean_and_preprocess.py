import re
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

class AgribusinessCleaner(BaseEstimator, TransformerMixin):
    """Deterministic domain cleaner for APMC Mandi & Agro-Climatic datasets (Week 2 Task)."""
    def __init__(self, hampel_k: float = 3.0, winsor_limits: tuple = (0.01, 0.99)):
        self.hampel_k = hampel_k
        self.winsor_limits = winsor_limits

    @staticmethod
    def _parse_arrival_to_mt(val) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val).replace(",", "").strip().lower()
        m = re.match(r"([-+]?\d*\.?\d+)\s*([a-z]*)", s)
        if not m:
            return np.nan
        qty, unit = float(m.group(1)), m.group(2)
        if qty < 0:
            return np.nan
        if unit in ("qtl", "quintal", "quintals"):
            return qty * 0.10
        if unit in ("kg", "kgs", "kilogram"):
            return qty * 0.001
        if unit in ("bag", "bags"):
            return qty * 0.05
        return qty

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        prices = pd.to_numeric(df["modal_price"], errors="coerce")
        self.median_price_ = prices.median()
        self.mad_price_ = 1.4826 * np.median(np.abs(prices.dropna() - self.median_price_))
        valid_p = prices[np.abs(prices - self.median_price_) <= 5.0 * self.mad_price_]
        self.p_low_, self.p_high_ = valid_p.quantile(self.winsor_limits)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy().drop_duplicates()
        df["arrival_mt"] = df["raw_arrival"].apply(self._parse_arrival_to_mt)
        df["modal_price"] = pd.to_numeric(df["modal_price"], errors="coerce")
        z_mad = np.abs(df["modal_price"] - self.median_price_) / self.mad_price_
        df.loc[z_mad > 8.0, "modal_price"] = np.nan
        df["modal_price"] = df["modal_price"].interpolate(method="linear").clip(self.p_low_, self.p_high_)
        df.loc[~df["soil_ph"].between(4.5, 9.5), "soil_ph"] = np.nan
        return df
