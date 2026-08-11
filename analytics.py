"""
Core analytics for StockPulse: variance computation, IQR anomaly detection,
and aggregation helpers. Pure pandas — no side effects, easy to unit test.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    "sku",
    "product_name",
    "category",
    "location",
    "expected_qty",
    "counted_qty",
    "unit_cost",
    "count_date",
]


def compute_variance(df: pd.DataFrame) -> pd.DataFrame:
    """Add quantity, percentage, and value variance columns."""
    out = df.copy()

    # Coerce numerics defensively.
    for col in ("expected_qty", "counted_qty", "unit_cost"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["qty_variance"] = out["counted_qty"] - out["expected_qty"]

    # Percentage variance relative to expected; guard divide-by-zero.
    out["pct_variance"] = (
        out["qty_variance"]
        / out["expected_qty"].replace(0, pd.NA)
        * 100
    ).fillna(0.0)

    out["value_variance"] = out["qty_variance"] * out["unit_cost"]
    return out


def detect_anomalies(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """
    Flag statistical outliers on value_variance using the IQR rule.

    A row is an anomaly if its value_variance is below Q1 - k*IQR or
    above Q3 + k*IQR.
    """
    out = df.copy()
    series = out["value_variance"]
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    out["is_anomaly"] = (series < lower) | (series > upper)
    return out


def summary_metrics(df: pd.DataFrame) -> dict:
    """Headline KPIs for the current view."""
    records = len(df)
    if records == 0:
        return {
            "records": 0, "accuracy": 0.0, "net_value": 0.0,
            "abs_value": 0.0, "anomalies": 0,
        }
    exact = int((df["qty_variance"] == 0).sum())
    return {
        "records": records,
        "accuracy": exact / records * 100,
        "net_value": float(df["value_variance"].sum()),
        "abs_value": float(df["value_variance"].abs().sum()),
        "anomalies": int(df["is_anomaly"].sum()),
    }


def variance_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Aggregate value variance across a categorical dimension."""
    grouped = (
        df.groupby(dimension, as_index=False)["value_variance"]
        .sum()
        .sort_values("value_variance")
    )
    return grouped


def top_discrepancies(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the n line items with the largest absolute value variance."""
    return (
        df.reindex(df["value_variance"].abs().sort_values(ascending=False).index)
        .head(n)
        .copy()
    )
