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


# --------------------------------------------------------------------------- #
# Shrinkage intelligence
# --------------------------------------------------------------------------- #
def shrinkage_metrics(df: pd.DataFrame) -> dict:
    """
    Split variance into shortages (loss) and overages (surplus).

    Shortage  = counted < expected  -> value_variance < 0  (money walking out)
    Overage   = counted > expected  -> value_variance > 0  (receiving/count errors)
    """
    if len(df) == 0:
        return {
            "shrinkage_value": 0.0, "overage_value": 0.0,
            "shrinkage_units": 0, "overage_units": 0,
            "shortage_rate": 0.0, "shrinkage_pct_of_stock": 0.0,
        }

    short = df[df["qty_variance"] < 0]
    over = df[df["qty_variance"] > 0]

    stock_value = float((df["expected_qty"] * df["unit_cost"]).sum())
    shrinkage_value = float(short["value_variance"].sum())  # negative

    return {
        "shrinkage_value": shrinkage_value,
        "overage_value": float(over["value_variance"].sum()),
        "shrinkage_units": int(short["qty_variance"].sum()),
        "overage_units": int(over["qty_variance"].sum()),
        "shortage_rate": len(short) / len(df) * 100,
        "shrinkage_pct_of_stock": (
            abs(shrinkage_value) / stock_value * 100 if stock_value else 0.0
        ),
    }


def shrinkage_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Total shortage value (losses only) per dimension, most negative first."""
    short = df[df["qty_variance"] < 0]
    if short.empty:
        return pd.DataFrame({dimension: [], "shrinkage_value": []})
    grouped = (
        short.groupby(dimension, as_index=False)["value_variance"]
        .sum()
        .rename(columns={"value_variance": "shrinkage_value"})
        .sort_values("shrinkage_value")
    )
    return grouped


# --------------------------------------------------------------------------- #
# Multi-period trend analysis
# --------------------------------------------------------------------------- #
def has_multiple_periods(df: pd.DataFrame) -> bool:
    """True if the data spans more than one count date."""
    return df["count_date"].nunique() > 1


def accuracy_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per count_date: accuracy %, net value variance, and absolute value at risk.
    Sorted chronologically.
    """
    rows = []
    for date, grp in df.groupby("count_date"):
        exact = int((grp["qty_variance"] == 0).sum())
        rows.append({
            "count_date": date,
            "accuracy": exact / len(grp) * 100 if len(grp) else 0.0,
            "net_value_variance": float(grp["value_variance"].sum()),
            "abs_value_at_risk": float(grp["value_variance"].abs().sum()),
            "records": len(grp),
        })
    return pd.DataFrame(rows).sort_values("count_date").reset_index(drop=True)


def recurring_offenders(df: pd.DataFrame, min_periods: int = 2) -> pd.DataFrame:
    """
    Products that show a shortage in at least `min_periods` distinct count dates.

    Recurring shortages are the strongest signal of genuine shrinkage
    (theft, spoilage, systematic miscount) versus a one-off counting error.
    """
    short = df[df["qty_variance"] < 0].copy()
    if short.empty:
        return pd.DataFrame(
            columns=["product_name", "periods_short", "total_shrinkage_value",
                     "total_units_lost", "locations"]
        )

    agg = (
        short.groupby("product_name")
        .agg(
            periods_short=("count_date", "nunique"),
            total_shrinkage_value=("value_variance", "sum"),
            total_units_lost=("qty_variance", "sum"),
            locations=("location", lambda s: ", ".join(sorted(set(s)))),
        )
        .reset_index()
    )
    agg = agg[agg["periods_short"] >= min_periods]
    return agg.sort_values("total_shrinkage_value").reset_index(drop=True)
