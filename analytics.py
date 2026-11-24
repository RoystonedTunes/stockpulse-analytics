"""
Core analytics for StockPulse: variance computation, IQR anomaly detection,
and aggregation helpers. Pure pandas — no side effects, easy to unit test.
"""

from __future__ import annotations

import io
import zipfile

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
    One row per stock take (count_date), oldest first:

      accuracy            % of product lines where the count matched exactly
      loss_value          money lost on items that came up short (positive number)
      surplus_value       money "found" on items that came up over (positive number)
      net_value_variance  surplus minus loss (negative = overall loss)
      abs_value_at_risk   loss plus surplus: the total size of all mismatches
      records             how many product lines were counted
    """
    rows = []
    for date, grp in df.groupby("count_date"):
        exact = int((grp["qty_variance"] == 0).sum())
        rows.append({
            "count_date": date,
            "accuracy": exact / len(grp) * 100 if len(grp) else 0.0,
            "loss_value": -float(grp.loc[grp["qty_variance"] < 0, "value_variance"].sum()),
            "surplus_value": float(grp.loc[grp["qty_variance"] > 0, "value_variance"].sum()),
            "net_value_variance": float(grp["value_variance"].sum()),
            "abs_value_at_risk": float(grp["value_variance"].abs().sum()),
            "records": len(grp),
        })
    return pd.DataFrame(rows).sort_values("count_date").reset_index(drop=True)


def loss_over_time(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    Money lost on shortages at each stock take, split by `dimension`
    ("category" or "location"). Loss is a positive number. Every group appears
    at every date (0 when nothing was lost), so lines don't break.
    """
    short = df[df["qty_variance"] < 0]
    dates = sorted(df["count_date"].unique())
    groups = sorted(df[dimension].unique())
    loss = -short.groupby(["count_date", dimension])["value_variance"].sum()
    full = pd.MultiIndex.from_product([dates, groups], names=["count_date", dimension])
    return (
        loss.reindex(full, fill_value=0.0)
        .rename("loss_value")
        .reset_index()
    )


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


# --------------------------------------------------------------------------- #
# Benchmarking & alerts
# --------------------------------------------------------------------------- #
def _band(pct: float, amber: float, red: float) -> str:
    """Return a traffic-light band for a shrinkage percentage."""
    if pct >= red:
        return "red"
    if pct >= amber:
        return "amber"
    return "green"


def benchmark_by_dimension(
    df: pd.DataFrame, dimension: str, amber: float = 3.0, red: float = 5.0
) -> pd.DataFrame:
    """
    Shrinkage as a percentage of stock value for each value of `dimension`,
    with a traffic-light status band.

    Fresh-meat retail typically runs 2-5% shrink, so the default thresholds
    flag anything above 3% (amber) and 5% (red).
    """
    rows = []
    for key, grp in df.groupby(dimension):
        stock_value = float((grp["expected_qty"] * grp["unit_cost"]).sum())
        short = grp[grp["qty_variance"] < 0]
        shrink_value = float(short["value_variance"].sum())  # negative
        pct = abs(shrink_value) / stock_value * 100 if stock_value else 0.0
        rows.append({
            dimension: key,
            "shrinkage_value": shrink_value,
            "stock_value": stock_value,
            "shrinkage_pct": pct,
            "status": _band(pct, amber, red),
        })
    out = pd.DataFrame(rows).sort_values("shrinkage_pct", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Worsening-trend detection
# --------------------------------------------------------------------------- #
def shrinkage_trajectory(df: pd.DataFrame, min_periods: int = 2) -> pd.DataFrame:
    """
    For each product, track shrinkage value across count dates and classify the
    direction as worsening, improving, or stable.

    A product losing progressively more each period is 'worsening' — a live,
    accelerating problem that matters more than a steadily bad one. Direction is
    judged by comparing the latest period's shrinkage to the earliest.
    """
    periods = sorted(df["count_date"].dropna().unique())
    if len(periods) < min_periods:
        return pd.DataFrame(
            columns=["product_name", "trend", "first_value", "latest_value",
                     "change", "series"]
        )

    # Shrinkage (losses only) per product per date.
    short = df[df["qty_variance"] < 0]
    pivot = (
        short.groupby(["product_name", "count_date"])["value_variance"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(columns=periods, fill_value=0.0)
    )

    rows = []
    for product, series in pivot.iterrows():
        vals = [float(series[p]) for p in periods]  # each <= 0
        first, latest = vals[0], vals[-1]
        change = latest - first  # more negative => worsening
        if abs(change) < 0.01 * max(1.0, abs(first)) or abs(change) < 5:
            trend = "stable"
        elif change < 0:
            trend = "worsening"
        else:
            trend = "improving"
        # Only report products that have some shrinkage somewhere.
        if any(v < 0 for v in vals):
            rows.append({
                "product_name": product,
                "trend": trend,
                "first_value": first,
                "latest_value": latest,
                "change": change,
                "series": vals,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Worsening first, ordered by how much worse.
    order = {"worsening": 0, "stable": 1, "improving": 2}
    out["_o"] = out["trend"].map(order)
    out = out.sort_values(["_o", "change"]).drop(columns="_o")
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stock-out / low-stock risk
# --------------------------------------------------------------------------- #
def stockout_risk(df: pd.DataFrame, critical_pct: float = 20.0) -> pd.DataFrame:
    """
    Flag items whose most recent counted quantity is critically low relative to
    the level they are normally stocked at.

    NOTE: a true reorder model needs sales/depletion data, which a stock-take
    alone does not contain. This is a *low-stock* signal: for each product+shop
    it compares the latest counted quantity to that item's typical (median)
    expected level, and flags anything sitting below `critical_pct` of normal.
    """
    latest_date = sorted(df["count_date"].dropna().unique())[-1]
    latest = df[df["count_date"] == latest_date].copy()

    # Typical stock level per product+shop across all periods.
    typical = (
        df.groupby(["product_name", "location"])["expected_qty"]
        .median()
        .rename("typical_qty")
        .reset_index()
    )
    merged = latest.merge(typical, on=["product_name", "location"], how="left")
    merged["pct_of_typical"] = (
        merged["counted_qty"] / merged["typical_qty"].replace(0, pd.NA) * 100
    ).fillna(0.0)

    at_risk = merged[merged["pct_of_typical"] <= critical_pct].copy()
    at_risk = at_risk[[
        "product_name", "category", "location", "counted_qty",
        "typical_qty", "pct_of_typical", "unit_cost",
    ]].sort_values("pct_of_typical")
    return at_risk.reset_index(drop=True), latest_date


# --------------------------------------------------------------------------- #
# Store scorecard: one row per shop, the key numbers side by side
# --------------------------------------------------------------------------- #
SCORECARD_COLUMNS = [
    "location", "items_counted", "count_accuracy_pct", "loss_value",
    "loss_pct_of_stock", "surplus_value", "unusual_items", "status",
]


def store_scorecard(df: pd.DataFrame, amber: float = 3.0, red: float = 5.0) -> pd.DataFrame:
    """
    A simple report card for every shop, like a weekly KPI table.

    For each shop:
      items_counted       how many product lines were counted
      count_accuracy_pct  % of lines where the count matched exactly
      loss_value          money lost on items that came up short (positive number)
      loss_pct_of_stock   that loss as a % of the shop's expected stock value
      surplus_value       money "found" on items that came up over
      unusual_items       lines flagged as statistical anomalies
      status              green / amber / red, using the same 3% and 5% limits
    """
    if df.empty:
        return pd.DataFrame(columns=SCORECARD_COLUMNS)

    rows = []
    for shop, grp in df.groupby("location"):
        items = len(grp)
        exact = int((grp["qty_variance"] == 0).sum())
        stock_value = float((grp["expected_qty"] * grp["unit_cost"]).sum())
        loss = -float(grp.loc[grp["qty_variance"] < 0, "value_variance"].sum())
        surplus = float(grp.loc[grp["qty_variance"] > 0, "value_variance"].sum())
        loss_pct = loss / stock_value * 100 if stock_value else 0.0
        unusual = int(grp["is_anomaly"].sum()) if "is_anomaly" in grp else 0
        rows.append({
            "location": shop,
            "items_counted": items,
            "count_accuracy_pct": exact / items * 100,
            "loss_value": loss,
            "loss_pct_of_stock": loss_pct,
            "surplus_value": surplus,
            "unusual_items": unusual,
            "status": _band(loss_pct, amber, red),
        })
    out = pd.DataFrame(rows).sort_values("loss_pct_of_stock", ascending=False)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Movers & underperformers: what changed between two stock takes
# --------------------------------------------------------------------------- #
def period_comparison(df: pd.DataFrame, current_date, previous_date) -> pd.DataFrame:
    """
    Compare each product's money difference (value variance) between two
    stock takes.

      before  = money difference at the earlier stock take
      now     = money difference at the later stock take
      change  = now - before

    Money difference is negative when stock is missing, so:
      change > 0  -> the product got better (smaller loss)
      change < 0  -> the product got worse (bigger loss)
    """
    def total_for(date):
        return (
            df[df["count_date"] == date]
            .groupby(["product_name", "category"])["value_variance"]
            .sum()
        )

    out = pd.concat(
        [total_for(previous_date).rename("before"), total_for(current_date).rename("now")],
        axis=1,
    ).fillna(0.0).reset_index()
    out["change"] = out["now"] - out["before"]
    return out


def movers_and_underperformers(comparison: pd.DataFrame, n: int = 10):
    """
    Split a period comparison into:
      movers         products that improved the most (largest positive change)
      underperformers products that got worse the most (largest negative change)
    """
    movers = (
        comparison[comparison["change"] > 0]
        .sort_values("change", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    underperformers = (
        comparison[comparison["change"] < 0]
        .sort_values("change")
        .head(n)
        .reset_index(drop=True)
    )
    return movers, underperformers


# --------------------------------------------------------------------------- #
# Export for Power BI: the data split into a simple star schema
# --------------------------------------------------------------------------- #
def powerbi_tables(df: pd.DataFrame) -> dict:
    """
    Split the stock-take data into four tidy tables that load straight into
    Power BI as a star schema:

      fact_stocktake  the numbers, one row per product per shop per count
      dim_product     one row per product (name, category)
      dim_shop        one row per shop
      dim_date        one row per count date (year, month, week)

    The fact table links to the others through the *_key columns.
    """
    data = df.copy()
    data["count_date"] = pd.to_datetime(data["count_date"])

    dim_product = (
        data[["product_name", "category"]]
        .drop_duplicates()
        .sort_values(["category", "product_name"])
        .reset_index(drop=True)
    )
    dim_product.insert(0, "product_key", range(1, len(dim_product) + 1))

    dim_shop = (
        data[["location"]]
        .drop_duplicates()
        .sort_values("location")
        .reset_index(drop=True)
        .rename(columns={"location": "shop_name"})
    )
    dim_shop.insert(0, "shop_key", range(1, len(dim_shop) + 1))

    dates = data["count_date"].drop_duplicates().sort_values().reset_index(drop=True)
    dim_date = pd.DataFrame({
        "date_key": dates.dt.strftime("%Y%m%d").astype(int),
        "date": dates.dt.date,
        "year": dates.dt.year,
        "month_number": dates.dt.month,
        "month_name": dates.dt.strftime("%B"),
        "week_number": dates.dt.isocalendar().week.astype(int).values,
    })

    fact = (
        data.merge(dim_product, on=["product_name", "category"])
        .merge(dim_shop, left_on="location", right_on="shop_name")
    )
    fact["date_key"] = fact["count_date"].dt.strftime("%Y%m%d").astype(int)
    fact_cols = ["date_key", "shop_key", "product_key", "sku", "expected_qty",
                 "counted_qty", "unit_cost", "qty_variance", "value_variance"]
    if "is_anomaly" in fact:
        fact_cols.append("is_anomaly")
    fact_stocktake = fact[fact_cols].reset_index(drop=True)

    return {
        "fact_stocktake": fact_stocktake,
        "dim_product": dim_product,
        "dim_shop": dim_shop,
        "dim_date": dim_date,
    }


def tables_to_zip(tables: dict) -> bytes:
    """Pack each table into one CSV file inside a single ZIP download."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, table in tables.items():
            zf.writestr(f"{name}.csv", table.to_csv(index=False))
    return buffer.getvalue()
