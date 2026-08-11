"""Unit tests for the analytics module. Run with: pytest"""

import pandas as pd

from analytics import (
    compute_variance,
    detect_anomalies,
    summary_metrics,
    top_discrepancies,
    variance_by_dimension,
)


def _frame():
    return pd.DataFrame(
        {
            "sku": ["A", "B", "C", "D"],
            "product_name": ["p1", "p2", "p3", "p4"],
            "category": ["X", "X", "Y", "Y"],
            "location": ["W1", "W1", "W2", "W2"],
            "expected_qty": [100, 50, 0, 200],
            "counted_qty": [90, 50, 5, 500],
            "unit_cost": [10.0, 2.0, 4.0, 1.0],
            "count_date": ["2026-06-01"] * 4,
        }
    )


def test_compute_variance_basic():
    df = compute_variance(_frame())
    assert df.loc[0, "qty_variance"] == -10
    assert df.loc[0, "value_variance"] == -100.0
    assert round(df.loc[0, "pct_variance"], 1) == -10.0


def test_compute_variance_divide_by_zero():
    df = compute_variance(_frame())
    # expected_qty == 0 should not blow up; pct set to 0.
    assert df.loc[2, "pct_variance"] == 0.0
    assert df.loc[2, "value_variance"] == 20.0


def test_detect_anomalies_flags_outlier():
    df = detect_anomalies(compute_variance(_frame()))
    # Row D (+300 units, +$300) is the clear outlier.
    assert bool(df.loc[3, "is_anomaly"]) is True


def test_summary_metrics():
    m = summary_metrics(detect_anomalies(compute_variance(_frame())))
    assert m["records"] == 4
    assert m["accuracy"] == 25.0  # only row B is an exact match
    assert m["anomalies"] >= 1


def test_variance_by_dimension():
    df = compute_variance(_frame())
    agg = variance_by_dimension(df, "category")
    assert set(agg["category"]) == {"X", "Y"}


def test_top_discrepancies_order():
    df = compute_variance(_frame())
    top = top_discrepancies(df, n=2)
    # Largest absolute variance first (row D at $300).
    assert top.iloc[0]["sku"] == "D"


def test_empty_summary():
    empty = compute_variance(_frame()).iloc[0:0]
    empty = detect_anomalies(empty)
    m = summary_metrics(empty)
    assert m["records"] == 0
    assert m["accuracy"] == 0.0
