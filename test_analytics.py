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


# ---- v2: shrinkage & trend tests ---------------------------------------- #

def _multi_period_frame():
    import pandas as pd
    rows = []
    for date in ["2026-06-01", "2026-07-01", "2026-08-01"]:
        rows += [
            {"sku": "A", "product_name": "leaky", "category": "X", "location": "W1",
             "expected_qty": 100, "counted_qty": 90, "unit_cost": 10.0, "count_date": date},
            {"sku": "B", "product_name": "fine", "category": "Y", "location": "W2",
             "expected_qty": 50, "counted_qty": 50, "unit_cost": 5.0, "count_date": date},
        ]
    return pd.DataFrame(rows)


def test_shrinkage_metrics_splits_loss_and_surplus():
    from analytics import compute_variance, shrinkage_metrics
    df = compute_variance(_frame())
    s = shrinkage_metrics(df)
    # Row A is short (-$100), rows C/D are over.
    assert s["shrinkage_value"] < 0
    assert s["overage_value"] > 0
    assert 0 <= s["shortage_rate"] <= 100


def test_has_multiple_periods():
    from analytics import compute_variance, has_multiple_periods
    assert has_multiple_periods(compute_variance(_multi_period_frame())) is True
    assert has_multiple_periods(compute_variance(_frame())) is False


def test_accuracy_trend_is_chronological():
    from analytics import compute_variance, accuracy_trend
    t = accuracy_trend(compute_variance(_multi_period_frame()))
    assert list(t["count_date"]) == sorted(t["count_date"])
    assert len(t) == 3


def test_recurring_offenders_flags_persistent_shortage():
    from analytics import compute_variance, recurring_offenders
    ro = recurring_offenders(compute_variance(_multi_period_frame()))
    # "leaky" is short in all 3 periods; "fine" never is.
    assert "leaky" in set(ro["product_name"])
    assert "fine" not in set(ro["product_name"])
    assert int(ro.loc[ro["product_name"] == "leaky", "periods_short"].iloc[0]) == 3
