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


# ---- ops features: benchmark, trajectory, stockout ---------------------- #

def test_benchmark_bands():
    from analytics import compute_variance, benchmark_by_dimension
    b = benchmark_by_dimension(compute_variance(_frame()), "category")
    assert set(["category", "shrinkage_pct", "status"]).issubset(b.columns)
    assert set(b["status"]).issubset({"green", "amber", "red"})


def test_shrinkage_trajectory_detects_worsening():
    import pandas as pd
    from analytics import compute_variance, shrinkage_trajectory
    rows = []
    # 'decliner' loses progressively more; 'steady' is flat at zero variance.
    shorts = {"2026-06-01": 95, "2026-07-01": 90, "2026-08-01": 60}
    for date, counted in shorts.items():
        rows.append({"sku": "A", "product_name": "decliner", "category": "X",
                     "location": "W1", "expected_qty": 100, "counted_qty": counted,
                     "unit_cost": 10.0, "count_date": date})
    df = compute_variance(pd.DataFrame(rows))
    traj = shrinkage_trajectory(df)
    row = traj[traj["product_name"] == "decliner"].iloc[0]
    # loss went 50 -> 100 -> 400 in value terms; latest worse than first.
    assert row["trend"] == "worsening"


def test_stockout_risk_flags_low_item():
    import pandas as pd
    from analytics import compute_variance, stockout_risk
    rows = [
        {"sku": "A", "product_name": "p", "category": "X", "location": "W1",
         "expected_qty": 100, "counted_qty": 100, "unit_cost": 5.0, "count_date": "2026-06-01"},
        {"sku": "A", "product_name": "p", "category": "X", "location": "W1",
         "expected_qty": 100, "counted_qty": 5, "unit_cost": 5.0, "count_date": "2026-07-01"},
    ]
    df = compute_variance(pd.DataFrame(rows))
    risk, latest = stockout_risk(df)
    assert latest == "2026-07-01"
    assert "p" in set(risk["product_name"])


# ---- v3: scorecard, movers, Power BI export ------------------------------ #

def test_store_scorecard_one_row_per_shop():
    from analytics import compute_variance, detect_anomalies, store_scorecard
    sc = store_scorecard(detect_anomalies(compute_variance(_frame())))
    assert set(sc["location"]) == {"W1", "W2"}
    w1 = sc[sc["location"] == "W1"].iloc[0]
    # W1: row A short by 10 units at 10.0 each = 100 lost; stock value 1000 + 100.
    assert w1["loss_value"] == 100.0
    assert round(w1["loss_pct_of_stock"], 2) == round(100 / 1100 * 100, 2)
    assert w1["count_accuracy_pct"] == 50.0
    assert set(sc["status"]).issubset({"green", "amber", "red"})


def test_store_scorecard_empty():
    from analytics import compute_variance, store_scorecard
    empty = compute_variance(_frame()).iloc[0:0]
    assert store_scorecard(empty).empty


def test_movers_and_underperformers():
    import pandas as pd
    from analytics import compute_variance, period_comparison, movers_and_underperformers
    rows = []
    # 'better' loses 10 units then 2; 'worse' loses 1 unit then 8.
    for date, better, worse in [("2026-06-01", 90, 99), ("2026-07-01", 98, 92)]:
        rows += [
            {"sku": "A", "product_name": "better", "category": "X", "location": "W1",
             "expected_qty": 100, "counted_qty": better, "unit_cost": 10.0, "count_date": date},
            {"sku": "B", "product_name": "worse", "category": "X", "location": "W1",
             "expected_qty": 100, "counted_qty": worse, "unit_cost": 10.0, "count_date": date},
        ]
    df = compute_variance(pd.DataFrame(rows))
    comp = period_comparison(df, "2026-07-01", "2026-06-01")
    movers, under = movers_and_underperformers(comp)
    assert movers.iloc[0]["product_name"] == "better"
    assert movers.iloc[0]["change"] == 80.0      # -100 before, -20 now
    assert under.iloc[0]["product_name"] == "worse"
    assert under.iloc[0]["change"] == -70.0      # -10 before, -80 now


def test_powerbi_tables_star_schema():
    from analytics import compute_variance, detect_anomalies, powerbi_tables, tables_to_zip
    import io
    import zipfile
    df = detect_anomalies(compute_variance(_multi_period_frame()))
    t = powerbi_tables(df)
    assert set(t) == {"fact_stocktake", "dim_product", "dim_shop", "dim_date"}
    # Every fact row keeps its numbers and links to each dimension.
    assert len(t["fact_stocktake"]) == len(df)
    assert set(t["fact_stocktake"]["product_key"]) <= set(t["dim_product"]["product_key"])
    assert set(t["fact_stocktake"]["shop_key"]) <= set(t["dim_shop"]["shop_key"])
    assert set(t["fact_stocktake"]["date_key"]) <= set(t["dim_date"]["date_key"])
    assert len(t["dim_date"]) == 3
    names = zipfile.ZipFile(io.BytesIO(tables_to_zip(t))).namelist()
    assert sorted(names) == sorted(f"{k}.csv" for k in t)


# ---- v4: clearer trends ---------------------------------------------------- #

def test_accuracy_trend_splits_loss_and_surplus():
    from analytics import compute_variance, accuracy_trend
    t = accuracy_trend(compute_variance(_multi_period_frame()))
    # Loss and surplus are positive; net = surplus - loss.
    assert (t["loss_value"] >= 0).all() and (t["surplus_value"] >= 0).all()
    assert ((t["surplus_value"] - t["loss_value"]).round(6)
            == t["net_value_variance"].round(6)).all()
    assert list(t["count_date"]) == sorted(t["count_date"])


def test_loss_over_time_fills_every_group_and_date():
    from analytics import compute_variance, loss_over_time
    df = compute_variance(_multi_period_frame())
    lot = loss_over_time(df, "location")
    assert len(lot) == df["count_date"].nunique() * df["location"].nunique()
    assert (lot["loss_value"] >= 0).all()
    total_loss = -df.loc[df["qty_variance"] < 0, "value_variance"].sum()
    assert round(lot["loss_value"].sum(), 6) == round(total_loss, 6)
