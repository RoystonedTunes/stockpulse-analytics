"""
StockPulse Analytics v2 — inventory stock-take variance dashboard.

Adds multi-period trend analysis, shrinkage/loss intelligence, and an
AI-written executive briefing on top of the core variance dashboard.

Run locally:  streamlit run app.py
"""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    REQUIRED_COLUMNS,
    accuracy_trend,
    benchmark_by_dimension,
    compute_variance,
    detect_anomalies,
    has_multiple_periods,
    loss_over_time,
    movers_and_underperformers,
    period_comparison,
    powerbi_tables,
    recurring_offenders,
    shrinkage_by_dimension,
    shrinkage_metrics,
    shrinkage_trajectory,
    stockout_risk,
    store_scorecard,
    summary_metrics,
    tables_to_zip,
    top_discrepancies,
    variance_by_dimension,
)
import ai_analyst


# --------------------------------------------------------------------------- #
# Money formatting: every amount is shown in euro, with the minus sign first
# (for example -€120 rather than €-120) so it reads naturally.
# --------------------------------------------------------------------------- #
def euro(value: float, decimals: int = 0) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}€{abs(value):,.{decimals}f}"


def EURO0(value):  # whole euros, used in tables
    return euro(value, 0)


def EURO2(value):  # euros and cents, used in tables
    return euro(value, 2)


def date_label(dates: pd.Series) -> pd.Series:
    """Turn dates into short readable labels such as '1 Jul 2026'."""
    d = pd.to_datetime(dates)
    return d.dt.day.astype(str) + " " + d.dt.strftime("%b %Y")

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="StockPulse Analytics",
    page_icon="\U0001F4E6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- "The Ledger" design system -------------------------------------------- #
# The subject is reconciliation: what the books say vs. what's on the shelf.
# Colour encodes meaning — oxblood = loss/shortage, viridian = surplus/health.
INK = "#0F1A2E"       # deep ledger ink
PAPER = "#FBF9F4"     # warm paper
LOSS = "#A6362E"      # oxblood — shortages / money lost
GAIN = "#2F6E5E"      # viridian — overages / health
BRASS = "#C9922E"     # brass accent — highlights, neutral emphasis
RULE = "#E4DDCF"      # hairline ledger rule
MUTE = "#6B7688"      # muted slate for secondary text

# Diverging scale for value variance: loss (red) → neutral paper → gain (green)
SCALE = [[0, LOSS], [0.5, "#EFE9DD"], [1, GAIN]]

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

      .stApp {{ background: {PAPER}; }}

      /* Display headings — Fraunces, a characterful modern serif */
      h1, h2, h3 {{
        font-family: 'Fraunces', Georgia, serif !important;
        color: {INK} !important; font-weight: 600 !important;
        letter-spacing: -0.015em;
      }}
      h1 {{ font-size: 2.15rem !important; }}

      /* Body / captions — Inter */
      .stApp, p, label, .stMarkdown {{ font-family: 'Inter', system-ui, sans-serif; }}

      /* Every number lives in a monospace — like figures ruled into a ledger */
      .metric-value, .recon-nums, .kpi-fig {{
        font-family: 'IBM Plex Mono', ui-monospace, monospace;
        font-variant-numeric: tabular-nums;
      }}

      /* KPI card: a small reconciliation between expected and counted */
      .metric-card {{
        background: #FFFFFF; border: 1px solid {RULE};
        border-radius: 4px; padding: 1.05rem 1.2rem 1.15rem;
        box-shadow: 0 1px 0 {RULE}; height: 100%;
        position: relative; overflow: hidden;
      }}
      .metric-card::before {{
        content: ""; position: absolute; left: 0; top: 0; bottom: 0;
        width: 3px; background: var(--accent, {BRASS});
      }}
      .metric-label {{ color: {MUTE}; font-size: 0.7rem; text-transform: uppercase;
        letter-spacing: 0.09em; margin-bottom: 0.35rem; font-weight: 600; }}
      .metric-value {{ color: {INK}; font-size: 1.7rem; font-weight: 600;
        line-height: 1.1; letter-spacing: -0.02em; }}
      .metric-sub {{ color: {MUTE}; font-size: 0.78rem; margin-top: 0.15rem; }}

      /* Reconciliation bar — the signature element */
      .recon {{ margin-top: 0.6rem; }}
      .recon-track {{ height: 5px; background: {RULE}; border-radius: 3px;
        overflow: hidden; }}
      .recon-fill {{ height: 100%; border-radius: 3px; }}
      .recon-nums {{ display: flex; justify-content: space-between;
        font-size: 0.68rem; color: {MUTE}; margin-top: 0.25rem; }}

      /* Tabs — ruled, understated, brass underline on the active one */
      .stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid {RULE}; }}
      .stTabs [data-baseweb="tab"] {{ font-family: 'Inter'; font-weight: 500;
        color: {MUTE}; padding-bottom: 0.5rem; }}
      .stTabs [aria-selected="true"] {{ color: {INK} !important;
        border-bottom: 2px solid {BRASS} !important; }}

      /* Sidebar — a touch of ink */
      section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid {RULE}; }}

      /* Buttons */
      .stButton button, .stDownloadButton button {{
        font-family: 'Inter'; font-weight: 600; border-radius: 4px;
        border: 1px solid {INK}; background: {INK}; color: {PAPER};
      }}
      .stButton button:hover, .stDownloadButton button:hover {{
        background: {BRASS}; border-color: {BRASS}; color: {INK}; }}

      #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, sub: str = "", accent: str = BRASS,
                recon: tuple | None = None) -> str:
    """
    A KPI card. When `recon=(expected, counted)` is supplied, it draws the
    signature reconciliation bar showing how far counted sits from expected.
    """
    recon_html = ""
    if recon is not None:
        expected, counted = recon
        if expected > 0:
            ratio = max(0.0, min(counted / expected, 1.0))
            pct = ratio * 100
            fill_color = GAIN if counted >= expected else LOSS
            recon_html = (
                f'<div class="recon"><div class="recon-track">'
                f'<div class="recon-fill" style="width:{pct:.0f}%;background:{fill_color};"></div>'
                f'</div><div class="recon-nums"><span>counted {counted:,.0f}</span>'
                f'<span>expected {expected:,.0f}</span></div></div>'
            )
    return (
        f'<div class="metric-card" style="--accent:{accent};">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div>{recon_html}</div>'
    )


def bar(df, x, y, label, height=340, horizontal=False):
    if horizontal:
        fig = px.bar(df, x=y, y=x, orientation="h", color=y,
                     color_continuous_scale=SCALE, labels={y: label, x: ""})
        fig.update_layout(yaxis=dict(autorange="reversed"))
    else:
        fig = px.bar(df, x=x, y=y, color=y,
                     color_continuous_scale=SCALE, labels={y: label})
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
        height=height,
        font=dict(family="IBM Plex Mono, monospace", size=12, color=INK),
    )
    fig.update_xaxes(gridcolor=RULE, zerolinecolor=RULE)
    fig.update_yaxes(gridcolor=RULE, zerolinecolor=RULE)
    return fig


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_file(file_bytes: bytes, name: str) -> pd.DataFrame:
    buffer = io.BytesIO(file_bytes)
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(buffer)
    return pd.read_csv(buffer)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    return pd.read_csv("sample_data.csv")


# --------------------------------------------------------------------------- #
# Sidebar — input
# --------------------------------------------------------------------------- #
st.sidebar.markdown(
    f"<div style='font-family:Fraunces,serif;font-size:1.5rem;font-weight:600;"
    f"color:{INK};line-height:1;'>StockPulse</div>"
    f"<div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;"
    f"letter-spacing:0.12em;color:{MUTE};margin-top:0.3rem;text-transform:uppercase;'>"
    f"book vs. shelf, reconciled</div>",
    unsafe_allow_html=True,
)
st.sidebar.write("")

uploaded = st.sidebar.file_uploader(
    "Upload stock-take file", type=["csv", "xlsx", "xls"]
)
use_sample = st.sidebar.toggle("Use sample dataset", value=uploaded is None)

if uploaded is not None:
    raw = load_file(uploaded.getvalue(), uploaded.name)
    source_label = uploaded.name
elif use_sample:
    raw = load_sample()
    source_label = "sample_data.csv"
else:
    st.title("StockPulse Analytics")
    st.info("Upload a stock-take file in the sidebar, or switch on the sample dataset to explore.")
    st.markdown("**Expected columns:** " + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS))
    st.stop()

missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing:
    st.title("StockPulse Analytics")
    st.error(
        "The file is missing required columns: "
        + ", ".join(f"`{c}`" for c in missing)
        + f".\n\nFound: {', '.join(raw.columns)}"
    )
    st.stop()

df = detect_anomalies(compute_variance(raw))

# --------------------------------------------------------------------------- #
# Sidebar — filters
# --------------------------------------------------------------------------- #
st.sidebar.divider()
st.sidebar.subheader("Filters")

cats = sorted(df["category"].dropna().unique())
locs = sorted(df["location"].dropna().unique())
dates = sorted(df["count_date"].dropna().unique())

sel_cats = st.sidebar.multiselect("Category", cats, default=cats)
sel_locs = st.sidebar.multiselect("Location", locs, default=locs)
sel_dates = st.sidebar.multiselect("Count date", dates, default=dates)
only_flagged = st.sidebar.checkbox("Show anomalies only", value=False)

mask = (
    df["category"].isin(sel_cats)
    & df["location"].isin(sel_locs)
    & df["count_date"].isin(sel_dates)
)
if only_flagged:
    mask &= df["is_anomaly"]
view = df[mask].copy()

# If the filters leave nothing to show, say so clearly instead of showing errors.
if view.empty:
    st.title("Inventory Stock-Take Analytics")
    st.warning("No rows match the current filters. Select at least one category, "
               "location and count date in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Inventory Stock-Take Analytics")
st.caption(f"Source: **{source_label}**  ·  {len(view):,} of {len(df):,} rows in view")

with st.expander("New here? How to read this dashboard"):
    st.markdown(
        """
**What this shows:** at every stock take we compare what the system says
should be on the shelf (**expected**) with what was actually counted (**counted**).

- **Variance** = counted minus expected. A minus number means stock is missing.
- **Shrinkage / loss** = the money value of missing stock (units short x unit cost).
- **Surplus** = more stock found than expected, usually a delivery or counting mistake.
- **Count accuracy** = the share of products where the count matched exactly.
- **Traffic lights:** 🟢 under 3% of stock value lost, 🟡 3 to 5%, 🔴 over 5%
  (fresh-meat retail typically runs at 2 to 5%).

**Where to start:** use the filters on the left to pick shops, categories and
dates. Open **Store Scorecard** to see how each shop is doing, then
**Movers & Underperformers** to see what changed since the last stock take.

*The sample dataset is made-up demo data modelled on a butcher's shop, not real figures.*
        """
    )

m = summary_metrics(view)
s = shrinkage_metrics(view)
tot_expected = float(view["expected_qty"].sum())
tot_counted = float(view["counted_qty"].sum())
net_color = GAIN if m["net_value"] >= 0 else LOSS

c1, c2, c3, c4 = st.columns(4)
c1.markdown(
    metric_card("Count accuracy", f"{m['accuracy']:.1f}%", "exact matches",
                accent=BRASS, recon=(tot_expected, tot_counted)),
    unsafe_allow_html=True,
)
c2.markdown(metric_card("Shrinkage (losses)", f"€{abs(s['shrinkage_value']):,.0f}",
                        f"{s['shrinkage_pct_of_stock']:.1f}% of stock value",
                        accent=LOSS), unsafe_allow_html=True)
c3.markdown(metric_card("Net value variance", euro(m['net_value']),
                        "counted minus expected", accent=net_color), unsafe_allow_html=True)
c4.markdown(metric_card("Value at risk", f"€{m['abs_value']:,.0f}",
                        f"{m['anomalies']} anomalies flagged", accent=BRASS),
            unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
(tab_overview, tab_score, tab_movers, tab_trends, tab_shrink, tab_ops,
 tab_anoms, tab_ai, tab_data) = st.tabs(
    ["Overview", "Store Scorecard", "Movers & Underperformers", "Trends",
     "Shrinkage", "Operations", "Anomalies", "AI Briefing", "Data"]
)

STATUS_LABEL = {"green": "🟢 On target", "amber": "🟡 Watch", "red": "🔴 Investigate"}

# ---- Store Scorecard ------------------------------------------------------ #
with tab_score:
    st.subheader("Store scorecard")
    st.caption("One line per shop with the key numbers side by side, worst first. "
               "Loss % is the money lost on missing stock as a share of the shop's "
               "expected stock value.")
    score = store_scorecard(view)
    if score.empty:
        st.info("No data in the current view. Widen the filters on the left.")
    else:
        red_shops = int((score["status"] == "red").sum())
        amber_shops = int((score["status"] == "amber").sum())
        green_shops = int((score["status"] == "green").sum())
        s1, s2, s3 = st.columns(3)
        s1.markdown(metric_card("Investigate", f"{red_shops}", "shops over 5% loss",
                                accent=LOSS), unsafe_allow_html=True)
        s2.markdown(metric_card("Watch", f"{amber_shops}", "shops at 3 to 5% loss",
                                accent=BRASS), unsafe_allow_html=True)
        s3.markdown(metric_card("On target", f"{green_shops}", "shops under 3% loss",
                                accent=GAIN), unsafe_allow_html=True)
        st.write("")

        table = score.copy()
        table["status"] = table["status"].map(STATUS_LABEL)
        table = table.rename(columns={
            "location": "Shop", "items_counted": "Items counted",
            "count_accuracy_pct": "Count accuracy", "loss_value": "Loss",
            "loss_pct_of_stock": "Loss % of stock", "surplus_value": "Surplus",
            "unusual_items": "Unusual items", "status": "Status",
        })
        st.dataframe(
            table.style.format({
                "Count accuracy": "{:.0f}%", "Loss": EURO0,
                "Loss % of stock": "{:.1f}%", "Surplus": EURO0,
            }),
            use_container_width=True, hide_index=True,
        )
        worst = score.iloc[0]
        st.markdown(
            f"**In plain words:** {worst['location']} has the highest loss at "
            f"{worst['loss_pct_of_stock']:.1f}% of stock value "
            f"({euro(worst['loss_value'])}). "
            f"{green_shops} of {len(score)} shops are on target."
        )

# ---- Movers & Underperformers --------------------------------------------- #
with tab_movers:
    st.subheader("Movers & underperformers")
    st.caption("Compares two stock takes product by product. **Movers** are products "
               "whose losses went down the most; **underperformers** are products "
               "whose losses went up the most. Change = now minus before.")
    period_dates = sorted(view["count_date"].dropna().unique())
    if len(period_dates) < 2:
        st.info("This needs at least two stock-take dates in the current view. "
                "Widen the date filter on the left.")
    else:
        pick1, pick2 = st.columns(2)
        current_date = pick1.selectbox("This stock take", period_dates[::-1], index=0)
        earlier = [d for d in period_dates if d < current_date]
        if not earlier:
            st.info("Pick a later stock take so there is an earlier one to compare with.")
        else:
            previous_date = pick2.selectbox("Compare with", earlier[::-1], index=0)
            comparison = period_comparison(view, current_date, previous_date)
            movers, underperformers = movers_and_underperformers(comparison, n=10)

            better = int((comparison["change"] > 0).sum())
            worse = int((comparison["change"] < 0).sum())
            net_change = float(comparison["change"].sum())
            k1, k2, k3 = st.columns(3)
            k1.markdown(metric_card("Improved", f"{better}", "products with smaller losses",
                                    accent=GAIN), unsafe_allow_html=True)
            k2.markdown(metric_card("Got worse", f"{worse}", "products with bigger losses",
                                    accent=LOSS), unsafe_allow_html=True)
            k3.markdown(metric_card("Overall change", euro(net_change),
                                    "plus = better, minus = worse",
                                    accent=GAIN if net_change >= 0 else LOSS),
                        unsafe_allow_html=True)
            st.write("")

            def show_changes(frame, empty_message):
                if frame.empty:
                    st.success(empty_message)
                    return
                disp = frame.rename(columns={
                    "product_name": "Product", "category": "Category",
                    "before": f"Before ({previous_date})", "now": f"Now ({current_date})",
                    "change": "Change"})
                money_cols = [f"Before ({previous_date})", f"Now ({current_date})", "Change"]
                st.dataframe(disp.style.format({c: EURO0 for c in money_cols}),
                             use_container_width=True, hide_index=True)

            left, right = st.columns(2)
            with left:
                st.markdown("**Top 10 underperformers** (losses went up)")
                show_changes(underperformers, "No product got worse. Good news.")
            with right:
                st.markdown("**Top 10 movers** (losses went down)")
                show_changes(movers, "No product improved between these dates.")

            if not underperformers.empty:
                top = underperformers.iloc[0]
                st.markdown(
                    f"**In plain words:** the biggest problem since {previous_date} is "
                    f"**{top['product_name']}**, which is {euro(abs(top['change']))} worse. "
                    f"That is the first product to check at the next stock take."
                )

# ---- Overview ------------------------------------------------------------- #
with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Value variance by category")
        st.plotly_chart(
            bar(variance_by_dimension(view, "category"), "category",
                "value_variance", "Value variance (€)"),
            use_container_width=True,
        )
    with right:
        st.subheader("Count accuracy breakdown")
        exact = int((view["qty_variance"] == 0).sum())
        over = int((view["qty_variance"] > 0).sum())
        short = int((view["qty_variance"] < 0).sum())
        donut = go.Figure(go.Pie(
            labels=["Exact", "Overcount", "Shortage"],
            values=[exact, over, short], hole=0.62,
            marker=dict(colors=[GAIN, BRASS, LOSS]),
        ))
        donut.update_layout(paper_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10),
                            height=340, legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(donut, use_container_width=True)

    st.subheader("Value variance by location")
    st.plotly_chart(
        bar(variance_by_dimension(view, "location"), "location",
            "value_variance", "Value variance (€)", height=320),
        use_container_width=True,
    )

# ---- Trends --------------------------------------------------------------- #
with tab_trends:
    if not has_multiple_periods(view):
        st.info("Trend analysis needs more than one count date in the current view. "
                "Widen the date filter in the sidebar to compare periods.")
    else:
        st.caption("How things change from one stock take to the next. "
                   "Each point on the charts is one stock take.")
        trend = accuracy_trend(view)
        # Show dates as readable labels (e.g. "1 Jul 2026") in time order.
        trend["label"] = date_label(trend["count_date"])
        latest, previous = trend.iloc[-1], trend.iloc[-2]

        # --- Latest vs previous stock take, in three cards ---
        acc_change = latest["accuracy"] - previous["accuracy"]
        loss_change = latest["loss_value"] - previous["loss_value"]
        found_change = latest["surplus_value"] - previous["surplus_value"]
        t1, t2, t3 = st.columns(3)
        t1.markdown(metric_card(
            f"Count accuracy ({latest['label']})", f"{latest['accuracy']:.1f}%",
            f"{acc_change:+.1f} points vs {previous['label']}",
            accent=GAIN if acc_change >= 0 else LOSS), unsafe_allow_html=True)
        t2.markdown(metric_card(
            f"Money lost ({latest['label']})", euro(latest["loss_value"]),
            f"{'up' if loss_change > 0 else 'down'} {euro(abs(loss_change))} "
            f"vs {previous['label']}",
            accent=LOSS if loss_change > 0 else GAIN), unsafe_allow_html=True)
        t3.markdown(metric_card(
            f"Surplus found ({latest['label']})", euro(latest["surplus_value"]),
            f"{'up' if found_change > 0 else 'down'} {euro(abs(found_change))} "
            f"vs {previous['label']}", accent=BRASS), unsafe_allow_html=True)
        st.write("")

        # --- Money lost vs surplus found at each stock take ---
        st.subheader("Money lost vs surplus found")
        st.caption("**Lost** = value of stock missing at the count. **Found** = value "
                   "of extra stock found (usually a delivery or counting mistake). "
                   "The gap between them is the overall result.")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=trend["label"], y=trend["loss_value"], name="Lost",
            marker_color=LOSS, text=[euro(v) for v in trend["loss_value"]],
            textposition="outside"))
        fig.add_trace(go.Bar(
            x=trend["label"], y=trend["surplus_value"], name="Found",
            marker_color=GAIN, text=[euro(v) for v in trend["surplus_value"]],
            textposition="outside"))
        fig.update_layout(
            barmode="group", plot_bgcolor="white", paper_bgcolor="white", height=340,
            margin=dict(t=30, b=10, l=10, r=10), legend=dict(orientation="h", y=1.12),
            yaxis=dict(title="€", rangemode="tozero"),
            xaxis=dict(type="category", title=None))
        st.plotly_chart(fig, use_container_width=True)

        # --- Accuracy on a fixed 0 to 100% scale so small moves look small ---
        st.subheader("Count accuracy at each stock take")
        st.caption("The share of products where the count matched the system exactly. "
                   "Higher is better. The scale runs from 0 to 100% so small changes "
                   "are not exaggerated.")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=trend["label"], y=trend["accuracy"], mode="lines+markers+text",
            line=dict(color=INK, width=3), marker=dict(size=10, color=BRASS),
            text=[f"{v:.1f}%" for v in trend["accuracy"]], textposition="top center",
            name="Accuracy"))
        fig2.update_layout(
            plot_bgcolor="white", paper_bgcolor="white", height=300,
            margin=dict(t=10, b=10, l=10, r=10), showlegend=False,
            yaxis=dict(title="Accuracy %", range=[0, 105]),
            xaxis=dict(type="category", title=None))
        st.plotly_chart(fig2, use_container_width=True)

        # --- Where the losses come from over time ---
        st.subheader("Where the losses come from")
        split = st.radio("Break down by", ["Category", "Shop"], horizontal=True,
                         key="trend_split")
        dim = "category" if split == "Category" else "location"
        lot = loss_over_time(view, dim)
        lot["label"] = date_label(lot["count_date"])
        fig3 = px.line(lot, x="label", y="loss_value", color=dim, markers=True,
                       labels={"label": "", "loss_value": "Money lost (€)",
                               dim: split})
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360,
                           margin=dict(t=10, b=10, l=10, r=10),
                           xaxis=dict(type="category"), yaxis=dict(rangemode="tozero"))
        st.plotly_chart(fig3, use_container_width=True)

        # Plain-words summary of the biggest increase between the last two counts.
        last_two = lot[lot["count_date"].isin([previous["count_date"], latest["count_date"]])]
        pivot = last_two.pivot(index=dim, columns="count_date", values="loss_value")
        pivot["change"] = pivot[latest["count_date"]] - pivot[previous["count_date"]]
        worst = pivot["change"].idxmax()
        if pivot.loc[worst, "change"] > 0:
            st.markdown(
                f"**In plain words:** between {previous['label']} and {latest['label']}, "
                f"losses rose most in **{worst}** (up {euro(pivot.loc[worst, 'change'])}). "
                f"Overall, money lost went {'up' if loss_change > 0 else 'down'} by "
                f"{euro(abs(loss_change))}."
            )

        # --- Products that keep coming up short ---
        st.subheader("Products that keep coming up short")
        st.caption("Products missing stock at two or more stock takes. A product that "
                   "is short again and again is a stronger warning sign than a one-off "
                   "counting mistake.")
        ro = recurring_offenders(view)
        if ro.empty:
            st.success("No products show repeated shortages in the current view.")
        else:
            n_counts = view["count_date"].nunique()
            show = pd.DataFrame({
                "Product": ro["product_name"],
                "Short at": ro["periods_short"].map(lambda p: f"{p} of {n_counts} stock takes"),
                "Money lost": -ro["total_shrinkage_value"],
                "Units missing": -ro["total_units_lost"],
                "Shops": ro["locations"],
            })
            st.dataframe(
                show.style.format({"Money lost": EURO2, "Units missing": "{:,.0f}"}),
                use_container_width=True, hide_index=True,
            )

# ---- Shrinkage ------------------------------------------------------------ #
with tab_shrink:
    st.subheader("Loss vs. surplus")
    st.caption("Shortages (counted below expected) point to loss, theft, or spoilage. "
               "Overages usually mean receiving or counting errors.")
    a, b, c = st.columns(3)
    a.markdown(metric_card("Total shortage value", f"€{abs(s['shrinkage_value']):,.0f}",
                           f"{abs(s['shrinkage_units']):,} units short"), unsafe_allow_html=True)
    b.markdown(metric_card("Total overage value", f"€{s['overage_value']:,.0f}",
                           f"{s['overage_units']:,} units over"), unsafe_allow_html=True)
    c.markdown(metric_card("Shortage rate", f"{s['shortage_rate']:.0f}%",
                           "of line items short"), unsafe_allow_html=True)

    st.write("")
    st.subheader("Where losses concentrate")
    lcol, ccol = st.columns(2)
    with lcol:
        st.caption("Shortage value by location")
        sbl = shrinkage_by_dimension(view, "location")
        if sbl.empty:
            st.success("No shortages in view.")
        else:
            st.plotly_chart(bar(sbl, "location", "shrinkage_value",
                                "Shortage value (€)", height=300), use_container_width=True)
    with ccol:
        st.caption("Shortage value by category")
        sbc = shrinkage_by_dimension(view, "category")
        if sbc.empty:
            st.success("No shortages in view.")
        else:
            st.plotly_chart(bar(sbc, "category", "shrinkage_value",
                                "Shortage value (€)", height=300), use_container_width=True)

# ---- Operations ----------------------------------------------------------- #
with tab_ops:
    st.subheader("Shrinkage benchmark")
    st.caption("Fresh-meat retail typically runs 2-5% shrink. Shops and categories "
               "are graded green (under 3%), amber (3-5%), or red (over 5%).")

    bench_amber, bench_red = 3.0, 5.0
    status_color = {"green": GAIN, "amber": BRASS, "red": LOSS}
    status_word = {"green": "On target", "amber": "Watch", "red": "Investigate"}

    def status_row(df_bench, dim_label):
        for _, r in df_bench.iterrows():
            col = status_color[r["status"]]
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:0.8rem;"
                f"padding:0.5rem 0;border-bottom:1px solid {RULE};'>"
                f"<span style='width:9px;height:9px;border-radius:50%;"
                f"background:{col};display:inline-block;flex:none;'></span>"
                f"<span style='flex:1;font-weight:500;'>{r[dim_label]}</span>"
                f"<span style='font-family:IBM Plex Mono,monospace;color:{col};"
                f"font-weight:600;'>{r['shrinkage_pct']:.1f}%</span>"
                f"<span style='font-family:IBM Plex Mono,monospace;color:{MUTE};"
                f"font-size:0.8rem;width:110px;text-align:right;'>"
                f"€{abs(r['shrinkage_value']):,.0f} lost</span>"
                f"<span style='color:{col};font-size:0.78rem;width:90px;"
                f"text-align:right;font-weight:600;'>{status_word[r['status']]}</span>"
                f"</div>", unsafe_allow_html=True)

    bcol, ccol = st.columns(2)
    with bcol:
        st.markdown("**By shop**")
        status_row(benchmark_by_dimension(view, "location", bench_amber, bench_red), "location")
    with ccol:
        st.markdown("**By category**")
        status_row(benchmark_by_dimension(view, "category", bench_amber, bench_red), "category")

    st.divider()
    st.subheader("Worsening products")
    st.caption("Products whose losses are growing across the count dates — live, "
               "accelerating problems, ranked by how much worse they've become.")
    if not has_multiple_periods(view):
        st.info("Needs more than one count date. Widen the date filter to see trajectories.")
    else:
        traj = shrinkage_trajectory(view)
        if traj.empty:
            st.success("No shrinkage trajectories to report in the current view.")
        else:
            worsening = traj[traj["trend"] == "worsening"]
            improving = traj[traj["trend"] == "improving"]
            k1, k2, k3 = st.columns(3)
            k1.markdown(metric_card("Worsening", f"{len(worsening)}",
                                    "getting worse", accent=LOSS), unsafe_allow_html=True)
            k2.markdown(metric_card("Improving", f"{len(improving)}",
                                    "getting better", accent=GAIN), unsafe_allow_html=True)
            k3.markdown(metric_card("Stable", f"{int((traj['trend']=='stable').sum())}",
                                    "little change", accent=BRASS), unsafe_allow_html=True)
            st.write("")
            if not worsening.empty:
                show = worsening.head(12).copy()
                show["path"] = show["series"].apply(
                    lambda v: " → ".join(euro(abs(x)) for x in v))
                disp = show[["product_name", "path", "change"]].rename(columns={
                    "product_name": "Product", "path": "Loss by period",
                    "change": "Worsened by"})
                st.dataframe(disp.style.format({"Worsened by": EURO0}),
                             use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Low-stock risk")
    st.caption("Items whose most recent count sits well below their usual level — "
               "candidates to reorder before they run out.")
    risk, latest_date = stockout_risk(view)
    st.markdown(f"<span style='color:{MUTE};font-size:0.82rem;'>"
                f"Based on the latest count: {latest_date}</span>", unsafe_allow_html=True)
    st.write("")
    if risk.empty:
        st.success("No items are critically low in the latest count.")
    else:
        disp = risk.rename(columns={
            "product_name": "Product", "category": "Category", "location": "Shop",
            "counted_qty": "In stock", "typical_qty": "Usual level",
            "pct_of_typical": "% of usual", "unit_cost": "Unit cost"})
        st.dataframe(disp.style.format({"% of usual": "{:.0f}%", "Unit cost": EURO2,
                                        "Usual level": "{:.0f}"}),
                     use_container_width=True, hide_index=True)
    st.caption("Note: a stock-take shows levels, not sales speed. This flags items "
               "low relative to their normal stock, not a full demand forecast.")

# ---- Anomalies ------------------------------------------------------------ #
with tab_anoms:
    st.subheader("Statistical anomalies (IQR method)")
    st.caption("Line items whose value variance falls outside 1.5 x IQR of the "
               "distribution are flagged as statistical outliers.")
    flagged = view[view["is_anomaly"]].copy()
    if flagged.empty:
        st.success("No anomalies in the current view.")
    else:
        st.markdown(metric_card("Flagged items", f"{len(flagged):,}",
                                f"€{flagged['value_variance'].abs().sum():,.0f} absolute value impact"),
                    unsafe_allow_html=True)
        st.write("")
        show = flagged[["sku", "product_name", "category", "location", "count_date",
                        "expected_qty", "counted_qty", "qty_variance",
                        "pct_variance", "value_variance"]].sort_values(
            "value_variance", key=abs, ascending=False)
        st.dataframe(show.style.format({"pct_variance": "{:.1f}%",
                                        "value_variance": EURO2}),
                     use_container_width=True, hide_index=True)

    st.subheader("Largest discrepancies by value")
    top = top_discrepancies(view, n=10)
    fig3 = px.bar(top, x="value_variance", y="product_name", orientation="h",
                  color="value_variance", color_continuous_scale=SCALE,
                  labels={"value_variance": "Value variance (€)", "product_name": ""},
                  hover_data=["sku", "location", "count_date"])
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
                       height=420, yaxis=dict(autorange="reversed"),
                       font=dict(family="IBM Plex Mono, monospace", size=12, color=INK))
    st.plotly_chart(fig3, use_container_width=True)

# ---- AI Briefing ---------------------------------------------------------- #
with tab_ai:
    st.subheader("AI-written executive briefing")
    st.caption("An AI model reads the computed statistics (not raw rows) and drafts "
               "a manager-ready briefing with findings and recommendations. "
               "Powered by Google Gemini, which has a free tier.")

    api_key = ai_analyst.get_api_key()
    if not api_key:
        st.info(
            "**AI briefing not configured.** This optional feature uses Google "
            "Gemini's free tier. To enable it:\n\n"
            "1. Get a free key at https://aistudio.google.com/app/apikey\n"
            "2. In your Streamlit app, open **Settings → Secrets**\n"
            "3. Add this line, keeping the quotes:\n"
            "   `GEMINI_API_KEY = \"your-key-here\"`\n\n"
            "Every other tab works fully without this."
        )
    else:
        stats = ai_analyst.build_stats_payload(
            summary=m, shrink=s,
            by_category=variance_by_dimension(view, "category"),
            by_location=shrinkage_by_dimension(view, "location"),
            trend=accuracy_trend(view) if has_multiple_periods(view) else pd.DataFrame(),
            offenders=recurring_offenders(view),
            top=top_discrepancies(view, n=10),
        )

        if st.button("Generate briefing", type="primary"):
            with st.spinner("Generating the briefing..."):
                try:
                    report = ai_analyst.generate_report(stats, api_key)
                    st.session_state["ai_report"] = report
                except Exception as e:
                    st.error(f"Could not generate the briefing: {e}")

        if "ai_report" in st.session_state:
            st.markdown(st.session_state["ai_report"])
            st.download_button("Download briefing (Markdown)",
                               data=st.session_state["ai_report"],
                               file_name="stockpulse_briefing.md", mime="text/markdown")

        st.divider()
        st.markdown("**Ask a question about this stock-take**")
        q = st.text_input("e.g. Which location has the worst shrinkage and why?")
        if q:
            with st.spinner("Thinking..."):
                try:
                    st.markdown(ai_analyst.answer_question(stats, q, api_key))
                except Exception as e:
                    st.error(f"Could not answer: {e}")

# ---- Data ----------------------------------------------------------------- #
with tab_data:
    st.subheader("Processed records")
    st.caption("Full dataset with computed variance columns. Use the sidebar filters to narrow it.")
    display = view[["sku", "product_name", "category", "location", "count_date",
                    "expected_qty", "counted_qty", "qty_variance", "pct_variance",
                    "unit_cost", "value_variance", "is_anomaly"]]
    st.dataframe(display.style.format({"pct_variance": "{:.1f}%",
                                       "unit_cost": EURO2,
                                       "value_variance": EURO2}),
                 use_container_width=True, hide_index=True)
    st.download_button("Download filtered data (CSV)",
                       data=display.to_csv(index=False).encode("utf-8"),
                       file_name="stockpulse_variance.csv", mime="text/csv")

    st.divider()
    st.subheader("Export for Power BI")
    st.caption("Downloads the filtered data as four linked tables (a star schema) "
               "in one ZIP file, ready to load into Power BI.")
    if view.empty:
        st.info("No data in the current view to export.")
    else:
        pbi = powerbi_tables(view)
        st.download_button(
            "Download for Power BI (ZIP of 4 CSV files)",
            data=tables_to_zip(pbi),
            file_name="stockpulse_powerbi.zip", mime="application/zip",
        )
        with st.expander("What is inside, and how to load it into Power BI"):
            st.markdown(
                f"""
| File | What it holds | Rows |
| --- | --- | --- |
| `fact_stocktake.csv` | The numbers: expected, counted, variance, value | {len(pbi['fact_stocktake']):,} |
| `dim_product.csv` | One row per product, with its category | {len(pbi['dim_product']):,} |
| `dim_shop.csv` | One row per shop | {len(pbi['dim_shop']):,} |
| `dim_date.csv` | One row per stock-take date, with year, month and week | {len(pbi['dim_date']):,} |

**To load it:**
1. Unzip the file.
2. In Power BI Desktop choose **Get data → Text/CSV** and load all four files.
3. In **Model view**, link each `_key` column in `fact_stocktake` to the same
   column in its table: `product_key` → `dim_product`, `shop_key` → `dim_shop`,
   `date_key` → `dim_date` (one-to-many).
4. Build measures on the fact table, for example
   `Loss = -CALCULATE(SUM(fact_stocktake[value_variance]), fact_stocktake[qty_variance] < 0)`.
                """
            )

st.divider()
st.caption("StockPulse Analytics · variance = counted − expected · "
           "shrinkage = value of shortages · benchmarks graded 3% / 5% · "
           "anomalies via 1.5×IQR · all amounts in euro (€) · sample data is illustrative.")
