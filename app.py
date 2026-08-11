"""
StockPulse Analytics — inventory stock-take variance dashboard.

Upload a stock-take CSV/Excel with expected vs. counted quantities and get
variance analysis, statistical anomaly detection, and interactive charts.

Run locally:  streamlit run app.py
"""

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analytics import (
    REQUIRED_COLUMNS,
    compute_variance,
    detect_anomalies,
    summary_metrics,
    variance_by_dimension,
    top_discrepancies,
)

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="StockPulse Analytics",
    page_icon="\U0001F4E6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette: ink navy, signal amber, cool slate, off-white paper.
INK = "#12203A"
AMBER = "#E8A03D"
SLATE = "#5B7089"
PAPER = "#F6F4EF"
FLAG = "#C24B4B"
OK = "#3E8E7E"

st.markdown(
    f"""
    <style>
      .stApp {{ background: {PAPER}; }}
      h1, h2, h3 {{ color: {INK}; font-weight: 700; letter-spacing: -0.01em; }}
      .metric-card {{
        background: white; border: 1px solid #E4E0D8; border-radius: 10px;
        padding: 1rem 1.2rem; box-shadow: 0 1px 2px rgba(18,32,58,0.04);
      }}
      .metric-label {{ color: {SLATE}; font-size: 0.78rem; text-transform: uppercase;
        letter-spacing: 0.06em; margin-bottom: 0.25rem; }}
      .metric-value {{ color: {INK}; font-size: 1.6rem; font-weight: 700; }}
      .metric-sub {{ color: {SLATE}; font-size: 0.8rem; }}
      div[data-testid="stDataFrame"] {{ border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-sub">{sub}</div></div>'
    )


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
st.sidebar.title("StockPulse")
st.sidebar.caption("Inventory variance analytics")

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
    st.markdown(
        "**Expected columns:** "
        + ", ".join(f"`{c}`" for c in REQUIRED_COLUMNS)
    )
    st.stop()

# Validate
missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing:
    st.title("StockPulse Analytics")
    st.error(
        "The file is missing required columns: "
        + ", ".join(f"`{c}`" for c in missing)
        + f".\n\nFound: {', '.join(raw.columns)}"
    )
    st.stop()

df = compute_variance(raw)
df = detect_anomalies(df)

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

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Inventory Stock-Take Analytics")
st.caption(f"Source: **{source_label}**  ·  {len(view):,} of {len(df):,} rows in view")

m = summary_metrics(view)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(metric_card("Records", f"{m['records']:,}", "line items counted"), unsafe_allow_html=True)
c2.markdown(metric_card("Count accuracy", f"{m['accuracy']:.1f}%", "exact matches"), unsafe_allow_html=True)
c3.markdown(
    metric_card("Net value variance", f"${m['net_value']:,.0f}",
                "counted minus expected"),
    unsafe_allow_html=True,
)
c4.markdown(
    metric_card("Absolute value at risk", f"${m['abs_value']:,.0f}",
                f"{m['anomalies']} anomalies flagged"),
    unsafe_allow_html=True,
)

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_anomalies, tab_data = st.tabs(
    ["Overview", "Anomalies", "Data"]
)

with tab_overview:
    left, right = st.columns(2)

    with left:
        st.subheader("Value variance by category")
        by_cat = variance_by_dimension(view, "category")
        fig = px.bar(
            by_cat, x="category", y="value_variance",
            color="value_variance",
            color_continuous_scale=[[0, FLAG], [0.5, "#EDE8DF"], [1, OK]],
            labels={"value_variance": "Value variance ($)"},
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Count accuracy breakdown")
        exact = int((view["qty_variance"] == 0).sum())
        over = int((view["qty_variance"] > 0).sum())
        short = int((view["qty_variance"] < 0).sum())
        donut = go.Figure(
            go.Pie(
                labels=["Exact", "Overcount", "Shortage"],
                values=[exact, over, short],
                hole=0.62,
                marker=dict(colors=[OK, AMBER, FLAG]),
            )
        )
        donut.update_layout(
            paper_bgcolor="white", margin=dict(t=10, b=10, l=10, r=10),
            height=340, legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(donut, use_container_width=True)

    st.subheader("Value variance by location")
    by_loc = variance_by_dimension(view, "location")
    fig2 = px.bar(
        by_loc, x="location", y="value_variance",
        color="value_variance",
        color_continuous_scale=[[0, FLAG], [0.5, "#EDE8DF"], [1, OK]],
        labels={"value_variance": "Value variance ($)"},
    )
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
        height=320,
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab_anomalies:
    st.subheader("Statistical anomalies (IQR method)")
    st.caption(
        "Line items whose value variance falls outside 1.5 x IQR of the "
        "distribution are flagged as statistical outliers."
    )
    flagged = view[view["is_anomaly"]].copy()
    if flagged.empty:
        st.success("No anomalies in the current view.")
    else:
        st.markdown(
            metric_card(
                "Flagged items",
                f"{len(flagged):,}",
                f"${flagged['value_variance'].abs().sum():,.0f} absolute value impact",
            ),
            unsafe_allow_html=True,
        )
        st.write("")
        show = flagged[[
            "sku", "product_name", "category", "location", "count_date",
            "expected_qty", "counted_qty", "qty_variance",
            "pct_variance", "value_variance",
        ]].sort_values("value_variance", key=abs, ascending=False)
        st.dataframe(
            show.style.format({
                "pct_variance": "{:.1f}%",
                "value_variance": "${:,.2f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.subheader("Largest discrepancies by value")
    top = top_discrepancies(view, n=10)
    fig3 = px.bar(
        top, x="value_variance", y="product_name", orientation="h",
        color="value_variance",
        color_continuous_scale=[[0, FLAG], [0.5, "#EDE8DF"], [1, OK]],
        labels={"value_variance": "Value variance ($)", "product_name": ""},
        hover_data=["sku", "location", "count_date"],
    )
    fig3.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
        height=420, yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab_data:
    st.subheader("Processed records")
    st.caption("Full dataset with computed variance columns. Use the sidebar filters to narrow it.")
    display = view[[
        "sku", "product_name", "category", "location", "count_date",
        "expected_qty", "counted_qty", "qty_variance", "pct_variance",
        "unit_cost", "value_variance", "is_anomaly",
    ]]
    st.dataframe(
        display.style.format({
            "pct_variance": "{:.1f}%",
            "unit_cost": "${:,.2f}",
            "value_variance": "${:,.2f}",
        }),
        use_container_width=True, hide_index=True,
    )
    csv = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv,
        file_name="stockpulse_variance.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "StockPulse Analytics · variance = counted − expected · "
    "anomalies via 1.5×IQR on value variance."
)
