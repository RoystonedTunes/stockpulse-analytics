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
    compute_variance,
    detect_anomalies,
    has_multiple_periods,
    recurring_offenders,
    shrinkage_by_dimension,
    shrinkage_metrics,
    summary_metrics,
    top_discrepancies,
    variance_by_dimension,
)
import ai_analyst

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

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Inventory Stock-Take Analytics")
st.caption(f"Source: **{source_label}**  ·  {len(view):,} of {len(df):,} rows in view")

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
c2.markdown(metric_card("Shrinkage (losses)", f"${abs(s['shrinkage_value']):,.0f}",
                        f"{s['shrinkage_pct_of_stock']:.1f}% of stock value",
                        accent=LOSS), unsafe_allow_html=True)
c3.markdown(metric_card("Net value variance", f"${m['net_value']:,.0f}",
                        "counted minus expected", accent=net_color), unsafe_allow_html=True)
c4.markdown(metric_card("Value at risk", f"${m['abs_value']:,.0f}",
                        f"{m['anomalies']} anomalies flagged", accent=BRASS),
            unsafe_allow_html=True)

st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_overview, tab_trends, tab_shrink, tab_anoms, tab_ai, tab_data = st.tabs(
    ["Overview", "Trends", "Shrinkage", "Anomalies", "AI Briefing", "Data"]
)

# ---- Overview ------------------------------------------------------------- #
with tab_overview:
    left, right = st.columns(2)
    with left:
        st.subheader("Value variance by category")
        st.plotly_chart(
            bar(variance_by_dimension(view, "category"), "category",
                "value_variance", "Value variance ($)"),
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
            "value_variance", "Value variance ($)", height=320),
        use_container_width=True,
    )

# ---- Trends --------------------------------------------------------------- #
with tab_trends:
    if not has_multiple_periods(view):
        st.info("Trend analysis needs more than one count date in the current view. "
                "Widen the date filter in the sidebar to compare periods.")
    else:
        st.subheader("Accuracy over time")
        trend = accuracy_trend(view)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=trend["count_date"], y=trend["accuracy"], mode="lines+markers",
            line=dict(color=INK, width=3), marker=dict(size=9, color=BRASS),
            name="Accuracy %",
        ))
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          height=320, margin=dict(t=10, b=10, l=10, r=10),
                          yaxis=dict(title="Count accuracy %"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Value at risk over time")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=trend["count_date"], y=trend["abs_value_at_risk"],
                              marker_color=MUTE, name="Absolute value at risk"))
        fig2.add_trace(go.Scatter(x=trend["count_date"], y=trend["net_value_variance"],
                                  mode="lines+markers", line=dict(color=LOSS, width=3),
                                  name="Net value variance"))
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=320,
                           margin=dict(t=10, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Recurring shortage products")
        st.caption("Products short across two or more count dates — the strongest "
                   "signal of genuine shrinkage rather than a one-off miscount.")
        ro = recurring_offenders(view)
        if ro.empty:
            st.success("No products show recurring shortages in the current view.")
        else:
            show = ro.rename(columns={
                "product_name": "Product", "periods_short": "Periods short",
                "total_shrinkage_value": "Shrinkage value",
                "total_units_lost": "Units lost", "locations": "Location(s)",
            })
            st.dataframe(
                show.style.format({"Shrinkage value": "${:,.2f}"}),
                use_container_width=True, hide_index=True,
            )

# ---- Shrinkage ------------------------------------------------------------ #
with tab_shrink:
    st.subheader("Loss vs. surplus")
    st.caption("Shortages (counted below expected) point to loss, theft, or spoilage. "
               "Overages usually mean receiving or counting errors.")
    a, b, c = st.columns(3)
    a.markdown(metric_card("Total shortage value", f"${abs(s['shrinkage_value']):,.0f}",
                           f"{abs(s['shrinkage_units']):,} units short"), unsafe_allow_html=True)
    b.markdown(metric_card("Total overage value", f"${s['overage_value']:,.0f}",
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
                                "Shortage value ($)", height=300), use_container_width=True)
    with ccol:
        st.caption("Shortage value by category")
        sbc = shrinkage_by_dimension(view, "category")
        if sbc.empty:
            st.success("No shortages in view.")
        else:
            st.plotly_chart(bar(sbc, "category", "shrinkage_value",
                                "Shortage value ($)", height=300), use_container_width=True)

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
                                f"${flagged['value_variance'].abs().sum():,.0f} absolute value impact"),
                    unsafe_allow_html=True)
        st.write("")
        show = flagged[["sku", "product_name", "category", "location", "count_date",
                        "expected_qty", "counted_qty", "qty_variance",
                        "pct_variance", "value_variance"]].sort_values(
            "value_variance", key=abs, ascending=False)
        st.dataframe(show.style.format({"pct_variance": "{:.1f}%",
                                        "value_variance": "${:,.2f}"}),
                     use_container_width=True, hide_index=True)

    st.subheader("Largest discrepancies by value")
    top = top_discrepancies(view, n=10)
    fig3 = px.bar(top, x="value_variance", y="product_name", orientation="h",
                  color="value_variance", color_continuous_scale=SCALE,
                  labels={"value_variance": "Value variance ($)", "product_name": ""},
                  hover_data=["sku", "location", "count_date"])
    fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10),
                       height=420, yaxis=dict(autorange="reversed"),
                       font=dict(family="IBM Plex Mono, monospace", size=12, color=INK))
    st.plotly_chart(fig3, use_container_width=True)

# ---- AI Briefing ---------------------------------------------------------- #
with tab_ai:
    st.subheader("AI-written executive briefing")
    st.caption("Claude analyses the computed statistics (not raw rows) and drafts a "
               "manager-ready briefing with findings and recommendations.")

    api_key = ai_analyst.get_api_key()
    if not api_key:
        st.warning(
            "No Anthropic API key found. Add `ANTHROPIC_API_KEY` in your Streamlit "
            "app settings under **Secrets** (or as an environment variable locally) "
            "to enable AI features."
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
            with st.spinner("Claude is analysing the stock-take..."):
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
                                       "unit_cost": "${:,.2f}",
                                       "value_variance": "${:,.2f}"}),
                 use_container_width=True, hide_index=True)
    st.download_button("Download filtered data (CSV)",
                       data=display.to_csv(index=False).encode("utf-8"),
                       file_name="stockpulse_variance.csv", mime="text/csv")

st.divider()
st.caption("StockPulse Analytics v2 · variance = counted − expected · "
           "shrinkage = value of shortages · anomalies via 1.5×IQR.")
