"""
AI analyst layer for StockPulse.

Turns the *computed statistics* (never the raw rows) into a written executive
briefing with findings and recommendations. Feeding summarised numbers rather
than raw data keeps the model grounded and avoids hallucinated line items.

Requires an Anthropic API key, supplied via Streamlit secrets or the
ANTHROPIC_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os

import pandas as pd

MODEL = "claude-sonnet-4-6"


def get_api_key() -> str | None:
    """Look for the key in Streamlit secrets first, then the environment."""
    try:
        import streamlit as st

        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def build_stats_payload(
    summary: dict,
    shrink: dict,
    by_category: pd.DataFrame,
    by_location: pd.DataFrame,
    trend: pd.DataFrame,
    offenders: pd.DataFrame,
    top: pd.DataFrame,
) -> dict:
    """Assemble a compact, JSON-serialisable summary of the analysis."""
    return {
        "overview": {
            "records": summary["records"],
            "count_accuracy_pct": round(summary["accuracy"], 1),
            "net_value_variance": round(summary["net_value"], 2),
            "absolute_value_at_risk": round(summary["abs_value"], 2),
            "anomalies_flagged": summary["anomalies"],
        },
        "shrinkage": {
            "total_shortage_value": round(shrink["shrinkage_value"], 2),
            "total_overage_value": round(shrink["overage_value"], 2),
            "shortage_rate_pct": round(shrink["shortage_rate"], 1),
            "shrinkage_pct_of_stock_value": round(shrink["shrinkage_pct_of_stock"], 2),
        },
        "value_variance_by_category": by_category.to_dict(orient="records"),
        "shrinkage_by_location": by_location.to_dict(orient="records"),
        "accuracy_trend": trend.to_dict(orient="records"),
        "recurring_shortage_products": offenders.head(10).to_dict(orient="records"),
        "largest_discrepancies": top[
            ["product_name", "location", "count_date", "value_variance"]
        ].to_dict(orient="records"),
    }


SYSTEM_PROMPT = (
    "You are a senior inventory and loss-prevention analyst writing a briefing "
    "for a retail operations manager. You are given pre-computed statistics from "
    "a physical stock-take (never raw records). Base every claim strictly on the "
    "numbers provided; do not invent SKUs, figures, or trends that are not in the "
    "data. Write in clear, confident business prose. Use concrete dollar figures "
    "from the data. Distinguish shortages (possible loss/theft) from overages "
    "(likely receiving or counting errors). Where a product shows shortages across "
    "multiple periods, treat that as a stronger shrinkage signal than a one-off. "
    "Be specific and actionable; avoid generic filler."
)

USER_TEMPLATE = (
    "Here are the computed stock-take statistics as JSON:\n\n{stats}\n\n"
    "Write a briefing in Markdown with exactly these sections:\n\n"
    "## Executive summary\n"
    "3-4 sentences on overall inventory health, accuracy, and the headline "
    "financial exposure.\n\n"
    "## Key findings\n"
    "4-6 bullet points, each citing a specific number from the data (a category, "
    "location, product, or trend).\n\n"
    "## Shrinkage & loss risk\n"
    "A short paragraph on where losses concentrate and which recurring-shortage "
    "products deserve investigation.\n\n"
    "## Recommendations\n"
    "4-6 prioritised, concrete actions the manager could take next week.\n\n"
    "Keep the whole briefing under 450 words."
)


def generate_report(stats: dict, api_key: str) -> str:
    """Call Claude and return the Markdown briefing. Raises on API error."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    stats=json.dumps(stats, indent=2, default=str)
                ),
            }
        ],
    )
    return "".join(
        block.text for block in message.content if block.type == "text"
    )


def answer_question(stats: dict, question: str, api_key: str) -> str:
    """Answer a free-text question grounded only in the computed stats."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=(
            SYSTEM_PROMPT
            + " Answer the user's question using only the provided statistics. "
            "If the answer is not derivable from the data, say so plainly rather "
            "than guessing."
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Statistics:\n{json.dumps(stats, indent=2, default=str)}\n\n"
                    f"Question: {question}"
                ),
            }
        ],
    )
    return "".join(
        block.text for block in message.content if block.type == "text"
    )
