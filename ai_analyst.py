"""
AI analyst layer for StockPulse (Google Gemini, free-tier friendly).

Turns the *computed statistics* (never the raw rows) into a written executive
briefing with findings and recommendations, and answers free-text questions.
Feeding summarised numbers rather than raw data keeps the model grounded and
avoids hallucinated line items.

Uses Google's Gemini API, which offers a free tier. Supply an API key via
Streamlit secrets (GEMINI_API_KEY) or the GEMINI_API_KEY environment variable.
Get a free key at https://aistudio.google.com/app/apikey
"""

from __future__ import annotations

import json
import os

import pandas as pd
import requests

# Gemini model + endpoint. If Google changes model names, update MODEL only.
MODEL = "gemini-1.5-flash"
ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)
TIMEOUT = 45


def get_api_key() -> str | None:
    """Look for the Gemini key in Streamlit secrets first, then the environment."""
    try:
        import streamlit as st

        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if name in st.secrets:
                return st.secrets[name]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


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
    "for a retail butcher's operations manager. You are given pre-computed "
    "statistics from a physical stock-take (never raw records). Base every claim "
    "strictly on the numbers provided; do not invent products, figures, or trends "
    "that are not in the data. Write in clear, confident business prose. Use "
    "concrete currency figures from the data. Distinguish shortages (possible "
    "loss/theft/spoilage) from overages (likely receiving or counting errors). "
    "Where a product shows shortages across multiple periods, treat that as a "
    "stronger shrinkage signal than a one-off. Be specific and actionable; avoid "
    "generic filler."
)

REPORT_INSTRUCTIONS = (
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


def _call_gemini(api_key: str, system: str, user: str, max_tokens: int) -> str:
    """Low-level Gemini REST call. Raises RuntimeError with a clear message."""
    url = ENDPOINT.format(model=MODEL)
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0.4},
    }
    try:
        resp = requests.post(
            url, params={"key": api_key}, json=payload, timeout=TIMEOUT
        )
    except requests.RequestException as e:
        raise RuntimeError(f"Network error contacting Gemini: {e}") from e

    if resp.status_code != 200:
        # Surface Google's error message so the user can act on it.
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("message", "")
        except Exception:
            detail = resp.text[:300]
        raise RuntimeError(
            f"Gemini API returned {resp.status_code}: {detail or 'unknown error'}"
        )

    data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError):
        # e.g. blocked content or an empty candidate.
        raise RuntimeError(
            "Gemini returned no usable text. It may have blocked the request or "
            "hit a token limit. Try again or shorten the question."
        )


def generate_report(stats: dict, api_key: str) -> str:
    """Return the Markdown briefing. Raises RuntimeError on API error."""
    user = (
        "Here are the computed stock-take statistics as JSON:\n\n"
        f"{json.dumps(stats, indent=2, default=str)}\n\n"
        f"{REPORT_INSTRUCTIONS}"
    )
    return _call_gemini(api_key, SYSTEM_PROMPT, user, max_tokens=1500)


def answer_question(stats: dict, question: str, api_key: str) -> str:
    """Answer a free-text question grounded only in the computed stats."""
    system = (
        SYSTEM_PROMPT
        + " Answer the user's question using only the provided statistics. If the "
        "answer is not derivable from the data, say so plainly rather than guessing."
    )
    user = (
        f"Statistics:\n{json.dumps(stats, indent=2, default=str)}\n\n"
        f"Question: {question}"
    )
    return _call_gemini(api_key, system, user, max_tokens=800)
