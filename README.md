# StockPulse Analytics

A Streamlit dashboard for inventory stock-take variance analysis, loss
prevention, and AI-written executive briefings. Upload a stock-take file
(CSV or Excel) with expected vs. counted quantities and get variance analysis,
shrinkage intelligence, multi-period trends, statistical anomaly detection,
and a manager-ready briefing drafted by Claude.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-E8A03D)

## Features

- **Variance analysis** — quantity, percentage, and value variance per line item
- **Shrinkage intelligence** — separates shortages (loss/theft/spoilage) from
  overages (receiving/counting errors) and puts a dollar figure on each
- **Multi-period trends** — accuracy and value-at-risk over time, plus a
  "recurring shortage products" table that flags items short across multiple
  counts (the strongest genuine-shrinkage signal)
- **Operations tab** — shrinkage benchmarked against 3%/5% retail thresholds with green/amber/red status by shop and category; a worsening-products list that flags losses growing across count dates; and a low-stock risk list of items sitting well below their usual level
- **Anomaly detection** — statistical outliers via the 1.5xIQR rule
- **AI executive briefing** — Claude reads the *computed statistics* (never raw
  rows) and drafts findings + prioritised recommendations, downloadable as
  Markdown, plus a free-text Q&A box
- **Interactive dashboard** — six tabs, filters by category/location/date/anomaly
- **Sample data** bundled so the app works out of the box

## Expected data format

| Column          | Type    | Description                          |
|-----------------|---------|--------------------------------------|
| `sku`           | text    | Stock keeping unit                   |
| `product_name`  | text    | Product name                         |
| `category`      | text    | Product category                     |
| `location`      | text    | Warehouse / location                 |
| `expected_qty`  | number  | Expected (system) quantity           |
| `counted_qty`   | number  | Physically counted quantity          |
| `unit_cost`     | number  | Cost per unit                        |
| `count_date`    | date    | Date of the stock take               |

A working example lives in `sample_data.csv`. Multi-period features light up
when the file contains more than one `count_date`.

## Run locally

```bash
git clone <your-repo-url>
cd stockpulse-analytics

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

## Enabling the AI briefing (optional, free)

The AI tab uses Google Gemini, which has a free tier. Every other tab works without it.

1. Get a free API key at https://aistudio.google.com/app/apikey
2. **Locally:** `export GEMINI_API_KEY="your-key"` (Windows: `set GEMINI_API_KEY=your-key`), then run the app.
3. **On Streamlit Cloud:** open the app, go to **Settings → Secrets**, and add:
   ```toml
   GEMINI_API_KEY = "your-key"
   ```
   Save; the app restarts with the AI feature enabled.

Free-tier note: Google rate-limits free usage and may use submitted prompts to improve their models, so avoid sending real confidential data through the free tier. The bundled sample data is synthetic, so it's safe.


## Tests

```bash
pytest
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to share.streamlit.io, sign in with GitHub.
3. **New app**, pick the repo, main file `app.py`, deploy.
4. (Optional) add `ANTHROPIC_API_KEY` under Settings > Secrets for the AI tab.

## Project structure

```
stockpulse-analytics/
├── app.py               # Streamlit UI (six tabs)
├── analytics.py         # Variance, shrinkage & trend logic (pure pandas)
├── ai_analyst.py        # Anthropic-powered briefing + Q&A
├── test_analytics.py    # Unit tests
├── sample_data.csv      # 150-row synthetic dataset
├── requirements.txt
├── .gitignore
├── .streamlit/config.toml
└── README.md
```

## How the numbers work

- **Quantity variance** = `counted_qty - expected_qty`
- **Percentage variance** = `qty_variance / expected_qty x 100` (0 when expected is 0)
- **Value variance** = `qty_variance x unit_cost`
- **Shrinkage** = summed value of rows where counted < expected (a loss)
- **Recurring offender** = a product short in 2+ distinct count dates
- **Anomaly** = value variance outside `Q1 - 1.5xIQR` .. `Q3 + 1.5xIQR`

## Grounding the AI

The AI layer is deliberately fed a compact JSON summary of the *computed*
statistics — totals, per-category and per-location aggregates, the trend
table, and the top-ten discrepancies — never the raw rows. This keeps the
model anchored to real figures and avoids invented SKUs or numbers.

## License

MIT
