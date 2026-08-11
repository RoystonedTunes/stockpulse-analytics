# StockPulse Analytics

A Streamlit dashboard for inventory stock-take variance analysis. Upload a
stock-take file (CSV or Excel) with expected vs. counted quantities and get
variance analysis, statistical anomaly detection, and interactive charts.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-E8A03D)

## Features

- **File upload** — CSV or Excel, with column validation
- **Variance analysis** — quantity, percentage, and value variance per line item
- **Anomaly detection** — statistical outliers via the 1.5×IQR rule on value variance
- **Interactive dashboard** — variance by category and location, accuracy donut,
  largest discrepancies
- **Filters** — by category, location, count date, and anomaly status
- **Export** — download the filtered, processed dataset as CSV
- **Sample data** — a 150-row synthetic dataset is bundled so the app works out of the box

## Expected data format

Your file needs these columns:

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

A working example lives in `sample_data.csv`.

## Run locally

```bash
git clone <your-repo-url>
cd stockpulse-analytics

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

The app opens at http://localhost:8501.

## Tests

```bash
pytest
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, pick this repo, set the main file to `app.py`, and deploy.

No secrets are required — the app runs entirely on uploaded/sample data.

## Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: StockPulse Analytics"
git branch -M main
git remote add origin <your-new-github-repo-url>
git push -u origin main
```

## Project structure

```
stockpulse-analytics/
├── app.py               # Streamlit UI
├── analytics.py         # Variance + anomaly logic (pure pandas)
├── test_analytics.py    # Unit tests
├── sample_data.csv      # 150-row synthetic dataset
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── config.toml      # Theme
└── README.md
```

## How the numbers work

- **Quantity variance** = `counted_qty − expected_qty`
- **Percentage variance** = `qty_variance / expected_qty × 100` (0 when expected is 0)
- **Value variance** = `qty_variance × unit_cost`
- **Anomaly** = value variance below `Q1 − 1.5×IQR` or above `Q3 + 1.5×IQR`

## License

MIT
