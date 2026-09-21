# Dengue prediction for Lambayeque (Peru)

Research pipeline to estimate **monthly dengue cases by district** in Lambayeque and present results in a local decision-support dashboard.

## Requirements

- Python 3.10+ (3.12 recommended)
- Internet access for Open-Meteo climate download (step 2)
- Input files under `data/`:
  - `cdc_dengue_surveillance_2000_2024.csv` — CDC/MINSA open dengue surveillance
  - `icen_coastal_el_nino.txt` — coastal El Niño index / ICEN (ENFEN)
  - Optional SENAMHI station series (for experiments only):
    - `senamhi_lambayeque_daily.txt` (ex-code qc00000301)
    - `senamhi_reque_daily.txt` (ex-code qc00000332)
    - `senamhi_jayanca_daily.txt` (ex-code qc00000333)

## Setup

```powershell
cd D:\repos\dengue-lambayeque
py -3.12 -m pip install -r requirements.txt
```

## Run the full pipeline

Execute the scripts **in order**:

```powershell
# 1) Build monthly district panel (missing months filled with 0)
py -3.12 01_prepare_data.py

# 2) Download Open-Meteo climate by node and merge with ICEN
py -3.12 02_fetch_climate.py

# 3) Train district model, evaluate 2024, export dashboard JSON + plots
py -3.12 03_train_district_model.py

# 4) Open the local dashboard
py -3.12 04_serve_dashboard.py
```

Dashboard URL: [http://127.0.0.1:8080](http://127.0.0.1:8080)

## Pipeline overview

| Step | Script | Output |
|------|--------|--------|
| 1 | `01_prepare_data.py` | `data/lambayeque_districts_monthly.csv` |
| 2 | `02_fetch_climate.py` | `data/lambayeque_districts_climate.csv` |
| 3 | `03_train_district_model.py` | `models/`, `web/data/dashboard.json` |
| 4 | `04_serve_dashboard.py` | Local web server |

Supporting module: `district_coords.py` (district centroids and climate-node mapping).

## Data sources

| Data | Source | File |
|------|--------|------|
| Dengue cases | CDC/MINSA open surveillance | `cdc_dengue_surveillance_2000_2024.csv` |
| Climate (precip, tmax, tmin, humidity) | [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) — no API key | produced by step 2 |
| Coastal El Niño (ICEN) | ENFEN | `icen_coastal_el_nino.txt` |
| SENAMHI stations (optional) | SENAMHI quality-controlled daily series | `senamhi_*_daily.txt` |

## Model (main result)

- **Unit of analysis:** district–month panel (ubigeo)
- **Target:** `log1p(cases)`
- **Algorithm:** `HistGradientBoostingRegressor`
- **Features:** case lags (1–3), climate lags, humidity, ICEN, seasonal sin/cos, province/district IDs
- **Evaluation:** hold-out year 2024 (one-month-ahead with observed lags)
- **Operational forecast:** 3-month horizon using month climatology and recursive predicted cases

## Project layout

```
01_prepare_data.py
02_fetch_climate.py
03_train_district_model.py
04_serve_dashboard.py
district_coords.py
requirements.txt
data/                 # inputs and intermediate panels
models/               # trained model + metadata
web/                  # dashboard with district charts (index.html + dashboard.json)
experiments/          # earlier department-level baselines (optional)
```

Charts (department aggregate and per-district backtest + forecast) are shown in the web dashboard after step 4 — no separate PNG exports are produced by the main pipeline.
## Optional experiments

Department-level baselines (not required for the dashboard):

```powershell
py -3.12 experiments/train_dept_baseline.py
py -3.12 experiments/merge_senamhi_climate.py
py -3.12 experiments/train_dept_with_climate.py
```

`merge_senamhi_climate.py` uses `data/senamhi_lambayeque_daily.txt`.
