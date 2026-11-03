# Dengue prediction for Lambayeque (Peru)

Repository: [github.com/monarc-x/lambayeque-dengue-outbreak-prediction](https://github.com/monarc-x/lambayeque-dengue-outbreak-prediction)

Research project *"Modelo de machine learning para el pronóstico anual de dengue asociado a El Niño costero en José Leonardo Ortiz, Lambayeque"*. The main pipeline estimates **annual dengue cases** in José Leonardo Ortiz (ubigeo 140105) with a model trained on 212 coastal districts and the coastal El Niño index (ICEN), and presents the 2024 validation and the 2026–2027 forecast in a decision-support dashboard.

## Main result (annual El Niño model)

```powershell
py -3.12 01_prepare_data.py           # district coordinates used by the dashboard map
py -3.12 05_annual_elnino_model.py    # train, validate 2024, forecast 2026–2027
py -3.12 04_serve_dashboard.py        # open the dashboard
```

- **Unit of analysis:** district–year, 2000–2025 (Tumbes, Piura, Lambayeque, La Libertad, Áncash, Ica)
- **Algorithm:** `HistGradientBoostingRegressor` with Poisson loss and monotonic constraints
- **Features:** max ICEN Jan–May, mean ICEN Jun–Aug of the previous year, log cases two and three years before, historical endemicity
- **Validation 2024 (trained 2003–2023):** 736 real vs 866 predicted cases, 17.6 % error
- **Forecast:** 2026 = 350 cases (observed ICEN), 2027 = 1,291 cases (ENFEN extraordinary El Niño scenario)
- **Outputs:** `web/data/dashboard.json`, `models/modelo_anual_elnino.pkl`, `outputs/forecasts/validacion_anual_jose_leonardo_ortiz.csv`
- **District dataset:** `data/jlo/` (individual, monthly and annual cases of José Leonardo Ortiz)

Report helpers: `scripts/export_formulas.py` (equation images in `outputs/formulas/`), `scripts/add_cronograma_docx.py` and `scripts/add_resultados_docx.py` (schedule and results sections of the report).

The sections below describe the earlier monthly district model (`02`–`03`), kept for reference.

## Requirements

- Python 3.10+ (3.12 recommended)
- Internet access for Open-Meteo climate download (step 2)
- Input files under `data/`:
  - `cdc_dengue_surveillance_2000_2025.csv` — CDC/MINSA open dengue surveillance
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

# 3) Train (2015–2024), backtest 2025, forecast 2026–2027
py -3.12 03_train_district_model.py

# 4) Open the local dashboard
py -3.12 04_serve_dashboard.py
```

Dashboard URL: [http://127.0.0.1:8080](http://127.0.0.1:8080)

Forecast CSVs: `outputs/forecasts/` (`backtest_*_2025.csv`, `forecast_*_2026_2027.csv`).

## Pipeline overview

| Step | Script | Output |
|------|--------|--------|
| 1 | `01_prepare_data.py` | `data/lambayeque_districts_monthly.csv` |
| 2 | `02_fetch_climate.py` | `data/lambayeque_districts_climate.csv` |
| 3 | `03_train_district_model.py` | `models/`, `web/data/dashboard.json`, `outputs/forecasts/` |
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
