"""Train a district-level dengue model, evaluate on 2025, and forecast 2026–2027.

Model: HistGradientBoostingRegressor on log1p(cases), district panel.
Climate: Open-Meteo nodes + coastal El Niño index (ICEN / ENFEN).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "lambayeque_districts_climate.csv"
MODEL = BASE / "models" / "modelo_dengue_distritos.pkl"
META = BASE / "models" / "modelo_meta.json"
OUT_JSON = BASE / "web" / "data" / "dashboard.json"
OUT_DIR = BASE / "outputs" / "forecasts"
OUT_BACKTEST = OUT_DIR / "backtest_distritos_2025.csv"
OUT_FORECAST = OUT_DIR / "forecast_distritos_2026_2027.csv"
OUT_DEP = OUT_DIR / "backtest_departamento_2025.csv"
OUT_FORECAST_DEP = OUT_DIR / "forecast_departamento_2026_2027.csv"

TEST_YEAR = 2025
HORIZON = 24

FEATURE_COLS = [
    "mes", "mes_sin", "mes_cos",
    "provincia_id", "distrito_id",
    "casos_lag1", "casos_lag2", "casos_lag3",
    "casos_roll3",
    "precip_lag1", "precip_lag2", "precip_lag3",
    "tmax_lag1", "tmax_lag2",
    "tmin_lag1",
    "humidity_lag1", "humidity_lag2",
    "icen_lag1", "icen_lag2", "icen_lag3",
]


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonal encoding of calendar month."""
    df = df.copy()
    angle = 2 * np.pi * (df["mes"] - 1) / 12
    df["mes_sin"] = np.sin(angle)
    df["mes_cos"] = np.cos(angle)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """Within-district lag and rolling features (no leakage across ubigeos)."""
    df = df.sort_values(["ubigeo", "anio", "mes"]).copy()
    grouped = df.groupby("ubigeo", group_keys=False)
    for k in (1, 2, 3):
        df[f"casos_lag{k}"] = grouped["casos"].shift(k)
        df[f"precip_lag{k}"] = grouped["precip"].shift(k)
        df[f"icen_lag{k}"] = grouped["icen"].shift(k)
    for k in (1, 2):
        df[f"tmax_lag{k}"] = grouped["tmax"].shift(k)
        df[f"humidity_lag{k}"] = grouped["humidity"].shift(k)
    df["tmin_lag1"] = grouped["tmin"].shift(1)
    df["casos_roll3"] = grouped["casos"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    return df


def encode_categories(
    df: pd.DataFrame,
    fit_maps: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Map province/district names to integer IDs."""
    df = df.copy()
    maps = fit_maps or {}
    if "provincia" not in maps:
        maps["provincia"] = {p: i for i, p in enumerate(sorted(df["provincia"].unique()))}
    if "distrito" not in maps:
        maps["distrito"] = {d: i for i, d in enumerate(sorted(df["distrito"].unique()))}
    df["provincia_id"] = df["provincia"].map(maps["provincia"]).fillna(-1).astype(int)
    df["distrito_id"] = df["distrito"].map(maps["distrito"]).fillna(-1).astype(int)
    return df, maps


def risk_level(pred: float, p66: float, p90: float) -> str:
    if pred >= p90:
        return "alto"
    if pred >= p66:
        return "medio"
    return "bajo"


def evaluate_backtest(
    model: HistGradientBoostingRegressor,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    pred_log = model.predict(test[FEATURE_COLS])
    pred = np.expm1(pred_log).clip(min=0)
    real = test["casos"].to_numpy()

    mae = mean_absolute_error(real, pred)
    rmse = float(np.sqrt(mean_squared_error(real, pred)))

    dep_real = test.groupby("mes")["casos"].sum()
    dep_pred = pd.Series(pred, index=test.index).groupby(test["mes"]).sum()
    mae_dep = mean_absolute_error(dep_real, dep_pred.reindex(dep_real.index).fillna(0))
    rmse_dep = float(
        np.sqrt(mean_squared_error(dep_real, dep_pred.reindex(dep_real.index).fillna(0)))
    )

    comparison = test[
        ["ubigeo", "provincia", "distrito", "anio", "mes", "casos", "lat", "lon"]
    ].copy()
    comparison["predicho"] = np.round(pred, 1)
    comparison["error"] = (comparison["casos"] - comparison["predicho"]).round(1)

    metrics = {
        "mae": float(mae),
        "rmse": rmse,
        "mae_dep": float(mae_dep),
        "rmse_dep": rmse_dep,
    }
    return comparison, metrics


def recursive_forecast(
    model: HistGradientBoostingRegressor,
    work: pd.DataFrame,
    cat_maps: dict,
    last_year: int,
    last_month: int,
) -> pd.DataFrame:
    """h-step ahead forecast using month climatology and predicted cases recursively."""
    forecast_rows: list[pd.DataFrame] = []
    sim = work.copy()

    climate_hist = (
        work.groupby(["nodo_clima", "mes"], as_index=False)
        .agg(
            precip=("precip", "mean"),
            tmax=("tmax", "mean"),
            tmin=("tmin", "mean"),
            humidity=("humidity", "mean"),
            icen=("icen", "mean"),
        )
    )

    for h in range(1, HORIZON + 1):
        month = last_month + h
        year = last_year
        while month > 12:
            month -= 12
            year += 1

        base_dist = (
            work.drop_duplicates("ubigeo")[
                ["ubigeo", "distrito", "provincia", "lat", "lon", "nodo_clima"]
            ].copy()
        )
        base_dist["anio"] = year
        base_dist["mes"] = month
        base_dist["casos"] = np.nan
        base_dist = base_dist.merge(climate_hist, on=["nodo_clima", "mes"], how="left")
        base_dist = add_calendar(base_dist)
        base_dist, _ = encode_categories(base_dist, fit_maps=cat_maps)

        sim = pd.concat([sim, base_dist], ignore_index=True)
        sim = sim.sort_values(["ubigeo", "anio", "mes"]).reset_index(drop=True)
        sim = add_lags(sim)

        mask = (sim["anio"] == year) & (sim["mes"] == month)
        Xh = sim.loc[mask, FEATURE_COLS]
        if Xh.isna().any().any():
            Xh = Xh.fillna(0)
        yh = np.expm1(model.predict(Xh)).clip(min=0)
        sim.loc[mask, "casos"] = yh

        chunk = sim.loc[
            mask, ["ubigeo", "distrito", "provincia", "lat", "lon", "anio", "mes"]
        ].copy()
        chunk["predicho"] = np.round(yh, 1)
        chunk["horizonte"] = h
        forecast_rows.append(chunk)

    return pd.concat(forecast_rows, ignore_index=True)


def fit_model(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(train[FEATURE_COLS], train["y"])
    return model


def pack_backtest(comparison: pd.DataFrame, metrics: dict[str, float], year: int) -> dict:
    top_mae = (
        comparison.assign(abs_err=lambda x: x["error"].abs())
        .groupby(["ubigeo", "distrito", "provincia"], as_index=False)
        .agg(
            casos_real=("casos", "sum"),
            casos_pred=("predicho", "sum"),
            mae=("abs_err", "mean"),
        )
        .sort_values("casos_real", ascending=False)
    )
    dep_series = (
        comparison.groupby("mes", as_index=False)
        .agg(real=("casos", "sum"), predicho=("predicho", "sum"))
    )
    annual_real = float(comparison["casos"].sum())
    annual_pred = float(comparison["predicho"].sum())
    annual_pct = abs(annual_real - annual_pred) / max(annual_real, 1.0) * 100
    return {
        "anio": year,
        "mae_distrito": round(metrics["mae"], 2),
        "rmse_distrito": round(metrics["rmse"], 2),
        "mae_departamento": round(metrics["mae_dep"], 2),
        "rmse_departamento": round(metrics["rmse_dep"], 2),
        "casos_anual_real": round(annual_real, 1),
        "casos_anual_predicho": round(annual_pred, 1),
        "error_anual_pct": round(annual_pct, 2),
        "n_distritos": int(comparison["ubigeo"].nunique()),
        "departamento": dep_series.to_dict(orient="records"),
        "backtest_distritos": comparison.to_dict(orient="records"),
        "resumen_distritos": top_mae.round(2).to_dict(orient="records"),
    }


def build_forecast_block(
    forecast: pd.DataFrame,
    p66: dict,
    p90: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    forecast_out = forecast.copy()
    forecast_out["riesgo"] = [
        risk_level(p, p66.get(u, 0), p90.get(u, 0))
        for u, p in zip(forecast_out["ubigeo"], forecast_out["predicho"])
    ]
    h1 = forecast_out[forecast_out["horizonte"] == 1].copy()
    # Short-horizon ranking (first 3 months) for the action panel
    near = forecast_out[forecast_out["horizonte"] <= 3]
    ranking = (
        near.groupby(
            ["ubigeo", "distrito", "provincia", "lat", "lon"], as_index=False
        )
        .agg(pred_promedio=("predicho", "mean"), pred_max=("predicho", "max"))
        .sort_values("pred_promedio", ascending=False)
    )
    ranking["riesgo"] = [
        risk_level(p, p66.get(u, 0), p90.get(u, 0))
        for u, p in zip(ranking["ubigeo"], ranking["pred_promedio"])
    ]
    return forecast_out, h1, ranking


def main() -> None:
    print("Loading district–climate panel...")
    raw = pd.read_csv(DATA, dtype={"ubigeo": str})
    raw["ubigeo"] = raw["ubigeo"].astype(str).str.zfill(6)
    raw = add_calendar(raw)
    raw, cat_maps = encode_categories(raw)
    raw = add_lags(raw)

    model_df = raw.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    model_df["y"] = np.log1p(model_df["casos"])

    BACKTEST_YEARS = (2024, 2025)
    validaciones: dict[str, dict] = {}
    last_model = None
    last_comparison = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in BACKTEST_YEARS:
        train = model_df[model_df["anio"] < year]
        test = model_df[model_df["anio"] == year]
        print(f"\n=== Train < {year} | Test {year}: {len(train)} / {len(test)} ===")
        model = fit_model(train)
        comparison, metrics = evaluate_backtest(model, test)
        pack = pack_backtest(comparison, metrics, year)
        validaciones[str(year)] = pack
        print(
            f"MAE dist {pack['mae_distrito']} | MAE dep {pack['mae_departamento']} | "
            f"annual {pack['casos_anual_real']:.0f} vs {pack['casos_anual_predicho']:.1f} "
            f"({pack['error_anual_pct']:.1f}%)"
        )
        last_model = model
        last_comparison = comparison
        comparison.to_csv(OUT_DIR / f"backtest_distritos_{year}.csv", index=False)
        (
            comparison.groupby(["anio", "mes"], as_index=False)
            .agg(real=("casos", "sum"), predicho=("predicho", "sum"))
            .to_csv(OUT_DIR / f"backtest_departamento_{year}.csv", index=False)
        )

    assert last_model is not None and last_comparison is not None

    BASE.joinpath("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": last_model, "features": FEATURE_COLS, "cat_maps": cat_maps},
        MODEL,
    )
    print(f"\nModel (trained through {TEST_YEAR - 1}): {MODEL}")

    work = raw.sort_values(["ubigeo", "anio", "mes"]).copy()
    last_year = int(work["anio"].max())
    last_month = int(work.loc[work["anio"] == last_year, "mes"].max())
    print(f"Forecast origin: {last_year}-{last_month:02d} | horizon: {HORIZON} months")

    forecast = recursive_forecast(last_model, work, cat_maps, last_year, last_month)
    hist = work.groupby("ubigeo")["casos"]
    p66 = hist.quantile(0.66).to_dict()
    p90 = hist.quantile(0.90).to_dict()
    forecast_out, h1, ranking = build_forecast_block(forecast, p66, p90)

    forecast_out.to_csv(OUT_FORECAST, index=False)
    (
        forecast_out.groupby(["anio", "mes"], as_index=False)
        .agg(predicho=("predicho", "sum"))
        .sort_values(["anio", "mes"])
        .to_csv(OUT_FORECAST_DEP, index=False)
    )
    print("\n=== Forecast totals by year ===")
    print(
        forecast_out.groupby("anio")["predicho"].sum().round(1).to_string()
    )

    # Default view = latest backtest year; keep legacy keys for older markup
    latest = validaciones[str(TEST_YEAR)]
    payload = {
        "meta": {
            "titulo": "Vigilancia predictiva de dengue — Lambayeque",
            "anio_test": TEST_YEAR,
            "anios_validacion": list(BACKTEST_YEARS),
            "horizonte_meses": HORIZON,
            "ultimo_dato": f"{last_year}-{last_month:02d}",
            "fuente_casos": "CDC/MINSA datos abiertos vigilancia dengue",
            "fuente_clima": "Open-Meteo Archive API + ICEN (ENFEN)",
            "modelo": "HistGradientBoostingRegressor (log1p casos) por panel distrital",
            "mae_distrito": latest["mae_distrito"],
            "rmse_distrito": latest["rmse_distrito"],
            "mae_departamento": latest["mae_departamento"],
            "rmse_departamento": latest["rmse_departamento"],
            "n_distritos": latest["n_distritos"],
            "nota": (
                "Backtests 2024 and 2025 = one-month-ahead with observed lags "
                "(each year trained only on prior years). "
                "Forecast 2026–2027 uses month climatology and recursive predicted cases."
            ),
        },
        "validaciones": validaciones,
        "departamento_backtest": latest["departamento"],
        "departamento_2024": validaciones["2024"]["departamento"],
        "backtest_distritos": latest["backtest_distritos"],
        "resumen_distritos_backtest": latest["resumen_distritos"],
        "resumen_distritos_2024": validaciones["2024"]["resumen_distritos"],
        "pronostico": forecast_out.to_dict(orient="records"),
        "ranking_accion": ranking.round(2).to_dict(orient="records"),
        "mapa_pronostico_h1": h1[
            ["ubigeo", "distrito", "provincia", "lat", "lon", "anio", "mes", "predicho", "riesgo"]
        ].to_dict(orient="records"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dashboard JSON: {OUT_JSON}")

    META.write_text(
        json.dumps(
            {
                "features": FEATURE_COLS,
                "cat_maps": cat_maps,
                "metrics": payload["meta"],
                "validaciones": {
                    y: {
                        "mae_distrito": v["mae_distrito"],
                        "mae_departamento": v["mae_departamento"],
                        "casos_anual_real": v["casos_anual_real"],
                        "casos_anual_predicho": v["casos_anual_predicho"],
                        "error_anual_pct": v["error_anual_pct"],
                    }
                    for y, v in validaciones.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Charts are available in the web dashboard (04_serve_dashboard.py).")


if __name__ == "__main__":
    main()
