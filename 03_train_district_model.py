"""Train a district-level dengue model, evaluate on 2024, and export dashboard JSON.

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

TEST_YEAR = 2024
HORIZON = 3  # operational months ahead

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


def build_dashboard_payload(
    comparison: pd.DataFrame,
    forecast: pd.DataFrame,
    metrics: dict[str, float],
    last_year: int,
    last_month: int,
    p66: dict,
    p90: dict,
) -> dict:
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

    month_ref = int(comparison["mes"].max())
    map_backtest = (
        comparison[comparison["mes"] == month_ref][
            ["ubigeo", "distrito", "provincia", "lat", "lon", "casos", "predicho"]
        ].copy()
    )
    map_backtest["tipo"] = "backtest"
    map_backtest["periodo"] = f"{TEST_YEAR}-{month_ref:02d}"

    h1 = forecast[forecast["horizonte"] == 1].copy()
    h1["riesgo"] = [
        risk_level(p, p66.get(u, 0), p90.get(u, 0))
        for u, p in zip(h1["ubigeo"], h1["predicho"])
    ]

    forecast_out = forecast.copy()
    forecast_out["riesgo"] = [
        risk_level(p, p66.get(u, 0), p90.get(u, 0))
        for u, p in zip(forecast_out["ubigeo"], forecast_out["predicho"])
    ]

    ranking = (
        forecast_out.groupby(
            ["ubigeo", "distrito", "provincia", "lat", "lon"], as_index=False
        )
        .agg(pred_promedio=("predicho", "mean"), pred_max=("predicho", "max"))
        .sort_values("pred_promedio", ascending=False)
    )
    ranking["riesgo"] = [
        risk_level(p, p66.get(u, 0), p90.get(u, 0))
        for u, p in zip(ranking["ubigeo"], ranking["pred_promedio"])
    ]

    return {
        "meta": {
            "titulo": "Vigilancia predictiva de dengue — Lambayeque",
            "anio_test": TEST_YEAR,
            "horizonte_meses": HORIZON,
            "ultimo_dato": f"{last_year}-{last_month:02d}",
            "fuente_casos": "CDC/MINSA datos abiertos vigilancia dengue",
            "fuente_clima": "Open-Meteo Archive API + ICEN (ENFEN)",
            "modelo": "HistGradientBoostingRegressor (log1p casos) por panel distrital",
            "mae_distrito": round(metrics["mae"], 2),
            "rmse_distrito": round(metrics["rmse"], 2),
            "mae_departamento": round(metrics["mae_dep"], 2),
            "rmse_departamento": round(metrics["rmse_dep"], 2),
            "n_distritos": int(comparison["ubigeo"].nunique()),
            "nota": (
                "Backtest 2024 = one-month-ahead with observed lags. "
                "Operational forecast uses month climatology and recursive predicted cases."
            ),
        },
        "departamento_2024": dep_series.to_dict(orient="records"),
        "backtest_distritos": comparison.to_dict(orient="records"),
        "resumen_distritos_2024": top_mae.round(2).to_dict(orient="records"),
        "pronostico": forecast_out.to_dict(orient="records"),
        "ranking_accion": ranking.round(2).to_dict(orient="records"),
        "mapa_mes_ref": map_backtest.to_dict(orient="records"),
        "mapa_pronostico_h1": h1[
            ["ubigeo", "distrito", "provincia", "lat", "lon", "anio", "mes", "predicho", "riesgo"]
        ].to_dict(orient="records"),
    }


def main() -> None:
    print("Loading district–climate panel...")
    raw = pd.read_csv(DATA, dtype={"ubigeo": str})
    raw["ubigeo"] = raw["ubigeo"].astype(str).str.zfill(6)
    raw = add_calendar(raw)
    raw, cat_maps = encode_categories(raw)
    raw = add_lags(raw)

    # After zero-filling months, the first 3 months per district lack full lags
    model_df = raw.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    model_df["y"] = np.log1p(model_df["casos"])

    train = model_df[model_df["anio"] < TEST_YEAR]
    test = model_df[model_df["anio"] == TEST_YEAR]
    print(f"Train: {len(train)} | Test {TEST_YEAR}: {len(test)}")

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.06,
        max_iter=400,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(train[FEATURE_COLS], train["y"])

    comparison, metrics = evaluate_backtest(model, test)
    print(f"\n=== District backtest {TEST_YEAR} ===")
    print(f"District MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f}")
    print(
        f"Department MAE (sum): {metrics['mae_dep']:.2f} | "
        f"RMSE: {metrics['rmse_dep']:.2f}"
    )

    BASE.joinpath("models").mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "features": FEATURE_COLS, "cat_maps": cat_maps},
        MODEL,
    )
    print(f"Model: {MODEL}")

    work = raw.sort_values(["ubigeo", "anio", "mes"]).copy()
    last_year = int(work["anio"].max())
    last_month = int(work.loc[work["anio"] == last_year, "mes"].max())

    forecast = recursive_forecast(model, work, cat_maps, last_year, last_month)

    hist = work.groupby("ubigeo")["casos"]
    p66 = hist.quantile(0.66).to_dict()
    p90 = hist.quantile(0.90).to_dict()

    payload = build_dashboard_payload(
        comparison, forecast, metrics, last_year, last_month, p66, p90
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Dashboard JSON: {OUT_JSON}")

    META.write_text(
        json.dumps(
            {
                "features": FEATURE_COLS,
                "cat_maps": cat_maps,
                "metrics": payload["meta"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Charts are available in the web dashboard (04_serve_dashboard.py).")


if __name__ == "__main__":
    main()
