"""Train on all districts; validate and forecast Chiclayo only.

Model: HistGradientBoostingRegressor on log1p(cases), district panel.
Validation: train on years < TEST_YEAR (2015–2023), evaluate Chiclayo in 2024.
Forecast: refit on every year through 2025, then forecast 1–3 months ahead.
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
MODEL = BASE / "models" / "modelo_dengue_chiclayo.pkl"
META = BASE / "models" / "modelo_meta_chiclayo.json"
OUT_JSON = BASE / "web" / "data" / "dashboard_corto_plazo.json"
OUT_DIR = BASE / "outputs" / "forecasts"
OUT_FORECAST = OUT_DIR / "forecast_chiclayo_corto_plazo.csv"

TEST_YEAR = 2024
HORIZON = 3  # operational short horizon (1–3 months)
TARGET_UBIGEO = "140101"  # Chiclayo
TARGET_NAME = "CHICLAYO"

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


def evaluate_target(comparison: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Keep focus-district rows and compute district-only metrics."""
    target = comparison[comparison["ubigeo"] == TARGET_UBIGEO].copy()
    if target.empty:
        raise ValueError(f"No backtest rows for {TARGET_NAME} ({TARGET_UBIGEO})")
    mae = mean_absolute_error(target["casos"], target["predicho"])
    rmse = float(np.sqrt(mean_squared_error(target["casos"], target["predicho"])))
    return target, {"mae": mae, "rmse": rmse}


def pack_backtest(target: pd.DataFrame, metrics: dict[str, float], year: int) -> dict:
    top_mae = (
        target.assign(abs_err=lambda x: x["error"].abs())
        .groupby(["ubigeo", "distrito", "provincia"], as_index=False)
        .agg(
            casos_real=("casos", "sum"),
            casos_pred=("predicho", "sum"),
            mae=("abs_err", "mean"),
        )
        .sort_values("casos_real", ascending=False)
    )
    series = (
        target.groupby("mes", as_index=False)
        .agg(real=("casos", "sum"), predicho=("predicho", "sum"))
    )
    series["error_pct"] = (
        (series["real"] - series["predicho"]).abs() / series["real"].clip(lower=1) * 100
    ).round(1)
    annual_real = float(target["casos"].sum())
    annual_pred = float(target["predicho"].sum())
    annual_pct = abs(annual_real - annual_pred) / max(annual_real, 1.0) * 100
    # WAPE: monthly errors are summed in absolute value, so months cannot offset each other
    monthly_pct = (series["real"] - series["predicho"]).abs().sum() / max(annual_real, 1.0) * 100
    return {
        "anio": year,
        "foco": TARGET_NAME,
        "ubigeo": TARGET_UBIGEO,
        "mae_distrito": round(metrics["mae"], 2),
        "rmse_distrito": round(metrics["rmse"], 2),
        # Kept for dashboard KPI compatibility (focus-district monthly series)
        "mae_departamento": round(metrics["mae"], 2),
        "rmse_departamento": round(metrics["rmse"], 2),
        "casos_anual_real": round(annual_real, 1),
        "casos_anual_predicho": round(annual_pred, 1),
        "error_anual_pct": round(annual_pct, 2),
        "error_mensual_pct": round(monthly_pct, 1),
        "n_distritos": 1,
        "departamento": series.to_dict(orient="records"),
        "backtest_distritos": target.to_dict(orient="records"),
        "resumen_distritos": top_mae.round(2).to_dict(orient="records"),
    }


def rolling_validation(
    model: HistGradientBoostingRegressor,
    raw: pd.DataFrame,
    cat_maps: dict,
    year: int,
) -> dict:
    """Forecast h=1..HORIZON from each origin in `year` using only data up to the origin."""
    key = raw["anio"] * 100 + raw["mes"]
    origins = [(year - 1, 12)] + [(year, m) for m in range(1, 13 - HORIZON)]
    actual = raw.set_index(["ubigeo", "anio", "mes"])["casos"]

    rows = []
    for oy, om in origins:
        work = raw[key <= oy * 100 + om].copy()
        fc = recursive_forecast(model, work, cat_maps, oy, om)
        fc = fc[fc["ubigeo"] == TARGET_UBIGEO].copy()
        fc["real"] = [actual.get((u, a, m), np.nan) for u, a, m in zip(fc["ubigeo"], fc["anio"], fc["mes"])]
        fc["origen"] = f"{oy}-{om:02d}"
        rows.append(fc)
    res = pd.concat(rows, ignore_index=True)
    res["abs_err"] = (res["real"] - res["predicho"]).abs()

    by_h = (
        res.groupby("horizonte", as_index=False)["abs_err"].mean()
        .rename(columns={"abs_err": "mae"})
    )
    windows = (
        res.groupby("origen", as_index=False)
        .agg(real=("real", "sum"), predicho=("predicho", "sum"))
    )
    windows["error"] = windows["real"] - windows["predicho"]
    real_total = float(windows["real"].sum())
    pred_total = float(windows["predicho"].sum())
    return {
        "horizonte_meses": HORIZON,
        "mae_por_horizonte": by_h.round(2).to_dict(orient="records"),
        "ventanas": windows.round(1).to_dict(orient="records"),
        "real_acumulado": round(real_total, 1),
        "predicho_acumulado": round(pred_total, 1),
        "error_acumulado_pct": round(abs(real_total - pred_total) / max(real_total, 1.0) * 100, 1),
        "detalle": res[["origen", "horizonte", "anio", "mes", "real", "predicho"]].round(1).to_dict(orient="records"),
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

    # Train on ALL districts; report / forecast ONLY Chiclayo (short horizon)
    BACKTEST_YEARS = (TEST_YEAR,)
    validaciones: dict[str, dict] = {}
    last_target = None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for year in BACKTEST_YEARS:
        train = model_df[model_df["anio"] < year]
        test = model_df[model_df["anio"] == year]
        print(
            f"\n=== Train ALL districts < {year} ({len(train)} rows) | "
            f"Validate {TARGET_NAME} {year} ==="
        )
        model = fit_model(train)
        comparison, _ = evaluate_backtest(model, test)
        target, metrics = evaluate_target(comparison)
        pack = pack_backtest(target, metrics, year)
        validaciones[str(year)] = pack
        print(
            f"{TARGET_NAME} MAE {pack['mae_distrito']} | "
            f"annual {pack['casos_anual_real']:.0f} vs {pack['casos_anual_predicho']:.1f} "
            f"({pack['error_anual_pct']:.1f}%)"
        )
        rolling = rolling_validation(model, raw, cat_maps, year)
        pack["rolling"] = rolling
        print(
            f"Rolling h=1..{HORIZON}: MAE "
            + ", ".join(f"h{r['horizonte']}={r['mae']}" for r in rolling["mae_por_horizonte"])
            + f" | acumulado {rolling['real_acumulado']:.0f} vs {rolling['predicho_acumulado']:.1f}"
            f" ({rolling['error_acumulado_pct']:.1f}%)"
        )
        last_target = target
        target.to_csv(OUT_DIR / f"backtest_chiclayo_{year}.csv", index=False)
        pd.DataFrame(rolling["detalle"]).to_csv(
            OUT_DIR / f"rolling_chiclayo_{year}.csv", index=False
        )

    assert last_target is not None

    work = raw.sort_values(["ubigeo", "anio", "mes"]).copy()
    last_year = int(work["anio"].max())
    last_month = int(work.loc[work["anio"] == last_year, "mes"].max())

    # After validation, refit on every available year for the operational forecast
    final_model = fit_model(model_df[model_df["anio"] <= last_year])
    BASE.joinpath("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURE_COLS,
            "cat_maps": cat_maps,
            "target_ubigeo": TARGET_UBIGEO,
            "target_name": TARGET_NAME,
            "horizonte_meses": HORIZON,
        },
        MODEL,
    )
    print(f"\nFinal model (all districts through {last_year}): {MODEL}")
    print(f"Forecast origin: {last_year}-{last_month:02d} | short horizon: {HORIZON} months")

    # Recursive forecast on full panel, then keep Chiclayo only (h+1..h+3).
    forecast = recursive_forecast(final_model, work, cat_maps, last_year, last_month)
    hist = work.groupby("ubigeo")["casos"]
    p66 = hist.quantile(0.66).to_dict()
    p90 = hist.quantile(0.90).to_dict()
    forecast_all, h1_all, ranking_all = build_forecast_block(forecast, p66, p90)
    forecast_out = forecast_all[forecast_all["ubigeo"] == TARGET_UBIGEO].copy()
    h1 = h1_all[h1_all["ubigeo"] == TARGET_UBIGEO].copy()
    ranking = ranking_all[ranking_all["ubigeo"] == TARGET_UBIGEO].copy()

    forecast_out.to_csv(OUT_FORECAST, index=False)
    print(f"\n=== {TARGET_NAME} short-horizon forecast ===")
    print(
        forecast_out[["horizonte", "anio", "mes", "predicho", "riesgo"]]
        .sort_values("horizonte")
        .to_string(index=False)
    )

    latest = validaciones[str(TEST_YEAR)]
    short_total = float(forecast_out["predicho"].sum())
    payload = {
        "meta": {
            "titulo": f"Vigilancia predictiva de dengue — {TARGET_NAME.title()}",
            "foco": TARGET_NAME,
            "ubigeo": TARGET_UBIGEO,
            "anio_test": TEST_YEAR,
            "anios_validacion": list(BACKTEST_YEARS),
            "horizonte_meses": HORIZON,
            "producto": "pronostico_corto_plazo",
            "ultimo_dato": f"{last_year}-{last_month:02d}",
            "fuente_casos": "CDC/MINSA datos abiertos vigilancia dengue",
            "fuente_clima": "Open-Meteo Archive API + ICEN (ENFEN)",
            "modelo": (
                "HistGradientBoostingRegressor (log1p casos); "
                f"validado con todos los distritos 2015–{TEST_YEAR - 1} contra {TEST_YEAR}; "
                f"reentrenado 2015–{last_year} para el pronóstico 1–{HORIZON} meses de {TARGET_NAME}"
            ),
            "mae_distrito": latest["mae_distrito"],
            "rmse_distrito": latest["rmse_distrito"],
            "mae_departamento": latest["mae_departamento"],
            "rmse_departamento": latest["rmse_departamento"],
            "casos_corto_plazo": round(short_total, 1),
            "n_distritos": 1,
            "nota": (
                f"Validación: entrenado con los 37 distritos de Lambayeque (años < {TEST_YEAR}), "
                f"evaluado solo en {TARGET_NAME} {TEST_YEAR}. "
                f"Pronóstico: modelo reentrenado hasta {last_year}, horizonte de {HORIZON} meses (no total anual)."
            ),
        },
        "validaciones": validaciones,
        "departamento_backtest": latest["departamento"],
        "backtest_distritos": latest["backtest_distritos"],
        "resumen_distritos_backtest": latest["resumen_distritos"],
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
