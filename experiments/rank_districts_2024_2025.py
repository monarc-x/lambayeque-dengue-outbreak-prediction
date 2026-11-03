"""Rank districts by forecast quality in both 2024 and 2025, plus Jan–Mar 2026 output.

For each test year Y the model is trained on years < Y and evaluated two ways:
  - one-step: each month predicted with the real previous month (annual % error)
  - rolling:  from each origin month, forecast h=1..3 without seeing real cases
              (accumulated % error and mean absolute error at h=1)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train", BASE / "03_train_district_model.py")
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)

H = 3
YEARS = (2024, 2025)


def pct(real: float, pred: float) -> float:
    return abs(real - pred) / max(real, 1.0) * 100


def evaluate_year(raw: pd.DataFrame, model_df: pd.DataFrame, cat_maps: dict, year: int):
    model = train.fit_model(model_df[model_df["anio"] < year])

    test = model_df[model_df["anio"] == year]
    comp, _ = train.evaluate_backtest(model, test)
    one = comp.groupby("distrito").agg(real=("casos", "sum"), pred=("predicho", "sum"))
    one[f"err1_{year}"] = [pct(r, p) for r, p in zip(one["real"], one["pred"])]

    key = raw["anio"] * 100 + raw["mes"]
    origins = [(year - 1, 12)] + [(year, m) for m in range(1, 13 - H)]
    actual = raw.set_index(["ubigeo", "anio", "mes"])["casos"]
    parts = []
    for oy, om in origins:
        fc = train.recursive_forecast(model, raw[key <= oy * 100 + om].copy(), cat_maps, oy, om)
        fc["real"] = [actual.get((u, a, m), np.nan) for u, a, m in zip(fc["ubigeo"], fc["anio"], fc["mes"])]
        parts.append(fc)
    roll = pd.concat(parts, ignore_index=True)
    roll["abs_err"] = (roll["real"] - roll["predicho"]).abs()
    acc = roll.groupby("distrito").agg(r=("real", "sum"), p=("predicho", "sum"))
    acc[f"errR_{year}"] = [pct(r, p) for r, p in zip(acc["r"], acc["p"])]
    mae1 = roll[roll["horizonte"] == 1].groupby("distrito")["abs_err"].mean().rename(f"mae1_{year}")

    out = pd.concat([
        one["real"].rename(f"casos_{year}"),
        one[f"err1_{year}"],
        acc[f"errR_{year}"],
        mae1,
    ], axis=1)
    return out, model


def main() -> None:
    raw = pd.read_csv(train.DATA, dtype={"ubigeo": str})
    raw["ubigeo"] = raw["ubigeo"].astype(str).str.zfill(6)
    raw = train.add_calendar(raw)
    raw, cat_maps = train.encode_categories(raw)
    raw = train.add_lags(raw)
    model_df = raw.dropna(subset=train.FEATURE_COLS).reset_index(drop=True)
    model_df["y"] = np.log1p(model_df["casos"])
    train.HORIZON = H

    tables = []
    model_2025 = None
    for y in YEARS:
        t, m = evaluate_year(raw, model_df, cat_maps, y)
        tables.append(t)
        model_2025 = m

    # Jan–Mar 2026 from Dec-2025, with the model trained on years < 2025 (as in the panel)
    ly, lm = int(raw["anio"].max()), 12
    fc26 = train.recursive_forecast(model_2025, raw.copy(), cat_maps, ly, lm)
    q1_2026 = fc26.groupby("distrito")["predicho"].sum().rename("pred_ene_mar_2026")
    q1_hist = (
        raw[(raw["mes"] <= 3) & (raw["anio"] <= 2025)]
        .groupby(["distrito", "anio"])["casos"].sum()
        .groupby("distrito").median()
        .rename("mediana_ene_mar_hist")
    )

    df = pd.concat(tables + [q1_2026, q1_hist], axis=1)
    df["errR_promedio"] = df[["errR_2024", "errR_2025"]].mean(axis=1)
    df["err1_promedio"] = df[["err1_2024", "err1_2025"]].mean(axis=1)
    df = df.sort_values("errR_promedio")

    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 60)
    cols = ["casos_2024", "casos_2025",
            "err1_2024", "err1_2025", "errR_2024", "errR_2025", "errR_promedio",
            "mae1_2024", "mae1_2025", "pred_ene_mar_2026", "mediana_ene_mar_hist"]
    print("=== Todos los distritos (errR = pronóstico 1–3 meses acumulado; err1 = mes siguiente) ===")
    print(df[cols].round(1).to_string())

    sel = df[(df["casos_2024"] >= 100) & (df["casos_2025"] >= 15)]
    print("\n=== Con volumen (>=100 casos 2024 y >=15 casos 2025) ===")
    print(sel[cols].round(1).to_string())
    df.round(2).to_csv(BASE / "outputs" / "forecasts" / "ranking_distritos_2024_2025.csv")


if __name__ == "__main__":
    main()
