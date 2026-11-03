"""Month-by-month error per district for 2024 and 2025 (stability check).

For each test year Y the model is trained on years < Y. Monthly error is
WAPE = sum|real - pred| / sum(real), so months do not offset each other.
  - wape1:  one-step (each month predicted with the real previous month)
  - wapeH1: rolling h=1 (forecast next month from origin, climatology climate)
  - wapeH3: rolling h=3 (forecast 3 months ahead without seeing real cases)
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


def wape(df: pd.DataFrame, real: str, pred: str) -> pd.Series:
    g = df.assign(abs_err=(df[real] - df[pred]).abs()).groupby("distrito")
    return g["abs_err"].sum() / g[real].sum().clip(lower=1) * 100


def evaluate_year(raw, model_df, cat_maps, year):
    model = train.fit_model(model_df[model_df["anio"] < year])
    comp, _ = train.evaluate_backtest(model, model_df[model_df["anio"] == year])

    key = raw["anio"] * 100 + raw["mes"]
    origins = [(year - 1, 12)] + [(year, m) for m in range(1, 13 - H)]
    actual = raw.set_index(["ubigeo", "anio", "mes"])["casos"]
    parts = []
    for oy, om in origins:
        fc = train.recursive_forecast(model, raw[key <= oy * 100 + om].copy(), cat_maps, oy, om)
        fc["real"] = [actual.get((u, a, m), np.nan) for u, a, m in zip(fc["ubigeo"], fc["anio"], fc["mes"])]
        parts.append(fc)
    roll = pd.concat(parts, ignore_index=True)

    return pd.concat([
        comp.groupby("distrito")["casos"].sum().rename(f"casos_{year}"),
        wape(comp, "casos", "predicho").rename(f"wape1_{year}"),
        wape(roll[roll["horizonte"] == 1], "real", "predicho").rename(f"wapeH1_{year}"),
        wape(roll[roll["horizonte"] == H], "real", "predicho").rename(f"wapeH3_{year}"),
    ], axis=1)


def main() -> None:
    raw = pd.read_csv(train.DATA, dtype={"ubigeo": str})
    raw["ubigeo"] = raw["ubigeo"].astype(str).str.zfill(6)
    raw = train.add_calendar(raw)
    raw, cat_maps = train.encode_categories(raw)
    raw = train.add_lags(raw)
    model_df = raw.dropna(subset=train.FEATURE_COLS).reset_index(drop=True)
    model_df["y"] = np.log1p(model_df["casos"])
    train.HORIZON = H

    df = pd.concat([evaluate_year(raw, model_df, cat_maps, y) for y in YEARS], axis=1)
    df["wape1_prom"] = df[["wape1_2024", "wape1_2025"]].mean(axis=1)
    df["wapeH1_prom"] = df[["wapeH1_2024", "wapeH1_2025"]].mean(axis=1)
    sel = df[(df["casos_2024"] >= 100) & (df["casos_2025"] >= 15)].sort_values("wape1_prom")

    pd.set_option("display.width", 240)
    print(sel.round(1).to_string())
    df.round(2).to_csv(BASE / "outputs" / "forecasts" / "error_mensual_distritos_2024_2025.csv")


if __name__ == "__main__":
    main()
