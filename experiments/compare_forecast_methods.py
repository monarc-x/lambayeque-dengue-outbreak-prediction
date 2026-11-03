"""Honest multi-step test: forecast all of 2025 from Dec-2024 using several methods.

Unlike the one-month-ahead backtest (observed lags), this mimics the real
2026 use case: no observed cases after the forecast origin.
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

TEST_YEAR = 2025
MIN_CASES = 15


def annual_err(real: float, pred: float) -> float:
    return abs(real - pred) / max(real, 1.0) * 100


def main() -> None:
    raw = pd.read_csv(train.DATA, dtype={"ubigeo": str})
    raw["ubigeo"] = raw["ubigeo"].astype(str).str.zfill(6)
    raw = train.add_calendar(raw)
    raw, cat_maps = train.encode_categories(raw)
    raw = train.add_lags(raw)

    model_df = raw.dropna(subset=train.FEATURE_COLS).reset_index(drop=True)
    model_df["y"] = np.log1p(model_df["casos"])
    model = train.fit_model(model_df[model_df["anio"] < TEST_YEAR])

    hist = raw[raw["anio"] < TEST_YEAR]
    actual = raw[raw["anio"] == TEST_YEAR].groupby(["ubigeo", "distrito"])["casos"].sum()

    # A) Recursive HGB from Dec-2024 (what the 2026 forecast does)
    train.HORIZON = 12
    rec = train.recursive_forecast(model, hist.copy(), cat_maps, TEST_YEAR - 1, 12)
    pred_rec = rec.groupby(["ubigeo", "distrito"])["predicho"].sum()

    # B) Seasonal median of the same month, 2015–2024
    pred_med = hist.groupby(["ubigeo", "distrito", "mes"])["casos"].median().groupby(
        ["ubigeo", "distrito"]
    ).sum()

    # C) Median annual total (robust to the 2023 epidemic)
    annual_hist = hist.groupby(["ubigeo", "distrito", "anio"])["casos"].sum()
    pred_med_year = annual_hist.groupby(["ubigeo", "distrito"]).median()

    # D) Mean of the 3 most recent "non-epidemic" years (exclude 2023)
    recent = annual_hist.reset_index()
    recent = recent[(recent["anio"] >= 2019) & (recent["anio"] != 2023)]
    pred_recent = recent.groupby(["ubigeo", "distrito"])["casos"].median()

    df = pd.DataFrame({
        "real_2025": actual,
        "hgb_recursivo": pred_rec,
        "mediana_mensual": pred_med,
        "mediana_anual": pred_med_year,
        "mediana_2019_24_sin2023": pred_recent,
    }).fillna(0).reset_index()

    methods = ["hgb_recursivo", "mediana_mensual", "mediana_anual", "mediana_2019_24_sin2023"]
    for m in methods:
        df[f"err_{m}"] = [annual_err(r, p) for r, p in zip(df["real_2025"], df[m])]

    big = df[df["real_2025"] >= MIN_CASES].copy()
    pd.set_option("display.width", 200)
    cols = ["distrito", "real_2025"] + [c for m in methods for c in (m, f"err_{m}")]
    print(f"=== 2025 forecast from Dec-2024 (districts with >= {MIN_CASES} cases) ===")
    print(big[cols].round(1).sort_values("real_2025", ascending=False).to_string(index=False))

    print("\n=== Median annual error by method ===")
    for m in methods:
        print(f"{m:28s} {big[f'err_{m}'].median():6.1f}%")

    print("\n=== Districts with any method <= 20% ===")
    for m in methods:
        ok = big[big[f"err_{m}"] <= 20]
        names = ", ".join(f"{d} ({e:.0f}%)" for d, e in zip(ok["distrito"], ok[f"err_{m}"]))
        print(f"{m:28s} {names or '-'}")


if __name__ == "__main__":
    main()
