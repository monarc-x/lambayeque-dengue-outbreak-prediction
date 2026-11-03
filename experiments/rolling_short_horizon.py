"""Rolling-origin validation for the short-horizon product (h = 1..3) in 2025.

For each origin month (Dec-2024 .. Sep-2025) only data up to the origin is
used; the next 3 months are forecast recursively, exactly like the 2026 run.
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
H = 3
MIN_CASES = 15


def main() -> None:
    raw = pd.read_csv(train.DATA, dtype={"ubigeo": str})
    raw["ubigeo"] = raw["ubigeo"].astype(str).str.zfill(6)
    raw = train.add_calendar(raw)
    raw, cat_maps = train.encode_categories(raw)
    raw = train.add_lags(raw)

    model_df = raw.dropna(subset=train.FEATURE_COLS).reset_index(drop=True)
    model_df["y"] = np.log1p(model_df["casos"])
    model = train.fit_model(model_df[model_df["anio"] < TEST_YEAR])

    train.HORIZON = H
    key = raw["anio"] * 100 + raw["mes"]
    origins = [(TEST_YEAR - 1, 12)] + [(TEST_YEAR, m) for m in range(1, 13 - H)]
    actual = raw.set_index(["ubigeo", "anio", "mes"])["casos"]

    rows = []
    for oy, om in origins:
        work = raw[key <= oy * 100 + om].copy()
        fc = train.recursive_forecast(model, work, cat_maps, oy, om)
        fc["real"] = [actual.get((u, a, m), np.nan) for u, a, m in zip(fc["ubigeo"], fc["anio"], fc["mes"])]
        fc["origen"] = f"{oy}-{om:02d}"
        rows.append(fc)
    res = pd.concat(rows, ignore_index=True)
    res["abs_err"] = (res["real"] - res["predicho"]).abs()
    res.to_csv(BASE / "outputs" / "forecasts" / "rolling_short_horizon_2025.csv", index=False)

    # Per district / horizon: MAE and normalized MAE (MAE / mean monthly cases)
    mean_2025 = raw[raw["anio"] == TEST_YEAR].groupby("distrito")["casos"].mean()
    total_2025 = raw[raw["anio"] == TEST_YEAR].groupby("distrito")["casos"].sum()
    per_h = res.groupby(["distrito", "horizonte"])["abs_err"].mean().unstack()
    per_h.columns = [f"mae_h{h}" for h in per_h.columns]

    # Error of each 3-month window sum (the quantity shown in the panel)
    win = res.groupby(["distrito", "origen"]).agg(real=("real", "sum"), pred=("predicho", "sum"))
    win["err_pct"] = (win["real"] - win["pred"]).abs() / win["real"].clip(lower=1) * 100
    win_err = win.groupby("distrito")["err_pct"].median().rename("err_ventana3_mediana")
    win_sum = win.groupby("distrito").agg(real3=("real", "sum"), pred3=("pred", "sum"))
    win_sum["err_acumulado_pct"] = (win_sum["real3"] - win_sum["pred3"]).abs() / win_sum["real3"].clip(lower=1) * 100

    out = pd.concat([total_2025.rename("casos_2025"), mean_2025.rename("media_mes"), per_h, win_err, win_sum["err_acumulado_pct"]], axis=1)
    for h in range(1, H + 1):
        out[f"nmae_h{h}"] = out[f"mae_h{h}"] / out["media_mes"].clip(lower=0.1) * 100
    out = out[out["casos_2025"] >= MIN_CASES].sort_values("err_acumulado_pct")

    pd.set_option("display.width", 220)
    cols = ["casos_2025", "media_mes", "mae_h1", "mae_h2", "mae_h3", "nmae_h1", "nmae_h3",
            "err_ventana3_mediana", "err_acumulado_pct"]
    print(f"=== Rolling 2025, h=1..{H}, districts with >= {MIN_CASES} cases ===")
    print(out[cols].round(1).to_string())


if __name__ == "__main__":
    main()
