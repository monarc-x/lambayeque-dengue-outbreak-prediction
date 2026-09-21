"""Department-level model with SENAMHI climate + ICEN lags.

Exploratory script for comparison with the district Open-Meteo model.
Requires: data/lambayeque_department_climate.csv (from merge_senamhi_climate.py).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "lambayeque_department_climate.csv"
MODEL = BASE / "models" / "modelo_dengue_clima.pkl"
PLOT = BASE / "outputs" / "real_vs_pred_2024_clima.png"


def main() -> None:
    df = pd.read_csv(DATA).sort_values(["anio", "mes"]).reset_index(drop=True)

    df["casos_lag1"] = df["casos"].shift(1)
    df["casos_lag2"] = df["casos"].shift(2)
    df["precip_lag1"] = df["precip"].shift(1)
    df["tmax_lag1"] = df["tmax"].shift(1)
    df["icen_lag1"] = df["icen"].shift(1)
    df = df.dropna().reset_index(drop=True)

    features = ["mes", "casos_lag1", "casos_lag2", "precip_lag1", "tmax_lag1", "icen_lag1"]
    train = df[df["anio"] <= 2023]
    test = df[df["anio"] == 2024]

    print(f"Train: {len(train)} months | Test 2024: {len(test)} months")

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(train[features], train["casos"])

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL)
    print(f"Model saved: {MODEL}")

    pred = model.predict(test[features])
    mae = mean_absolute_error(test["casos"], pred)
    rmse = float(np.sqrt(mean_squared_error(test["casos"], pred)))

    print("\n=== Model WITH climate (2024) ===")
    comparison = test[["anio", "mes", "casos"]].copy()
    comparison["predicho"] = pred.round(1)
    comparison["error"] = (comparison["casos"] - comparison["predicho"]).round(1)
    print(comparison.to_string(index=False))
    print(f"\nMAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("\nCompare with no-climate baseline: MAE 394.90 | RMSE 708.09")

    PLOT.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(comparison["mes"], comparison["casos"], marker="o", label="Observed 2024")
    plt.plot(
        comparison["mes"],
        comparison["predicho"],
        marker="o",
        label="Predicted 2024 (with climate)",
    )
    plt.title("Dengue Lambayeque 2024: observed vs predicted (with climate)")
    plt.xlabel("Month")
    plt.ylabel("Cases")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT, dpi=150)
    print(f"Plot saved: {PLOT}")


if __name__ == "__main__":
    main()
