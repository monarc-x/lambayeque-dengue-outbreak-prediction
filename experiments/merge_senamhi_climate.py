"""Merge SENAMHI station climate + ICEN with the department case series.

Exploratory alternative to Open-Meteo (see 02_fetch_climate.py).
Uses the Lambayeque city station (formerly SENAMHI code qc00000301).

Requires:
  - data/lambayeque_department_monthly.csv
  - data/senamhi_lambayeque_daily.txt
  - data/icen_coastal_el_nino.txt
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CASES = BASE / "data" / "lambayeque_department_monthly.csv"
SENAMHI = BASE / "data" / "senamhi_lambayeque_daily.txt"
ICEN = BASE / "data" / "icen_coastal_el_nino.txt"
OUT = BASE / "data" / "lambayeque_department_climate.csv"


def main() -> None:
    cases = pd.read_csv(CASES)

    climate = pd.read_csv(
        SENAMHI,
        sep=r"\s+",
        header=None,
        names=["anio", "mes", "dia", "precip", "tmax", "tmin"],
        engine="python",
    )

    # SENAMHI uses -99.9 as missing
    for col in ["precip", "tmax", "tmin"]:
        climate[col] = pd.to_numeric(climate[col], errors="coerce")
        climate.loc[climate[col] <= -99, col] = np.nan

    climate_m = (
        climate.groupby(["anio", "mes"], as_index=False)
        .agg(
            precip=("precip", "sum"),
            tmax=("tmax", "mean"),
            tmin=("tmin", "mean"),
        )
    )

    icen = pd.read_csv(
        ICEN,
        sep=r"\s+",
        comment="%",
        header=None,
        names=["anio", "mes", "icen"],
        engine="python",
    )

    df = (
        cases.merge(climate_m, on=["anio", "mes"], how="left")
        .merge(icen, on=["anio", "mes"], how="left")
        .sort_values(["anio", "mes"])
    )

    df.to_csv(OUT, index=False)
    print(f"Saved: {OUT}")
    print(df.head(12).to_string(index=False))
    print("\nMissing values by column:")
    print(df.isna().sum())
    print(f"\nRows: {len(df)}")


if __name__ == "__main__":
    main()
