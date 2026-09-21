"""Fetch monthly climate features from Open-Meteo and merge with the district panel.

Source: https://archive-api.open-meteo.com/v1/archive
Variables: precipitation_sum, temperature_2m_max/min, relative_humidity_2m_mean
Coastal El Niño index (ICEN): data/icen_coastal_el_nino.txt (ENFEN)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

from district_coords import CLIMATE_NODES

BASE = Path(__file__).resolve().parent
CASES = BASE / "data" / "lambayeque_districts_monthly.csv"
ICEN = BASE / "data" / "icen_coastal_el_nino.txt"
OUT = BASE / "data" / "lambayeque_districts_climate.csv"
OUT_NODES = BASE / "data" / "climate_nodes_monthly.csv"

START, END = "2015-01-01", "2024-12-31"


def log(message: str) -> None:
    print(message, flush=True)


def fetch_all_nodes() -> pd.DataFrame:
    """Download daily climate for all nodes in one multi-location request."""
    names = list(CLIMATE_NODES.keys())
    latitudes = ",".join(str(CLIMATE_NODES[n][0]) for n in names)
    longitudes = ",".join(str(CLIMATE_NODES[n][1]) for n in names)
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={latitudes}&longitude={longitudes}"
        f"&start_date={START}&end_date={END}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "relative_humidity_2m_mean"
        "&timezone=America%2FLima"
    )
    log(f"Downloading {len(names)} Open-Meteo nodes in one request...")
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    payload = response.json()
    # Multi-location returns a list; single location returns one dict
    if isinstance(payload, dict) and "daily" in payload:
        payload = [payload]

    frames: list[pd.DataFrame] = []
    for name, location in zip(names, payload):
        daily = location["daily"]
        climate = pd.DataFrame(
            {
                "fecha": pd.to_datetime(daily["time"]),
                "tmax": daily["temperature_2m_max"],
                "tmin": daily["temperature_2m_min"],
                "precip": daily["precipitation_sum"],
                "humidity": daily["relative_humidity_2m_mean"],
            }
        )
        climate["anio"] = climate["fecha"].dt.year
        climate["mes"] = climate["fecha"].dt.month
        monthly = (
            climate.groupby(["anio", "mes"], as_index=False)
            .agg(
                precip=("precip", "sum"),
                tmax=("tmax", "mean"),
                tmin=("tmin", "mean"),
                humidity=("humidity", "mean"),
            )
        )
        monthly["nodo_clima"] = name
        frames.append(monthly)
        log(f"  ok {name}: {len(monthly)} months")
    return pd.concat(frames, ignore_index=True)


def load_icen(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="%",
        header=None,
        names=["anio", "mes", "icen"],
        engine="python",
    )


def main() -> None:
    nodes = fetch_all_nodes()
    OUT_NODES.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_csv(OUT_NODES, index=False)
    log(f"Climate nodes saved: {OUT_NODES}")

    cases = pd.read_csv(CASES, dtype={"ubigeo": str})
    cases["ubigeo"] = cases["ubigeo"].astype(str).str.zfill(6)
    icen = load_icen(ICEN)

    panel = (
        cases.merge(nodes, on=["nodo_clima", "anio", "mes"], how="left")
        .merge(icen, on=["anio", "mes"], how="left")
        .sort_values(["ubigeo", "anio", "mes"])
        .reset_index(drop=True)
    )

    panel.to_csv(OUT, index=False)
    log(f"Saved: {OUT}")
    log(f"Rows: {len(panel)}")
    log("Missing climate values:")
    log(str(panel[["precip", "tmax", "tmin", "humidity", "icen"]].isna().sum()))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
