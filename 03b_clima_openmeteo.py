from pathlib import Path
import pandas as pd
import requests

BASE = Path(r"D:\repos\dengue-lambayeque")
CASOS = BASE / "data" / "lambayeque_mensual.csv"
ICEN = BASE / "data" / "ICEN.txt"
OUT = BASE / "data" / "lambayeque_mensual_clima.csv"

# Chiclayo (Lambayeque)
LAT, LON = -6.7714, -79.8409

url = (
    "https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={LAT}&longitude={LON}"
    "&start_date=2015-01-01"
    "&end_date=2024-12-31"
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
    "&timezone=America%2FLima"
)

print("Descargando clima Open-Meteo...")
r = requests.get(url, timeout=60)
r.raise_for_status()
data = r.json()["daily"]

clima = pd.DataFrame({
    "fecha": pd.to_datetime(data["time"]),
    "tmax": data["temperature_2m_max"],
    "tmin": data["temperature_2m_min"],
    "precip": data["precipitation_sum"],
})
clima["anio"] = clima["fecha"].dt.year
clima["mes"] = clima["fecha"].dt.month

clima_m = (
    clima.groupby(["anio", "mes"], as_index=False)
    .agg(precip=("precip", "sum"), tmax=("tmax", "mean"), tmin=("tmin", "mean"))
)

casos = pd.read_csv(CASOS)

icen = pd.read_csv(
    ICEN, sep=r"\s+", comment="%", header=None,
    names=["anio", "mes", "icen"], engine="python",
)

df = (
    casos.merge(clima_m, on=["anio", "mes"], how="left")
         .merge(icen, on=["anio", "mes"], how="left")
         .sort_values(["anio", "mes"])
)

df.to_csv(OUT, index=False)

print("Guardado:", OUT)
print(df.head(12).to_string(index=False))
print("\nNulos por columna:")
print(df.isna().sum())
print("\nFilas:", len(df))