from pathlib import Path
import pandas as pd

BASE = Path(r"D:\repos\dengue-lambayeque")
CSV = BASE / "data" / "datos_abiertos_vigilancia_dengue_2000_2024.csv"
OUT = BASE / "data" / "lambayeque_mensual.csv"

print("Leyendo CSV (puede tardar un poco)...")
df = pd.read_csv(CSV, sep=";")

print("Filas totales:", len(df))
print("Columnas:", list(df.columns))

# Solo Lambayeque
df = df[df["departamento"].str.upper() == "LAMBAYEQUE"].copy()
print("Filas Lambayeque:", len(df))

# Cada fila = 1 caso. Agrupar por año-mes
df["fecha"] = pd.to_datetime(
    df["ano"].astype(str) + df["semana"].astype(str) + "1",
    format="%Y%W%w",
    errors="coerce",
)

mensual = (
    df.dropna(subset=["fecha"])
      .assign(anio=lambda x: x["fecha"].dt.year,
              mes=lambda x: x["fecha"].dt.month)
      .groupby(["anio", "mes"], as_index=False)
      .size()
      .rename(columns={"size": "casos"})
      .sort_values(["anio", "mes"])
)

mensual = mensual[(mensual["anio"] >= 2015) & (mensual["anio"] <= 2024)]
mensual.to_csv(OUT, index=False)

print("\nGuardado en:", OUT)
print(mensual.head(12))
print("\nResumen por año:")
print(mensual.groupby("anio")["casos"].sum())