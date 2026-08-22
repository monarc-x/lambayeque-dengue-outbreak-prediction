from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(r"D:\repos\dengue-lambayeque")
CASOS = BASE / "data" / "lambayeque_mensual.csv"
SENAMHI = BASE / "data" / "qc00000301.txt"
ICEN = BASE / "data" / "ICEN.txt"
OUT = BASE / "data" / "lambayeque_mensual_clima.csv"

# 1) Casos
casos = pd.read_csv(CASOS)

# 2) SENAMHI diario -> mensual
clima = pd.read_csv(
    SENAMHI,
    sep=r"\s+",
    header=None,
    names=["anio", "mes", "dia", "precip", "tmax", "tmin"],
    engine="python",
)

# -99.9 = faltante en SENAMHI
for col in ["precip", "tmax", "tmin"]:
    clima[col] = pd.to_numeric(clima[col], errors="coerce")
    clima.loc[clima[col] <= -99, col] = np.nan

clima_m = (
    clima.groupby(["anio", "mes"], as_index=False)
    .agg(
        precip=("precip", "sum"),
        tmax=("tmax", "mean"),
        tmin=("tmin", "mean"),
    )
)

# 3) ICEN
icen = pd.read_csv(
    ICEN,
    sep=r"\s+",
    comment="%",
    header=None,
    names=["anio", "mes", "icen"],
    engine="python",
)

# 4) Unir
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