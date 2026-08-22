from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(r"D:\repos\dengue-lambayeque")
DATA = BASE / "data" / "lambayeque_mensual_clima.csv"
MODEL = BASE / "models" / "modelo_dengue_clima.pkl"
PLOT = BASE / "outputs" / "real_vs_pred_2024_clima.png"

df = pd.read_csv(DATA).sort_values(["anio", "mes"]).reset_index(drop=True)

# Variables de entrada (casos anteriores + clima)
df["casos_lag1"] = df["casos"].shift(1)
df["casos_lag2"] = df["casos"].shift(2)
df["precip_lag1"] = df["precip"].shift(1)
df["tmax_lag1"] = df["tmax"].shift(1)
df["icen_lag1"] = df["icen"].shift(1)
df = df.dropna().reset_index(drop=True)

features = ["mes", "casos_lag1", "casos_lag2", "precip_lag1", "tmax_lag1", "icen_lag1"]
target = "casos"

train = df[df["anio"] <= 2023]
test = df[df["anio"] == 2024]

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

print("Train:", len(train), "meses | Test 2024:", len(test), "meses")

modelo = RandomForestRegressor(n_estimators=300, random_state=42)
modelo.fit(X_train, y_train)
joblib.dump(modelo, MODEL)
print("Modelo guardado en:", MODEL)

pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n=== Modelo CON clima (2024) ===")
comp = test[["anio", "mes", "casos"]].copy()
comp["predicho"] = pred.round(1)
comp["error"] = (comp["casos"] - comp["predicho"]).round(1)
print(comp.to_string(index=False))
print(f"\nMAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print("\nComparar con modelo SIN clima: MAE 394.90 | RMSE 708.09")

plt.figure(figsize=(10, 5))
plt.plot(comp["mes"], comp["casos"], marker="o", label="Real 2024")
plt.plot(comp["mes"], comp["predicho"], marker="o", label="Predicho 2024 (con clima)")
plt.title("Dengue Lambayeque 2024: real vs predicho (con clima)")
plt.xlabel("Mes")
plt.ylabel("Casos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT, dpi=150)
print("Gráfico guardado en:", PLOT)