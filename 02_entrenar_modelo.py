from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE = Path(r"D:\repos\dengue-lambayeque")
DATA = BASE / "data" / "lambayeque_mensual.csv"
MODEL = BASE / "models" / "modelo_dengue.pkl"
PLOT = BASE / "outputs" / "real_vs_pred_2024.png"

df = pd.read_csv(DATA).sort_values(["anio", "mes"]).reset_index(drop=True)

# Features simples: mes + casos de meses anteriores
df["casos_lag1"] = df["casos"].shift(1)
df["casos_lag2"] = df["casos"].shift(2)
df["casos_lag3"] = df["casos"].shift(3)
df = df.dropna().reset_index(drop=True)

features = ["mes", "casos_lag1", "casos_lag2", "casos_lag3"]
target = "casos"

train = df[df["anio"] <= 2023].copy()
test = df[df["anio"] == 2024].copy()

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

print("Train:", len(train), "meses | Test 2024:", len(test), "meses")

# AQUÍ NACÉ EL MODELO
modelo = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
)
modelo.fit(X_train, y_train)

# Guardar modelo
joblib.dump(modelo, MODEL)
print("Modelo guardado en:", MODEL)

# Predecir 2024 y comparar con real
pred = modelo.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n=== Comparación 2024 ===")
comparacion = test[["anio", "mes", "casos"]].copy()
comparacion["predicho"] = pred.round(1)
comparacion["error"] = (comparacion["casos"] - comparacion["predicho"]).round(1)
print(comparacion.to_string(index=False))
print(f"\nMAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Gráfico real vs predicho
plt.figure(figsize=(10, 5))
plt.plot(comparacion["mes"], comparacion["casos"], marker="o", label="Real 2024")
plt.plot(comparacion["mes"], comparacion["predicho"], marker="o", label="Predicho 2024")
plt.title("Dengue Lambayeque 2024: real vs predicho")
plt.xlabel("Mes")
plt.ylabel("Casos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT, dpi=150)
print("Gráfico guardado en:", PLOT)