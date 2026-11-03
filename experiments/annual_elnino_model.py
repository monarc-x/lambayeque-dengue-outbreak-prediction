"""Annual dengue model per district driven by the coastal El Niño index (ICEN).

Unit: district-year (2001–2025, 37 Lambayeque districts, CDC weekly cases).
Target: log1p(annual cases). Features known before the season starts:
  - icen_season: mean ICEN Jan–Apr of the forecast year (from ENFEN outlook)
  - icen_prev: mean ICEN Jun–Aug of the previous year (already observed)
  - casos_y2, casos_y3: cases two and three years before (Y-1 is not yet published)
  - hist: mean log cases of the district in all earlier years (endemicity)
Validation is expanding-window: to predict year Y, train only on years < Y.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor

BASE = Path(__file__).resolve().parents[1]
CDC = BASE / "data" / "cdc_dengue_surveillance_2000_2025.csv"
ICEN = BASE / "data" / "icen_coastal_el_nino.txt"
FEATURES = ["icen_season", "icen_prev", "log_y2", "log_y3", "hist"]


def load_panel(departments: tuple[str, ...]) -> pd.DataFrame:
    cdc = pd.read_csv(CDC, usecols=["departamento", "distrito", "ano", "ubigeo"], dtype={"ubigeo": str})
    cdc = cdc[cdc["departamento"].isin(departments)]
    cdc["ubigeo"] = cdc["ubigeo"].str.zfill(6)
    names = cdc.drop_duplicates("ubigeo").set_index("ubigeo")[["distrito", "departamento"]]
    idx = pd.MultiIndex.from_product([names.index, range(2000, 2026)], names=["ubigeo", "anio"])
    panel = cdc.groupby(["ubigeo", "ano"]).size().rename("casos")
    panel.index.names = ["ubigeo", "anio"]
    panel = panel.reindex(idx, fill_value=0).reset_index()
    return panel.join(names, on="ubigeo")


def load_icen() -> pd.DataFrame:
    icen = pd.read_csv(ICEN, sep=r"\s+", comment="%", header=None, names=["anio", "mes", "icen"])
    season = icen[icen["mes"] <= 5].groupby("anio")["icen"].max().rename("icen_season")
    prev = icen[icen["mes"].between(6, 8)].groupby("anio")["icen"].mean().rename("icen_prev")
    prev.index = prev.index + 1
    return pd.concat([season, prev], axis=1)


def add_features(panel: pd.DataFrame, icen: pd.DataFrame) -> pd.DataFrame:
    df = panel.sort_values(["ubigeo", "anio"]).copy()
    df["log"] = np.log1p(df["casos"])
    g = df.groupby("ubigeo")["log"]
    df["log_y2"] = g.shift(2)
    df["log_y3"] = g.shift(3)
    df["hist"] = g.transform(lambda s: s.shift(2).expanding().mean())
    return df.merge(icen, left_on="anio", right_index=True, how="left")


def models():
    return {
        "hgb": lambda: HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15,
            monotonic_cst=[1, 1, 0, 0, 1], random_state=42),
        "poisson": lambda: HistGradientBoostingRegressor(
            loss="poisson", max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=15,
            monotonic_cst=[1, 1, 0, 0, 1], random_state=42),
    }


def fit_predict(name, train, test):
    model = models()[name]()
    if name == "poisson":
        model.fit(train[FEATURES], train["casos"])
        return model, model.predict(test[FEATURES])
    model.fit(train[FEATURES], train["log"])
    return model, np.expm1(model.predict(test[FEATURES])).clip(min=0)


def main() -> None:
    import sys
    depts = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("LAMBAYEQUE",)
    print("Departamentos de entrenamiento:", depts)
    df = add_features(load_panel(depts), load_icen()).dropna(subset=FEATURES)
    rows = []
    for year in range(2010, 2026):
        train, test = df[df["anio"] < year], df[df["anio"] == year].copy()
        for name in models():
            _, pred = fit_predict(name, train, test)
            test[f"pred_{name}"] = pred
        rows.append(test)
    res = pd.concat(rows)

    ch = res[res["ubigeo"] == "140101"][["anio", "icen_season", "casos", "pred_hgb", "pred_poisson"]]
    print("=== Chiclayo: pronóstico anual sin ver el año anterior (entrenado solo con años previos) ===")
    print(ch.round(1).to_string(index=False))

    dept = res[res["departamento"] == "LAMBAYEQUE"].groupby("anio")[["casos", "pred_hgb", "pred_poisson"]].sum()
    print("\n=== Departamento (suma de distritos) ===")
    print(dept.round(0).to_string())

    # 2027: Y-1 (2026) cases unpublished; Y-2 = 2025, Y-3 = 2024; ICEN Jun–Aug 2026 observed
    full = add_features(load_panel(depts), load_icen())
    icen = load_icen()
    model, _ = fit_predict("poisson", df, df.head(1))
    last = full[full["anio"] == 2025].set_index("ubigeo")
    prev = full[full["anio"] == 2024].set_index("ubigeo")
    x = pd.DataFrame({
        "icen_prev": icen.loc[2027, "icen_prev"],
        "log_y2": last["log"],
        "log_y3": prev["log"],
        "hist": full[full["anio"] <= 2025].groupby("ubigeo")["log"].mean(),
    })
    x = x.join(last[["distrito", "departamento"]])
    print(f"\n=== 2027 (ICEN jun–ago 2026 = {icen.loc[2027, 'icen_prev']:.2f}) ===")
    for scen in (1.0, 2.0, 3.0, 4.0):
        x["icen_season"] = scen
        x["pred"] = model.predict(x[FEATURES])
        lam = x[x["departamento"] == "LAMBAYEQUE"]
        print(f"ICEN máx ene–may 2027 = {scen}: Chiclayo {x.loc['140101', 'pred']:.0f} | Lambayeque depto {lam['pred'].sum():.0f}")


if __name__ == "__main__":
    main()
