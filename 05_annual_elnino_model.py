"""Annual dengue model driven by the coastal El Niño index; focus José Leonardo Ortiz.

Unit: district-year, 2000–2025, every district of the north/central coast
(Tumbes, Piura, Lambayeque, La Libertad, Áncash, Ica) from CDC/MINSA open data.
Target: annual cases (HistGradientBoosting with Poisson loss).
Features known before the season:
  - icen_season: max ICEN Jan–May of the forecast year (ENFEN outlook in operation)
  - icen_prev:   mean ICEN Jun–Aug of the previous year (observed)
  - log_y2, log_y3: log cases two and three years before (Y-1 is not yet published)
  - hist: mean log cases of the district up to Y-2 (endemicity)
Validation: expanding window, each year predicted with a model trained on earlier years only.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

BASE = Path(__file__).resolve().parent
CDC = BASE / "data" / "cdc_dengue_surveillance_2000_2025.csv"
ICEN = BASE / "data" / "icen_coastal_el_nino.txt"
COORDS = BASE / "data" / "lambayeque_districts_monthly.csv"
TARGET_DIR = BASE / "data" / "jlo"
MODEL = BASE / "models" / "modelo_anual_elnino.pkl"
OUT_JSON = BASE / "web" / "data" / "dashboard.json"
OUT_DIR = BASE / "outputs" / "forecasts"

DEPARTMENTS = ("TUMBES", "PIURA", "LAMBAYEQUE", "LA LIBERTAD", "ANCASH", "ICA")
TARGET_UBIGEO = "140105"
TARGET_NAME = "JOSÉ LEONARDO ORTIZ"
TEST_YEAR = 2024
FIRST_EVAL_YEAR = 2010
LAST_CASES_YEAR = 2025
FEATURES = ["icen_season", "icen_prev", "log_y2", "log_y3", "hist"]
SCENARIOS_2027 = [
    (1.0, "Niño débil", False),
    (2.0, "Niño fuerte", False),
    (3.0, "Niño extraordinario (pronóstico ENFEN)", True),
]


def load_panel() -> pd.DataFrame:
    cdc = pd.read_csv(CDC, usecols=["departamento", "distrito", "ano", "ubigeo"], dtype={"ubigeo": str})
    cdc = cdc[cdc["departamento"].isin(DEPARTMENTS)]
    cdc["ubigeo"] = cdc["ubigeo"].str.zfill(6)
    names = cdc.drop_duplicates("ubigeo").set_index("ubigeo")[["distrito", "departamento"]]
    idx = pd.MultiIndex.from_product(
        [names.index, range(2000, LAST_CASES_YEAR + 1)], names=["ubigeo", "anio"]
    )
    panel = cdc.groupby(["ubigeo", "ano"]).size().rename("casos")
    panel.index.names = ["ubigeo", "anio"]
    panel = panel.reindex(idx, fill_value=0).reset_index()
    return panel.join(names, on="ubigeo")


def load_icen_monthly() -> pd.DataFrame:
    return pd.read_csv(ICEN, sep=r"\s+", comment="%", header=None, names=["anio", "mes", "icen"]).dropna()


def icen_features(icen: pd.DataFrame) -> pd.DataFrame:
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


def fit_model(train: pd.DataFrame) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="poisson", max_depth=3, learning_rate=0.05, max_iter=300,
        min_samples_leaf=15, monotonic_cst=[1, 1, 0, 0, 1], random_state=42,
    )
    return model.fit(train[FEATURES], train["casos"])


def future_features(df: pd.DataFrame, year: int, icen_season: float, icen_prev: float) -> pd.DataFrame:
    """Features for a year whose cases are not published yet (Y-1 may be missing too)."""
    logs = df.pivot(index="ubigeo", columns="anio", values="log")
    return pd.DataFrame({
        "icen_season": icen_season,
        "icen_prev": icen_prev,
        "log_y2": logs[year - 2],
        "log_y3": logs[year - 3],
        "hist": logs.loc[:, :year - 2].mean(axis=1),
    })


def export_target_data() -> None:
    """Raw CDC rows plus monthly and annual counts for the target district only."""
    cdc = pd.read_csv(CDC, dtype={"ubigeo": str, "localcod": str})
    cdc["ubigeo"] = cdc["ubigeo"].str.zfill(6)
    raw = cdc[cdc["ubigeo"] == TARGET_UBIGEO].sort_values(["ano", "semana"])
    first, last = int(raw["ano"].min()), int(raw["ano"].max())
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(TARGET_DIR / f"cdc_dengue_jlo_{first}_{last}.csv", index=False)

    weeks = pd.to_datetime(
        raw["ano"].astype(str) + raw["semana"].astype(str).str.zfill(2) + "1", format="%Y%W%w"
    )
    monthly = (
        raw.assign(anio=raw["ano"], mes=weeks.dt.month.where(weeks.dt.year == raw["ano"], 12))
        .groupby(["anio", "mes"]).size()
        .reindex(pd.MultiIndex.from_product([range(2000, last + 1), range(1, 13)], names=["anio", "mes"]), fill_value=0)
        .rename("casos").reset_index()
    )
    monthly.to_csv(TARGET_DIR / f"jlo_casos_mensuales_2000_{last}.csv", index=False)
    monthly.groupby("anio", as_index=False)["casos"].sum().to_csv(
        TARGET_DIR / f"jlo_casos_anuales_2000_{last}.csv", index=False
    )
    print(f"JLO dataset: {len(raw)} casos ({first}–{last}) -> {TARGET_DIR}")


def pct_error(real: float, pred: float) -> float:
    return abs(real - pred) / max(real, 1.0) * 100


def main() -> None:
    export_target_data()
    icen_m = load_icen_monthly()
    icen = icen_features(icen_m)
    full = add_features(load_panel(), icen)
    df = full.dropna(subset=FEATURES).reset_index(drop=True)
    print(f"Panel: {full['ubigeo'].nunique()} distritos, {len(df)} distrito-años con features")

    history = []
    for year in range(FIRST_EVAL_YEAR, LAST_CASES_YEAR + 1):
        model = fit_model(df[df["anio"] < year])
        row = df[(df["anio"] == year) & (df["ubigeo"] == TARGET_UBIGEO)].iloc[0]
        pred = float(model.predict(row[FEATURES].to_frame().T.astype(float))[0])
        history.append({
            "anio": year,
            "icen_season": round(float(row["icen_season"]), 2),
            "real": int(row["casos"]),
            "predicho": round(pred, 1),
            "error_pct": round(pct_error(row["casos"], pred), 1),
        })
    hist_df = pd.DataFrame(history)
    print(f"\n=== {TARGET_NAME}: cada año pronosticado sin verlo ===")
    print(hist_df.to_string(index=False))
    validation = next(h for h in history if h["anio"] == TEST_YEAR)

    final_model = fit_model(df)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "features": FEATURES, "departamentos": DEPARTMENTS}, MODEL)

    target_cases = full[full["ubigeo"] == TARGET_UBIGEO].set_index("anio")["casos"]
    p66, p90 = target_cases.quantile(0.66), target_cases.quantile(0.90)

    def risk(v: float) -> str:
        return "alto" if v >= p90 else "medio" if v >= p66 else "bajo"

    def predict_target(year: int, icen_season: float) -> float:
        x = future_features(full, year, icen_season, float(icen.loc[year, "icen_prev"]))
        return float(final_model.predict(x.loc[[TARGET_UBIGEO], FEATURES])[0])

    season_2026 = float(icen.loc[2026, "icen_season"])
    pred_2026 = predict_target(2026, season_2026)
    forecast_2026 = {
        "anio": 2026, "icen_season": round(season_2026, 2),
        "predicho": round(pred_2026, 0), "riesgo": risk(pred_2026),
    }
    scenarios = []
    for value, label, enfen in SCENARIOS_2027:
        pred = predict_target(2027, value)
        scenarios.append({
            "anio": 2027, "icen_season": value, "etiqueta": label, "enfen": enfen,
            "predicho": round(pred, 0), "riesgo": risk(pred),
        })
    print(f"\n2026 (ICEN máx ene–may observado {season_2026:.2f}): {pred_2026:.0f}")
    for s in scenarios:
        print(f"2027 {s['etiqueta']} (ICEN {s['icen_season']}): {s['predicho']:.0f} [{s['riesgo']}]")

    last_icen = icen_m.iloc[-1]
    coords = pd.read_csv(COORDS, dtype={"ubigeo": str})
    coords = coords[coords["ubigeo"].str.zfill(6) == TARGET_UBIGEO].iloc[0]
    payload = {
        "meta": {
            "titulo": f"Vigilancia predictiva de dengue — {TARGET_NAME.title()}",
            "foco": TARGET_NAME,
            "ubigeo": TARGET_UBIGEO,
            "provincia": "CHICLAYO",
            "lat": float(coords["lat"]),
            "lon": float(coords["lon"]),
            "anio_validacion": TEST_YEAR,
            "anio_inicio_entrenamiento": int(df["anio"].min()),
            "ultimo_dato_casos": LAST_CASES_YEAR,
            "icen_ultimo": {"anio": int(last_icen["anio"]), "mes": int(last_icen["mes"]), "valor": float(last_icen["icen"])},
            "departamentos_entrenamiento": list(DEPARTMENTS),
            "n_distritos_entrenamiento": int(full["ubigeo"].nunique()),
            "fuente_casos": "CDC/MINSA datos abiertos vigilancia dengue 2000–2025",
            "fuente_clima": "ICEN (IGP/ENFEN), met.igp.gob.pe/datos/ICEN.txt",
            "modelo": (
                "HistGradientBoosting (pérdida Poisson), casos anuales por distrito; "
                f"entrenado con {full['ubigeo'].nunique()} distritos de la costa norte y centro"
            ),
            "nota": (
                f"Validación {TEST_YEAR}: modelo entrenado solo con años anteriores y con el ICEN observado de ese año. "
                "Los casos 2026 aún no están publicados en datos abiertos; "
                "2027 depende del ICEN que pronostica ENFEN, por eso se presenta como escenarios."
            ),
        },
        "validacion": validation,
        "casos_anuales": [{"anio": int(a), "casos": int(c)} for a, c in target_cases.items()],
        "pronostico_2026": forecast_2026,
        "escenarios_2027": scenarios,
        "icen_reciente": icen_m[icen_m["anio"] >= 2025].astype({"anio": int, "mes": int}).to_dict(orient="records"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hist_df.to_csv(OUT_DIR / "validacion_anual_jose_leonardo_ortiz.csv", index=False)
    print(f"\nDashboard JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
