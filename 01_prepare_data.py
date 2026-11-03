"""Build a complete monthly dengue case panel by district (ubigeo).

Reads CDC/MINSA open surveillance data, filters to Lambayeque, aggregates
to month × district, and fills missing months with zero cases so lag
features are well-defined for every district.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from district_coords import DISTRICTS

BASE = Path(__file__).resolve().parent
CDC_CSV = BASE / "data" / "cdc_dengue_surveillance_2000_2025.csv"
OUT_DISTRICTS = BASE / "data" / "lambayeque_districts_monthly.csv"
OUT_DEPARTMENT = BASE / "data" / "lambayeque_department_monthly.csv"
OUT_TARGET_CASES = BASE / "data" / "cdc_dengue_chiclayo_2015_2025.csv"
OUT_TARGET_MONTHLY = BASE / "data" / "chiclayo_monthly_2015_2025.csv"

YEAR_START, YEAR_END = 2015, 2025
TARGET_UBIGEO = "140101"  # Chiclayo
TARGET_NAME = "CHICLAYO"
TARGET_TRAIN_END = 2025


def load_cdc_cases(path: Path) -> pd.DataFrame:
    """Load and filter CDC dengue surveillance rows for Lambayeque."""
    print("Reading CDC surveillance CSV...")
    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(
        path,
        sep=sep,
        dtype={"ubigeo": str, "ano": str, "semana": str},
        low_memory=False,
        encoding="utf-8",
    )
    df = df[df["departamento"].str.upper() == "LAMBAYEQUE"].copy()
    print(f"Lambayeque rows: {len(df)}")

    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)
    df["fecha"] = pd.to_datetime(
        df["ano"].astype(str) + df["semana"].astype(str).str.zfill(2) + "1",
        format="%Y%W%w",
        errors="coerce",
    )
    df = df.dropna(subset=["fecha"])
    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    return df[(df["anio"] >= YEAR_START) & (df["anio"] <= YEAR_END)]


def district_metadata() -> pd.DataFrame:
    """Catalog of districts with coordinates and climate-node assignment."""
    rows = [
        {
            "ubigeo": ubigeo,
            "distrito": district,
            "provincia": province,
            "lat": lat,
            "lon": lon,
            "nodo_clima": climate_node,
        }
        for ubigeo, (district, province, lat, lon, climate_node) in DISTRICTS.items()
    ]
    return pd.DataFrame(rows)


def build_panel(cases: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Complete district × month panel with zero-filled missing months."""
    monthly = (
        cases.groupby(["ubigeo", "anio", "mes"], as_index=False)
        .size()
        .rename(columns={"size": "casos"})
    )

    months = pd.MultiIndex.from_product(
        [meta["ubigeo"], range(YEAR_START, YEAR_END + 1), range(1, 13)],
        names=["ubigeo", "anio", "mes"],
    ).to_frame(index=False)

    panel = (
        months.merge(monthly, on=["ubigeo", "anio", "mes"], how="left")
        .merge(meta, on="ubigeo", how="left")
    )
    panel["casos"] = panel["casos"].fillna(0).astype(int)
    panel = panel.sort_values(["provincia", "distrito", "anio", "mes"]).reset_index(drop=True)

    extra = monthly[~monthly["ubigeo"].isin(meta["ubigeo"])]
    if len(extra):
        print(
            "Ubigeos outside catalog (omitted from panel):",
            extra["ubigeo"].unique().tolist(),
        )
    return panel


def export_target_cases(path: Path) -> pd.DataFrame:
    """Case-level extract for the focus district (2015–2024) from the main CDC file."""
    print(f"Extracting {TARGET_NAME} case-level rows (2015–{TARGET_TRAIN_END})...")
    sample = path.read_text(encoding="utf-8", errors="ignore")[:2048]
    sep = ";" if sample.count(";") > sample.count(",") else ","
    df = pd.read_csv(
        path,
        sep=sep,
        dtype={"ubigeo": str, "ano": str, "semana": str},
        low_memory=False,
        encoding="utf-8",
    )
    df["ubigeo"] = df["ubigeo"].astype(str).str.zfill(6)
    df["fecha"] = pd.to_datetime(
        df["ano"].astype(str) + df["semana"].astype(str).str.zfill(2) + "1",
        format="%Y%W%w",
        errors="coerce",
    )
    df = df.dropna(subset=["fecha"])
    df["anio_calendario"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    target = df[
        (df["ubigeo"] == TARGET_UBIGEO)
        & (df["anio_calendario"] >= YEAR_START)
        & (df["anio_calendario"] <= TARGET_TRAIN_END)
    ].copy()
    return target.sort_values(["fecha"]).reset_index(drop=True)


def main() -> None:
    cases = load_cdc_cases(CDC_CSV)
    meta = district_metadata()
    panel = build_panel(cases, meta)

    OUT_DISTRICTS.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_DISTRICTS, index=False)
    print(f"Saved: {OUT_DISTRICTS}")
    print(f"Panel rows: {len(panel)} | Districts: {panel['ubigeo'].nunique()}")
    print(f"Total cases in panel: {int(panel['casos'].sum())}")
    print("\nTop 10 districts by cases (2015–2024):")
    print(
        panel.groupby(["distrito", "provincia"], as_index=False)["casos"]
        .sum()
        .sort_values("casos", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    # Department aggregate kept for legacy / exploratory scripts
    department = (
        panel.groupby(["anio", "mes"], as_index=False)["casos"]
        .sum()
        .sort_values(["anio", "mes"])
    )
    department.to_csv(OUT_DEPARTMENT, index=False)
    print(f"\nAlso updated: {OUT_DEPARTMENT}")
    print(department.groupby("anio")["casos"].sum().to_string())

    # Focus-district extracts (Chiclayo) for the single-district experiment
    target_cases = export_target_cases(CDC_CSV)
    target_cases.to_csv(OUT_TARGET_CASES, index=False)
    print(f"\nSaved {TARGET_NAME} cases: {OUT_TARGET_CASES} ({len(target_cases)} rows)")

    target_monthly = (
        panel[
            (panel["ubigeo"] == TARGET_UBIGEO)
            & (panel["anio"] >= YEAR_START)
            & (panel["anio"] <= TARGET_TRAIN_END)
        ]
        .sort_values(["anio", "mes"])
        .reset_index(drop=True)
    )
    target_monthly.to_csv(OUT_TARGET_MONTHLY, index=False)
    print(
        f"Saved {TARGET_NAME} monthly: {OUT_TARGET_MONTHLY} "
        f"({int(target_monthly['casos'].sum())} cases, 2015–{TARGET_TRAIN_END})"
    )


if __name__ == "__main__":
    main()
