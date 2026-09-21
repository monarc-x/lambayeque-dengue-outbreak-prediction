# -*- coding: utf-8 -*-
"""Export figures for the research Word document from dashboard.json."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

BASE = Path(__file__).resolve().parents[1]
JSON = BASE / "web" / "data" / "dashboard.json"
OUT = BASE / "outputs" / "figuras_informe"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load():
    return json.loads(JSON.read_text(encoding="utf-8"))


def fig1_departamento(data):
    dep = pd.DataFrame(data["departamento_2024"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(dep["mes"], dep["real"], marker="o", label="Real 2024", color="#1f2937")
    ax.plot(dep["mes"], dep["predicho"], marker="s", label="Predicho 2024", color="#0f766e")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Casos mensuales")
    ax.set_title("Lambayeque 2024: casos reales vs predichos (suma de distritos)")
    ax.set_xticks(range(1, 13))
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = OUT / "fig1_departamento_real_vs_pred_2024.png"
    fig.savefig(path)
    plt.close(fig)
    print("OK", path.name)


def fig2_top_distritos(data, n=6):
    resumen = pd.DataFrame(data["resumen_distritos_2024"]).head(n)
    back = pd.DataFrame(data["backtest_distritos"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharex=True)
    for ax, row in zip(axes.ravel(), resumen.itertuples(index=False)):
        sub = back[back["ubigeo"] == row.ubigeo].sort_values("mes")
        ax.plot(sub["mes"], sub["casos"], marker="o", ms=4, label="Real", color="#1f2937")
        ax.plot(sub["mes"], sub["predicho"], marker="s", ms=4, label="Predicho", color="#0f766e")
        ax.set_title(str(row.distrito).title(), fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(range(1, 13, 2))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Backtest 2024 en los distritos con más casos", y=1.01)
    fig.supxlabel("Mes")
    fig.supylabel("Casos")
    path = OUT / "fig2_top6_distritos_2024.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("OK", path.name)


def fig3_barras_error(data, n=10):
    df = pd.DataFrame(data["resumen_distritos_2024"]).head(n).copy()
    df["distrito"] = df["distrito"].str.title()
    df = df.sort_values("casos_real")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    y = np.arange(len(df))
    ax.barh(y - 0.2, df["casos_real"], height=0.4, label="Casos reales 2024", color="#334155")
    ax.barh(y + 0.2, df["casos_pred"], height=0.4, label="Suma predicha 2024", color="#0f766e")
    ax.set_yticks(y)
    ax.set_yticklabels(df["distrito"])
    ax.set_xlabel("Casos anuales (2024)")
    ax.set_title("Comparación anual real vs predicha — principales distritos")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    path = OUT / "fig3_barras_distritos_anual_2024.png"
    fig.savefig(path)
    plt.close(fig)
    print("OK", path.name)


def fig4_mae_distritos(data, n=12):
    df = pd.DataFrame(data["resumen_distritos_2024"]).head(n).copy()
    df["distrito"] = df["distrito"].str.title()
    df = df.sort_values("mae")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(df["distrito"], df["mae"], color="#b45309")
    ax.set_xlabel("MAE mensual 2024 (casos)")
    ax.set_title("Error absoluto medio por distrito (principales por carga)")
    ax.grid(True, axis="x", alpha=0.3)
    path = OUT / "fig4_mae_por_distrito.png"
    fig.savefig(path)
    plt.close(fig)
    print("OK", path.name)


def fig5_mapa_scatter(data):
    m = pd.DataFrame(data["mapa_pronostico_h1"])
    colors = {"alto": "#b91c1c", "medio": "#c2410c", "bajo": "#0f766e"}
    fig, ax = plt.subplots(figsize=(7.5, 7))
    for riesgo, g in m.groupby("riesgo"):
        sizes = 40 + np.sqrt(g["predicho"].clip(lower=0)) * 18
        ax.scatter(
            g["lon"], g["lat"],
            s=sizes,
            c=colors.get(riesgo, "#666"),
            label=riesgo.capitalize(),
            alpha=0.85,
            edgecolors="white",
            linewidths=0.6,
        )
    # labels for top predictions
    top = m.nlargest(6, "predicho")
    for r in top.itertuples(index=False):
        ax.annotate(
            str(r.distrito).title(),
            (r.lon, r.lat),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
        )
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title("Mapa de riesgo distrital — pronóstico mes siguiente (h+1)")
    ax.legend(title="Riesgo")
    ax.grid(True, alpha=0.25)
    path = OUT / "fig5_mapa_riesgo_h1.png"
    fig.savefig(path)
    plt.close(fig)
    print("OK", path.name)


def fig6_ranking(data, n=10):
    df = pd.DataFrame(data["ranking_accion"]).head(n).copy()
    df["distrito"] = df["distrito"].str.title()
    df = df.sort_values("pred_promedio")
    colors = df["riesgo"].map({"alto": "#b91c1c", "medio": "#c2410c", "bajo": "#0f766e"})
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(df["distrito"], df["pred_promedio"], color=colors)
    ax.set_xlabel("Predicción promedio (horizonte 3 meses)")
    ax.set_title("Ranking de prioridad distrital (panel operativo)")
    ax.grid(True, axis="x", alpha=0.3)
    path = OUT / "fig6_ranking_prioridad.png"
    fig.savefig(path)
    plt.close(fig)
    print("OK", path.name)


def main():
    data = load()
    fig1_departamento(data)
    fig2_top_distritos(data)
    fig3_barras_error(data)
    fig4_mae_distritos(data)
    fig5_mapa_scatter(data)
    fig6_ranking(data)
    print("Carpeta:", OUT)


if __name__ == "__main__":
    main()
