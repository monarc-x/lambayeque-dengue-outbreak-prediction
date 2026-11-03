"""Insert the RESULTADOS section (text, tables and figures) before the CRONOGRAMA of the report.

Run after add_cronograma_docx.py; the document passed is modified in place.
"""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt
from docx.text.paragraph import Paragraph

BASE = Path(__file__).resolve().parent.parent
DASHBOARD = BASE / "web" / "data" / "dashboard.json"
FIG_DIR = BASE / "outputs" / "figures"


def fmt(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ")


def figure_validation(v: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    bars = ax.bar(["Casos reales", "Casos pronosticados"], [v["real"], v["predicho"]],
                  color=["#4472C4", "#ED7D31"], width=0.5)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15, fmt(bar.get_height()),
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Casos de dengue")
    ax.set_ylim(0, max(v["real"], v["predicho"]) * 1.18)
    ax.set_title(f"José Leonardo Ortiz, {v['anio']} (error {v['error_pct']} %)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def figure_forecast(data: dict, p66: float, p90: float, path: Path) -> None:
    hist = pd.DataFrame(data["casos_anuales"])
    f26 = data["pronostico_2026"]
    f27 = next(s for s in data["escenarios_2027"] if s["enfen"])
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    ax.bar(hist["anio"], hist["casos"], color="#A5A5A5", label="Casos notificados (CDC-MINSA)")
    ax.bar([2026, 2027], [f26["predicho"], f27["predicho"]], color="#ED7D31", hatch="//",
           edgecolor="white", label="Pronóstico del modelo")
    for year, value in [(2023, 2801), (2024, 736), (2026, f26["predicho"]), (2027, f27["predicho"])]:
        ax.text(year, value + 40, fmt(value), ha="center", va="bottom", fontsize=8)
    ax.axhline(p90, color="#C00000", ls="--", lw=1, label=f"Umbral de riesgo alto (P90 = {p90:.0f})")
    ax.axhline(p66, color="#BF9000", ls=":", lw=1, label=f"Umbral de riesgo medio (P66 = {p66:.0f})")
    ax.set_xticks(range(2000, 2028, 3))
    ax.set_ylabel("Casos de dengue")
    ax.set_ylim(0, 3100)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


class Writer:
    """Inserts paragraphs, tables and images before `anchor`, copying formats from the document."""

    def __init__(self, doc: Document, anchor: Paragraph, heading: Paragraph, subheading: Paragraph,
                 body: Paragraph):
        self.doc, self.anchor = doc, anchor
        self.heading, self.subheading, self.body = heading, subheading, body

    def _clone(self, template: Paragraph) -> Paragraph:
        element = deepcopy(template._p)
        for run in element.findall(qn("w:r")):
            element.remove(run)
        self.anchor._p.addprevious(element)
        return Paragraph(element, self.anchor._parent)

    def heading1(self, text: str) -> None:
        run = self._clone(self.heading).add_run(text)
        template_props = self.heading.runs[0]._r.rPr
        if template_props is not None:
            run._r.insert(0, deepcopy(template_props))

    def heading2(self, text: str) -> None:
        self._clone(self.subheading).add_run(text).bold = True

    def para(self, text: str) -> Paragraph:
        para = self._clone(self.body)
        para.add_run(text)
        return para

    def caption(self, label: str, title: str) -> None:
        para = self._clone(self.body)
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        para.paragraph_format.space_after = Pt(2)
        para.paragraph_format.keep_with_next = True
        para.add_run(label + ". ").bold = True
        para.add_run(title).italic = True

    def source(self) -> None:
        para = self._clone(self.body)
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run = para.add_run("Fuente: elaboración propia con datos del CDC-MINSA y del IGP.")
        run.font.size = Pt(9)

    def image(self, path: Path, width_cm: float) -> None:
        para = self._clone(self.body)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para.paragraph_format.first_line_indent = Cm(0)
        para.paragraph_format.keep_with_next = True
        para.add_run().add_picture(str(path), width=Cm(width_cm))

    def table(self, rows: list[list[str]], widths: list[float], highlight: int | None = None) -> None:
        table = self.doc.add_table(rows=len(rows), cols=len(rows[0]))
        table.style = self.doc.tables[0].style
        table.autofit = False
        for r, values in enumerate(rows):
            for c, text in enumerate(values):
                cell = table.cell(r, c)
                cell.width = Cm(widths[c])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                para.paragraph_format.line_spacing = 1.0
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT if c == 0 else WD_ALIGN_PARAGRAPH.CENTER
                run = para.add_run(text)
                run.font.size = Pt(10)
                run.bold = r == 0 or r == highlight
                if r == 0:
                    shade(cell, "D9E2F3")
        for col, width in zip(table._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            col.set(qn("w:w"), str(Cm(width).twips))
        self.anchor._p.addprevious(table._tbl)
        self.para("")


def shade(cell, color: str) -> None:
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), color)
    cell._tc.get_or_add_tcPr().append(fill)


def main() -> None:
    path = Path(sys.argv[1])
    doc = Document(path)
    paragraphs = doc.paragraphs
    headings = [p for p in paragraphs if p.style.name == "Heading 1"]
    if any(p.text.strip().upper() == "RESULTADOS" for p in headings):
        sys.exit("El documento ya tiene la sección RESULTADOS.")
    crono = next(p for p in headings if p.text.strip().upper() == "CRONOGRAMA")
    subheading = next(p for p in paragraphs if p.text.startswith("4.1. Enfoque"))
    body = next(p for p in paragraphs if p.text.startswith("La investigación tiene enfoque cuantitativo"))
    for stray in [p for p in paragraphs if p.text.strip() == "."]:
        stray._p.getparent().remove(stray._p)

    data = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    v, f26 = data["validacion"], data["pronostico_2026"]
    f27 = next(s for s in data["escenarios_2027"] if s["enfen"])
    others = [s for s in data["escenarios_2027"] if not s["enfen"]]
    cases = pd.Series({d["anio"]: d["casos"] for d in data["casos_anuales"]})
    p66, p90 = cases.quantile(0.66), cases.quantile(0.90)
    n_dist = data["meta"]["n_distritos_entrenamiento"]
    years = len(cases)
    abs_err = abs(v["real"] - v["predicho"])
    jlo_total = int(cases.sum())
    pre_2023_max = int(cases.loc[:2022].max())

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1, fig2 = FIG_DIR / "resultados_validacion_2024.png", FIG_DIR / "resultados_pronostico_2026_2027.png"
    figure_validation(v, fig1)
    figure_forecast(data, p66, p90, fig2)

    w = Writer(doc, crono, crono, subheading, body)
    w.heading1("RESULTADOS")

    w.heading2("5.1. Integración de datos")
    w.para(
        f"Se obtuvo un panel anual de {n_dist} distritos de Tumbes, Piura, Lambayeque, La Libertad, Áncash e "
        f"Ica para el periodo 2000–2025, con {fmt(n_dist * years)} registros distrito-año, a los que se unieron "
        "los dos indicadores anuales del ICEN (ecuaciones 2 y 3). La serie del ICEN se actualizó hasta agosto "
        f"de 2026 ({data['meta']['icen_ultimo']['valor']:.2f} °C). El dataset de José Leonardo Ortiz reúne "
        f"{fmt(jlo_total)} casos notificados entre 2001 y 2025, en tres archivos: casos individuales, "
        "mensuales y anuales."
    )
    w.para(
        f"La serie del distrito muestra que, hasta 2022, ningún año superó los {pre_2023_max} casos; en "
        f"2023 se notificaron {fmt(cases[2023])} casos y en 2024, {fmt(cases[2024])}, ambos con El Niño "
        f"costero activo, mientras que en 2025 bajaron a {fmt(cases[2025])}. Esto confirma la relación entre "
        "el calentamiento del mar y los brotes que motiva la investigación."
    )

    w.heading2("5.2. Entrenamiento del modelo")
    w.para(
        f"El modelo HistGradientBoostingRegressor con pérdida Poisson se entrenó con {fmt(n_dist * 21)} "
        f"registros distrito-año (2003–2023) para la validación, y con {fmt(n_dist * 23)} registros "
        "(2003–2025) para el pronóstico final. Se usaron 300 iteraciones, tasa de aprendizaje 0.05, "
        "profundidad máxima 3 y un mínimo de 15 registros por hoja, con restricciones monotónicas crecientes "
        "para los dos indicadores del ICEN y la endemicidad histórica (ecuación 9). El modelo entrenado se "
        "guardó en el archivo modelo_anual_elnino.pkl, que usa el panel web."
    )

    w.heading2("5.3. Validación con los casos de 2024")
    w.para(
        f"Con el modelo entrenado solo hasta 2023 y el ICEN observado de la temporada 2024 "
        f"({v['icen_season']:.2f} °C), se pronosticaron {fmt(v['predicho'])} casos para José Leonardo Ortiz, "
        f"frente a {fmt(v['real'])} casos reales. El modelo sobreestimó en {fmt(abs_err)} casos, con un error "
        f"porcentual de {v['error_pct']} % (ecuaciones 10 y 11), como se muestra en la Tabla 1 y la Figura 1."
    )
    w.caption("Tabla 1", "Resultados de la validación del modelo en José Leonardo Ortiz, 2024")
    w.table([
        ["Indicador", "Valor"],
        ["Periodo de entrenamiento", "2003–2023"],
        ["Año validado", str(v["anio"])],
        ["ICEN de la temporada (máximo enero–mayo)", f"{v['icen_season']:.2f} °C"],
        ["Casos reales", fmt(v["real"])],
        ["Casos pronosticados", fmt(v["predicho"])],
        ["Error absoluto", f"{fmt(abs_err)} casos"],
        ["Error porcentual", f"{v['error_pct']} %"],
    ], [8.5, 5.0], highlight=7)
    w.caption("Figura 1", "Casos reales y pronosticados de dengue en José Leonardo Ortiz, 2024")
    w.image(fig1, 9.5)
    w.source()

    w.heading2("5.4. Pronóstico 2026 y 2027")
    w.para(
        "Con el modelo reentrenado hasta 2025 se estimaron los casos de 2026, con el ICEN ya observado de "
        "enero a mayo, y los de 2027, con el ICEN de 3.0 °C que corresponde al evento extraordinario que "
        f"pronostica el ENFEN. Según los umbrales del distrito (P66 = {p66:.0f} y P90 = {p90:.0f} casos "
        f"anuales, ecuación 12), ambos años se clasifican en riesgo alto (Tabla 2 y Figura 2)."
    )
    w.caption("Tabla 2", "Pronóstico anual de casos de dengue en José Leonardo Ortiz")
    w.table([
        ["Año", "ICEN de la temporada", "Origen del ICEN", "Casos pronosticados", "Riesgo"],
        ["2026", f"{f26['icen_season']:.2f} °C", "Observado (IGP)", fmt(f26["predicho"]), f26["riesgo"].capitalize()],
        ["2027", f"{f27['icen_season']:.1f} °C", "Pronóstico ENFEN", fmt(f27["predicho"]), f27["riesgo"].capitalize()],
    ], [1.6, 3.4, 3.6, 3.4, 2.0])
    w.caption("Figura 2", "Casos anuales notificados (2000–2025) y pronóstico (2026–2027) en José Leonardo Ortiz")
    w.image(fig2, 15.5)
    w.source()
    w.para(
        "El pronóstico de 2027 depende de la intensidad que alcance El Niño: con un ICEN de "
        f"{others[0]['icen_season']:.1f} °C (Niño débil) el modelo estima {fmt(others[0]['predicho'])} casos "
        f"y con {others[1]['icen_season']:.1f} °C (Niño fuerte), {fmt(others[1]['predicho'])} casos. Por eso, "
        "el pronóstico debe actualizarse con cada comunicado oficial del ENFEN."
    )

    w.heading2("5.5. Panel web de apoyo a la decisión")
    w.para(
        "Se implementó un panel web con dos vistas. La vista Pronóstico 2026–2027 muestra los casos estimados, "
        "el nivel de riesgo, el último valor del ICEN y la ubicación del distrito en un mapa. La vista "
        "Validación 2024 muestra el periodo de entrenamiento, los casos reales frente a los pronosticados y el "
        "error del modelo. El panel lee los resultados del modelo desde un archivo JSON, por lo que se "
        "actualiza al volver a ejecutar el pipeline con datos nuevos. El código fuente, los datos del distrito "
        "y el panel están disponibles en el repositorio "
        "https://github.com/monarc-x/lambayeque-dengue-outbreak-prediction."
    )

    w.heading2("5.6. Contrastación de la hipótesis y limitaciones")
    w.para(
        f"El error porcentual obtenido en 2024 ({v['error_pct']} %) es menor al 20 % planteado en la "
        "hipótesis, por lo que esta se acepta: el modelo entrenado con los casos históricos de la costa "
        "peruana y el ICEN estima el número anual de casos de José Leonardo Ortiz con un error aceptable y "
        "permite proyectar un riesgo alto para 2026 y 2027."
    )
    w.para(
        "Los resultados tienen limitaciones. La validación corresponde a un solo año, por lo que el error "
        "puede variar en otros años. La magnitud de la epidemia de 2023 no tenía precedentes en la serie del "
        "distrito y el modelo no la habría anticipado. Además, el pronóstico de 2027 depende del pronóstico "
        "del ENFEN, y los datos de vigilancia pueden presentar subregistro de casos."
    )
    crono.paragraph_format.page_break_before = True

    doc.save(path)
    print(path)


if __name__ == "__main__":
    main()
