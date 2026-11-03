"""Insert the 14-week schedule (cronograma) before the bibliography of the project report."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

WEEKS = 14
SCHEDULE = [
    (
        "Integrar en un panel anual reproducible los casos de dengue del CDC-MINSA (2000–2025) "
        "de los 212 distritos costeros con el ICEN, y extraer el dataset de José Leonardo Ortiz\n"
        "Producto: panel distrito-año y dataset de JLO",
        [
            ("(i) Descargar la base de vigilancia de dengue del CDC-MINSA, depurarla y conservar los "
             "registros de Tumbes, Piura, Lambayeque, La Libertad, Áncash e Ica.", [1, 2]),
            ("(ii) Agregar los casos por distrito y año (ec. 1) y completar con cero los años sin "
             "notificación.", [2, 3]),
            ("(iii) Descargar la serie mensual del ICEN (IGP), calcular sus indicadores anuales "
             "(ec. 2 y 3) y unirlos al panel.", [3, 4]),
            ("(iv) Extraer el dataset de José Leonardo Ortiz (casos individuales, mensuales y anuales).", [4]),
        ],
    ),
    (
        "Entrenar un modelo HistGradientBoostingRegressor con pérdida Poisson (2003–2023) para "
        "estimar los casos anuales por distrito\n"
        "Producto: modelo entrenado",
        [
            ("(i) Construir las variables de entrada: transformación logarítmica, rezagos a dos y tres "
             "años y endemicidad histórica (ec. 4–6).", [5]),
            ("(ii) Configurar y entrenar el modelo con pérdida Poisson y restricciones monotónicas "
             "(ec. 7–9) con los 212 distritos, periodo 2003–2023.", [6, 7]),
        ],
    ),
    (
        "Evaluar el error del modelo frente a 2024 en José Leonardo Ortiz, implementar el panel web "
        "y pronosticar 2026 y 2027\n"
        "Producto: reporte de validación, pronóstico 2026–2027 y panel web",
        [
            ("(i) Pronosticar los casos de 2024 de José Leonardo Ortiz y calcular el error absoluto y "
             "porcentual (ec. 10 y 11).", [8]),
            ("(ii) Reentrenar el modelo con 2003–2025, pronosticar 2026 (ICEN observado) y 2027 "
             "(escenario ENFEN) y clasificar el riesgo (ec. 12).", [9, 10]),
            ("(iii) Implementar el panel web de apoyo a la decisión con las vistas de pronóstico y "
             "validación.", [10, 11, 12]),
            ("(iv) Redactar el informe final y preparar la sustentación de resultados.", [13, 14]),
        ],
    ),
]


def shade(cell, color: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    fill = OxmlElement("w:shd")
    fill.set(qn("w:val"), "clear")
    fill.set(qn("w:color"), "auto")
    fill.set(qn("w:fill"), color)
    props.append(fill)


def write(cell, text: str, bold: bool = False, center: bool = False, size: float = 8) -> None:
    cell.text = ""
    lines = text.split("\n")
    for k, line in enumerate(lines):
        para = cell.paragraphs[0] if k == 0 else cell.add_paragraph()
        para.paragraph_format.space_after = Pt(0)
        para.paragraph_format.space_before = Pt(0)
        para.paragraph_format.line_spacing = 1.0
        para.paragraph_format.first_line_indent = Cm(0)
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        run = para.add_run(line)
        run.font.size = Pt(size)
        run.bold = bold or (k > 0)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def build_table(doc: Document):
    n_rows = 1 + sum(len(acts) for _, acts in SCHEDULE)
    table = doc.add_table(rows=n_rows, cols=2 + WEEKS)
    table.style = doc.tables[0].style
    widths = [Cm(3.5), Cm(5.3)] + [Cm(0.56)] * WEEKS

    header = ["Objetivo específico / Producto", "Actividades que se realizan para cumplir el objetivo específico"]
    for c, text in enumerate(header + [str(w) for w in range(1, WEEKS + 1)]):
        write(table.cell(0, c), text, bold=True, center=c >= 2, size=7 if c >= 2 else 8)
        shade(table.cell(0, c), "D9E2F3")

    row = 1
    for objective, activities in SCHEDULE:
        first = row
        for text, weeks in activities:
            write(table.cell(row, 1), text)
            for w in range(1, WEEKS + 1):
                cell = table.cell(row, 1 + w)
                write(cell, "X" if w in weeks else "", bold=True, center=True)
                if w in weeks:
                    shade(cell, "BDD7EE")
            row += 1
        merged = table.cell(first, 0).merge(table.cell(row - 1, 0))
        write(merged, objective)

    table.autofit = False
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    table._tbl.tblPr.append(layout)
    margins = OxmlElement("w:tblCellMar")
    for side, width in (("left", Cm(0.06)), ("right", Cm(0.06))):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(width.twips))
        node.set(qn("w:type"), "dxa")
        margins.append(node)
    table._tbl.tblPr.append(margins)
    grid = table._tbl.tblGrid
    for col, width in zip(grid.findall(qn("w:gridCol")), widths):
        col.set(qn("w:w"), str(width.twips))
    for r in table.rows:
        for c, width in enumerate(widths):
            r.cells[c].width = width
    return table


def clone_before(anchor, template, text: str):
    """Insert a copy of `template` (keeping its numbering and fonts) before `anchor`, with new text."""
    element = deepcopy(template._p)
    anchor._p.addprevious(element)
    para = Paragraph(element, anchor._parent)
    for run in para.runs[1:]:
        run._r.getparent().remove(run._r)
    para.runs[0].text = text
    return para


def main() -> None:
    src = Path(sys.argv[1])
    dst = src.with_name(src.stem + "_cronograma.docx")
    doc = Document(src)
    headings = [p for p in doc.paragraphs if p.style.name == "Heading 1"]
    bib = next(p for p in headings if p.text.strip().upper() == "BIBLIOGRAFÍA")
    body = next(p for p in doc.paragraphs if p.style.name == "Normal" and p.text.startswith("El problema se alinea"))
    clone_before(bib, bib, "CRONOGRAMA")
    note = clone_before(
        bib, body, "Duración: 14 semanas (ciclo 2026-2). Las ecuaciones citadas corresponden a la sección 4.5."
    )
    table = build_table(doc)
    note._p.addnext(table._tbl)
    clone_before(bib, body, "")
    doc.save(dst)
    print(dst)


if __name__ == "__main__":
    main()
