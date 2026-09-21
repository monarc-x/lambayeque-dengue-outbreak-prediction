# -*- coding: utf-8 -*-
"""Actualiza Contreras_Campos_API2: objetivos, estado del arte, metodología y cronograma."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph

SRC = Path(r"c:\Users\jorgi\Downloads\Contreras_Campos_API2.docx")
OUT = Path(r"c:\Users\jorgi\Downloads\Contreras_Campos_API2_actualizado.docx")


def set_run_font(run, size=11, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def fill_paragraph(p, text, *, bold=False, italic=False, size=11, justify=True):
    p.clear()
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    if justify and not bold:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def delete_paragraph(paragraph):
    el = paragraph._element
    parent = el.getparent()
    if parent is not None:
        parent.remove(el)


def insert_paragraph_after(paragraph, text="", *, style=None, bold=False, justify=True):
    new_el = OxmlElement("w:p")
    paragraph._element.addnext(new_el)
    p = Paragraph(new_el, paragraph._parent)
    if style is not None:
        p.style = style
    if text:
        run = p.add_run(text)
        set_run_font(run, bold=bold)
        if justify and not bold:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    return p


def remove_table(table):
    tbl = table._tbl
    parent = tbl.getparent()
    if parent is not None:
        parent.remove(tbl)


def set_cell_text(cell, text, *, bold=False, size=8, center=True):
    cell.text = ""
    p = cell.paragraphs[0]
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold)
    if center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def find_para(doc, startswith: str):
    for p in doc.paragraphs:
        if p.text.strip().startswith(startswith):
            return p
    raise KeyError(startswith)


def delete_between(doc, start_heading: str, end_heading: str, keep_start=True):
    """Delete paragraphs strictly between two Heading 1 sections."""
    paras = list(doc.paragraphs)
    i_start = next(i for i, p in enumerate(paras) if p.text.strip().startswith(start_heading))
    i_end = next(i for i, p in enumerate(paras) if p.text.strip().startswith(end_heading))
    # delete from i_start+1 to i_end-1
    for p in paras[i_start + 1 : i_end]:
        delete_paragraph(p)
    return find_para(doc, start_heading)


def add_body_after(anchor, texts, style="Normal"):
    last = anchor
    for t in texts:
        if isinstance(t, tuple):
            kind, content = t
            if kind == "h2":
                last = insert_paragraph_after(last, content, style="List Paragraph", bold=True, justify=False)
            elif kind == "p":
                last = insert_paragraph_after(last, content, style=style, justify=True)
            elif kind == "blank":
                last = insert_paragraph_after(last, "", style=style, justify=False)
        else:
            last = insert_paragraph_after(last, t, style=style, justify=True)
    return last


ESTADO_ARTE = [
    (
        "p",
        "La predicción temprana de dengue se ha consolidado como un problema interdisciplinario "
        "en la intersección de la epidemiología, la climatología y la ciencia de datos. En contextos "
        "tropicales, la incidencia responde a condiciones ambientales que favorecen al vector Aedes "
        "aegypti (temperatura, precipitación, humedad) y a forzantes de mayor escala como El Niño, "
        "al tiempo que la vigilancia sanitaria genera series temporales irregulares, con subregistro "
        "y cambios de régimen entre años epidémicos y no epidémicos. El estado del arte reciente "
        "muestra dos tendencias complementarias: (i) modelos climáticos–espaciales orientados a "
        "explicar y anticipar brotes con horizonte corto, y (ii) benchmarks de machine learning y "
        "deep learning sobre datos abiertos, con énfasis creciente en prototipos desplegables para "
        "autoridades locales.",
    ),
    ("blank", ""),
    ("h2", "Modelado climático y pronóstico en departamentos endémicos del Perú"),
    (
        "p",
        "Mills y Donnelly [1] desarrollaron un marco espaciotemporal basado en clima para modelar "
        "la incidencia mensual de dengue en Piura, Tumbes y Lambayeque (2010–2021). Caracterizaron "
        "efectos no lineales y rezagados de la temperatura máxima media, precipitación extrema, "
        "sequía sostenida e indicadores peruanos asociados a El Niño–Oscilación del Sur. Con un "
        "horizonte de un mes (2018–2021), reportaron clasificación correcta del 100 % de brotes con "
        "tasa de incidencia ≥ 50 (o ≥ 150) por 100 000 habitantes, con tasas de falsa alarma de 0,12 "
        "y 0,05 respectivamente. Su aporte es doble: evidencia sólida de que el clima aporta señal "
        "predictiva en la costa norte peruana —incluido Lambayeque— y una orientación explícita "
        "hacia sistemas de alerta temprana y asignación de control vectorial. No obstante, el "
        "enfoque es departamental/agregado, con fuerte componente estadístico bayesiano, y no "
        "constituye un prototipo informático de bajo costo orientado a distritos ni a una interfaz "
        "operativa local reproducible con datos abiertos posteriores a 2021.",
    ),
    ("blank", ""),
    ("h2", "Series temporales univariadas y proyecciones multi-anuales con Prophet"),
    (
        "p",
        "Rojas-Paucar et al. [2] aplicaron Prophet a notificaciones MINSA 2000–2024 para departamentos "
        "clasificados con “brote explosivo”, entrenando hasta 2023, validando en 2024 (RMSE, MAE y "
        "MAPE) y proyectando 2025–2027. En Lambayeque, Callao y Tumbes el ajuste fue relativamente "
        "favorable; en La Libertad y Lima subestimaron de forma grave (> 16 000 casos de error), "
        "mientras que en Madre de Dios y Ucayali sobreestimaron a partir de extremos históricos. "
        "Las proyecciones 2025–2027 sugieren incrementos anuales moderados (~10–20 %) bajo ciclos "
        "similares a periodos previos, pero los autores advierten que un modelo univariado no "
        "captura bien brotes repentinos. Este trabajo refuerza la utilidad de validar contra 2024 "
        "y de estimar periodos posteriores, al tiempo que deja abierto el vacío de integrar "
        "covariables climáticas y una desagregación espacial más fina para Lambayeque.",
    ),
    ("blank", ""),
    ("h2", "Benchmarks abiertos: ARIMA, Prophet, LSTM y BiLSTM (DengueAI)"),
    (
        "p",
        "Mayta Quispe y Alemán Gonzales [3] presentaron DengueAI, un marco de código abierto que "
        "compara familias clásicas de series temporales con arquitecturas de deep learning usando "
        "más de 25 años de datos abiertos nacionales del Perú (notificaciones semanales 1999–2024 "
        "enriquecidas con covariables meteorológicas). Evaluaron ARIMA, Prophet, LSTM y BiLSTM. "
        "En horizontes cortos (1–4 semanas), la mejor configuración ARIMA alcanzó R² = 0,9707 y "
        "MAE = 88,7 casos, superando a las redes neuronales en 5–8 %; las LSTM capturaron mejor "
        "tendencias no lineales de mediano plazo cuando se incluyeron múltiples variables climáticas. "
        "El sistema se desplegó en un dashboard Streamlit con pronósticos, mapas de calor y "
        "artefactos descargables para oficinas de salud. La conclusión central —que baselines "
        "estadísticos bien ajustados pueden igualar o superar a deep learning en escenarios con "
        "datos escasos o ruidosos, mientras que estrategias híbridas son pragmáticas para alerta "
        "temprana— es especialmente relevante para un proyecto universitario local: prioriza "
        "reproducibilidad, costo computacional moderado y utilidad operativa frente a la sola "
        "búsqueda de complejidad algorítmica.",
    ),
    ("blank", ""),
    ("h2", "Síntesis, vacío y justificación del presente proyecto"),
    (
        "p",
        "En conjunto, la literatura reciente confirma que: (a) el clima y El Niño aportan "
        "información predictiva en la costa norte peruana [1]; (b) es viable y pertinente validar "
        "modelos con 2024 y proyectar periodos posteriores, aunque los enfoques solo univariados "
        "fallan ante extremos [2]; y (c) en datos abiertos peruanos, modelos clásicos o ensembles "
        "bien calibrados pueden ser competitivos frente a deep learning, siempre que el producto "
        "final sea usable por personal de salud [3]. Persiste, sin embargo, un problema abierto "
        "para Lambayeque: disponer de un prototipo informático reproducible, de bajo costo, que "
        "integre vigilancia CDC/MINSA con clima abierto (p. ej. Open-Meteo e ICEN), opere a "
        "escala distrital —donde se decide el control vectorial—, cuantifique error (MAE/RMSE) "
        "en 2024 y deje explícita la etapa de estimación de periodos posteriores para alerta "
        "temprana. Ese vacío justifica el presente proyecto de investigación en Ingeniería de "
        "Sistemas e Informática, alineado con aplicaciones TIC para la salud y computación "
        "científica.",
    ),
]

METODOLOGIA = [
    (
        "p",
        "El proyecto adopta un enfoque cuantitativo, aplicado y experimental-computacional "
        "basado en datos secundarios abiertos (sin recolección primaria ni encuesta a personas). "
        "La unidad de análisis es el distrito–mes (ubigeo) del departamento de Lambayeque, en el "
        "periodo 2015–2024. El desarrollo se organiza en cuatro etapas asociadas a los objetivos "
        "específicos: (I) integración del panel epidemiológico–climático, (II) entrenamiento del "
        "modelo de machine learning, (III) evaluación en 2024 y panel de visualización, y "
        "(IV) estimación de periodos posteriores (pronóstico operativo). Cada etapa se describe a continuación.",
    ),
    ("blank", ""),
    ("h2", "I. Integración de casos mensuales por distrito y variables climáticas (OE1)"),
    (
        "p",
        "Se utilizarán los datos abiertos de vigilancia de dengue del CDC-MINSA "
        "(notificaciones con departamento, provincia, distrito, ubigeo, año y semana epidemiológica). "
        "Tras filtrar Lambayeque, cada registro se interpretará como un caso y se agregará a conteos "
        "mensuales por ubigeo. Se construirá un panel completo distrito × año × mes (2015–2024), "
        "imputando con cero los meses sin notificación para no romper los rezagos temporales. "
        "Las variables climáticas mensuales (precipitación acumulada, temperatura máxima y mínima "
        "medias, humedad relativa media) se obtendrán mediante la API Archive de Open-Meteo en "
        "nodos geográficos representativos del departamento, asignados a cada distrito por proximidad; "
        "el índice costero El Niño (ICEN, ENFEN) se unirá por año–mes. El producto de esta etapa "
        "es un conjunto de datos mensual reproducible (panel distrital con clima e ICEN), versionado "
        "en el repositorio del proyecto.",
    ),
    ("blank", ""),
    ("h2", "II. Ingeniería de características y entrenamiento del modelo (OE2)"),
    (
        "p",
        "Sobre el panel se generarán, por distrito, rezagos de casos (1–3 meses), media móvil de "
        "casos, rezagos de precipitación, temperatura, humedad e ICEN, además de codificación "
        "estacional del mes (seno/coseno) e identificadores de provincia y distrito. La variable "
        "objetivo será log1p(casos) para estabilizar la escala de brotes extremos. El modelo "
        "principal será un HistGradientBoostingRegressor (ensemble de gradiente), entrenado con "
        "meses de anio ≤ 2023; como baseline de control se mantendrá un Random Forest a escala "
        "departamental o con el mismo panel, a fin de contrastar desempeño. Los hiperparámetros "
        "se fijarán de forma conservadora (profundidad, tasa de aprendizaje, regularización) para "
        "reducir sobreajuste en series cortas. El producto será el modelo serializado y la "
        "especificación de features utilizada en inferencia.",
    ),
    ("blank", ""),
    ("h2", "III. Evaluación frente a 2024 y panel para autoridades (OE3 – validación)"),
    (
        "p",
        "La evaluación usará 2024 como hold-out temporal: predicción un mes adelante con rezagos "
        "observados. Se reportarán MAE y RMSE a nivel distrito y, por agregación, a nivel "
        "departamento, junto con tablas y gráficos real vs. predicho. Los resultados se "
        "expondrán en un panel web local (mapa de riesgo, ranking de distritos prioritarios y "
        "detalle filtrable) orientado a personal de vigilancia y control. Esta etapa cierra el "
        "ciclo de validación cuantitativa y de comunicación de resultados.",
    ),
    ("blank", ""),
    ("h2", "IV. Estimación de periodos posteriores / pronóstico operativo (OE3 – proyección)"),
    (
        "p",
        "Como etapa final del objetivo específico 3 —prevista en el diseño, aunque su ejecución "
        "completa se programará al cierre del ciclo— se utilizará el modelo validado para "
        "estimar casos mensuales de periodos posteriores a 2024 mediante pronóstico recursivo "
        "a corto plazo (horizonte operativo de varios meses): los rezagos de casos se actualizarán "
        "con predicciones previas y el clima futuro se aproximará con climatología del mes "
        "(o escenarios simples). Los productos serán series proyectadas por distrito, niveles de "
        "riesgo relativos al historial local y actualización del panel, como apoyo a alerta "
        "temprana. Se explicitarán supuestos y limitaciones (incertidumbre creciente con el "
        "horizonte; ausencia de intervenciones o movilidad).",
    ),
    ("blank", ""),
    ("h2", "Análisis, interpretación y presentación de resultados"),
    (
        "p",
        "Los resultados se analizarán comparando error de validación 2024, importancia relativa "
        "de grupos de variables (rezagos de casos vs. clima) y coherencia espacial de los "
        "distritos de mayor riesgo. La interpretación contrastará los hallazgos con el estado "
        "del arte [1]–[3]: se esperará que el clima aporte señal, que el error sea mayor en "
        "picos extremos (p. ej. legado de 2023) y que un modelo interpretable/ensemble resulte "
        "competitivo frente a alternativas más complejas en un escenario de datos abiertos "
        "locales. La presentación incluirá el repositorio reproducible, métricas, visualizaciones "
        "del panel y el informe del proyecto de investigación.",
    ),
]


def build_gantt(doc, after_para):
    """Insert 18-week Gantt table after the given paragraph."""
    # Remove old example table if any remain near cronograma
    rows_spec = [
        (
            "OE1. Integrar panel distrital + clima",
            "(i) Revisar fuentes CDC/MINSA, Open-Meteo e ICEN; definir ubigeos y nodos climáticos.",
            list(range(1, 3)),
        ),
        (
            "OE1. Integrar panel distrital + clima",
            "(ii) Preparar script de agregación mensual por distrito y completar meses faltantes con 0.",
            list(range(2, 5)),
        ),
        (
            "OE1. Integrar panel distrital + clima",
            "(iii) Descargar/unir clima e ICEN; validar nulos y consistencia del panel.",
            list(range(4, 7)),
        ),
        (
            "OE2. Entrenar modelo ML",
            "(i) Ingeniería de features (lags, estacionalidad, IDs espaciales, log1p).",
            list(range(6, 9)),
        ),
        (
            "OE2. Entrenar modelo ML",
            "(ii) Entrenar HistGradientBoosting (2015–2023) y baseline Random Forest.",
            list(range(8, 11)),
        ),
        (
            "OE2. Entrenar modelo ML",
            "(iii) Ajuste fino, serialización del modelo y registro de metadatos.",
            list(range(10, 12)),
        ),
        (
            "OE3. Evaluar 2024 y panel",
            "(i) Calcular MAE/RMSE distrital y departamental; tablas real vs. predicho.",
            list(range(11, 14)),
        ),
        (
            "OE3. Evaluar 2024 y panel",
            "(ii) Implementar/actualizar panel web (mapa, ranking, detalle por distrito).",
            list(range(12, 15)),
        ),
        (
            "OE3. Pronóstico periodos posteriores",
            "(i) Diseñar protocolo de pronóstico recursivo y supuestos climáticos.",
            list(range(14, 16)),
        ),
        (
            "OE3. Pronóstico periodos posteriores",
            "(ii) Ejecutar proyección de periodos posteriores e integrar resultados al panel.",
            list(range(15, 18)),
        ),
        (
            "Cierre del proyecto",
            "Documentación final, informe, repositorio reproducible y exposición.",
            list(range(16, 19)),
        ),
    ]

    n_weeks = 18
    table = doc.add_table(rows=1 + len(rows_spec), cols=2 + n_weeks)
    table.style = "Table Grid"

    hdr = table.rows[0].cells
    set_cell_text(hdr[0], "Objetivo específico / Producto", bold=True, size=7)
    set_cell_text(hdr[1], "Actividades", bold=True, size=7)
    for w in range(1, n_weeks + 1):
        set_cell_text(hdr[1 + w], str(w), bold=True, size=7)

    for r_i, (obj, act, weeks) in enumerate(rows_spec, start=1):
        cells = table.rows[r_i].cells
        set_cell_text(cells[0], obj, bold=False, size=7, center=False)
        set_cell_text(cells[1], act, bold=False, size=7, center=False)
        for w in range(1, n_weeks + 1):
            mark = "X" if w in weeks else ""
            set_cell_text(cells[1 + w], mark, bold=True, size=7)

    # Move table to after cronograma heading content: python-docx adds at end.
    # Relocate XML element after anchor paragraph.
    tbl = table._tbl
    body = doc.element.body
    body.remove(tbl)
    after_para._element.addnext(tbl)
    return table


def main():
    doc = Document(str(SRC))

    # --- Objetivos / problema / hipótesis ---
    for p in doc.paragraphs:
        t = p.text.strip()
        if t.startswith("Existen estudios en el Perú con Prophet"):
            fill_paragraph(
                p,
                "Existen estudios en el Perú con Prophet a escala de varios departamentos, "
                "marcos climáticos regionales que incluyen Lambayeque y, más recientemente, "
                "benchmarks abiertos que comparan ARIMA, Prophet, LSTM y BiLSTM a nivel nacional. "
                "Sin embargo, no cubren del todo la necesidad de un prototipo informático local, "
                "reproducible y de bajo costo, validado con 2024 a escala distrital y orientado "
                "a estimar los periodos siguientes. Abordar este vacío es conveniente porque una "
                "predicción temprana puede apoyar una alerta más oportuna, sin recoger datos "
                "primarios ni encuestar personas.",
            )
        elif t.startswith("Pregunta de investigación"):
            fill_paragraph(
                p,
                "Pregunta de investigación: ¿De qué manera un modelo de machine learning "
                "entrenado con casos históricos de dengue por distrito y variables climáticas "
                "de datos abiertos permite predecir de forma temprana los brotes en el "
                "departamento de Lambayeque?",
            )
        elif t.startswith("Predecir de manera temprana"):
            fill_paragraph(
                p,
                "Predecir de manera temprana los brotes de dengue en el departamento de "
                "Lambayeque mediante un modelo de machine learning (HistGradientBoosting, "
                "con baseline Random Forest) que integre vigilancia epidemiológica y variables "
                "climáticas de datos abiertos oficiales, con alcance a casos mensuales "
                "notificados por distrito en 2015–2024, evaluación frente a 2024 y estimación "
                "de periodos posteriores como etapa final de alerta temprana.",
            )
        elif t.startswith("Integrar los casos mensuales"):
            fill_paragraph(
                p,
                "Integrar los casos mensuales de dengue notificados en Lambayeque (2015–2024) "
                "a nivel de distrito (ubigeo), completando meses sin notificación con cero, "
                "con variables climáticas de fuentes abiertas (precipitación, temperatura "
                "máxima/mínima, humedad e ICEN) en un panel mensual reproducible.",
            )
        elif t.startswith("Entrenar un modelo de Random Forest") or t.startswith(
            "Entrenar un modelo de HistGradientBoosting"
        ):
            fill_paragraph(
                p,
                "Entrenar un modelo de HistGradientBoostingRegressor (y un baseline Random "
                "Forest) con el periodo 2015–2023, usando como entradas el mes, identificadores "
                "espaciales, rezagos de casos y rezagos climáticos, para estimar los casos "
                "mensuales por distrito.",
            )
        elif t.startswith("Evaluar la precisión"):
            fill_paragraph(
                p,
                "Evaluar la precisión y el error del modelo (MAE y RMSE) respecto a los "
                "registros reales de 2024 a escala distrital y agregada departamental, "
                "implementar un panel web de apoyo a la decisión y, como etapa final del "
                "objetivo, estimar con el modelo validado los casos de los periodos posteriores "
                "(pronóstico operativo) como apoyo a una alerta temprana.",
            )
        elif t.startswith("Un modelo de Random Forest entrenado") or t.startswith(
            "Un modelo de machine learning entrenado"
        ):
            fill_paragraph(
                p,
                "Un modelo de machine learning entrenado con casos históricos de dengue por "
                "distrito y variables climáticas estima los casos mensuales en Lambayeque con "
                "un error cuantificable frente a 2024 y permite proyectar el riesgo a corto "
                "plazo en los periodos posteriores.",
            )

    # --- Estado del arte: replace instructional placeholders ---
    h_arte = delete_between(
        doc,
        "FUNDAMENTACIÓN (ESTADO DEL ARTE)",
        "DISEÑO METODOLÓGICO DE INVESTIGACIÓN",
    )
    add_body_after(h_arte, ESTADO_ARTE)

    # --- Metodología: remove example content ---
    h_met = delete_between(
        doc,
        "DISEÑO METODOLÓGICO DE INVESTIGACIÓN",
        "CRONOGRAMA DE EJECUCIÓN",
    )
    add_body_after(h_met, METODOLOGIA)

    # --- Cronograma ---
    h_cro = delete_between(doc, "CRONOGRAMA DE EJECUCIÓN", "BIBLIOGRAFÍA")
    intro = insert_paragraph_after(
        h_cro,
        "El cronograma corresponde al ciclo académico 2026-2 (18 semanas). Las actividades "
        "se organizan por objetivo específico; la proyección de periodos posteriores se "
        "programa al final del ciclo como cierre del OE3.",
        justify=True,
    )
    # Remove leftover example tables
    for t in list(doc.tables):
        # keep only if we just added; remove old 14-week example if still present
        first = t.rows[0].cells[0].text.strip() if t.rows else ""
        if "Objetivo específico/Producto" in first or first.startswith("Objetivo específico"):
            # could be old; if cols suggest 14 weeks old example
            if len(t.columns) <= 16:
                remove_table(t)

    build_gantt(doc, intro)
    note = doc.paragraphs[-1]
    # add note after table: find cronograma intro and add after table via body end is ok
    last_p = insert_paragraph_after(
        find_para(doc, "BIBLIOGRAFÍA"),
        "",
    )
    # Actually insert note before bibliografía
    # Move: create paragraph before bibliografía heading
    bib = find_para(doc, "BIBLIOGRAFÍA")
    note_p = OxmlElement("w:p")
    bib._element.addprevious(note_p)
    note_para = Paragraph(note_p, bib._parent)
    run = note_para.add_run(
        "Nota: “X” indica semanas planificadas de ejecución. La semana 1 corresponde al "
        "inicio del ciclo 2026-2 y la semana 18 al cierre académico del ciclo."
    )
    set_run_font(run, italic=True, size=10)

    # --- Bibliografía: append [3] ---
    # Find last bibliography paragraph and append after
    last_bib = None
    for p in doc.paragraphs:
        if p.text.strip().startswith("[2]") or "revibiomedica" in p.text.lower() or p.text.strip().startswith("https://revibiomedica"):
            last_bib = p
    # Walk to the last non-empty after [2]
    paras = list(doc.paragraphs)
    bib_idx = next(i for i, p in enumerate(paras) if p.text.strip().startswith("BIBLIOGRAFÍA"))
    last_bib = paras[-1]
    for p in paras[bib_idx + 1 :]:
        if p.text.strip():
            last_bib = p

    insert_paragraph_after(last_bib, "", justify=False)
    insert_paragraph_after(
        last_bib,
        "[3] M. F. Mayta Quispe y L. Alemán Gonzales, «DengueAI: Comparative Time-Series and "
        "Deep-Learning Modeling for Early Prediction of Dengue Outbreaks in Rural Peru Using "
        "Open Data», en 2025 IEEE XXXII International Conference on Electronics, Electrical "
        "Engineering and Computing (INTERCON), Arequipa, Perú, 20–22 ago. 2025, doi: "
        "10.1109/INTERCON67304.2025.11244382.",
        justify=True,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(OUT))
    print("Guardado:", OUT)


if __name__ == "__main__":
    main()
