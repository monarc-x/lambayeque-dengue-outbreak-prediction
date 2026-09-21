# Prompt para otro chat (Cursor) — exportar figuras del panel dengue-lambayeque al Word

Copia y pega esto en el otro chat:

---

Trabaja en el repo `D:\repos\dengue-lambayeque`.

Objetivo: generar imágenes PNG de alta calidad para insertar en el Word del proyecto de investigación (Contreras_Campos), a partir de `web/data/dashboard.json` y/o capturas del panel en `http://127.0.0.1:8080/`.

1) Si no está corriendo, levanta el panel:
   `py -3.12 04_serve_dashboard.py`

2) Ejecuta (o crea si falta) el script:
   `py -3.12 scripts/export_figuras_informe.py`
   Debe guardar en `outputs/figuras_informe/`:
   - fig1_departamento_real_vs_pred_2024.png
   - fig2_top6_distritos_2024.png
   - fig3_barras_distritos_anual_2024.png
   - fig4_mae_por_distrito.png
   - fig5_mapa_riesgo_h1.png
   - fig6_ranking_prioridad.png

3) Opcional: abre el panel en el browser y captura screenshots limpios del mapa y de la tabla de detalle (sin UI de Cursor), guardándolos en la misma carpeta como:
   - fig7_screenshot_mapa_panel.png
   - fig8_screenshot_tabla_distritos.png

4) Al terminar, lista las rutas absolutas de los PNG y confirma resolución (~200 dpi).

No modifiques el modelo ni reentrenes. Solo exportar figuras.

---
