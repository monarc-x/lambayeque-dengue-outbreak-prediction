"""Render the methodology formulas (section 4.5) as PNG images for the report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "outputs" / "formulas"
plt.rcParams["mathtext.fontset"] = "cm"
FONT = 22
DPI = 300

FORMULAS = {
    1: r"$y_{d,t} = \sum_{i=1}^{N} \mathbf{1}\left[\,\mathrm{ubigeo}_i = d \;\wedge\; \mathrm{a\tilde{n}o}_i = t\,\right]$",
    2: r"$\mathrm{ICEN}^{\mathrm{temp}}_{t} = \max_{m \in \{1,\ldots,5\}} \; \mathrm{ICEN}_{t,m}$",
    3: r"$\mathrm{ICEN}^{\mathrm{prev}}_{t} = \dfrac{1}{3}\sum_{m=6}^{8} \mathrm{ICEN}_{t-1,m}$",
    4: r"$z_{d,t} = \ln\left(1 + y_{d,t}\right)$",
    5: r"$H_{d,t} = \dfrac{1}{t-2001}\sum_{k=2000}^{t-2} z_{d,k}$",
    6: r"$\mathbf{x}_{d,t} = \left(\mathrm{ICEN}^{\mathrm{temp}}_{t},\; \mathrm{ICEN}^{\mathrm{prev}}_{t},\; z_{d,t-2},\; z_{d,t-3},\; H_{d,t}\right)$",
    7: r"$F_M(\mathbf{x}) = F_0 + \nu \sum_{j=1}^{M} f_j(\mathbf{x}), \qquad \hat{y} = e^{F_M(\mathbf{x})}$",
    8: r"$\mathcal{L} = \dfrac{1}{n}\sum_{i=1}^{n}\left(\hat{y}_i - y_i \ln \hat{y}_i\right)$",
    9: r"$\dfrac{\partial \hat{y}}{\partial\, \mathrm{ICEN}^{\mathrm{temp}}} \geq 0, \qquad \dfrac{\partial \hat{y}}{\partial\, \mathrm{ICEN}^{\mathrm{prev}}} \geq 0, \qquad \dfrac{\partial \hat{y}}{\partial H} \geq 0$",
    10: r"$EA = \left|\, y - \hat{y} \,\right|$",
    11: r"$EP = \dfrac{\left|\, y - \hat{y} \,\right|}{y} \times 100$",
}

RISK_CASES = [
    (r"$\mathrm{alto}$", r"$\mathrm{si}\ \ \hat{y} \geq P_{90}$"),
    (r"$\mathrm{medio}$", r"$\mathrm{si}\ \ P_{66} \leq \hat{y} < P_{90}$"),
    (r"$\mathrm{bajo}$", r"$\mathrm{si}\ \ \hat{y} < P_{66}$"),
]
GAP = 40  # pixels between pieces drawn side by side


def new_figure() -> tuple[plt.Figure, object]:
    fig = plt.figure(figsize=(12, 3), dpi=DPI)
    return fig, fig.canvas.get_renderer()


def put(fig, renderer, x: float, y: float, text: str, size: float = FONT, va: str = "baseline"):
    """Draw text at pixel coordinates (figure-relative so tight bbox sees it); return its pixel box."""
    w, h = fig.bbox.width, fig.bbox.height
    artist = fig.text(x / w, y / h, text, fontsize=size, va=va)
    return artist.get_window_extent(renderer)


def number_label(number: int) -> str:
    return rf"$({number})$"


def save(fig: plt.Figure, number: int) -> Path:
    path = OUT / f"ecuacion_{number:02d}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def render(number: int, tex: str) -> Path:
    fig, renderer = new_figure()
    box = put(fig, renderer, 50, 400, tex)
    put(fig, renderer, box.x1 + GAP * 3, 400, number_label(number))
    return save(fig, number)


def render_risk(number: int) -> Path:
    fig, renderer = new_figure()
    mid, step = 450, 110
    left = put(fig, renderer, 50, mid, r"$\mathrm{Riesgo}(\hat{y}) =$", va="center")
    brace = put(fig, renderer, left.x1 + GAP / 2, mid, r"$\{$", size=FONT * 3.4, va="center")
    labels_x = brace.x1 + GAP / 2
    label_boxes = [
        put(fig, renderer, labels_x, mid + step * (1 - k), label, va="center")
        for k, (label, _) in enumerate(RISK_CASES)
    ]
    cond_x = max(b.x1 for b in label_boxes) + GAP * 2
    cond_boxes = [
        put(fig, renderer, cond_x, mid + step * (1 - k), cond, va="center")
        for k, (_, cond) in enumerate(RISK_CASES)
    ]
    put(fig, renderer, max(b.x1 for b in cond_boxes) + GAP * 3, mid, number_label(number), va="center")
    return save(fig, number)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for number, tex in FORMULAS.items():
        print(render(number, tex))
    print(render_risk(12))


if __name__ == "__main__":
    main()
