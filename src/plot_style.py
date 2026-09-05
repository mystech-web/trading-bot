"""Tema oscuro/neón compartido para las gráficas de matplotlib (`monte_carlo_hist.png`,
`equity_oos.png`) -- para que combinen con el tema del dashboard (`dashboard.py`,
`.streamlit/config.toml`) en vez de quedar como recuadros blancos genéricos
incrustados en un panel oscuro."""
import matplotlib.pyplot as plt
from cycler import cycler

BG = "#060a12"
CARD_BG = "#0d1420"
ACCENT = "#00ffc8"
TEXT = "#d8faff"
MUTED_TEXT = "#7fd8c8"
GRID = "#123028"

# Misma paleta que NEON_COLORWAY en dashboard.py, para que una línea "momentum"
# tenga el mismo color en el PNG de matplotlib y en las gráficas de Plotly.
NEON_COLORWAY = ["#00ffc8", "#00b4ff", "#ff5fa2", "#ffd166", "#c77dff", "#8892a0", "#4dff88"]


def apply_neon_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.facecolor": CARD_BG,
        "axes.edgecolor": ACCENT,
        "axes.labelcolor": TEXT,
        "axes.prop_cycle": cycler(color=NEON_COLORWAY),
        "text.color": TEXT,
        "xtick.color": MUTED_TEXT,
        "ytick.color": MUTED_TEXT,
        "grid.color": GRID,
        "axes.grid": True,
        "grid.alpha": 0.6,
        "legend.facecolor": CARD_BG,
        "legend.edgecolor": ACCENT,
        "legend.labelcolor": TEXT,
        "font.family": "monospace",
    })
