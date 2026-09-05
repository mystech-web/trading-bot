"""Dashboard interactivo del bot: estado en vivo, curvas de equity, distribución
Monte Carlo, proyección hacia adelante, stress test, estabilidad de parámetros
y retornos mensuales -- todo leído de los archivos que genera `scripts/run_backtest.py`
(en `reports/`) y `scripts/run_live_once.py` (`reports/live_equity_log.csv`,
`reports/live_state.json`, `reports/live.log`).

No se conecta a Alpaca ni a ninguna red -- solo lee archivos locales, así que es
seguro correrlo en cualquier momento sin exponer tus claves de API.

Uso:
    streamlit run dashboard.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.monte_carlo import project_forward
from src.tracking import load_equity_log, load_state

ROOT = pathlib.Path(__file__).resolve().parent

st.set_page_config(page_title="Nexo -- panel de control", layout="wide", page_icon="⬡")

# ---------------------------------------------------------- Tema "centro de mando" ----
# Paleta neón consistente para todas las gráficas de Plotly + CSS para que los
# widgets nativos de Streamlit (metric, tabs, dataframe, alert boxes) combinen
# con el tema oscuro de .streamlit/config.toml en vez de quedar "genéricos".
NEON_BG = "#060a12"
NEON_CARD_BG = "#0d1420"
NEON_GRID = "rgba(0,255,200,0.12)"
NEON_TEXT = "#d8faff"
NEON_COLORWAY = ["#00ffc8", "#00b4ff", "#ff5fa2", "#ffd166", "#c77dff", "#8892a0", "#4dff88"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; }

.stApp {
    background: radial-gradient(ellipse 1400px 900px at 15% -10%, rgba(0,255,200,0.05) 0%, transparent 55%),
                radial-gradient(ellipse 1200px 800px at 110% 10%, rgba(0,150,255,0.04) 0%, transparent 55%),
                #070b13;
}

h1, h2, h3 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
    color: #eaf9f6 !important;
    letter-spacing: -0.01em;
}
h1 strong, h2 strong, h3 strong { color: #00e5b8; }

/* Logotipo "NEXO" -- ícono de nodos conectados + wordmark + tagline, en vez
   de un st.title() con emoji. El bloque completo hace de header de marca. */
.nexo-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 6px 0 22px 0;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.nexo-mark {
    width: 46px;
    height: 46px;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(160deg, rgba(0,229,184,0.16), rgba(0,229,184,0.02));
    border: 1px solid rgba(0,229,184,0.38);
    border-radius: 12px;
}
.nexo-wordmark-group { display: flex; flex-direction: column; gap: 2px; line-height: 1; }
.nexo-wordmark {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    letter-spacing: -0.02em;
    color: #f4fbfa;
}
.nexo-wordmark span { color: #00e5b8; }
.nexo-tagline {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.7rem;
    letter-spacing: 2.4px;
    text-transform: uppercase;
    color: #64888c;
}

/* Tarjetas de métricas -- números en monoespaciada (alineación tabular, look
   "terminal financiera"), etiquetas y resto del texto en Inter. */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-left: 2px solid rgba(0,229,184,0.55);
    border-radius: 10px;
    padding: 14px 18px 12px 18px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
[data-testid="stMetricValue"] {
    color: #eaf9f6 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] {
    color: #8ba3a8 !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
}

/* Pestañas -- subrayado limpio en vez de recuadros con borde por todos lados */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.08); }
[data-baseweb="tab"] {
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    border-radius: 6px 6px 0 0;
    color: #8ba3a8 !important;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    padding: 10px 16px;
    transition: color 0.15s ease, background 0.15s ease;
}
[data-baseweb="tab"]:hover { color: #cfe9e4 !important; background: rgba(255,255,255,0.03); }
[data-baseweb="tab"][aria-selected="true"] {
    color: #00e5b8 !important;
    background: rgba(0,229,184,0.06) !important;
    border-bottom: 2px solid #00e5b8 !important;
    font-weight: 600;
}

/* Selector de perfil / broker -- "segmented control" tipo iOS/macOS en vez de
   radios sueltos: contenedor con fondo, cada opción es un botón, la
   seleccionada queda con relleno sólido (look de botón presionado). */
[data-testid="stRadio"] > div[role="radiogroup"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 4px;
    gap: 2px !important;
}
[data-testid="stRadio"] label {
    border-radius: 7px;
    padding: 7px 16px !important;
    margin: 0 !important;
    background: transparent;
    transition: background 0.15s ease, color 0.15s ease;
    cursor: pointer;
}
[data-testid="stRadio"] label:hover { background: rgba(255,255,255,0.05); }
/* Oculta el punto de radio nativo -- el "botón" es el fondo del label */
[data-testid="stRadio"] label > div:first-child { display: none; }
[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] p {
    font-weight: 500;
    color: #b7c9cc;
    font-size: 0.92rem;
}
[data-testid="stRadio"] label:has(input:checked) {
    background: #00e5b8;
    box-shadow: 0 2px 10px rgba(0,229,184,0.30);
}
[data-testid="stRadio"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
    color: #04120e;
    font-weight: 700;
}

/* Dataframes / tablas */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    overflow: hidden;
}

/* Cajas de alerta (info/warning/error) */
[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.10);
    font-family: 'Inter', sans-serif;
}

/* Bloques de código / log -- monoespaciada, es contenido tipo terminal real */
[data-testid="stCodeBlock"] {
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 8px;
}

/* Sidebar (si se usa) */
[data-testid="stSidebar"] { background: #050810; border-right: 1px solid rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)


def neon_fig(fig: go.Figure, height: int = 450) -> go.Figure:
    """Aplica el tema oscuro/neón consistente a cualquier gráfica de Plotly del
    dashboard -- fondo transparente (se ve el degradado de .stApp detrás),
    paleta de colores neón, grid tenue. Centralizado acá para que las ~7
    gráficas del dashboard no diverjan una de otra."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=NEON_TEXT, size=12),
        colorway=NEON_COLORWAY,
        height=height,
        margin=dict(t=30, b=40, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=NEON_GRID, zerolinecolor=NEON_GRID)
    fig.update_yaxes(gridcolor=NEON_GRID, zerolinecolor=NEON_GRID)
    return fig


# ---------------------------------------------------- Nombres legibles ----
# Los CSV que genera run_backtest.py/run_crypto_backtest.py usan nombres
# TÉCNICOS como índice/columna (ej. "ENSEMBLE_OOS_walkforward",
# "momentum_in_sample (sesgado)") porque otro código los busca por ese nombre
# exacto (ver scripts/run_backtest.py y esta misma función más abajo, que sí
# usa el nombre técnico para el resumen de arriba). Esta función SOLO traduce
# para lo que se muestra en pantalla -- nunca toca los archivos en disco.
BASE_STRATEGY_LABELS = {
    "momentum": "Momentum",
    "mean_reversion": "Reversión a la media",
    "sector_rotation": "Rotación sectorial",
    "ensemble": "Ensamble, mezcla fija",
    "ensemble_dynamic_alloc": "Ensamble, asignación dinámica",
    "ensemble_dynamic_alloc_vol_target": "Ensamble, dinámico + vol-targeting",
}


def friendly_label(raw: str) -> str:
    name = raw.strip()
    if name.startswith("benchmark_"):
        ticker = name[len("benchmark_"):]
        if ticker.endswith("_buy_hold"):
            ticker = ticker[: -len("_buy_hold")]
        return f"{ticker} (comprar y mantener)"
    if name == "ENSEMBLE_OOS_dynamic_alloc":
        return "Ensamble, asignación dinámica — fuera de muestra"
    if name == "ENSEMBLE_OOS_dynamic_alloc_vol_target":
        return "Ensamble, dinámico + vol-targeting — fuera de muestra"
    if name.endswith("_in_sample (sesgado)"):
        base = name[: -len("_in_sample (sesgado)")].lower()
        return f"{BASE_STRATEGY_LABELS.get(base, base)} — histórico completo (sesgado)"
    if name.endswith("_OOS_walkforward"):
        base = name[: -len("_OOS_walkforward")].lower()
        return f"{BASE_STRATEGY_LABELS.get(base, base)} — fuera de muestra"
    return BASE_STRATEGY_LABELS.get(name, name)


METRIC_COLUMN_LABELS = {
    "cagr": "CAGR (%)",
    "ann_vol": "Vol. anualizada (%)",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Máx. drawdown (%)",
    "calmar": "Calmar",
    "avg_monthly_return": "Retorno mensual prom. (%)",
    "median_monthly_return": "Retorno mensual mediana (%)",
    "pct_positive_months": "Meses positivos (%)",
    "worst_month": "Peor mes (%)",
    "best_month": "Mejor mes (%)",
    "n_months": "Meses (n)",
}


def friendly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Copia de `df` con el índice y las columnas traducidos a nombres legibles
    -- para mostrar en `st.dataframe` sin tocar el `df` original (otro código
    de esta misma página sigue usando `summary.loc["ENSEMBLE_OOS_walkforward"]`,
    y `run_backtest.py` sigue escribiendo `summary.csv` con los nombres técnicos
    originales, que es lo que otras herramientas/scripts esperan)."""
    out = df.copy()
    out.index = [friendly_label(i) for i in out.index]
    out.index.name = "Estrategia"
    out = out.rename(columns=METRIC_COLUMN_LABELS)
    return out


st.markdown("""
<div class="nexo-header">
  <div class="nexo-mark">
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" xmlns="http://www.w3.org/2000/svg">
      <line x1="6" y1="20" x2="13" y2="6" stroke="#00e5b8" stroke-width="1.6" opacity="0.55"/>
      <line x1="20" y1="20" x2="13" y2="6" stroke="#00e5b8" stroke-width="1.6" opacity="0.55"/>
      <line x1="6" y1="20" x2="20" y2="20" stroke="#00e5b8" stroke-width="1.6" opacity="0.30"/>
      <circle cx="13" cy="6" r="3" fill="#00e5b8"/>
      <circle cx="6" cy="20" r="3" fill="#00e5b8" opacity="0.75"/>
      <circle cx="20" cy="20" r="3" fill="#00e5b8" opacity="0.45"/>
    </svg>
  </div>
  <div class="nexo-wordmark-group">
    <div class="nexo-wordmark">NEXO<span>·</span></div>
    <div class="nexo-tagline">Inversión sistemática</div>
  </div>
</div>
""", unsafe_allow_html=True)

profile_label = st.radio(
    "Perfil", ["🟢 Conservador", "🔴 Agresivo (apalancado -- más riesgo)", "🪙 Cripto (Binance)"],
    horizontal=True, label_visibility="collapsed",
)
profile = {"🔴": "aggressive", "🪙": "crypto"}.get(profile_label[0], "conservative")
REPORTS_DIR = ROOT / {"aggressive": "reports_aggressive", "crypto": "reports_crypto"}.get(profile, "reports")
PERIODS_PER_YEAR = 365 if profile == "crypto" else 252
DAYS_PER_MONTH = 30 if profile == "crypto" else 21

if profile == "aggressive":
    st.error(
        "**Perfil agresivo**: usa ETFs apalancados 3x y mayor volatilidad objetivo. Los números de abajo "
        "reflejan MUCHO más riesgo de drawdown severo que el perfil conservador -- compara el max drawdown "
        "y el percentil p95 del stress test antes de considerar usarlo, incluso en paper trading."
    )
elif profile == "crypto":
    st.warning(
        "**Módulo cripto**: mercado 24/7, mucho más volátil que acciones/ETFs, y sin la protección de la "
        "regulación de valores de EE.UU. Los pares de este universo son los de HOY -- si algún token pierde "
        "toda su liquidez o es deslistado de Binance, no lo vas a ver reflejado como \"riesgo de supervivencia\" "
        "en el backtest, a diferencia del bot de acciones."
    )


def _load_csv(name, **kwargs):
    path = REPORTS_DIR / name
    return pd.read_csv(path, **kwargs) if path.exists() else None


def _load_json(name):
    path = REPORTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


summary = _load_csv("summary.csv", index_col=0)
if summary is None:
    if profile == "crypto":
        cmd = "python scripts/run_crypto_backtest.py"
    elif profile == "aggressive":
        cmd = "python scripts/run_backtest.py --profile aggressive"
    else:
        cmd = "python scripts/run_backtest.py"
    st.warning(
        f"Todavía no hay reportes para este perfil. Corre primero:\n\n`{cmd}`\n\n"
        "(necesita datos reales de internet -- no funciona con datos sintéticos de prueba)."
    )
    st.stop()

mc = _load_json("monte_carlo.json")
oos_returns = _load_csv("oos_returns.csv", index_col=0, parse_dates=["date"])
monthly = _load_csv("ensemble_monthly_returns.csv", index_col=0, parse_dates=[0])
stress_insample = _load_csv("stress_test_insample.csv")
stress_oos = _load_csv("stress_test_oos.csv")
stability_scores = _load_json("param_stability_scores.json")
param_drift = _load_json("param_drift.json")

BROKERS_BY_PROFILE = {
    "conservative": ["alpaca", "virtual"],
    "aggressive": ["alpaca", "virtual"],
    "crypto": ["virtual", "binance", "bitso"],
}
BROKER_LABELS = {
    "alpaca": "Alpaca (paper trading)",
    "virtual": "Virtual (capital simulado)",
    "binance": "Binance (testnet)",
    "bitso": "Bitso (dinero real)",
}

tab_resumen, tab_mc, tab_stress, tab_stability, tab_monthly, tab_live = st.tabs(
    ["📈 Resumen", "🎲 Monte Carlo & Proyección", "💥 Stress Test",
     "🔧 Estabilidad de parámetros", "📅 Retornos mensuales", "🟢 En vivo"]
)

# ---------------------------------------------------------------- Resumen ----
with tab_resumen:
    ens = summary.loc["ENSEMBLE_OOS_walkforward"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR (OOS)", f"{ens['cagr']:.1f}%")
    c2.metric("Retorno mensual prom.", f"{ens['avg_monthly_return']:.2f}%")
    c3.metric("Sharpe", f"{ens['sharpe']:.2f}")
    c4.metric("Max Drawdown", f"{ens['max_drawdown']:.1f}%")
    c5.metric("% meses positivos", f"{ens['pct_positive_months']:.0f}%")

    if mc:
        prob = mc["prob_avg_monthly_in_target_0.5_2pct"] * 100
        st.info(f"**Probabilidad de que el retorno mensual promedio caiga en tu objetivo (0.5%-2%): "
                f"{prob:.1f}%** (según simulación Monte Carlo sobre el histórico out-of-sample).")

    st.subheader("Curvas de equity out-of-sample")
    if oos_returns is not None:
        equity = (1 + oos_returns.fillna(0)).cumprod()
        fig = go.Figure()
        for col in equity.columns:
            fig.add_trace(go.Scatter(x=equity.index, y=equity[col], name=friendly_label(col), mode="lines"))
        fig.update_layout(yaxis_title="Crecimiento de $1", xaxis_title="Fecha",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(neon_fig(fig), use_container_width=True)

    st.subheader("Tabla completa de métricas")
    st.dataframe(friendly_index(summary), use_container_width=True)

# ------------------------------------------------------- Monte Carlo tab ----
with tab_mc:
    if mc is None:
        st.warning("No hay datos de Monte Carlo todavía.")
    else:
        st.subheader("Distribución de escenarios plausibles (retorno mensual promedio)")
        am = mc["avg_monthly_return"]
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("p5 (pesimista)", "p5"), ("p25", "p25"), ("mediana", "p50"),
                                             ("p75", "p75"), ("p95 (optimista)", "p95")]):
            col.metric(label, f"{am[key] * 100:.2f}%")

        st.caption(f"Probabilidad de caer en el rango objetivo 0.5%-2%: "
                   f"**{mc['prob_avg_monthly_in_target_0.5_2pct'] * 100:.1f}%** · "
                   f"Probabilidad de que sea negativo: **{mc['prob_avg_monthly_negative'] * 100:.1f}%**")

        st.subheader("Proyección hacia adelante (cono de incertidumbre)")
        colA, colB = st.columns(2)
        start_capital = colA.number_input("Capital inicial ($)", min_value=100.0, value=10_000.0, step=500.0)
        months_ahead = colB.slider("Meses hacia adelante", min_value=6, max_value=60, value=24)

        if oos_returns is not None and "ensemble" in oos_returns.columns:
            proj = project_forward(oos_returns["ensemble"], months=months_ahead, start_capital=start_capital,
                                    block_size=DAYS_PER_MONTH, days_per_month=DAYS_PER_MONTH)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p95"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p5"], fill="tonexty", fillcolor="rgba(0,255,200,0.12)",
                                      line=dict(width=0), name="rango p5-p95"))
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p75"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p25"], fill="tonexty", fillcolor="rgba(0,255,200,0.28)",
                                      line=dict(width=0), name="rango p25-p75"))
            fig.add_trace(go.Scatter(x=proj.index, y=proj["p50"], line=dict(color="#00ffc8", width=2),
                                      name="mediana"))
            fig.update_layout(xaxis_title="Meses desde hoy", yaxis_title="Capital proyectado ($)")
            st.plotly_chart(neon_fig(fig), use_container_width=True)
            st.caption("Esto NO es una predicción -- es el rango de resultados plausibles si el futuro se parece "
                       "estadísticamente al período out-of-sample analizado. El futuro puede ser peor (o mejor) "
                       "que todo este rango.")

        st.image(str(REPORTS_DIR / "monte_carlo_hist.png"), use_container_width=True)

# ----------------------------------------------------------- Stress test ----
with tab_stress:
    st.subheader("Comportamiento en crashes conocidos")
    for label, df in [("Out-of-sample (walk-forward, honesto)", stress_oos),
                       ("In-sample (parámetros por defecto, referencia)", stress_insample)]:
        st.markdown(f"**{label}**")
        if df is None or df.empty:
            st.caption("Ningún crash conocido cae en este rango de datos.")
            continue
        return_cols = [c for c in df.columns if c.endswith("_return_%")]
        fig = go.Figure()
        for col in return_cols:
            strat = col.replace("_return_%", "")
            fig.add_trace(go.Bar(x=df["period"], y=df[col], name=friendly_label(strat)))
        fig.update_layout(barmode="group", yaxis_title="Retorno durante el período (%)")
        st.plotly_chart(neon_fig(fig, height=400), use_container_width=True)

        df_display = df.rename(columns={
            c: f"{friendly_label(c[:-len('_return_%')])} — retorno %" if c.endswith("_return_%")
            else f"{friendly_label(c[:-len('_max_dd_%')])} — drawdown máx. %" if c.endswith("_max_dd_%")
            else c
            for c in df.columns
        })
        st.dataframe(df_display, use_container_width=True)

# --------------------------------------------------- Estabilidad de params ----
with tab_stability:
    if param_drift:
        lines = [f"**{len(param_drift)} parámetro(s)** en `config/live_params.yaml` ya no coinciden con lo que "
                 f"eligió el walk-forward en su fold más reciente:"]
        for f in param_drift:
            lines.append(f"- **[{friendly_label(f['strategy'])}] {f['param']}**: en vivo = `{f['live_value']}` "
                          f"vs. walk-forward más reciente = `{f['latest_wf_value']}` "
                          f"(fold {f['fold_test_start']} → {f['fold_test_end']})")
        lines.append("\nEsto NO se aplica solo -- es tu decisión revisar la tabla de abajo y actualizar el YAML "
                      "a mano si te convence (un solo fold puede ser ruido, no una señal real).")
        st.warning("\n".join(lines))
    elif param_drift is not None:
        st.success("Sin drift: los parámetros en vivo coinciden con lo que eligió el fold más reciente del "
                   "walk-forward.")

    st.subheader("¿El parámetro 'óptimo' de cada fold es real, o es ruido?")
    st.caption("1.0 = el walk-forward siempre eligió el mismo valor en todos los folds (estable/confiable). "
               "Cerca de 0 = el óptimo salta de fold a fold (trátalo como ruido).")
    if stability_scores:
        rows = []
        for strat, scores in stability_scores.items():
            for param, score in scores.items():
                rows.append({"estrategia": strat, "parámetro": param, "estabilidad": score})
        df_scores = pd.DataFrame(rows)
        fig = go.Figure()
        for strat in df_scores["estrategia"].unique():
            sub = df_scores[df_scores["estrategia"] == strat]
            fig.add_trace(go.Bar(x=sub["parámetro"] + " (" + friendly_label(strat) + ")", y=sub["estabilidad"],
                                  name=friendly_label(strat)))
        fig.add_hline(y=0.5, line_dash="dash", annotation_text="umbral de aviso")
        fig.update_layout(yaxis_title="Estabilidad (0-1)", showlegend=False)
        st.plotly_chart(neon_fig(fig, height=400), use_container_width=True)

    for name in ["momentum", "mean_reversion", "sector_rotation"]:
        df = _load_csv(f"param_stability_{name}.csv", index_col=0)
        if df is not None:
            st.markdown(f"**{friendly_label(name)}** -- parámetro ganador por fold")
            st.dataframe(df, use_container_width=True)

# ------------------------------------------------------ Retornos mensuales ----
with tab_monthly:
    st.subheader("Retornos mensuales del ensamble (out-of-sample)")
    if monthly is not None:
        s = monthly.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        df_pivot = pd.DataFrame({"year": s.index.year, "month": s.index.month, "return": s.values * 100})
        pivot = df_pivot.pivot(index="year", columns="month", values="return")
        month_names = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
        pivot.columns = [month_names[m - 1] for m in pivot.columns]
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values, x=pivot.columns, y=pivot.index.astype(str),
            colorscale="RdYlGn", zmid=0, text=pivot.round(2).values, texttemplate="%{text}",
        ))
        fig.update_layout(yaxis_title="Año")
        st.plotly_chart(neon_fig(fig, height=max(300, 40 * len(pivot))), use_container_width=True)
    else:
        st.caption("No hay datos de retornos mensuales todavía.")

# ------------------------------------------------------------------ Vivo ----
with tab_live:
    st.subheader("Estado de la cuenta en vivo (paper trading)")

    available_brokers = BROKERS_BY_PROFILE.get(profile, ["alpaca", "virtual"])
    broker_choice = st.radio("Broker", available_brokers, horizontal=True, key="broker_choice",
                              format_func=lambda b: BROKER_LABELS.get(b, b))
    tracking_dir = REPORTS_DIR / broker_choice
    st.caption(f"Cada broker lleva su propio tracking, separado -- esto muestra únicamente "
               f"`{tracking_dir.relative_to(ROOT)}/`.")

    live_state = load_state(tracking_dir) if tracking_dir.exists() else None
    live_log = load_equity_log(tracking_dir) if tracking_dir.exists() else None

    if live_state and live_state.get("peak_equity") is not None:
        c1, c2 = st.columns(2)
        c1.metric("Pico de equity registrado", f"${live_state.get('peak_equity', 0):,.2f}")
        guard = live_state.get("guard_active", False)
        c2.metric("Guardia de drawdown", "🔴 ACTIVA (exposición al 50%)" if guard else "🟢 inactiva")
    else:
        cmd = f"python scripts/run_live_once.py --broker {broker_choice}" if profile != "crypto" \
            else f"python scripts/run_crypto_live_once.py --broker {broker_choice}"
        st.caption(f"Todavía no hay estado registrado para '{broker_choice}' -- corre `{cmd}` al menos una vez.")

    if live_log is not None and len(live_log):
        live_log = live_log.sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=live_log["date"], y=live_log["equity"], mode="lines+markers",
                                  name="Equity real", line=dict(color="#00ffc8", width=2.5),
                                  marker=dict(size=6, color="#00ffc8")))

        if oos_returns is not None and "ensemble" in oos_returns.columns and len(live_log) >= 2:
            months_span = max(1, int((live_log["date"].iloc[-1] - live_log["date"].iloc[0]).days / 30) + 3)
            proj = project_forward(oos_returns["ensemble"], months=months_span,
                                    start_capital=float(live_log["equity"].iloc[0]),
                                    block_size=DAYS_PER_MONTH, days_per_month=DAYS_PER_MONTH)
            proj_dates = pd.date_range(live_log["date"].iloc[0], periods=len(proj), freq="MS")
            fig.add_trace(go.Scatter(x=proj_dates, y=proj["p95"], line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=proj_dates, y=proj["p5"], fill="tonexty",
                                      fillcolor="rgba(0,180,255,0.15)", line=dict(width=0),
                                      name="banda esperada p5-p95"))

        fig.update_layout(xaxis_title="Fecha", yaxis_title="Equity ($)",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(neon_fig(fig), use_container_width=True)
        st.caption("Si la línea de equity real se sale por debajo de la banda esperada de forma sostenida, "
                   "el bot manda una alerta automática (ver `check_drift` en `src/tracking.py`).")
    else:
        st.caption("Todavía no hay historial de equity en vivo para este broker.")

    log_path = tracking_dir / "live.log"
    if log_path.exists():
        st.subheader("Últimas líneas del log")
        lines = log_path.read_text().splitlines()[-40:]
        st.code("\n".join(lines), language="log")
