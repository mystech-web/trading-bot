# Bot de inversión sistemática (0.5%–2% mensual objetivo)

## Antes que nada: expectativas realistas

Este bot combina 3 estrategias con lógica distinta (tendencia, reversión a la media,
rotación sectorial) y las valida con **walk-forward** (optimiza parámetros solo con
datos pasados, los prueba en datos que nunca vio) para evitar el error más común de
los "bots de bolsa": un backtest sobreajustado que se ve espectacular y en vivo pierde
dinero. El reporte que genera distingue explícitamente:

- **In-sample** (optimizado y evaluado con el mismo período): siempre se ve mejor de lo
  que es. Solo para referencia.
- **Out-of-sample / walk-forward**: la estimación honesta de qué hubieras logrado
  operando esto en tiempo real durante los últimos años.

Ningún bot garantiza 0.5–2% mensual de forma consistente. El objetivo de este proyecto
es maximizar la probabilidad de estar cerca de ese rango con el menor drawdown posible,
no prometerlo. **Empieza siempre en paper trading (dinero simulado) durante al menos
3-6 meses antes de considerar dinero real.**

### Sobre apuntar a 3%+ mensual

3% mensual compuesto es ~43% anual sostenido -- terreno de los mejores fondos del
mundo (con apalancamiento masivo y ventajas que un retail no tiene), no algo que un
ajuste de parámetros vaya a lograr de forma honesta. Empujar el backtest hasta que
muestre 3% casi siempre significa sobreajuste a ruido, no una ventaja real. Este
proyecto SÍ incluye un **perfil agresivo opcional** (apalancamiento vía ETFs 3x, ver
más abajo) que empuja el retorno esperado más arriba a cambio de mucho más riesgo de
drawdown -- pero no existe una versión de este bot que dé 3%+/mes de forma sostenida
sin ese riesgo. Compara honestamente ambos perfiles en el dashboard antes de decidir
cuál (si alguno) te hace sentido.

## Qué incluye

- `src/strategies/momentum.py` — cruce de medias móviles + tamaño de posición por
  **paridad de riesgo** (inverse-vol: un activo el doble de volátil recibe la mitad
  de peso relativo), **ajustada por correlación** (un activo muy correlacionado con
  el resto del universo -- redundante -- recibe menos peso relativo; uno que
  diversifica, correlación baja o negativa, recibe más), escalado por volatilidad
  objetivo del portafolio, y con **confirmación multi-horizonte**: antes de confiar
  en el cruce de medias (que puede dar falsas señales en mercados laterales,
  "whipsaw"), exige que el momentum también sea positivo en al menos el 75% de un
  conjunto de horizontes (~1, 3, 6 y 12 meses) -- si el precio subió mucho en los
  últimos 21 días pero viene cayendo en los últimos 6-12 meses, no es una tendencia
  real todavía, y el cruce de medias por sí solo no distingue eso.
- `src/strategies/mean_reversion.py` — RSI(2) en tendencia alcista de fondo (estilo
  Connors), con límite de días en posición y tamaño también por paridad de riesgo.
  El stop es **TRAILING por defecto** (`use_trailing_stop=True`): se mide desde el
  máximo alcanzado DESDE LA ENTRADA, no desde el precio de entrada -- si la posición
  sube y después retrocede, protege la ganancia ya generada en vez de esperar a que
  retroceda todo el camino de vuelta al precio de entrada. `use_trailing_stop=False`
  recupera el stop fijo (`stop_loss_pct`, medido siempre desde la entrada).
- `src/strategies/sector_rotation.py` — dual momentum entre ETFs sectoriales, con
  filtro de cash (si nada le gana al T-Bill, el portafolio se va a cash) y el mismo
  **ajuste por correlación** entre los sectores elegidos: si dos de los `top_n`
  sectores seleccionados suelen moverse juntos (ej. XLK y XLY en un rally amplio),
  reciben menos peso relativo que uno que diversifica dentro de la canasta elegida.
  El ranking entre los sectores que califican es **ajustado a riesgo por defecto**
  (`risk_adjusted_ranking=True`): ordena por momentum/volatilidad (tipo Sharpe), no
  por momentum crudo -- rankear por retorno puro favorece sistemáticamente a
  sectores más volátiles (suben más rápido, pero también caen más rápido), y
  termina persiguiendo el sector más "ruidoso" en vez del que mejor recompensa el
  riesgo que toma. `risk_adjusted_ranking=False` recupera el ranking por momentum
  crudo.
- `src/backtest.py` — motor de backtest con costos de transacción (planos o
  sensibles a liquidez, ver `src/costs.py`), filtro de régimen macro opcional
  (`src/regime.py`), y DOS guardias de drawdown a nivel portafolio: una REACTIVA
  (reduce exposición a la mitad si el drawdown ya pasó -15%) y una PROACTIVA (reduce
  exposición si la volatilidad realizada de corto plazo se dispara muy por encima de
  su nivel normal, ej. 2x -- así suelen empezar los crashes grandes, con un salto de
  volatilidad ANTES de que el precio caiga en serio). Ambas comparten la misma
  histéresis: para volver a exposición completa hace falta que AMBAS condiciones se
  normalicen, no solo una. La REACTIVACIÓN es **gradual** (`guard_ramp_days=5` por
  defecto): una vez liberada, la exposición sube en pasos iguales día a día hasta
  volver a 100%, en vez de saltar de golpe -- evita quedar totalmente expuesto de
  nuevo justo cuando la recuperación todavía podría ser un rebote falso. Si la
  guardia se reactiva a mitad de la rampa, la exposición vuelve de golpe al mínimo
  (sin "crédito parcial" por la recuperación anterior). Mismo mecanismo en vivo, ver
  `src/tracking.py -> update_drawdown_guard`. `guard_ramp_days=0` recupera el salto
  instantáneo de antes.
- `src/costs.py` — modelo de costos sensible a liquidez: más caro mover una posición
  grande en un activo poco líquido que en uno muy líquido (spread base + impacto de
  mercado según qué tan grande es la orden respecto al volumen diario típico).
- `src/regime.py` — filtro de régimen macro **proactivo**, con hasta TRES señales
  independientes combinadas tomando siempre la más conservadora (el mínimo de las
  tres) día a día: (1) **tendencia de precio** -- reduce la exposición suavemente
  cuando el benchmark cae bajo su media móvil de 200 días; (2) **volatilidad
  REALIZADA relativa** -- reduce la exposición cuando la volatilidad realizada
  reciente (20 días) se dispara muy por encima de su nivel normal, incluso si el
  precio TODAVÍA no cayó bajo la media -- la volatilidad suele dispararse ANTES de
  una caída grande, no después; (3) **volatilidad IMPLÍCITA (VIX, opcional, solo
  acciones)** -- a diferencia de la señal 2 (qué tan volátil ESTUVO el mercado), el
  VIX es lo que el mercado de OPCIONES espera hacia adelante, y a veces se adelanta
  a la volatilidad realizada. Umbrales absolutos (VIX<20 tranquilo, VIX>35 estrés
  serio), no una razón. Sin equivalente líquido y gratuito para cripto, así que no
  se usa ahí. Ninguna de las tres llega a exposición cero por completo -- eso se lo
  deja a la guardia de drawdown reactiva de `backtest.py`, que sí puede
  justificarlo con una caída ya confirmada.
- `src/walk_forward.py` — validación fuera de muestra, con **embargo** de 5 días entre
  train/test y reporte de **estabilidad de parámetros** entre folds (si el "óptimo"
  cambia de fold a fold, te lo marca como posible ruido, no como ventaja real).
- `src/monte_carlo.py` — block bootstrap sobre el ensamble OOS: en vez de un solo
  número de retorno mensual, te da un rango de escenarios plausibles (percentiles), la
  probabilidad de caer dentro del objetivo 0.5%-2%, y una **proyección hacia adelante**
  (cono de incertidumbre) que alimenta el dashboard.
- `src/stress_test.py` — comportamiento de cada estrategia en crashes conocidos
  (2015, Q4-2018, COVID-2020, bear 2022), porque el Sharpe promedio puede esconder un
  mal momento justo cuando más importa.
- `src/tax.py` — estimación aproximada del drag fiscal (turnover alto -> mayoría de
  ganancias de corto plazo -> tasa marginal más alta). No es asesoría fiscal.
- `src/tax_loss_harvesting.py` — **cosecha automática de pérdidas fiscales**
  (opt-in, `tax.harvest_losses_enabled` en `config/live_params.yaml`, `false` por
  defecto): en cada corrida de `run_live_once.py`, si una posición tiene una
  pérdida no realizada por debajo de un umbral (`harvest_loss_threshold_pct`, -5%
  por defecto), se vende para realizar esa pérdida -- aunque la señal normal
  quisiera mantenerla -- y queda bloqueada para recompra automática durante
  `wash_sale_days` (31 por defecto), para no invalidar la pérdida ya cosechada
  (regla de "wash sale" de EE.UU.). **Alcance honestamente limitado**: solo
  funciona con `--broker virtual` o `alpaca`, los únicos que exponen cost basis
  por posición (`VirtualBroker` lo calcula él mismo con costo promedio ponderado;
  Alpaca lo trae nativo). Binance/Bitso NO lo soportan -- llevar cost basis ahí
  requeriría un ledger propio fuera de alcance de este proyecto, y el tratamiento
  fiscal de cripto varía demasiado por país para aproximarlo bien (ver `tax.py`
  arriba). La guardia de wash sale es una aproximación razonable, NO una garantía
  de cumplimiento con el IRS -- confirma el tratamiento con tu contador.
- `src/portfolio_overlays.py` — TRES overlays aplicados a los pesos de CADA
  estrategia, antes del backtest/combinación (`portfolio_overlays` en
  `config/live_params*.yaml`, los tres activos por defecto), en este orden:
  1. **Topes de posición dinámicos por correlación** (`dynamic_caps_enabled`):
     cuando el universo entero se mueve muy correlacionado (poca
     diversificación real pese a tener muchos activos), aprieta los topes de
     posición (`position_caps`) más allá de lo estático -- fuerza más
     diversificación justo cuando más importa.
  2. **Entrada escalonada de posiciones nuevas** (`ramp_in_enabled`): limita
     cuánto puede SUBIR el peso de un ticker en un solo día
     (`ramp_max_daily_increase`, 2 puntos porcentuales por defecto) -- entra a
     una posición nueva o creciente en varios días en vez de todo de una vez,
     reduciendo el riesgo de "comprar justo el techo" de un movimiento de
     corto plazo. Las BAJADAS de peso nunca se limitan (salir de riesgo es
     siempre instantáneo, mismo principio que la reentrada gradual de la
     guardia de drawdown).
  3. **Barrido de cash ocioso** (`cash_sweep_enabled`): el capital que quedó
     libre en los dos pasos anteriores, más el que la estrategia ya dejaba sin
     invertir por su cuenta (menos activos "en tendencia" de los que caben,
     vol-targeting que reduce exposición a propósito, o `sector_rotation`
     cuando nada le gana al cash), por defecto ganaba 0% -- ni siquiera la
     tasa libre de riesgo. Ahora se "barre" hacia el proxy de cash (BIL en
     acciones) para que gane su retorno real en vez de quedar fuera del
     portafolio ganando nada.

  Ninguno de los tres se aplica cuando el filtro de régimen o la guardia de
  drawdown reducen exposición -- esa reducción es una decisión de riesgo
  explícita, no una ineficiencia de cash, concentración, o velocidad de
  entrada (ver el docstring del módulo para el razonamiento completo). Un
  cuarto overlay relacionado, el blackout de eventos macro, vive en
  `src/event_blackout.py` (ver abajo) pero se compone en el mismo punto del
  pipeline.
- `src/event_blackout.py` — dos blackouts (opt-in, `event_blackout` en
  `config/live_params*.yaml`, activos por defecto) para evitar operar
  alrededor de eventos binarios que pueden mover el precio de golpe sin ser
  tendencia real:
  - **Blackout de FOMC** (`fomc_blackout_enabled`): el portafolio COMPLETO no
    rebalancea el día de una reunión de la Reserva Federal (fechas en
    `config/macro_calendar.yaml`) -- mantiene los pesos del día anterior.
    Funciona en backtest Y en vivo. **Nota honesta**: la lista de fechas es un
    punto de partida (cubre 2024-2025), no un calendario mantenido solo --
    revísala contra el calendario oficial de la Fed antes de confiar en el
    backtest para medir su efecto histórico completo, y agrega fechas nuevas
    a medida que la Fed las publique.
  - **Blackout de earnings** (`earnings_blackout_enabled`): acciones
    individuales, cerca de su reporte de resultados (consulta "best effort" a
    yfinance). **SOLO EN VIVO** -- yfinance solo expone la PRÓXIMA fecha de
    reporte de cada ticker, no un historial point-in-time confiable, así que
    esta guardia nunca aparece en ningún backtest (una discrepancia real y
    documentada entre backtest y ejecución en vivo, no escondida). Si la
    consulta falla, no bloquea nada (falla del lado seguro).
- `src/ensemble.py` — combina las 3 estrategias en un solo portafolio con una
  mezcla **fija** (`DEFAULT_ALLOCATION`, ajustable) o una asignación **dinámica**
  (`optimize_ensemble_weights`) que le da más capital, fold a fold, a la
  estrategia con menor volatilidad OOS reciente Y menos correlacionada con las
  demás (mismo ajuste por correlación de `momentum.py`/`sector_rotation.py`, ahora
  un nivel arriba, entre ESTRATEGIAS -- dos estrategias que ganan/pierden juntas en
  el mismo tramo son redundantes aunque tengan la misma volatilidad individual) --
  calculada SOLO con folds anteriores (nunca mira el fold actual ni futuros, para
  no hacer trampa). Sobre el ensamble dinámico se aplica además un tercer overlay,
  **vol-targeting a nivel de portafolio** (`apply_portfolio_vol_target`): cada
  estrategia individual ya apunta a su propia volatilidad objetivo, pero si las 3
  coinciden en modo agresivo al mismo tiempo el ensamble combinado puede terminar
  más volátil de lo que cualquiera apunta por separado -- este overlay escala la
  exposición del ensamble YA COMBINADO (de forma causal, con `shift(1)`, nunca usa
  el retorno de hoy para decidir la escala de hoy) para que la volatilidad
  anualizada del portafolio completo ronde el objetivo configurado
  (`portfolio_vol_target` en `config/live_params.yaml`), sin apalancar nunca por
  encima de `max_gross_exposure` (**1.3 en el conservador** -- ver la nota de
  riesgo real/margen en `config/live_params.yaml`; 1.0 en el agresivo, que ya
  usa ETFs 3x). El reporte del backtest
  incluye las tres variantes (`ENSEMBLE_OOS_walkforward`,
  `ENSEMBLE_OOS_dynamic_alloc`, `ENSEMBLE_OOS_dynamic_alloc_vol_target`) para
  comparar -- ninguna es automáticamente "mejor", compara CAGR, vol y drawdown.
- `src/param_drift.py` — después de cada backtest, compara lo que el walk-forward
  eligió en su fold MÁS RECIENTE contra lo que está fijo en
  `config/live_params.yaml` (lo que `run_live_once.py` usa de verdad -- los
  parámetros no se re-optimizan solos, a propósito). Si divergen más de 20%, lo
  marca como aviso (`reports/param_drift.json`, visible en el dashboard, pestaña
  "Estabilidad de parámetros") -- nunca actualiza el YAML solo, solo avisa para que
  tú decidas si vale la pena revisarlo.
- `src/data_quality.py` — detecta y limpia probables **errores de datos** (bad
  ticks del proveedor) en los precios antes de calcular cualquier señal: un
  precio inválido (<=0), o un salto extremo en un solo día que se REVIERTE casi
  por completo en los días siguientes (la firma típica de un dato erróneo que se
  autocorrige). A propósito NO toca un crash real (caída grande que no revierte
  rápido) -- el criterio combina "movimiento extremo" CON "reversión rápida",
  nunca solo lo primero, para no borrar accidentalmente los días de crash que el
  stress test necesita ver.
- `src/notify.py` + `src/tracking.py` — logging estructurado, alertas por
  email/Telegram, guardia de drawdown persistente entre corridas, y detección de
  "decay" (cuando el retorno real en vivo se sale de la banda que predijo el Monte
  Carlo del backtest).
- `src/live/alpaca_broker.py` — ejecución en **paper trading** vía Alpaca con
  **órdenes límite** (no de mercado, para acotar el slippage) y verificación de que
  el mercado esté abierto antes de enviar nada real. Después de enviar cada orden
  hace **polling de su estado real** hasta que se llena o se agota un timeout (60s
  por defecto), momento en el que la CANCELA explícitamente -- una orden límite
  enviada no es lo mismo que una orden ejecutada, y antes el bot no tenía forma de
  distinguirlas. **División de órdenes grandes tipo TWAP** (`twap_threshold_usd`,
  $5000 por defecto, ver `execution` en `config/live_params*.yaml`): una orden por
  encima del umbral se divide en hasta `twap_max_slices` (5 por defecto) porciones
  más chicas, enviadas en SECUENCIA con una pausa entre cada una
  (`twap_slice_delay_sec`), en vez de una sola orden grande de una vez -- reduce el
  impacto de mercado de golpear el libro de órdenes con todo el monto junto. El
  resultado se agrega en un solo registro por ticker (mismas claves de siempre, más
  `n_slices`/`slice_fills` con el detalle), así que el resto del bot no necesita
  saber que una orden se dividió. Solo aplica a Alpaca (`VirtualBroker` simula
  fills instantáneos, no tiene libro de órdenes que impactar).
  `src/live/binance_broker.py` y `src/live/bitso_broker.py` hacen lo
  mismo (ver abajo, módulo cripto, sin TWAP todavía). Los 4 brokers (Alpaca, Binance, Bitso, y el
  virtual) respetan una **banda de no-operación** (`min_weight_drift`, 2% por
  defecto): no rebalancean un ticker si su peso actual ya está a menos de esa
  fracción del objetivo, sin importar cuántos dólares represente esa diferencia --
  sin esto, una cuenta grande podía rebalancear posiciones cada corrida por drifts
  de peso irrelevantes (30.0% vs 30.3%), pagando costos de transacción e impuestos
  de corto plazo sin ningún beneficio real.
- `src/live/virtual_broker.py` — broker simulado local con su propio capital
  inicial, sin cuenta ni API key -- para correr los 2 perfiles en paralelo, cada uno
  con su propio dinero (ej. $1000), sin necesitar 2 cuentas de Alpaca. Ver más abajo.
- `dashboard.py` — panel interactivo (Streamlit + Plotly): equity, distribución
  Monte Carlo, proyección hacia adelante, stress test, estabilidad de parámetros,
  calendario de retornos mensuales, y estado de la cuenta en vivo. Ver más abajo.
- `config/live_params_aggressive.yaml` — **perfil agresivo opcional** (apalancamiento
  vía ETFs 3x, mayor volatilidad objetivo). Ver la sección dedicada más abajo --
  léela ANTES de usarlo.
- **Diversificación internacional y por clase de activo, DESACTIVADA por default**
  (`config/universe.yaml` -> `international_etfs`: EFA/EEM; `diversifier_etfs`:
  VNQ/DBC/IEF -- REITs, commodities amplios, bonos 7-10 años). La idea original
  era buena (sin esto, todo el universo es 100% acciones de EE.UU.), pero un
  backtest real (2015-2026, walk-forward) mostró que estos 5 tickers diluían
  `avg_monthly_return` del ensamble por debajo del 0.5% objetivo -- ni siquiera
  reduciendo su tope de posición a tamaño "satélite" (8%/12%) se recuperaba del
  todo, así que el problema era más de selección dentro de momentum/mean_reversion
  que de tamaño. Por eso `include_satellite_etfs` (`config/live_params*.yaml`)
  quedó en `false`: estos 5 tickers siguen definidos en `universe.yaml` y se
  siguen descargando, pero NO entran al universo de `momentum`/`mean_reversion`
  (nunca entraron a `sector_rotation`, que es específicamente sectores de EE.UU.)
  ni a `run_backtest.py` ni a `run_live_once.py` -- mismo flag, mismo efecto en
  los dos, ver `tests/diversifier_etfs_test.py` / `tests/international_diversification_test.py`.
  Si prefieres esa diversificación a cambio de algo de retorno, pon el flag en
  `true` -- pero corre `python scripts/run_backtest.py --include-satellite`
  primero y compara `avg_monthly_return` en `reports/summary.csv` contra el
  default antes de decidir.

### Sesgo de supervivencia (acciones individuales)

El universo de acciones individuales (`config/universe.yaml` -> `liquid_stocks`) es
la lista de HOY -- no incluye empresas que en los últimos 10 años quebraron o fueron
excluidas del índice, lo que infla artificialmente el backtest de esas posiciones (no
existe un dataset gratuito con membresía histórica punto-en-el-tiempo del S&P 500).
Mitigación aplicada: `config/live_params.yaml -> position_caps` limita cuánto puede
concentrarse el portafolio en una sola acción individual (8% por defecto) frente a un
ETF (20%-40%), y el reporte de `run_backtest.py` te lo recuerda. Si te preocupa, la
opción más simple es vaciar `liquid_stocks` en `universe.yaml` y operar solo con ETFs.

## Instalación con un clic (recomendado)

Requiere Python 3.10+ instalado (si no lo tienes, el script te avisa y te manda al
instalador oficial: https://www.python.org/downloads/).

- **Mac**: doble clic en `start_mac.command`.
  - La primera vez, macOS puede bloquearlo por venir de "un desarrollador no
    identificado" -- clic derecho sobre el archivo → **Abrir** → confirmar. Solo
    hace falta una vez.
- **Windows**: doble clic en `start_windows.bat`.

La primera vez crea el entorno virtual e instala todo (tarda unos minutos); las
siguientes veces abre directo el dashboard en tu navegador
(`http://localhost:8501`). También crea `.env` desde `.env.example` si no existe
(lo necesitas solo para paper trading en vivo, no para el dashboard ni el backtest).

Para cerrar el dashboard: vuelve a la ventana de Terminal/CMD que se abrió y
presiona `Ctrl+C`.

### Instalación manual (alternativa)

```bash
git clone <este-repo>   # o copia la carpeta a tu máquina
cd trading-bot
python3 -m venv venv

# Mac/Linux:
source venv/bin/activate
# Windows (PowerShell):
venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Cuenta de paper trading (gratis)

1. Crea una cuenta en https://alpaca.markets (broker con API, regulado en EE.UU.).
2. En el dashboard, genera claves de **Paper Trading** (no las de dinero real).
3. Copia `.env.example` a `.env` y pon tus claves:

```bash
cp .env.example .env
# edita .env con ALPACA_API_KEY y ALPACA_SECRET_KEY
```

`.env` está en `.gitignore` — nunca lo subas a un repositorio.

## Uso

### 1. Validar que el código corre bien (sin datos reales, con ruido sintético)

```bash
python tests/smoke_test.py               # prueba cada módulo por separado (acciones)
python tests/integration_smoke.py        # corre scripts/run_backtest.py completo, de punta a punta
python tests/live_smoke.py               # valida run_live_once.py + AlpacaBroker con un broker falso
python tests/crypto_smoke_test.py        # ídem, módulo cripto
python tests/crypto_integration_smoke.py # corre run_crypto_backtest.py completo, de punta a punta
python tests/crypto_live_smoke.py        # valida run_crypto_live_once.py + BinanceBroker
python tests/bitso_live_smoke.py         # valida BitsoBroker (firmado HMAC, mapeo de símbolos, órdenes)
python tests/data_incremental_test.py    # valida la descarga incremental (acciones y cripto)
python tests/ensemble_optimize_test.py   # valida la asignación dinámica (vol + correlación)
python tests/momentum_correlation_test.py # valida el ajuste por correlación de la paridad de riesgo
python tests/data_quality_test.py        # valida que distingue bad ticks de crashes reales
python tests/no_trade_band_test.py       # valida la banda de no-operación en los 4 brokers
python tests/sector_rotation_correlation_test.py  # valida el ajuste por correlación entre sectores
python tests/vol_acceleration_guard_test.py       # valida la guardia proactiva por aceleración de vol
python tests/param_drift_test.py         # valida la detección de drift de parámetros
python tests/regime_volatility_test.py   # valida la señal de volatilidad relativa del filtro de régimen
python tests/momentum_multi_horizon_test.py  # valida la confirmación multi-horizonte en momentum
python tests/portfolio_vol_target_test.py    # valida el vol-targeting a nivel de ensamble
python tests/international_diversification_test.py  # valida que EFA/EEM entran al universo (backtest y en vivo)
python tests/tax_loss_harvesting_test.py     # valida la cosecha de pérdidas fiscales y la guardia de wash sale
python tests/trailing_stop_test.py           # valida el stop trailing de mean_reversion (protege ganancias)
python tests/gradual_reentry_test.py         # valida la reentrada gradual tras la guardia de drawdown
python tests/sector_risk_adjusted_ranking_test.py  # valida el ranking por momentum/volatilidad en sector_rotation
python tests/portfolio_overlays_test.py      # valida el barrido de cash, topes dinámicos, y ramp-in de posiciones
python tests/regime_vix_test.py              # valida la señal de volatilidad implícita (VIX) del filtro de régimen
python tests/event_blackout_test.py          # valida el blackout de FOMC (backtest+vivo) y de earnings (solo vivo)
python tests/twap_order_split_test.py        # valida la división de órdenes grandes tipo TWAP (AlpacaBroker)
python tests/diversifier_etfs_test.py        # valida que VNQ/DBC/IEF entran al universo (backtest y en vivo)
python tests/exclude_satellite_test.py       # valida el flag de diagnóstico --exclude-satellite
python tests/vol_target_max_exposure_test.py # valida el flag de diagnóstico --vol-target-max-exposure
```

Esto no te dice si la estrategia es rentable, solo que no hay bugs en el pipeline (ambos
usan datos sintéticos generados localmente, no bajan nada de internet).

### 2. Backtest completo con datos reales (10 años, in-sample + walk-forward)

```bash
python scripts/run_backtest.py                       # perfil conservador (default)
python scripts/run_backtest.py --profile aggressive   # perfil agresivo -- lee la sección de abajo primero
python scripts/run_backtest.py --jobs 1               # sin paralelismo (default: todos los núcleos menos uno)
python scripts/run_backtest.py --include-satellite    # diagnóstico: corre CON EFA/EEM/VNQ/DBC/IEF (ver nota abajo)
python scripts/run_backtest.py --vol-target-max-exposure 1.0   # diagnóstico: ver nota abajo
```

Dos flags de diagnóstico (no son opciones para uso normal, no tocan `config/*.yaml`
ni el bot en vivo -- solo sirven para comparar `summary.csv` entre corridas):

- `--include-satellite` / `--exclude-satellite`: sobreescriben `include_satellite_etfs`
  (`config/live_params*.yaml`, default real `false`) solo para esa corrida, sin tocar
  el config. `--exclude-satellite` hoy es redundante con el default (ya están fuera);
  `--include-satellite` sirve para comparar contra el default y decidir si la
  diversificación de EFA/EEM/VNQ/DBC/IEF vale el costo en retorno que mostró el backtest.
- `--vol-target-max-exposure`: sobreescribe `portfolio_vol_target.max_gross_exposure`
  (default real en el conservador: **1.3**, ver la nota de riesgo en
  `config/live_params.yaml`) solo para esa corrida. Con 1.0, el overlay de
  vol-targeting del ensamble (`src/ensemble.py -> apply_portfolio_vol_target`) SOLO
  puede reducir exposición cuando la volatilidad realizada del ensamble supera el
  objetivo -- si la volatilidad realizada ya corre POR DEBAJO del objetivo (revisa
  `ann_vol` de `ENSEMBLE_OOS_dynamic_alloc` en `summary.csv` contra
  `portfolio_vol_target.vol_target`), el overlay queda inerte y
  `ENSEMBLE_OOS_dynamic_alloc_vol_target` sale idéntico a `ENSEMBLE_OOS_dynamic_alloc`.
  Un backtest real mostró exactamente eso (volatilidad realizada ~5.5% vs objetivo 10%),
  y subir el tope a 1.3 (apalancamiento moderado, real en vivo) cruzó el objetivo de
  0.5-2%/mes -- por eso quedó así en el conservador. Usa este flag con un valor distinto
  (ej. 1.0) para comparar sin tocar el config.

El walk-forward corre en paralelo por defecto (`--jobs`, ver la sección "Rendimiento"
más abajo) -- en una Mac de 8 núcleos esto corta el tiempo de la corrida varias veces
respecto a la versión secuencial, sin cambiar ningún número del resultado.

Descarga datos de Yahoo Finance (gratis, sin API key) para el universo definido en
`config/universe.yaml`, corre las 3 estrategias + el ensamble, y genera (en `reports/`
para el perfil conservador, `reports_aggressive/` para el agresivo):

- `reports/summary.csv` — tabla comparativa de métricas (CAGR, Sharpe, drawdown máximo,
  retorno mensual promedio, % de meses positivos, etc.) -- incluye tanto
  `ENSEMBLE_OOS_walkforward` (mezcla fija) como `ENSEMBLE_OOS_dynamic_alloc`
  (asignación dinámica entre estrategias, ver `src/ensemble.py` arriba).
- `reports/ensemble_dynamic_allocations.json` — qué % de capital le tocó a cada
  estrategia en cada fold de la asignación dinámica.
- `reports/equity_oos.png` — gráfico de las curvas de equity fuera de muestra
  (incluye ambos ensambles).
- `reports/ensemble_monthly_returns.csv` — retorno mes a mes del ensamble (mezcla fija).
- `reports/param_stability_<estrategia>.csv` — qué parámetro ganó en cada fold del
  walk-forward. Si salta de un extremo a otro, ese "óptimo" es ruido, no una ventaja real.
- `reports/monte_carlo.json` + `monte_carlo_hist.png` — 1000 escenarios simulados
  (block bootstrap) del ensamble OOS: percentiles del retorno mensual promedio y
  probabilidad de caer en el rango objetivo 0.5%-2%. Este archivo también lo usa
  `run_live_once.py` para detectar si el desempeño real se desvía de lo esperado.
- `reports/stress_test_insample.csv` / `stress_test_oos.csv` — cómo le fue a cada
  estrategia en crashes conocidos (Q4-2018, COVID-2020, bear 2022).
- `reports/data_quality_outliers.csv` (solo si se detectó alguno) — precios que
  `src/data_quality.py` identificó como probables errores de datos y limpió antes
  de calcular cualquier señal (ver `src/data_quality.py` arriba).
- `reports/param_drift.json` — parámetros de `config/live_params.yaml` que ya no
  coinciden con el fold más reciente del walk-forward (ver `src/param_drift.py`
  arriba). Vacío si no hay drift relevante.

Revisa la fila `ENSEMBLE_OOS_walkforward` de `summary.csv` y, sobre todo, el rango de
percentiles de `monte_carlo.json`: esa es la estimación honesta, con incertidumbre
explícita, no un solo número optimista. Si la mayoría del rango cae fuera de 0.5%–2%,
ajusta parámetros/universo en `config/universe.yaml` o `scripts/run_backtest.py` (los
grids `MOMENTUM_GRID`, `MEAN_REV_GRID`, `ROTATION_GRID`) y vuelve a correr — pero **no
sobre-ajustes mirando el número final**; deja que el walk-forward y la estabilidad de
parámetros decidan.

### 3. Paper trading diario (una vez que el backtest te convence)

```bash
python scripts/run_live_once.py              # perfil conservador, dry-run: solo muestra qué haría
python scripts/run_live_once.py --execute    # perfil conservador, ejecuta de verdad en la cuenta PAPER
python scripts/run_live_once.py --profile aggressive --execute   # perfil agresivo -- ver sección de abajo
```

Los parámetros que usa en vivo están en `config/live_params.yaml` (no se
re-optimizan solos en cada corrida a propósito — actualízalos manualmente cada
3-6 meses revisando qué ganó en el walk-forward más reciente).

Cada corrida además:
- registra el equity de la cuenta en `reports/<broker>/tracking.sqlite3` (una
  subcarpeta por broker -- ver "Rendimiento y confiabilidad" más abajo);
- mantiene una **guardia de drawdown persistente** entre corridas (no solo dentro de
  un backtest) que reduce la exposición a la mitad si el drawdown desde el pico pasa -15%;
- compara el retorno real contra la banda esperada del Monte Carlo (`reports/monte_carlo.json`,
  generado por `run_backtest.py`) y avisa si el desempeño se sale de lo plausible;
- manda una alerta si algo de esto pasa, o si la corrida falla — configura `SMTP_*`
  y/o `TELEGRAM_*` en `.env` (ver `.env.example`); si no configuras nada, solo queda
  registrado en `reports/<broker>/live.log`.

Ejecuta órdenes **límite** (no de mercado) con un slippage máximo configurable
(`config/live_params.yaml -> execution.max_slippage_pct`), y verifica que el mercado
esté abierto antes de mandar nada real -- si corres el script fuera de horario con
`--execute`, simplemente no hace nada (lo deja loggeado) en vez de fallar.

### 3b. Los 2 perfiles en paralelo, cada uno con su propio capital (ej. $1000)

Alpaca solo da UN balance de paper trading por cuenta -- para correr conservador y
agresivo a la vez, cada uno con $1000, sin crear dos cuentas de Alpaca, usa el
**broker virtual** (`--broker virtual`): simula día a día con precios de cierre
reales y su propio capital inicial, sin ninguna cuenta ni API key de por medio.

```bash
# Perfil conservador, $1000, persiste entre corridas:
python scripts/run_live_once.py --broker virtual --starting-cash 1000 --execute

# Perfil agresivo, otros $1000, completamente aislado del anterior:
python scripts/run_live_once.py --broker virtual --starting-cash 1000 --profile aggressive --execute
```

`--starting-cash` solo importa la primera vez que corre para ese perfil -- después
el capital y las posiciones quedan guardados en
`reports/virtual/virtual_broker_state.json` (o `reports_aggressive/virtual/...`) y
cada corrida parte de ahí, no reinicia. Corre esto una vez al día (a mano, o
automatízalo -- ver más abajo) y en el dashboard, pestaña **En vivo**, cambia entre
"🟢 Conservador" y "🔴 Agresivo" (y ahí, entre broker "alpaca"/"virtual") para ver
los equity curves avanzando de forma independiente, cada uno desde sus $1000.

Cada combinación perfil+broker escribe en su propia subcarpeta (`reports/virtual/`,
`reports/alpaca/`, `reports_aggressive/virtual/`, etc.) -- corre `--broker alpaca` y
`--broker virtual` del mismo perfil al mismo tiempo si quieres, no comparten guardia
de drawdown ni historial de equity entre sí.

Es una simulación honesta con datos de mercado reales día a día (no un backtest
histórico), pero no pasa por ningún exchange real -- si en algún punto quieres pasar
uno de los dos a Alpaca (paper) de verdad, usa `--broker alpaca` (el default) con esa
cuenta.

Para automatizarlo diario, agrega estas líneas a tu crontab (Mac/Linux) -- o
duplica la lógica del `.plist` de la sección de automatización, apuntando a estos
mismos comandos:

```
32 9 * * 1-5 cd /ruta/a/trading-bot && venv/bin/python scripts/run_live_once.py --broker virtual --starting-cash 1000 --execute >> reports/virtual_cron.log 2>&1
33 9 * * 1-5 cd /ruta/a/trading-bot && venv/bin/python scripts/run_live_once.py --broker virtual --starting-cash 1000 --profile aggressive --execute >> reports_aggressive/virtual_cron.log 2>&1
```

### 4. Dashboard (gráficas, estadísticas, proyecciones)

```bash
streamlit run dashboard.py
```

Abre un panel en tu navegador (`http://localhost:8501`) leyendo lo que ya generaron
`run_backtest.py` y `run_live_once.py` en `reports/` -- no se conecta a ninguna red,
así que es seguro dejarlo abierto todo el día. Incluye:

- **Resumen**: métricas clave (CAGR, Sharpe, drawdown, % meses positivos) y curvas de
  equity out-of-sample de cada estrategia + el ensamble.
- **Monte Carlo & Proyección**: distribución de escenarios plausibles y un **cono de
  incertidumbre** -- proyecta tu capital inicial hacia adelante (6 a 60 meses,
  ajustable) usando la misma simulación por block bootstrap, mostrando el rango
  p5-p95, no una sola línea falsamente precisa.
- **Stress Test**: barras comparando cómo le fue a cada estrategia en cada crash
  conocido.
- **Estabilidad de parámetros**: qué tan consistente fue el "óptimo" del walk-forward
  entre folds.
- **Retornos mensuales**: mapa de calor año x mes.
- **En vivo**: selector de broker (cada uno con su propio tracking) + tu equity real
  superpuesto sobre la banda esperada del Monte Carlo, estado de la guardia de
  drawdown, y las últimas líneas del log -- para ver de un vistazo si el bot está
  haciendo lo que se esperaba.

## Automatizar la corrida diaria (para que "el bot se encargue de todo")

### Con un clic (recomendado)

- **Mac**: doble clic en `install_daily_automation_mac.command` (requiere haber
  corrido `start_mac.command` al menos una vez antes).
- **Windows**: doble clic en `install_daily_automation_windows.bat` (requiere haber
  corrido `start_windows.bat` al menos una vez antes).

Instala DOS tareas programadas: una diaria (días hábiles, 9:35am hora de Nueva York
-- ajusta si tu zona horaria es distinta) que calcula la señal del bot, y una semanal
(sábados) que re-corre el backtest completo para refrescar el dashboard y las bandas
de alerta.

**Por seguridad, quedan instaladas en modo SIMULACIÓN (dry-run) por defecto** -- no
envían ninguna orden real, ni siquiera en paper trading, hasta que tú edites la tarea
manualmente y agregues `--execute` (el instalador te dice exactamente cómo, al final).
Esto es intencional: automatizar el envío de órdenes reales debe ser una decisión
explícita tuya, no algo que un script active solo.

Para desinstalar, cada script imprime el comando exacto al final de su instalación.

### Manual (si prefieres no usar los instaladores, o usas Linux)

#### macOS (launchd — más confiable que cron en Mac porque sobrevive reinicios/sleep)

Crea `~/Library/LaunchAgents/com.tradingbot.daily.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.tradingbot.daily</string>
  <key>ProgramArguments</key>
  <array>
    <string>/ruta/a/trading-bot/venv/bin/python</string>
    <string>/ruta/a/trading-bot/scripts/run_live_once.py</string>
    <string>--execute</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>9</integer>
    <key>Minute</key><integer>35</integer>
  </dict>
  <key>StandardOutPath</key><string>/ruta/a/trading-bot/reports/live.log</string>
  <key>StandardErrorPath</key><string>/ruta/a/trading-bot/reports/live_error.log</string>
</dict>
</plist>
```

Luego: `launchctl load ~/Library/LaunchAgents/com.tradingbot.daily.plist`

(9:35am hora de Nueva York, 5 minutos después de la apertura, para evitar la
volatilidad del primer minuto — ajusta la hora según tu zona horaria).

#### Windows (Task Scheduler)

1. Abre "Task Scheduler" → "Create Basic Task".
2. Trigger: diario, hora de tu elección (ej. 9:35am ET convertido a tu zona horaria).
3. Action: "Start a program" →
   - Program: `C:\ruta\a\trading-bot\venv\Scripts\python.exe`
   - Arguments: `scripts\run_live_once.py --execute`
   - Start in: `C:\ruta\a\trading-bot`
4. En "Conditions", desmarca "Start the task only if the computer is on AC power" si
   usas laptop.

#### Alternativa simple (cron, Mac/Linux, si tu máquina siempre está encendida)

```
35 9 * * 1-5 cd /ruta/a/trading-bot && venv/bin/python scripts/run_live_once.py --execute >> reports/live.log 2>&1
```

#### Segunda tarea programada: refrescar el backtest semanalmente

`run_live_once.py` corre TODOS los días hábiles y usa los parámetros fijos de
`config/live_params.yaml`. Para que el dashboard, el Monte Carlo y la banda de drift
se mantengan al día (no que se queden congelados con datos de hace meses), agrega una
**segunda** tarea programada, semanal, que re-corra `run_backtest.py`:

```
# cron (Mac/Linux), ej. sábados 8am:
0 8 * * 6 cd /ruta/a/trading-bot && venv/bin/python scripts/run_backtest.py >> reports/backtest.log 2>&1
```

En macOS con launchd, duplica el `.plist` de arriba con otro `Label`, apuntando a
`scripts/run_backtest.py` y con `StartCalendarInterval` semanal (`Weekday: 6`). En
Windows Task Scheduler, un segundo trigger semanal igual que el diario pero apuntando
a `run_backtest.py`.

Con ambas tareas corriendo, el bot queda completamente desatendido: se re-evalúa a sí
mismo cada semana, opera cada día, y avisa (alerta) si algo se sale de lo esperado --
tú solo entras a `streamlit run dashboard.py` quincenalmente para revisar cómo va.

## Perfil agresivo (apalancado) -- lee esto completo antes de tocarlo

`config/live_params_aggressive.yaml` es un segundo perfil de riesgo, pensado para
quien entiende que "más retorno esperado" y "más riesgo de pérdida" son la misma
perilla, no dos cosas separadas. Comparado con el perfil conservador (default):

- Agrega **ETFs apalancados 3x** (TQQQ, SPXL, SOXL) al universo de la estrategia de
  **momentum únicamente** -- nunca a mean reversion, porque el "decay" de los
  apalancados en mercados laterales es especialmente dañino para una estrategia de
  entradas y salidas cortas.
- Sube el objetivo de volatilidad de momentum de 10% a 30% anualizado (3x).
- Concentra más la rotación sectorial (`top_n: 2` en vez de `3`).
- Sube los topes de posición por tipo de activo.

Lo que **no** cambia, a propósito: la guardia de drawdown y el filtro de régimen
macro siguen igual de estrictos que en el conservador -- son la única razón por la
que este perfil no es una ruleta rusa total. Aun así, con apalancamiento 3x, un mal
tramo de mercado puede producir un drawdown mucho más profundo y mucho más rápido
que en el perfil conservador.

**Cómo evaluarlo (en este orden, sin saltarte ninguno):**

1. `python scripts/run_backtest.py --profile aggressive`
2. `streamlit run dashboard.py` → selector "🔴 Agresivo" → compara, contra el
   conservador: el `max_drawdown` de `summary.csv`, el percentil p95 (peor caso) del
   Monte Carlo, y las columnas `_max_dd_%` del stress test en cada crash conocido.
3. Si después de ver esos números todavía te parece un riesgo aceptable, corre
   `python scripts/run_live_once.py --profile aggressive` (dry-run) durante varias
   semanas antes de agregar `--execute`.
4. Nunca lo conectes a una cuenta con dinero real sin haberlo corrido en paper
   trading, con este perfil, durante varios meses -- y sin haber vivido al menos un
   drawdown real con él (paper) para saber si puedes tolerarlo emocionalmente.

## Módulo cripto (Binance)

Un track completamente separado del bot de acciones -- comparte el motor (backtest,
walk-forward, Monte Carlo, stress test, ensamble, alertas, broker virtual) pero con su
propia fuente de datos, universo, y parámetros calibrados para un mercado que cotiza
**365 días al año** (no 252 días hábiles) y es varias veces más volátil que un ETF.

### Binance, Bitso y por qué no GBM (todavía)

Antes de construir esto investigué las tres cuentas que mencionaste:

- **Binance**: API pública muy completa y documentada, con testnet oficial para paper
  trading real (`testnet.binance.vision`) -- por eso se construyó primero. Fuente de
  datos históricos para las señales de AMBOS brokers (ver abajo).
- **Bitso**: API REST documentada (`docs.bitso.com`), pero **sin testnet/sandbox
  oficial** -- a diferencia de Alpaca y Binance, no hay forma de hacer "paper trading
  real" contra el exchange. `src/live/bitso_broker.py` lo integra igual (firmado
  HMAC-SHA256 verificado contra la documentación oficial), pero con una guardia extra:
  exige `BITSO_CONFIRM_REAL_MONEY=true` explícito en `.env`, porque cualquier orden
  que no sea dry-run mueve pesos mexicanos de verdad. El broker virtual sigue siendo
  el default y la forma recomendada de probar la estrategia sin riesgo.
- **GBM**: lo que expone públicamente (`gbm.com`) es una API de **Open Banking**, de
  solo lectura (consultar saldos/movimientos) bajo la Ley Fintech mexicana -- no
  encontré evidencia de una API de envío de órdenes de autoservicio para retail. No es
  integrable para trading automático con lo que hay disponible públicamente; tendrías
  que confirmar con GBM directamente si existe algo distinto para cuentas
  institucionales.

Bitso no expone velas históricas públicas (no hay endpoint de klines/OHLC), así que
`run_crypto_backtest.py` sigue usando datos de Binance para validar la estrategia --
el movimiento relativo de BTC/ETH/etc. es prácticamente el mismo en cualquier exchange
grande (el arbitraje lo mantiene así), así que la señal es igual de válida. Lo que sí
viene de Bitso, en vivo, es el **precio de referencia en MXN** al momento de calcular
cada orden (su propio ticker público) -- así que el tamaño de cada orden es exacto
para tu cuenta en pesos, no una conversión aproximada desde USDT.

### Instalación y universo

Ya incluido en `requirements.txt` (`python-binance`, `requests`) -- si ya corriste
`start_mac.command`/`start_windows.bat` o `pip install -r requirements.txt`, no hay
nada extra que instalar.

`config/crypto_universe.yaml`: 2 "majors" (BTC, ETH) + 8 altcoins líquidas contra
USDT. `USDT` se trata como "cash" (serie sintética plana, no se descarga -- no es un
par tradeable contra sí mismo). Edítalo para agregar/quitar monedas.

`config/crypto_live_params.yaml`: mismo formato que `live_params.yaml`, con las
diferencias que importan para cripto ya documentadas en comentarios dentro del
archivo -- `periods_per_year: 365`, volatilidad objetivo más alta, costos calibrados
al fee spot de Binance (~0.1%), y **sin estimación de impuestos por defecto** (el
tratamiento fiscal de cripto varía demasiado por país, y en México no está bien
definido -- consulta a tu contador).

### Backtest y paper trading

```bash
python scripts/run_crypto_backtest.py       # backtest completo con datos reales de Binance
streamlit run dashboard.py                   # selector de perfil -> "🪙 Cripto (Binance)"
```

Genera el mismo tipo de reporte honesto que el bot de acciones (walk-forward, Monte
Carlo, estabilidad de parámetros) pero con stress test contra crashes **cripto**
específicos: mayo 2021, colapso de Terra/Luna (mayo 2022), colapso de FTX
(noviembre 2022), y el bear market completo de 2022 -- ver `src/stress_test.py ->
CRYPTO_CRISIS_PERIODS`.

```bash
# Con capital propio simulado, sin ninguna cuenta (recomendado para empezar):
python scripts/run_crypto_live_once.py --starting-cash 1000 --execute

# Contra el testnet real de Binance (requiere BINANCE_API_KEY/SECRET_KEY en .env):
python scripts/run_crypto_live_once.py --broker binance --execute

# Contra Bitso -- SIN TESTNET, esto mueve MXN real. Requiere BITSO_API_KEY/
# SECRET_KEY Y BITSO_CONFIRM_REAL_MONEY=true en .env, o lanza una excepción:
python scripts/run_crypto_live_once.py --broker bitso --execute
```

Como no hay "mercado cerrado" en cripto, no existe el chequeo de horario que sí tiene
`run_live_once.py` -- pero sigue usando órdenes límite con slippage máximo
configurable (`config/crypto_live_params.yaml -> execution.max_slippage_pct`).

**Antes de usar `--broker bitso` con dinero real:** corre semanas con `--broker
virtual` primero, revisa el Monte Carlo y el stress test en el dashboard, y cuando
decidas usar Bitso de verdad, hazlo primero con el monto mínimo posible (el código no
conoce los mínimos exactos por libro de Bitso -- usa un piso configurable de $50 MXN
por orden, `min_order_mxn`, que es una aproximación, no el mínimo real del exchange;
una orden por debajo del mínimo real de Bitso simplemente será rechazada y se
loggeará el error sin detener la corrida).

### Automatizar

Mismo patrón que el bot de acciones (ver sección de automatización más arriba) --
agrega una línea a tu crontab/launchd/Task Scheduler apuntando a
`scripts/run_crypto_live_once.py` en vez de `run_live_once.py`. Como cripto cotiza
24/7, puedes correrlo más de una vez al día si quieres (ej. cada 6-12 horas) en vez de
una sola corrida diaria -- ajusta la frecuencia según qué tan seguido quieres que
rebalancee (más frecuencia = más turnover = más costos, no necesariamente mejor).

### Diferencia importante con el bot de acciones: sesgo de "supervivencia" cripto

El universo de altcoins es el de HOY. A diferencia de las acciones (donde al menos el
S&P 500 tiene índices históricos), en cripto no hay un dataset gratuito de "qué
monedas estaban entre las top-10 por market cap en 2019" -- si un token relevante de
hace unos años ya no existe o perdió toda su liquidez, simplemente no aparece en este
backtest. Trátalo como una limitación conocida, no como algo que se pueda arreglar
fácilmente.

## Rendimiento y confiabilidad

Mejoras internas, transparentes para el uso normal (no cambian ningún resultado
del backtest, solo qué tan rápido/seguro se llega a él):

- **Descargas en paralelo, e INCREMENTALES** (`src/data.py`, `src/crypto_data.py`):
  lo que ya está en caché se lee del disco normal; lo que hace falta bajar de Yahoo
  Finance o Binance se pide con varios hilos a la vez (son llamadas de red, no de
  CPU). Además, en corridas diarias (`--refresh-data`, o `force=True` que usa
  `run_live_once.py`/`run_crypto_live_once.py`) ya NO se vuelve a descargar el
  historial completo de cada ticker -- solo los días desde el último dato en caché
  (con unos días de margen por si el proveedor revisa un cierre reciente), y se
  pegan al caché existente sin duplicar fechas. Antes, cada corrida diaria bajaba
  de nuevo 10+ años de historia por cada ticker/símbolo; ahora son unos pocos días.

- **`mean_reversion.py` acelerado con numba** (opcional, ver `requirements.txt`):
  esta estrategia es una "state machine" día a día (necesita saber si hay una
  posición abierta, cuánto lleva, si tocó el stop) que no se puede vectorizar con
  operaciones normales de pandas/numpy -- pero SÍ se puede compilar. Si `numba` está
  instalado, ese loop corre compilado a código máquina (10-50x más rápido que Python
  puro interpretado); si falla instalar `numba` en tu máquina, el bot cae de vuelta
  automáticamente al mismo loop en Python normal -- mismo resultado exacto, solo más
  lento. Nunca es un requisito duro.

- **Walk-forward sin cómputo redundante**: antes, el backtest de la combinación de
  parámetros ganadora de cada fold se corría dos veces (una para elegirla, otra para
  sacar el tramo de test) -- ahora se cachea y se reusa.

- **Walk-forward en paralelo** (`--jobs`, default: todos los núcleos menos uno): la
  parte más cara -- generar los pesos de cada combinación única del grid de
  parámetros (especialmente mean_reversion, que hace un loop por ticker) -- se reparte
  entre procesos con `multiprocessing`, usando el contexto `spawn` explícitamente
  (el default en macOS/Windows) para que funcione igual en tu Mac que en Linux. Por
  eso las funciones que arma `weights_fn` en `run_backtest.py`/`run_crypto_backtest.py`
  usan `functools.partial` en vez de lambdas -- un lambda no se puede mandar a otro
  proceso con `spawn`. `--jobs 1` desactiva el paralelismo si alguna vez lo necesitas
  (debug, por ejemplo).

- **Tracking en SQLite, separado por broker** (`src/tracking.py`): antes, el equity
  en vivo y la guardia de drawdown se guardaban en un CSV/JSON simple por perfil.
  Dos problemas reales que esto tenía: (1) un `to_csv()` que trunca y reescribe todo
  el archivo puede corromperse si dos corridas escriben a la vez -- con varios
  perfiles y brokers corriendo por cron, posiblemente al mismo tiempo, no era solo
  teórico; (2) **si corrías `--broker alpaca` y `--broker virtual` del mismo perfil
  (o `virtual`/`binance`/`bitso` en cripto), compartían el mismo archivo de
  tracking y se mezclaban sin que nada lo avisara** -- un bug real, no solo una
  mejora de rendimiento. Ahora cada perfil+broker tiene su propia base SQLite en
  `reports/<broker>/tracking.sqlite3` (con `PRAGMA busy_timeout` -- un escritor
  concurrente espera en vez de fallar o corromper el archivo) y su propio
  `live.log`. El dashboard tiene un selector de broker en la pestaña "En vivo" para
  elegir cuál ver.

## Estructura del proyecto

```
trading-bot/
  start_mac.command                            # doble clic: instala todo + abre el dashboard (Mac)
  start_windows.bat                             # lo mismo, Windows
  install_daily_automation_mac.command           # doble clic: instala la automatización diaria/semanal (Mac)
  install_daily_automation_windows.bat            # lo mismo, Windows
  config/
    universe.yaml         # qué activos operar (incluye leveraged_etfs, international_etfs, diversifier_etfs)
    live_params.yaml       # perfil conservador (default) -- riesgo/costos/régimen, vivo Y backtest
    live_params_aggressive.yaml  # perfil agresivo opcional -- LEE la sección dedicada antes de usarlo
    macro_calendar.yaml     # fechas de FOMC para el blackout de eventos macro (lista de partida, revisar)
  src/
    data.py                 # descarga (completa o incremental) y cachea precios+volumen (Yahoo Finance)
    data_quality.py          # detecta y limpia bad ticks -- nunca toca crashes reales
    indicators.py           # SMA, EMA, RSI, ATR, Bollinger, momentum
    costs.py                 # modelo de costos sensible a liquidez
    regime.py                 # filtro de régimen macro proactivo (tendencia + vol. realizada + VIX opcional)
    backtest.py                # motor de backtest + guardia de drawdown (reactiva con reentrada gradual + proactiva por vol)
    metrics.py                  # CAGR, Sharpe, drawdown, retornos mensuales
    monte_carlo.py                # block bootstrap + proyección hacia adelante
    stress_test.py                 # comportamiento en crashes conocidos
    walk_forward.py                 # validación fuera de muestra + estabilidad (paralelo, --jobs)
    tax.py                           # estimación aproximada de drag fiscal
    tax_loss_harvesting.py           # cosecha automática de pérdidas fiscales (opt-in) + guardia de wash sale
    portfolio_overlays.py             # topes dinámicos por correlación + ramp-in de posiciones + barrido de cash
    event_blackout.py                  # blackout de FOMC (backtest+vivo) y de earnings (solo vivo)
    ensemble.py                       # combina las 3 estrategias -- mezcla fija, dinámica, o + vol-targeting de portafolio
    param_drift.py                     # avisa si config/live_params.yaml quedó desactualizado vs. el walk-forward
    notify.py                           # logging + alertas email/Telegram
    tracking.py                          # equity en vivo + guardia persistente (drawdown con reentrada gradual) + drift
    strategies/
      momentum.py                          # paridad de riesgo + ajuste por correlación + confirmación multi-horizonte
      mean_reversion.py                    # RSI(2) + stop TRAILING (protege ganancias)
      sector_rotation.py                   # dual momentum + ranking ajustado a riesgo + ajuste por correlación
    live/
      alpaca_broker.py                   # órdenes límite + fill + banda de no-operación + división TWAP
      binance_broker.py                   # cripto, órdenes límite + filtros del exchange (testnet) + fill
      bitso_broker.py                      # cripto/MXN, firmado HMAC -- SIN testnet, mueve dinero real + fill
      virtual_broker.py                   # simulación local, capital propio, sin cuenta (acciones Y cripto)
  scripts/
    run_backtest.py                        # reporte completo con datos reales (acciones)
    run_live_once.py                        # corrida diaria (cron/Task Scheduler, acciones)
    run_crypto_backtest.py                   # lo mismo, módulo cripto (Binance)
    run_crypto_live_once.py                   # corrida diaria/frecuente, módulo cripto
  dashboard.py                               # panel interactivo, 3 perfiles (streamlit run dashboard.py)
  tests/
    smoke_test.py                            # valida cada módulo con datos sintéticos (acciones)
    integration_smoke.py                      # corre run_backtest.py completo, sintético (ambos perfiles)
    live_smoke.py                              # valida run_live_once.py + broker, sin red
    crypto_smoke_test.py                        # ídem, módulo cripto
    crypto_integration_smoke.py                  # corre run_crypto_backtest.py completo, sintético
    crypto_live_smoke.py                          # valida run_crypto_live_once.py + BinanceBroker, sin red
    bitso_live_smoke.py                            # valida BitsoBroker (firmado HMAC, mapeo, órdenes), sin red
    data_incremental_test.py                         # valida la descarga incremental (acciones y cripto)
    ensemble_optimize_test.py                         # valida la asignación dinámica (vol + correlación)
    momentum_correlation_test.py                       # valida el ajuste por correlación de la paridad de riesgo
    data_quality_test.py                                # valida que distingue bad ticks de crashes reales
    no_trade_band_test.py                                # valida la banda de no-operación en los 4 brokers
    sector_rotation_correlation_test.py                   # valida el ajuste por correlación entre sectores
    vol_acceleration_guard_test.py                         # valida la guardia proactiva por aceleración de vol
    param_drift_test.py                                     # valida la detección de drift de parámetros
    regime_volatility_test.py                                 # valida la señal de volatilidad del filtro de régimen
    momentum_multi_horizon_test.py                             # valida la confirmación multi-horizonte en momentum
    portfolio_vol_target_test.py                                # valida el vol-targeting a nivel de ensamble
    international_diversification_test.py                       # valida EFA/EEM en el universo (backtest y en vivo)
    tax_loss_harvesting_test.py                                   # valida la cosecha de pérdidas y la guardia de wash sale
    trailing_stop_test.py                                          # valida el stop trailing de mean_reversion
    gradual_reentry_test.py                                         # valida la reentrada gradual tras la guardia de drawdown
    sector_risk_adjusted_ranking_test.py                             # valida el ranking por momentum/volatilidad
    portfolio_overlays_test.py                                       # valida topes dinámicos, ramp-in y barrido de cash
    regime_vix_test.py                                                # valida la señal de volatilidad implícita (VIX)
    event_blackout_test.py                                             # valida el blackout de FOMC y de earnings
    twap_order_split_test.py                                           # valida la división de órdenes tipo TWAP
    diversifier_etfs_test.py                                            # valida VNQ/DBC/IEF en el universo (backtest y en vivo)
    exclude_satellite_test.py                                           # valida el flag de diagnóstico --exclude-satellite
    vol_target_max_exposure_test.py                                     # valida el flag de diagnóstico --vol-target-max-exposure
  config/
    crypto_universe.yaml                     # universo cripto (Binance)
    crypto_live_params.yaml                   # parámetros del módulo cripto
  reports/                                     # perfil conservador (acciones) -- se genera al correr los scripts
    summary.csv, monte_carlo.json, ...           # del backtest -- compartido entre brokers de este perfil
    alpaca/tracking.sqlite3, alpaca/live.log       # tracking en vivo -- una subcarpeta POR BROKER
    virtual/tracking.sqlite3, virtual/live.log
  reports_aggressive/                           # perfil agresivo (acciones) -- misma estructura
  reports_crypto/                               # módulo cripto -- misma estructura (virtual/binance/bitso)
```

## Próximos pasos razonables (no incluidos todavía)

- Re-optimización periódica automática de `live_params.yaml` con un walk-forward
  incremental (hoy es manual a propósito, para forzarte a revisar los resultados).
- Consultar filtros exactos por libro en Bitso (mínimos de orden reales, no la
  aproximación de `min_order_mxn`) si vas a operar montos chicos ahí.
- Modelo fiscal más preciso: `src/tax_loss_harvesting.py` ya cosecha pérdidas
  automáticamente (ver "Qué incluye"), pero con costo promedio ponderado, no por
  LOTE individual (FIFO/lote específico) -- y no lleva pérdidas compensables entre
  años. Además, sigue limitado a `--broker virtual`/`alpaca` (Binance/Bitso no
  exponen cost basis). Un motor contable completo por lote resolvería ambas cosas
  si el drag estimado en `src/tax.py` te importa lo suficiente como para
  justificarlo.
- Universo de acciones individuales point-in-time real (requiere un dataset de pago
  con membresía histórica del índice) para eliminar del todo el sesgo de
  supervivencia, más allá del tope de posición que ya mitiga su impacto.
- El barrido de cash ocioso (`src/portfolio_overlays.py`) en el módulo CRIPTO
  barre hacia `quote_currency` (USDT), que es una serie sintética PLANA -- a
  diferencia de BIL (con retorno real de T-Bills), esto no cambia ningún número
  del backtest cripto todavía, solo hace explícito en los reportes cuánto quedó
  en stablecoin. Modelar un retorno real de staking/lending de USDT (si algún día
  se opera a través de un exchange que lo ofrezca) le daría efecto económico real.
- `config/macro_calendar.yaml` (blackout de FOMC, ver "Qué incluye") solo cubre
  fechas de reunión de la Fed, no CPI ni otros datos macro -- y su lista de
  fechas (2024-2025) hay que revisarla/ampliarla contra el calendario oficial
  antes de confiar en el backtest para medir el efecto histórico completo, o
  antes de dejarla correr en vivo por mucho tiempo sin agregar fechas nuevas.
- Blackout de earnings vía yfinance (ver "Qué incluye") es "best effort" -- sin
  reintentos ni caché entre corridas; si la consulta falla justo ese día, ese
  ticker simplemente no queda protegido esa corrida (falla del lado seguro, no
  bloquea nada, pero tampoco garantiza cobertura).
- TWAP (`src/live/alpaca_broker.py`) reparte una orden grande en varias
  porciones dentro de la MISMA corrida (segundos/minutos), no a lo largo de
  varias horas del día -- un TWAP real de mesa institucional se extiende por
  más tiempo. Suficiente para el patrón de este bot (una corrida diaria vía
  cron), no para trading intradía. Tampoco existe todavía para Binance/Bitso
  (solo Alpaca).
- Notificaciones push nativas (hoy es email/Telegram) o una versión del dashboard
  desplegada (hoy es local, `streamlit run dashboard.py`).

## Disclaimer

Esto no es asesoría financiera. Los resultados de backtest, incluso walk-forward, no
garantizan resultados futuros. Los mercados cambian de régimen; una estrategia que
funcionó en los últimos 10 años puede dejar de funcionar. Usa dinero que puedas
permitirte perder, empieza en paper trading, y entiende cada línea de este código
antes de conectarlo a una cuenta con dinero real.
