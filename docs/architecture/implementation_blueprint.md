# Implementation Blueprint: Option Strategy Infrastructure

## 1. Proposito

Este documento define el primer blueprint de implementacion para convertir el
proyecto de cuantitativas en una infraestructura local, modular y reproducible
para investigacion y ejecucion analitica de estrategias con opciones usando
ThetaData como fuente principal de datos de mercado.

La topologia objetivo es:

```text
Modular Monolith + Pipeline local reproducible
```

El sistema debe ser suficientemente simple para trabajarse dentro de este
repositorio, pero con fronteras logicas claras para que en el futuro pueda
evolucionar hacia una arquitectura mas productiva.

## 2. Principios De Arquitectura

### 2.1 Todo es trade-off

El sistema priorizara reproducibilidad, claridad de contratos y velocidad de
investigacion sobre escalabilidad distribuida. SQLite, CSV y ejecucion local son
suficientes para la primera etapa, aunque no sean la solucion final para alto
volumen o produccion.

### 2.2 Por que > como

El objetivo no es solo descargar datos. El objetivo es poder defender, auditar y
comparar estrategias con opciones. Por eso los datos, modelos, estrategias y
reportes deben quedar separados.

### 2.3 Espectro de decisiones

Algunas decisiones se fijan desde el inicio:

- `src/modulos` sera el nucleo nuevo de codigo importable.
- Los notebooks no deben consultar directamente ThetaData.
- Toda estrategia debe consumir contratos de datos procesados.
- Toda corrida debe producir metricas comparables.

Otras decisiones quedan abiertas:

- SQLite vs Parquet como almacenamiento principal de largo plazo.
- Frecuencia de rebalanceo.
- Universo inicial de tickers.
- Modelo principal para valuacion avanzada: Heston, Merton jump diffusion u otro.
- Costos de transaccion y supuestos de liquidez.

## 3. Estado Inicial Del Repositorio

| Elemento | Estado | Evidencia | Siguiente accion |
|---|---|---|---|
| ThetaData local | Parcial | `src/proyect/clean_data.ipynb`, `docs/basic_commands.md` | Extraer a modulos Python |
| Yahoo/yfinance options | Implementado | `src/cuantis_utils/get_prices_options.py` | Evaluar si se migra o queda como provider alternativo |
| SQLite de opciones | Implementado parcial | `src/data/options/option_chain_history.sqlite` | Definir schema comun para providers |
| Notebooks del curso | Implementado pero mezclado | `src/01`, `src/02`, `docs/01`, `docs/02` | Migrar a `course_notes` |
| Proyecto option hedging | Implementado exploratorio | `src/proyect/opction_hedging.ipynb` | Crear version publicable en `projects/option_hedging` |
| Arquitectura nueva | Planeada | Este documento | Crear estructura y contratos |

Nota: esta tabla representa el estado previo a la Fase 1 de migracion. El
registro operativo de cambios estructurales vive en `migration_log.md`.

## 4. Estructura Final Propuesta

La estructura objetivo separa material de curso, documentacion, proyectos
publicables, datos y codigo reusable.

```text
cuantitativas/
  course_notes/
    01/
      notes/
      notebooks/
      outputs/
      work_files/
    02/
      notes/
      notebooks/
      outputs/
    legacy_projects/
      option_hedging_original/
      merton_original/

  docs/
    architecture/
      implementation_blueprint.md
      data_architecture.md
      strategy_architecture.md
    adrs/
    workflows/
      market_data_ingestion.md
      option_strategy_research.md
    datasets/
      option_contracts.md
      data_quality_rules.md

  projects/
    option_hedging/
      notebooks/
      reports/
      figures/
      README.md

  src/
    modulos/
      data_sources/
        base.py
        thetadata_client.py
        thetadata_options.py
        thetadata_stocks.py
        yahoo_options.py
      schemas/
        market_data.py
        strategies.py
      storage/
        sqlite_repository.py
        csv_repository.py
      validation/
        option_data_checks.py
        market_data_quality.py
      pipelines/
        market_data_pipeline.py
        hedging_dataset_pipeline.py
        volatility_surface_pipeline.py
      models/
        black_scholes.py
        greeks.py
        heston.py
        jump_diffusion.py
      strategies/
        short_call.py
        delta_hedging.py
        delta_gamma_hedging.py
      evaluation/
        pnl.py
        risk_metrics.py
        hedge_error.py

  data/
    raw/
      thetadata/
      yahoo/
    processed/
      options/
      volatility_surfaces/
      strategy_datasets/
    sqlite/
      option_market_data.sqlite

  configs/
    tickers.yaml
    data_sources.yaml
    strategy_runs.yaml
```

Nota: `src/cuantis_utils` puede mantenerse temporalmente como legado mientras se
extraen las piezas utiles hacia `src/modulos`.

## 5. Mapa De Migracion

| Origen actual | Destino propuesto | Tipo | Criterio |
|---|---|---|---|
| `src/01` | `course_notes/01` | Curso | Notebooks, datos y outputs de la unidad 01 |
| `src/02` | `course_notes/02` | Curso | Notebooks, datos y outputs de la unidad 02 |
| `docs/01` | `course_notes/01` | Curso | PDFs, notas y archivos de trabajo |
| `docs/02` | `course_notes/02` | Curso | PDFs, notas y outputs |
| `src/proyect` | `course_notes/legacy_projects/option_hedging_original` | Legado | Mantener version historica sin romper nada |
| `src/proyect/opction_hedging.ipynb` | `projects/option_hedging/notebooks/option_hedging_research.ipynb` | Proyecto publicable | Copia limpia y refactor posterior |
| `src/proyect/clean_data.ipynb` | `docs/workflows/market_data_ingestion.md` y `src/modulos/data_sources` | Extraccion | Convertir celdas en modulos Python |
| `src/cuantis_utils/get_prices_options.py` | `src/modulos/data_sources/yahoo_options.py` y `src/modulos/storage` | Refactor | Separar provider Yahoo de almacenamiento SQLite |
| `src/data/options` | `data/sqlite` o `data/processed/options` | Datos | Evitar datos persistentes dentro de `src` |

Regla de migracion: primero copiar o mover con trazabilidad; despues refactorizar
imports y notebooks. No mezclar movimiento de archivos con cambios de logica.

## 6. Contratos De Datos

Los contratos definen los DataFrames que circulan entre providers, pipelines,
estrategias y reportes.

### 6.1 `StockEOD`

| Campo | Tipo | Requerido | Descripcion |
|---|---|---:|---|
| `ticker` | string | Si | Simbolo del subyacente |
| `date` | date | Si | Fecha de mercado |
| `open` | float | No | Precio de apertura |
| `high` | float | No | Maximo diario |
| `low` | float | No | Minimo diario |
| `close` | float | Si | Cierre diario |
| `volume` | float | No | Volumen |
| `source` | string | Si | Fuente, por ejemplo `ThetaData` |
| `downloaded_at_utc` | datetime | Si | Momento de descarga |

Validaciones minimas:

- `ticker` no vacio.
- `date` no nulo.
- `close > 0`.
- No debe haber duplicados por `ticker, date, source`.

### 6.2 `OptionEOD`

| Campo | Tipo | Requerido | Descripcion |
|---|---|---:|---|
| `ticker` | string | Si | Subyacente |
| `date` | date | Si | Fecha de observacion |
| `expiration_date` | date | Si | Vencimiento |
| `option_type` | string | Si | `call` o `put` |
| `strike` | float | Si | Strike |
| `bid` | float | No | Mejor bid |
| `ask` | float | No | Mejor ask |
| `mid` | float | Si | Precio medio calculado o provisto |
| `last_price` | float | No | Ultimo precio |
| `volume` | float | No | Volumen |
| `open_interest` | float | No | Interes abierto |
| `underlying_price` | float | Si | Precio del subyacente |
| `source` | string | Si | Fuente |
| `downloaded_at_utc` | datetime | Si | Momento de descarga |

Validaciones minimas:

- `expiration_date > date`.
- `strike > 0`.
- `underlying_price > 0`.
- Si existen `bid` y `ask`, entonces `ask >= bid`.
- `mid` debe ser positivo y estar entre `bid` y `ask` cuando ambos existan.

### 6.3 `OptionGreeks`

| Campo | Tipo | Requerido | Descripcion |
|---|---|---:|---|
| `ticker` | string | Si | Subyacente |
| `date` | date | Si | Fecha de observacion |
| `expiration_date` | date | Si | Vencimiento |
| `option_type` | string | Si | `call` o `put` |
| `strike` | float | Si | Strike |
| `delta` | float | Si | Delta |
| `gamma` | float | No | Gamma |
| `vega` | float | No | Vega |
| `theta` | float | No | Theta |
| `rho` | float | No | Rho |
| `implied_volatility` | float | No | Volatilidad implicita |
| `source` | string | Si | Fuente o modelo |

Validaciones minimas:

- `delta` debe estar en rango razonable segun tipo de opcion.
- `gamma >= 0` cuando venga de modelos vanilla estandar.
- `implied_volatility > 0` cuando exista.

### 6.4 `HedgingDataset`

| Campo | Tipo | Requerido | Descripcion |
|---|---|---:|---|
| `ticker` | string | Si | Subyacente |
| `date` | date | Si | Fecha de rebalanceo |
| `expiration_date` | date | Si | Vencimiento |
| `option_type` | string | Si | `call` o `put` |
| `strike` | float | Si | Strike |
| `option_mid` | float | Si | Precio usado para valuacion o trade |
| `underlying_price` | float | Si | Precio spot |
| `time_to_maturity` | float | Si | Tiempo en anos |
| `risk_free_rate` | float | Si | Tasa libre de riesgo |
| `implied_volatility` | float | No | IV de mercado |
| `model_volatility` | float | No | Volatilidad usada por modelo |
| `delta` | float | No | Delta usada para cobertura |
| `gamma` | float | No | Gamma usada para cobertura |

Consumidores:

- `short_call.py`
- `delta_hedging.py`
- `delta_gamma_hedging.py`
- notebooks publicables de `projects/option_hedging`

### 6.5 `StrategyResult`

| Campo | Tipo | Requerido | Descripcion |
|---|---|---:|---|
| `run_id` | string | Si | Identificador de corrida |
| `strategy_name` | string | Si | Nombre de estrategia |
| `ticker` | string | Si | Subyacente |
| `date` | date | Si | Fecha de evaluacion |
| `portfolio_value` | float | Si | Valor del portafolio |
| `cash` | float | No | Caja |
| `underlying_position` | float | No | Posicion en subyacente |
| `option_position` | float | No | Posicion en opciones |
| `pnl` | float | Si | P&L diario o acumulado |
| `transaction_cost` | float | No | Costos |
| `delta_exposure` | float | No | Exposicion delta neta |
| `gamma_exposure` | float | No | Exposicion gamma neta |

## 7. Flujo De Trabajo Real

### 7.1 Ingesta De Mercado

```text
configs/tickers.yaml
  -> ThetaData Terminal local
  -> src/modulos/data_sources/thetadata_client.py
  -> raw data
  -> validation
  -> normalized OptionEOD and StockEOD
  -> storage
```

### 7.2 Construccion De Dataset Para Estrategia

```text
OptionEOD + StockEOD + rates + model assumptions
  -> hedging_dataset_pipeline
  -> HedgingDataset
  -> processed storage
```

### 7.3 Ejecucion De Estrategias

```text
HedgingDataset
  -> strategy module
  -> rebalance logic
  -> StrategyResult
  -> evaluation metrics
```

### 7.4 Reporte Publicable

```text
StrategyResult + figures + assumptions
  -> projects/option_hedging/notebooks
  -> reports
  -> README
  -> LinkedIn/GitHub narrative
```

## 8. Fases De Implementacion

### Fase 0: Blueprint y aprobacion

Entregables:

- `docs/architecture/implementation_blueprint.md`
- Lista de decisiones abiertas
- Mapa de migracion

Criterio de exito:

- El blueprint define estructura, contratos, fases y riesgos antes de tocar
  codigo.

### Fase 0: Reorganizacion de carpetas

Entregables:

- Crear `course_notes`
- Crear `projects/option_hedging`
- Crear `docs/architecture`, `docs/workflows`, `docs/datasets`, `docs/adrs`
- Mover material de curso fuera de `src`

Criterio de exito:

- `src` queda reservado para codigo importable.
- Los notebooks originales se conservan como legado.
- No se cambia logica de calculo.

### Fase 1: Esqueleto de `src/modulos`

Entregables:

- Paquetes vacios con `__init__.py`
- Estructura base para `data_sources`, `schemas`, `storage`, `validation`,
  `pipelines`, `models`, `strategies` y `evaluation`

Criterio de exito:

- Se puede importar `src/modulos` sin ejecutar descargas ni notebooks.

### Fase 2: Contratos y validaciones

Entregables:

- Definicion de columnas esperadas
- Funciones de validacion para `StockEOD`, `OptionEOD`, `HedgingDataset`
- Fixtures pequenos para pruebas locales

Criterio de exito:

- Un dataset invalido falla con errores claros.

### Fase 3: Provider ThetaData

Entregables:

- `thetadata_client.py`
- `thetadata_options.py`
- `thetadata_stocks.py`
- Verificacion de salud de Theta Terminal

Criterio de exito:

- El sistema consulta varios tickers desde ThetaData sin editar notebooks.

### Fase 4: Storage local

Entregables:

- Repositorio SQLite o CSV inicial
- Registro de `ingestion_runs`
- Separacion `raw` vs `processed`

Criterio de exito:

- Cada dataset puede trazarse a ticker, rango de fechas, fuente y hora de
  descarga.

### Fase 5: Pipeline de hedging

Entregables:

- `market_data_pipeline.py`
- `hedging_dataset_pipeline.py`
- Dataset reproducible para `AMZN`, `AAPL` o `CVX`

Criterio de exito:

- Una llamada prepara el dataset que necesita una estrategia.

### Fase 6: Estrategias y evaluacion

Entregables:

- Estrategia sin cobertura
- Delta hedging
- Delta-gamma hedging
- Metricas de P&L, drawdown, hedge error, turnover y costos

Criterio de exito:

- Las estrategias se comparan bajo el mismo dataset y supuestos.

### Fase 7: Proyecto publicable

Entregables:

- Notebook limpio en `projects/option_hedging/notebooks`
- Figuras exportadas
- README del proyecto
- Reporte explicativo

Criterio de exito:

- El proyecto se puede leer como investigacion aplicada, no como exploracion
  suelta.

## 9. Criterios De Exito Globales

| Dimension | Criterio |
|---|---|
| Reproducibilidad | Una corrida puede repetirse con la misma configuracion |
| Calidad de datos | Datos invalidos no llegan a estrategias |
| Escalabilidad local | Se pueden procesar varios tickers sin editar notebooks |
| Separacion de responsabilidades | Providers, pipelines, modelos y estrategias no se mezclan |
| Comparabilidad | Todas las estrategias producen `StrategyResult` |
| Publicabilidad | Los notebooks finales solo narran y analizan |
| Evolucion futura | Un modulo puede migrar a servicio sin reescribir todo |

## 10. Riesgos

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Theta Terminal no esta activo | No hay ingesta | Health check antes de descargar |
| Datos muy pesados por `expiration=*` | Descargas lentas o archivos enormes | Usar `strike_range`, `max_dte` y ventanas cortas |
| Notebooks con rutas hardcodeadas | Ruptura al mover carpetas | Migracion en dos pasos y variables de proyecto |
| Mezclar refactor con logica nueva | Dificil depurar errores | Separar PRs o commits por fase |
| Contratos de datos incompletos | Estrategias inconsistentes | Validaciones antes de storage procesado |
| Costos de transaccion ignorados | Resultados irreales | Incluir costos desde la evaluacion inicial |
| Sobreajuste de estrategias | Falsa optimizacion | Evaluar por ticker, periodo y sensibilidad |
| Datos licenciados | Reproducibilidad publica limitada | Documentar fuente y no publicar datos restringidos |

## 11. Decisiones Abiertas

| Decision | Opciones | Recomendacion inicial |
|---|---|---|
| Storage principal | SQLite, CSV, Parquet | SQLite para metadata y CSV/Parquet para datasets |
| Nombre final del paquete | `src/modulos`, paquete instalable | Iniciar con `src/modulos` como pidio el usuario |
| Manejo de `src/cuantis_utils` | Mantener, migrar, archivar | Mantener temporalmente y migrar por partes |
| Datos dentro o fuera del repo | Versionar muestras, ignorar grandes | Versionar muestras pequenas, ignorar bases grandes |
| Configuracion | YAML, TOML, Python | YAML para tickers y corridas |
| Tests | Unitarios, notebooks, smoke tests | Unitarios para contratos y smoke tests para providers |
| Tasas libres de riesgo | Constante, curva externa, proxy | Constante inicial, curva despues |
| Optimizacion | Heuristica, grid search, optimizacion numerica | Empezar con comparacion controlada |

## 12. Primer Corte De Implementacion

La primera implementacion real deberia limitarse a:

1. Crear estructura documental y de carpetas.
2. Mover material de curso a `course_notes`.
3. Crear esqueleto de `src/modulos`.
4. Definir contratos de datos en documentacion y codigo.
5. Extraer ThetaData desde `clean_data.ipynb`.

No se deberia implementar Heston, jumps ni optimizacion avanzada hasta que el
pipeline base de datos y hedging sea reproducible.

## 13. Topologia Evolutiva

### Etapa 1: Local research system

```text
Notebook -> src/modulos -> local files
```

### Etapa 2: Local pipeline system

```text
Config -> pipeline -> validation -> storage -> strategy -> report
```

### Etapa 3: Production-ready local system

```text
CLI -> scheduled ingestion -> versioned datasets -> strategy runs -> reports
```

### Etapa 4: Servicio separable

```text
Data service + strategy service + reporting layer
```

La arquitectura inicial no implementa servicios separados, pero sus fronteras
logicas preparan ese camino.
