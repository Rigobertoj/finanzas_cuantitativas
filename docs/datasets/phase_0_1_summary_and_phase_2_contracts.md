# Fase 0/1 y Preparacion De Fase 2: Contratos De Datos

## 1. Proposito Del Documento

Este documento resume los cambios realizados durante la Fase 0 y la Fase 1 de
la reestructura del proyecto `cuantitativas`, y define el detalle de
implementacion de la siguiente etapa: la Fase 2, enfocada en contratos de datos.

La Fase 2 no debe empezar creando pipelines ni estrategias. Primero debe fijar
los contratos que permitiran que ThetaData, storage, validaciones, modelos y
estrategias hablen el mismo idioma.

## 2. Resumen Ejecutivo

Durante las fases iniciales se separo el material academico del codigo
reutilizable, se archivo el proyecto exploratorio de option hedging, se creo la
frontera nueva `src/modulos` y se formalizo la arquitectura objetivo mediante
documentacion.

La siguiente etapa consiste en transformar la arquitectura propuesta en
contratos de datos concretos. Estos contratos definiran columnas, tipos,
validaciones, productores, consumidores y riesgos de cambio para los datasets
que alimentaran estrategias con opciones.

## 3. Cambios Realizados Hasta Ahora

### 3.1 Serie De Commits

| Commit | Tipo | Proposito |
|---|---|---|
| `76127af` | Reestructura | Migro material academico desde `src/01`, `src/02`, `docs/01` y `docs/02` hacia `course_notes`. |
| `c3e0d8f` | Research | Archivo el proyecto exploratorio de option hedging en `course_notes/legacy_projects`. |
| `74e957e` | Implementacion | Creo el esqueleto importable de `src/modulos`. |
| `14dd344` | Research | Creo el contenedor publicable `projects/option_hedging`. |
| `7f18ca5` | Documentacion | Formalizo el blueprint y el log de migracion. |
| `2176af3` | Documentacion | Creo areas documentales para ADRs, workflows y datasets. |

### 3.2 Nueva Separacion De Responsabilidades

| Area | Responsabilidad |
|---|---|
| `course_notes` | Material historico del curso, notas, notebooks, PDFs y outputs. |
| `course_notes/legacy_projects` | Proyectos exploratorios que se conservan como referencia. |
| `projects/option_hedging` | Proyecto publicable y narrativo sobre estrategias con opciones. |
| `src/modulos` | Nucleo nuevo de codigo importable. |
| `docs/architecture` | Decisiones y diseno arquitectonico. |
| `docs/workflows` | Flujos operativos de ingesta, preparacion, ejecucion y reporte. |
| `docs/datasets` | Contratos, reglas de calidad y lineage de datos. |
| `docs/adrs` | Registro de decisiones arquitectonicas. |

### 3.3 Estado Actual De Implementacion

| Elemento | Estado | Evidencia | Riesgo / Siguiente accion |
|---|---|---|---|
| Blueprint de arquitectura | Implementado | `docs/architecture/implementation_blueprint.md` | Mantenerlo alineado con decisiones reales. |
| Log de migracion | Implementado | `docs/architecture/migration_log.md` | Actualizarlo cuando cambie la estructura. |
| Material academico en `course_notes` | Implementado | `course_notes/01`, `course_notes/02` | Notebooks pueden conservar rutas antiguas. |
| Proyecto exploratorio archivado | Implementado | `course_notes/legacy_projects/option_hedging_original` | Rutas internas pueden apuntar a `src/proyect`. |
| Nucleo `src/modulos` | Implementado inicial | Paquetes `data_sources`, `schemas`, `storage`, `validation`, `pipelines`, `models`, `strategies`, `evaluation` | Aun no contiene contratos ni logica. |
| Proyecto publicable | Implementado inicial | `projects/option_hedging/README.md` | Falta notebook limpio y datasets procesados. |
| ThetaData | Parcial | Notebook legado `clean_data.ipynb`, comandos en `docs/basic_commands.md` | Falta extraerlo a provider formal. |

## 4. Frontera De La Fase 2

La Fase 2 debe definir los contratos de datos antes de implementar providers,
pipelines o estrategias. Su objetivo es producir una base comun para todo el
sistema.

### 4.1 Incluye

- Definir schemas documentados para datasets de mercado y estrategia.
- Crear representacion inicial de contratos en `src/modulos/schemas`.
- Crear validaciones minimas en `src/modulos/validation`.
- Crear fixtures pequenos para probar contratos.
- Documentar productores, consumidores y riesgos de cada contrato.

### 4.2 No Incluye

- Descargar datos reales desde ThetaData.
- Implementar pipelines completos.
- Implementar estrategias de hedging.
- Implementar Heston o procesos con saltos.
- Refactorizar notebooks legados.
- Optimizar estrategias.

## 5. Contratos A Definir En Fase 2

### 5.1 `StockEOD`

Representa una observacion diaria del subyacente.

| Campo | Tipo esperado | Requerido | Regla |
|---|---|---:|---|
| `ticker` | string | Si | No vacio, normalizado a mayusculas. |
| `date` | date | Si | Fecha de mercado. |
| `open` | float | No | Positivo si existe. |
| `high` | float | No | Mayor o igual que `low` si ambos existen. |
| `low` | float | No | Positivo si existe. |
| `close` | float | Si | Mayor que cero. |
| `volume` | float | No | Mayor o igual que cero si existe. |
| `source` | string | Si | Fuente de datos. |
| `downloaded_at_utc` | datetime | Si | Timestamp de descarga. |

Productores esperados:

- `thetadata_stocks.py`
- `yahoo_options.py` o un provider Yahoo futuro

Consumidores esperados:

- `hedging_dataset_pipeline.py`
- `volatility_surface_pipeline.py`
- modelos de retornos y volatilidad

### 5.2 `OptionEOD`

Representa precios diarios de contratos de opciones.

| Campo | Tipo esperado | Requerido | Regla |
|---|---|---:|---|
| `ticker` | string | Si | Subyacente normalizado. |
| `date` | date | Si | Fecha de observacion. |
| `expiration_date` | date | Si | Debe ser posterior a `date`. |
| `option_type` | string | Si | `call` o `put`. |
| `strike` | float | Si | Mayor que cero. |
| `bid` | float | No | Mayor o igual que cero si existe. |
| `ask` | float | No | Mayor o igual que `bid` si ambos existen. |
| `mid` | float | Si | Positivo; idealmente entre `bid` y `ask`. |
| `last_price` | float | No | Positivo si existe. |
| `volume` | float | No | Mayor o igual que cero si existe. |
| `open_interest` | float | No | Mayor o igual que cero si existe. |
| `underlying_price` | float | Si | Mayor que cero. |
| `source` | string | Si | Fuente de datos. |
| `downloaded_at_utc` | datetime | Si | Timestamp de descarga. |

Productores esperados:

- `thetadata_options.py`
- provider Yahoo heredado o migrado

Consumidores esperados:

- `hedging_dataset_pipeline.py`
- `volatility_surface_pipeline.py`
- estrategias de short call, delta hedging y delta-gamma hedging

### 5.3 `OptionGreeks`

Representa sensibilidades de opciones, ya sea de ThetaData o de modelos
internos.

| Campo | Tipo esperado | Requerido | Regla |
|---|---|---:|---|
| `ticker` | string | Si | Subyacente normalizado. |
| `date` | date | Si | Fecha de observacion. |
| `expiration_date` | date | Si | Posterior a `date`. |
| `option_type` | string | Si | `call` o `put`. |
| `strike` | float | Si | Mayor que cero. |
| `delta` | float | Si | Rango razonable segun tipo de opcion. |
| `gamma` | float | No | No negativo para opciones vanilla estandar. |
| `vega` | float | No | No negativo bajo convencion estandar. |
| `theta` | float | No | Puede ser positivo o negativo. |
| `rho` | float | No | Puede ser positivo o negativo. |
| `implied_volatility` | float | No | Mayor que cero si existe. |
| `source` | string | Si | `ThetaData`, `BlackScholes`, `Heston`, etc. |

Productores esperados:

- `thetadata_options.py`
- `models/greeks.py`
- `models/black_scholes.py`

Consumidores esperados:

- estrategias de cobertura
- metricas de hedge error
- notebooks publicables

### 5.4 `HedgingDataset`

Dataset procesado que conecta mercado, modelo y estrategia.

| Campo | Tipo esperado | Requerido | Regla |
|---|---|---:|---|
| `ticker` | string | Si | Subyacente normalizado. |
| `date` | date | Si | Fecha de rebalanceo. |
| `expiration_date` | date | Si | Posterior a `date`. |
| `option_type` | string | Si | `call` o `put`. |
| `strike` | float | Si | Mayor que cero. |
| `option_mid` | float | Si | Precio usado para trade o valuacion. |
| `underlying_price` | float | Si | Mayor que cero. |
| `time_to_maturity` | float | Si | En anos, mayor que cero. |
| `risk_free_rate` | float | Si | Convencion decimal. |
| `implied_volatility` | float | No | Mayor que cero si existe. |
| `model_volatility` | float | No | Mayor que cero si existe. |
| `delta` | float | No | Requerido para delta hedging. |
| `gamma` | float | No | Requerido para delta-gamma hedging. |

Productores esperados:

- `hedging_dataset_pipeline.py`

Consumidores esperados:

- `strategies/short_call.py`
- `strategies/delta_hedging.py`
- `strategies/delta_gamma_hedging.py`
- `evaluation/pnl.py`
- `evaluation/hedge_error.py`

### 5.5 `StrategyResult`

Resultado normalizado de una estrategia.

| Campo | Tipo esperado | Requerido | Regla |
|---|---|---:|---|
| `run_id` | string | Si | Identificador unico de corrida. |
| `strategy_name` | string | Si | Nombre canonico de estrategia. |
| `ticker` | string | Si | Subyacente. |
| `date` | date | Si | Fecha de evaluacion. |
| `portfolio_value` | float | Si | Valor total del portafolio. |
| `cash` | float | No | Caja disponible. |
| `underlying_position` | float | No | Acciones o unidades del subyacente. |
| `option_position` | float | No | Numero de contratos o unidades normalizadas. |
| `pnl` | float | Si | P&L diario o acumulado segun convencion. |
| `transaction_cost` | float | No | Costo estimado. |
| `delta_exposure` | float | No | Exposicion delta neta. |
| `gamma_exposure` | float | No | Exposicion gamma neta. |

Productores esperados:

- modulos en `src/modulos/strategies`

Consumidores esperados:

- `evaluation/risk_metrics.py`
- notebooks de `projects/option_hedging`
- reportes publicables

## 6. Implementacion Propuesta De Fase 2

### 6.1 Estructura De Archivos

```text
src/modulos/
  schemas/
    __init__.py
    market_data.py
    strategy_data.py
  validation/
    __init__.py
    market_data_checks.py
    strategy_data_checks.py

docs/datasets/
  phase_0_1_summary_and_phase_2_contracts.md
  option_contracts.md
  data_quality_rules.md

tests/
  fixtures/
    stock_eod_sample.csv
    option_eod_sample.csv
    hedging_dataset_sample.csv
  test_market_data_contracts.py
  test_strategy_data_contracts.py
```

Si no queremos introducir `tests/` todavia, la alternativa minima es crear
fixtures en `docs/datasets/examples/` y validar con scripts temporales. La
recomendacion es crear `tests/` desde esta fase porque los contratos son la base
del sistema.

### 6.2 Orden De Trabajo

1. Crear `docs/datasets/option_contracts.md` con la referencia canonica de
   contratos.
2. Crear `docs/datasets/data_quality_rules.md` con reglas transversales de
   calidad.
3. Crear `src/modulos/schemas/market_data.py` con definiciones de columnas y
   tipos esperados.
4. Crear `src/modulos/schemas/strategy_data.py` con contratos de estrategia.
5. Crear validadores en `src/modulos/validation`.
6. Crear fixtures pequenos.
7. Crear pruebas unitarias para validaciones.
8. Validar imports y pruebas.

### 6.3 Criterios De Aceptacion

La Fase 2 se considera lista cuando:

- Los contratos estan documentados en `docs/datasets`.
- `src/modulos/schemas` expone contratos importables.
- `src/modulos/validation` detecta errores de columnas faltantes, tipos basicos
  y reglas financieras minimas.
- Existen fixtures pequenos para `StockEOD`, `OptionEOD` y `HedgingDataset`.
- Las pruebas pasan sin depender de ThetaData ni de internet.
- Ningun notebook necesita ejecutarse para validar contratos.

## 7. Reglas De Calidad Transversales

Estas reglas deben aplicar a todos los contratos de mercado:

- Los tickers se normalizan a mayusculas.
- Las fechas deben parsearse a tipo fecha o datetime de forma explicita.
- Los precios no pueden ser negativos.
- Las columnas requeridas no pueden estar ausentes.
- Los duplicados deben definirse por llave natural.
- La fuente de datos debe preservarse.
- Las validaciones deben fallar temprano y con mensajes claros.

## 8. Dependencias Futuras

La Fase 2 desbloquea:

- Fase 3: provider formal de ThetaData.
- Fase 4: storage local reproducible.
- Fase 5: pipeline de hedging.
- Fase 6: estrategias y evaluacion.
- Fase 7: notebook publicable de option hedging.

Sin contratos estables, esas fases quedarian acopladas a nombres de columnas y
supuestos dispersos en notebooks.

## 9. Riesgos Y Mitigaciones

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Sobrediseno de schemas | La fase se vuelve lenta | Empezar con DataFrames y validadores simples. |
| Contratos incompletos | Pipelines futuros rompen | Documentar campos opcionales y requeridos. |
| Mezclar ThetaData en la fase | Se pierde foco | No llamar API externa en Fase 2. |
| No crear pruebas | Validadores fragiles | Crear fixtures minimos desde el inicio. |
| Diferencias entre Yahoo y ThetaData | Providers incompatibles | Normalizar a contratos internos comunes. |

## 10. Decision Recomendada

Implementar la Fase 2 como una capa ligera de contratos basada en pandas:

- contratos como listas de columnas, tipos esperados y llaves naturales;
- validadores que reciban y devuelvan `pd.DataFrame`;
- errores explicitos mediante `ValueError`;
- fixtures pequenos para pruebas.

Esto mantiene el sistema simple, compatible con notebooks y suficientemente
formal para crecer hacia pipelines productivos.
