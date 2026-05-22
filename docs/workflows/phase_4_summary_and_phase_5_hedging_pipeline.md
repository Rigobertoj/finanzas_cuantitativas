# Fase 4 y Preparacion De Fase 5: Pipeline De Hedging

## 1. Portada

**Titulo:** Resumen De Fase 4 Y Diseno De Implementacion Para Fase 5  
**Subtitulo:** De storage SQLite auditable a datasets reproducibles de hedging  
**Proyecto:** Infraestructura cuantitativa para estrategias con opciones  
**Fecha De Creacion:** 2026-05-22  
**Autores:** Rigo / Codex  

### Resumen

La Fase 4 dejo al proyecto con una base local de datos mas seria: storage SQLite,
tablas con llaves naturales, constraints de integridad, manifests completos por
corrida y un pipeline multi-ticker capaz de descargar, validar y persistir
datos de mercado. Con esto, los notebooks ya no necesitan descargar datos
directamente desde ThetaData para investigar estrategias.

La Fase 5 debe construir el siguiente puente: transformar `StockEOD` y
`OptionEOD` persistidos en SQLite en un `HedgingDataset` reproducible. Este
dataset sera la entrada estandar para estrategias posteriores como posicion sin
cobertura, delta hedging y delta-gamma hedging. La meta no es optimizar todavia
la estrategia, sino preparar un dataset limpio, trazable y consistente para que
las estrategias puedan compararse bajo los mismos supuestos.

## 2. Indice

1. Portada  
2. Indice  
3. Proposito Del Documento  
4. Resumen Ejecutivo De Fase 4  
5. Cambios Implementados En Fase 4  
6. Evidencia De Validacion  
7. Decisiones Que Quedan Fijas  
8. Objetivo De Fase 5  
9. Frontera De Fase 5  
10. Arquitectura Conceptual De Fase 5  
11. Componentes Logicos Propuestos  
12. Contratos De Datos Involucrados  
13. Flujos De Trabajo Esperados  
14. Diseno De Implementacion  
15. Estructura De Archivos Propuesta  
16. Pruebas Esperadas  
17. Criterios De Exito  
18. Riesgos Y Mitigaciones  
19. Decisiones Abiertas  
20. Orden Recomendado De Implementacion  

## 3. Proposito Del Documento

Este documento cumple dos funciones:

- cerrar conceptualmente la Fase 4, explicando que quedo implementado;
- definir con precision que debe abarcar la Fase 5 segun el blueprint de
  implementacion.

La Fase 5 debe evitar un error comun en proyectos cuantitativos: pasar
directamente de datos de mercado a una estrategia sin una capa intermedia de
dataset reproducible. Esa capa intermedia es la que permite comparar resultados,
auditar supuestos y explicar el trabajo como investigacion aplicada.

## 4. Resumen Ejecutivo De Fase 4

La Fase 4 implemento storage local con SQLite y un pipeline de ingesta
multi-ticker. El sistema ahora puede:

- descargar `StockEOD` y `OptionEOD` desde ThetaData;
- validar los DataFrames contra contratos internos;
- guardar datos en SQLite con llaves primarias naturales;
- evitar duplicados mediante `ON CONFLICT`;
- registrar manifests completos en la tabla `run_manifests`;
- capturar corridas exitosas, parciales o fallidas;
- cargar datos desde SQLite para notebooks o pipelines posteriores.

La ubicacion inicial del storage quedo definida como:

```text
data/sqlite/backtesting.sqlite
```

Este archivo es generado localmente y queda fuera del repositorio mediante
`.gitignore`.

## 5. Cambios Implementados En Fase 4

### 5.1 Storage

| Archivo | Responsabilidad |
|---|---|
| `src/modulos/storage/base.py` | Define `StorageKey`, `RunManifest`, `new_run_id` y `utc_timestamp`. |
| `src/modulos/storage/sqlite_market_data_repository.py` | Implementa `SQLiteMarketDataRepository` para datos de mercado y manifests. |
| `src/modulos/storage/__init__.py` | Expone la API publica de storage. |

### 5.2 Pipeline

| Archivo | Responsabilidad |
|---|---|
| `src/modulos/pipelines/market_data_ingestion.py` | Implementa `MarketDataIngestionPipeline` y `MarketDataIngestionResult`. |
| `src/modulos/pipelines/__init__.py` | Expone la API publica de pipelines. |

### 5.3 Configuracion Y Documentacion

| Archivo | Responsabilidad |
|---|---|
| `configs/market_data_ingestion.json` | Configuracion ejemplo para ingesta local. |
| `docs/workflows/local_market_data_pipeline.md` | Guia operativa del pipeline SQLite. |
| `docs/workflows/phase_3_summary_and_phase_4_storage_pipeline.md` | Documento actualizado con la implementacion real de Fase 4. |
| `.gitignore` | Excluye bases SQLite locales y artefactos WAL/SHM. |

### 5.4 Pruebas

| Archivo | Responsabilidad |
|---|---|
| `tests/test_sqlite_market_data_repository.py` | Prueba save/load, deduplicacion, constraints y manifests completos. |
| `tests/test_market_data_ingestion_pipeline.py` | Prueba ingesta multi-ticker, errores parciales y manifest. |

## 6. Evidencia De Validacion

La suite completa se ejecuto correctamente:

```text
python -m unittest discover -s tests
Ran 31 tests
OK
```

Tambien se ejecuto un smoke test real con ThetaData y SQLite:

```text
status success
rows_written {'stock_eod': 3, 'option_eod': 116}
manifest_status success
manifest_tickers ['AAPL']
stock_rows_loaded 3
option_rows_loaded 116
```

## 7. Decisiones Que Quedan Fijas

### 7.1 SQLite Como Storage Inicial

SQLite queda como backend local inicial porque entrega integridad, consultas por
rango y una ruta natural hacia backtesting sin agregar infraestructura externa.

### 7.2 Manifest Dentro De SQLite

El manifest no vive como archivo JSON suelto. Vive en `run_manifests`, junto a
los datos que audita. Esto reduce riesgo de desalineacion entre dataset y
metadata.

### 7.3 Providers Single-Ticker, Pipeline Multi-Ticker

Los providers siguen trabajando ticker por ticker. La coordinacion multi-ticker
vive en el pipeline. Esta separacion evita que los providers acumulen logica de
orquestacion, reintentos, errores parciales y persistencia.

### 7.4 Notebooks Como Consumidores

Los notebooks deben consumir datos desde SQLite o desde datasets procesados. No
deben convertirse en scripts de ingesta.

## 8. Objetivo De Fase 5

La Fase 5 debe construir el pipeline que transforma market data validada en un
dataset listo para hedging:

```text
SQLiteMarketDataRepository
  -> StockEOD + OptionEOD
  -> seleccion de contrato / calendario de rebalanceo
  -> calculo de columnas derivadas
  -> validate_hedging_dataset
  -> HedgingDataset reproducible
```

El objetivo principal es:

```text
Una llamada prepara el dataset que necesita una estrategia.
```

Este criterio viene directamente del blueprint. La estrategia todavia no es el
centro de Fase 5; el centro es construir una entrada confiable para las
estrategias de Fase 6.

## 9. Frontera De Fase 5

### 9.1 Incluye

- Cargar `StockEOD` y `OptionEOD` desde SQLite.
- Seleccionar contratos de opcion bajo reglas explicitas.
- Construir un `HedgingDataset` bajo contrato interno.
- Calcular `time_to_maturity`.
- Definir supuestos iniciales de `risk_free_rate`.
- Preparar columnas para `delta`, `gamma` e implied/model volatility aunque
  algunas queden opcionales.
- Validar con `validate_hedging_dataset`.
- Guardar o devolver el dataset para notebooks y estrategias.
- Probar el pipeline con fixtures o SQLite temporal.
- Documentar el flujo operativo.

### 9.2 No Incluye

- Ejecutar estrategias completas.
- Calcular P&L.
- Optimizar frecuencia de rebalanceo.
- Implementar Heston o jump diffusion.
- Calibrar volatilidad compleja.
- Reportes publicables finales.
- Comparacion entre estrategias.

## 10. Arquitectura Conceptual De Fase 5

La Fase 5 agrega una capa entre datos de mercado y estrategias:

```text
Market data SQLite
  -> HedgingDatasetPipeline
  -> contract selection
  -> feature engineering
  -> HedgingDataset
  -> future strategies
```

Topologia:

```text
Modular Monolith + Pipeline local reproducible
```

No se introduce un servicio externo ni una arquitectura distribuida. La
arquitectura sigue siendo local, modular y auditable.

## 11. Componentes Logicos Propuestos

### 11.1 `HedgingDatasetPipeline`

Responsabilidad:

- recibir ticker, fechas y parametros de seleccion;
- cargar market data desde SQLite;
- aplicar seleccion de contratos;
- construir columnas del contrato `HedgingDataset`;
- validar el resultado;
- devolver un DataFrame listo para estrategias.

### 11.2 `OptionSelectionPolicy`

Responsabilidad:

- elegir el contrato objetivo por fecha;
- definir si se prioriza call o put;
- definir rango de DTE;
- definir moneyness objetivo;
- resolver empates de liquidez o distancia al strike objetivo.

Esta pieza puede iniciar como una funcion simple. Solo debe convertirse en clase
si la logica empieza a crecer.

### 11.3 `RebalanceCalendar`

Responsabilidad:

- definir fechas de rebalanceo;
- filtrar observaciones por frecuencia diaria, semanal o personalizada;
- evitar fechas sin datos disponibles.

En el primer corte puede ser una funcion que conserva todas las fechas
disponibles.

### 11.4 `HedgingAssumptions`

Responsabilidad:

- guardar supuestos como `risk_free_rate`;
- definir si se usa implied volatility, model volatility o ambas;
- mantener supuestos fuera del notebook.

Puede empezar como dataclass ligera.

## 12. Contratos De Datos Involucrados

### 12.1 Inputs

`StockEOD`:

- `ticker`
- `date`
- `close`
- `source`
- `downloaded_at_utc`

`OptionEOD`:

- `ticker`
- `date`
- `expiration_date`
- `option_type`
- `strike`
- `mid`
- `underlying_price`
- `source`
- `downloaded_at_utc`

### 12.2 Output

`HedgingDataset`:

| Campo | Requerido | Descripcion |
|---|---:|---|
| `ticker` | Si | Subyacente. |
| `date` | Si | Fecha de rebalanceo. |
| `expiration_date` | Si | Vencimiento del contrato. |
| `option_type` | Si | `call` o `put`. |
| `strike` | Si | Strike seleccionado. |
| `option_mid` | Si | Precio medio de la opcion. |
| `underlying_price` | Si | Precio del subyacente. |
| `time_to_maturity` | Si | Tiempo a vencimiento en anos. |
| `risk_free_rate` | Si | Tasa libre de riesgo asumida. |
| `implied_volatility` | No | Volatilidad implicita si esta disponible. |
| `model_volatility` | No | Volatilidad estimada/modelada. |
| `delta` | No | Delta para estrategias de cobertura. |
| `gamma` | No | Gamma para delta-gamma hedging. |

## 13. Flujos De Trabajo Esperados

### 13.1 Construccion De Dataset Para Un Ticker

```text
Usuario define ticker y rango
  -> pipeline carga option_eod desde SQLite
  -> pipeline carga stock_eod desde SQLite
  -> selecciona contratos elegibles
  -> calcula time_to_maturity
  -> agrega risk_free_rate
  -> valida HedgingDataset
  -> devuelve dataset
```

### 13.2 Construccion Multi-Ticker

```text
Usuario define lista de tickers
  -> pipeline ejecuta construccion por ticker
  -> concatena datasets validados
  -> conserva manifest o resumen por ticker
  -> devuelve HedgingDataset consolidado
```

### 13.3 Consumo Desde Estrategias Futuras

```text
HedgingDataset
  -> estrategia sin cobertura
  -> delta hedging
  -> delta-gamma hedging
  -> StrategyResult
```

## 14. Diseno De Implementacion

### 14.1 Interfaz Recomendada

```python
from modulos.pipelines import HedgingDatasetPipeline
from modulos.storage import SQLiteMarketDataRepository

repository = SQLiteMarketDataRepository("data/sqlite/backtesting.sqlite")
pipeline = HedgingDatasetPipeline(repository)

dataset = pipeline.build(
    tickers=["AAPL"],
    start_date="20260518",
    end_date="20260520",
    option_type="call",
    min_dte=7,
    max_dte=45,
    target_moneyness=1.0,
    risk_free_rate=0.045,
)
```

### 14.2 Seleccion De Contratos

Regla inicial recomendada:

```text
Para cada ticker y fecha:
  filtrar contratos por option_type
  filtrar min_dte <= DTE <= max_dte
  calcular moneyness = strike / underlying_price
  elegir contrato con moneyness mas cercano a target_moneyness
  resolver empates con mayor volume y menor spread relativo
```

Esta regla es simple, defendible y suficiente para construir el primer dataset.

### 14.3 Calculos Derivados

Columnas derivadas minimas:

```text
option_mid = mid
time_to_maturity = (expiration_date - date).days / 365
risk_free_rate = parametro fijo inicial
```

Columnas derivadas opcionales:

```text
implied_volatility
model_volatility
delta
gamma
```

Si no estan disponibles, pueden quedar como nulas siempre que el contrato las
marque como opcionales.

## 15. Estructura De Archivos Propuesta

```text
src/modulos/
  pipelines/
    hedging_dataset_pipeline.py
  strategies/
    option_selection.py

docs/workflows/
  hedging_dataset_pipeline.md

tests/
  test_hedging_dataset_pipeline.py
  test_option_selection_policy.py
```

Si la seleccion de contratos se mantiene simple, `option_selection.py` puede
vivir temporalmente dentro de `pipelines`. Si crece, conviene moverla a
`strategies` o a un submodulo especifico.

## 16. Pruebas Esperadas

### 16.1 Dataset

- construye `HedgingDataset` desde SQLite temporal;
- valida columnas requeridas;
- calcula `time_to_maturity` correctamente;
- agrega `risk_free_rate`;
- rechaza datasets vacios cuando no hay contratos elegibles;
- mantiene tickers normalizados.

### 16.2 Seleccion De Contratos

- filtra por `option_type`;
- respeta `min_dte` y `max_dte`;
- elige moneyness mas cercano al objetivo;
- usa volumen/spread como desempate;
- no selecciona contratos vencidos.

### 16.3 Integracion Con Storage

- carga `StockEOD` y `OptionEOD` desde `SQLiteMarketDataRepository`;
- no llama ThetaData;
- no escribe datos crudos nuevos;
- puede construir dataset para multiples tickers.

## 17. Criterios De Exito

La Fase 5 se considera lista cuando:

- existe `HedgingDatasetPipeline`;
- el pipeline carga datos desde SQLite;
- una llamada prepara un `HedgingDataset`;
- el dataset pasa `validate_hedging_dataset`;
- la seleccion de contratos es explicita y testeada;
- las pruebas no dependen de ThetaData real;
- los notebooks pueden consumir el dataset sin conocer la logica interna.

## 18. Riesgos Y Mitigaciones

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Seleccion de contrato ambigua | Resultados dificiles de explicar | Definir una politica simple y documentada. |
| Dataset con contratos no comparables | Estrategias sesgadas | Filtrar DTE, moneyness y option type de forma consistente. |
| Falta de datos para algun ticker | Pipeline falla sin diagnostico | Errores claros por ticker y dataset vacio controlado. |
| Supuestos escondidos en notebooks | Baja reproducibilidad | Usar parametros o dataclass de supuestos. |
| Calcular Greeks demasiado pronto | Complejidad prematura | Mantener `delta` y `gamma` opcionales hasta Fase 6. |

## 19. Decisiones Abiertas

| Decision | Opciones | Recomendacion Inicial |
|---|---|---|
| Tipo de opcion inicial | Call, put, ambos | Call para primer corte; ambos despues. |
| Moneyness objetivo | ATM, OTM, ITM | ATM (`strike / underlying_price` cercano a 1). |
| Rango DTE | 7-45, 30-90, configurable | 7-45 para primer dataset. |
| Risk-free rate | Fija, curva externa, proxy | Fija por parametro en Fase 5. |
| Volatilidad | Implícita, historica, modelo | Opcional; no bloquear el dataset. |
| Persistencia del dataset | Solo DataFrame, SQLite, CSV | DataFrame primero; SQLite si se requiere reuso inmediato. |

## 20. Orden Recomendado De Implementacion

1. Crear helpers para calcular DTE, moneyness y time to maturity.
2. Crear politica simple de seleccion de contratos.
3. Crear `HedgingDatasetPipeline`.
4. Cargar `StockEOD` y `OptionEOD` desde SQLite.
5. Construir DataFrame con columnas `HedgingDataset`.
6. Validar con `validate_hedging_dataset`.
7. Agregar pruebas de seleccion de contratos.
8. Agregar pruebas del pipeline con SQLite temporal.
9. Documentar `docs/workflows/hedging_dataset_pipeline.md`.
10. Probar con datos reales ya guardados en `data/sqlite/backtesting.sqlite`.

## 21. Decision Recomendada

La Fase 5 debe ser un puente disciplinado entre storage y estrategias:

```text
SQLite market data -> HedgingDataset reproducible -> estrategias futuras
```

La decision recomendada es empezar con una seleccion de contratos ATM, rango DTE
configurable y tasa libre de riesgo fija. Esa combinacion mantiene el codigo
simple y permite llegar rapido a una primera comparacion de estrategias sin
adelantar complejidad de modelos como Heston o jump diffusion.
