# Fase 3 y Fase 4 Ejecutada: Storage SQLite Y Pipeline Local

## 1. Portada

**Titulo:** Resumen De Fase 3 Y Ejecucion De Fase 4  
**Subtitulo:** De provider ThetaData validado a storage SQLite y pipeline local reproducible  
**Proyecto:** Infraestructura cuantitativa para estrategias con opciones  
**Fecha De Creacion:** 2026-05-21  
**Autores:** Rigo / Codex  

### Resumen

Este documento cierra la Fase 3 del proyecto y registra la Fase 4 ejecutada. La Fase 3
construyo la capa formal de conexion con ThetaData: cliente HTTP, providers para
stock y opciones, mappers hacia contratos internos, pruebas unitarias, smoke
test real y reglas operativas de importacion desde notebooks.

La Fase 4 convirtio esa capacidad de descarga en un flujo reproducible:
descargar datos, validarlos, persistirlos en SQLite, registrar un manifest
completo por corrida y entregar datasets limpios a notebooks, backtesting y
pipelines posteriores. La decision ejecutada fue usar SQLite desde el inicio en
`data/sqlite/backtesting.sqlite`, priorizando integridad, trazabilidad y
preparacion para backtesting.

## 2. Indice

1. Portada  
2. Indice  
3. Proposito Del Documento  
4. Resumen Ejecutivo De Fase 3  
5. Cambios Realizados En Fase 3  
6. Validaciones Y Evidencia Operativa  
7. Decisiones De Arquitectura Que Quedan Fijas  
8. Problemas Detectados Y Correcciones Aplicadas  
9. Objetivo Ejecutado En Fase 4  
10. Frontera Ejecutada De Fase 4  
11. Arquitectura Conceptual Implementada  
12. Componentes Logicos Implementados  
13. Contratos Y Entidades De Persistencia Implementadas  
14. Flujos De Trabajo De La Aplicacion  
15. Diseno De Implementacion  
16. Estructura De Archivos Implementada  
17. Pruebas Ejecutadas  
18. Criterios De Exito  
19. Riesgos Y Mitigaciones  
20. Decisiones Abiertas  
21. Orden De Implementacion Ejecutado  
22. Decision Final Ejecutada  

## 3. Proposito Del Documento

El proposito de este documento es dejar trazabilidad entre dos etapas:

- La Fase 3, donde se logro consultar ThetaData y normalizar datos bajo
  contratos internos.
- La Fase 4, donde esos datos ya empiezan a almacenarse en SQLite y circular
  por un pipeline local reproducible.

La idea principal es evitar que los notebooks se conviertan otra vez en el lugar
donde vive la logica de descarga, limpieza, persistencia y validacion. Los
notebooks deben consumir capacidades ya implementadas en `src/modulos`.

## 4. Resumen Ejecutivo De Fase 3

La Fase 3 implemento el provider formal de ThetaData dentro de
`src/modulos/data_sources`. Esta capa traduce la API local de Theta Terminal a
DataFrames internos que cumplen los contratos definidos en Fase 2.

El flujo implementado es:

```text
Theta Terminal local
  -> ThetaDataClient
  -> ThetaDataStocks / ThetaDataOptions
  -> mapper provider-specific
  -> validate_stock_eod / validate_option_eod
  -> DataFrame normalizado
```

La decision mas importante fue mantener el provider limitado a consulta, mapeo y
validacion. No persiste archivos, no ejecuta estrategias, no calcula P&L y no
contiene logica de backtesting. Esto mantiene la arquitectura alineada al estilo
Modular Monolith: un solo proyecto local, pero con fronteras internas claras.

## 5. Cambios Realizados En Fase 3

### 5.1 Modulos Implementados

| Modulo | Responsabilidad |
|---|---|
| `src/modulos/data_sources/base.py` | Define errores propios, normalizacion de tickers, validacion de fechas `YYYYMMDD` y timestamp UTC. |
| `src/modulos/data_sources/thetadata_client.py` | Implementa cliente HTTP para Theta Terminal local con `get_csv`, `get_json` y `health_check`. |
| `src/modulos/data_sources/thetadata_stocks.py` | Descarga historico stock EOD, lo mapea a `StockEOD` y lo valida. |
| `src/modulos/data_sources/thetadata_options.py` | Descarga opciones EOD, calcula `mid`, mergea precio del subyacente y valida `OptionEOD`. |
| `src/modulos/data_sources/__init__.py` | Expone la API publica del modulo de providers. |

### 5.2 Pruebas Implementadas

| Artefacto | Proposito |
|---|---|
| `tests/test_thetadata_client.py` | Prueba construccion de requests, formato CSV y manejo de errores de conexion. |
| `tests/test_thetadata_mappers.py` | Prueba mapeo de respuestas ThetaData a contratos internos. |
| `tests/fixtures/thetadata_stock_eod_response.csv` | Fixture local para stock EOD simulado. |
| `tests/fixtures/thetadata_option_eod_response.csv` | Fixture local para option EOD simulado. |
| `tests/test_market_data_contracts.py` | Amplia reglas para opciones 0DTE y vencimientos invalidos. |

### 5.3 Documentacion Implementada

| Documento | Proposito |
|---|---|
| `docs/workflows/market_data_ingestion.md` | Guia operativa para consultar ThetaData desde los providers. |
| `docs/workflows/imports_and_environment.md` | Guia para instalar el proyecto editable y usar imports desde notebooks. |
| `docs/workflows/phase_2_summary_and_phase_3_thetadata.md` | Documento puente entre contratos de datos y provider ThetaData. |

### 5.4 Ajustes De Entorno

Se agrego `pyproject.toml` para instalar el proyecto en modo editable. Esto
permite importar:

```python
from modulos.data_sources import ThetaDataOptions
```

desde notebooks, scripts o carpetas fuera de `src`, siempre que el kernel use el
ambiente donde se ejecuto:

```bash
python3 -m pip install -e .
```

Tambien se registro el kernel `cuantitativas (env)` para que los notebooks del
proyecto usen el ambiente correcto.

## 6. Validaciones Y Evidencia Operativa

### 6.1 Pruebas Unitarias

La suite de pruebas paso correctamente:

```bash
python3 -m unittest discover -s tests
```

Resultado:

```text
Ran 20 tests
OK
```

### 6.2 Smoke Test Real Contra ThetaData

Con Theta Terminal activo se valido:

```text
health_check("AAPL") -> True
stock (3, 9)
option (116, 14)
```

Rango probado:

```text
ticker: AAPL
start_date: 20260518
end_date: 20260520
strike_range: 1
max_dte: 30
```

Esta prueba confirma que la infraestructura puede descargar datos reales de
stock y opciones desde ThetaData, normalizarlos y validarlos con los contratos
internos.

## 7. Decisiones De Arquitectura Que Quedan Fijas

### 7.1 Import Publico

El import publico del proyecto es:

```python
from modulos.data_sources import ThetaDataOptions
```

No se debe usar:

```python
from src.modulos.data_sources import ThetaDataOptions
```

`src` es una carpeta de layout, no el paquete publico.

### 7.2 Provider Sin Persistencia

ThetaData no debe guardar archivos ni escribir bases de datos. Su frontera es:

```text
request -> DataFrame crudo -> DataFrame normalizado y validado
```

La persistencia empieza en Fase 4, dentro de `src/modulos/storage` y
`src/modulos/pipelines`.

### 7.3 Validacion Como Frontera De Calidad

Todo DataFrame que salga del provider debe pasar por validadores. Esto evita que
datos incompletos, duplicados o financieramente inconsistentes avancen hacia
storage o estrategias.

### 7.4 Opciones 0DTE

ThetaData devuelve opciones cuyo vencimiento puede ser igual a la fecha de
observacion. Por eso la regla correcta es:

```text
expiration_date >= date
```

El sistema solo debe rechazar:

```text
expiration_date < date
```

### 7.5 Kernel Y Entorno Como Parte Del Workflow

Los notebooks deben correr con el ambiente del proyecto. Si un notebook falla
con `ModuleNotFoundError: No module named 'modulos'`, el problema esperado es
un kernel incorrecto, no el import.

## 8. Problemas Detectados Y Correcciones Aplicadas

| Problema | Causa | Correccion |
|---|---|---|
| `ModuleNotFoundError: No module named 'src'` | Se intentaba importar `src.modulos`, pero `src` no es paquete publico. | Definir import correcto como `modulos...`. |
| `ModuleNotFoundError: No module named 'modulos'` en notebook | El kernel apuntaba a otro ambiente. | Instalar editable en `env` y registrar kernel `cuantitativas (env)`. |
| Falla de validacion con opciones reales | ThetaData devolvio contratos 0DTE. | Permitir `expiration_date == date`. |
| `last_price` demasiado estricto | Algunos datos de mercado pueden reportar ultimo precio cero o ausente. | Validar `last_price` como no negativo, no como estrictamente positivo. |
| Ejecucion de notebook con runtime en Windows Temp | Permisos incompatibles para archivos de conexion Jupyter. | Usar runtime temporal Unix cuando se ejecuta desde entorno automatizado. |

## 9. Objetivo Ejecutado En Fase 4

La Fase 4 construyo el primer flujo local reproducible para datos de
mercado:

```text
provider validado
  -> pipeline de ingesta
  -> storage SQLite
  -> manifest completo de corrida
  -> datasets reutilizables por notebooks y estrategias
```

El objetivo ejecutado no fue crear un sistema distribuido ni un data lake
complejo. El objetivo fue que cada descarga pueda repetirse, auditarse y
reutilizarse sin volver a consultar ThetaData manualmente desde notebooks.

## 10. Frontera Ejecutada De Fase 4

### 10.1 Incluyo

- Crear `SQLiteMarketDataRepository` para `StockEOD`, `OptionEOD` y manifests.
- Guardar y cargar DataFrames validados desde SQLite.
- Aplicar integridad mediante primary keys naturales y `CHECK` constraints.
- Crear `MarketDataIngestionPipeline` como orquestador multi-ticker.
- Registrar manifest completo en SQLite en cada corrida normal.
- Usar `data/sqlite/backtesting.sqlite` como ubicacion inicial de storage.
- Crear configuracion ejemplo en `configs/market_data_ingestion.json`.
- Agregar pruebas unitarias con bases SQLite temporales y providers simulados.
- Documentar el flujo en `docs/workflows/local_market_data_pipeline.md`.

### 10.2 No Incluye

- Backtesting completo.
- Estrategias de hedging.
- Optimizacion de portafolios de opciones.
- Scheduler automatico.
- Descargas masivas multi-proceso.
- Modelos Heston o jump diffusion.
- Persistencia cloud.
- API externa.

## 11. Arquitectura Conceptual Implementada

La topologia recomendada sigue siendo:

```text
Modular Monolith + Pipeline local reproducible
```

El flujo conceptual es:

```text
Config
  -> MarketDataIngestionPipeline
  -> ThetaDataStocks / ThetaDataOptions
  -> Validators
  -> SQLiteMarketDataRepository
  -> tablas stock_eod / option_eod / run_manifests
  -> datasets listos para investigacion y backtesting
```

La Fase 4 introdujo dos conceptos nuevos:

- **Repository SQLite:** objeto encargado de guardar y cargar datos con
  integridad local.
- **Pipeline multi-ticker:** orquestador pequeno que ejecuta una ingesta de
  principio a fin y captura estados `success`, `partial` o `failed`.

## 12. Componentes Logicos Implementados

### 12.1 `storage`

Implementado en:

```text
src/modulos/storage/base.py
src/modulos/storage/sqlite_market_data_repository.py
```

Responsabilidad:

- guardar DataFrames validados;
- cargar datasets por contrato, ticker y rango de fechas;
- evitar duplicados por llave natural;
- mantener storage SQLite en `data/sqlite/backtesting.sqlite`;
- guardar manifests completos;
- no descargar datos;
- no calcular estrategias.

### 12.2 `pipelines`

Implementado en:

```text
src/modulos/pipelines/market_data_ingestion.py
```

Responsabilidad:

- recibir parametros de una corrida;
- soportar uno o multiples tickers;
- llamar providers;
- mandar resultados a storage;
- construir metadata de ejecucion;
- devolver resumen de la corrida.

### 12.3 `configs`

Implementado en:

```text
configs/market_data_ingestion.json
```

Responsabilidad:

- definir tickers;
- definir ventanas de fechas;
- definir parametros conservadores de opciones;
- separar configuracion de codigo.

### 12.4 `data`

Responsabilidad ejecutada:

- almacenar datos descargados y procesados;
- mantener archivos fuera de `src`;
- facilitar auditoria local;
- evitar subir snapshots pesados al repositorio.

## 13. Contratos Y Entidades De Persistencia Implementadas

### 13.1 `StorageKey`

Entidad implementada para ubicar un dataset:

| Campo | Tipo | Descripcion |
|---|---|---|
| `contract_name` | string | Nombre logico: `StockEOD` u `OptionEOD`. |
| `source` | string | Fuente de datos, por ejemplo `ThetaData`. |
| `ticker` | string | Simbolo del subyacente. |
| `start_date` | string | Fecha inicial en `YYYYMMDD`. |
| `end_date` | string | Fecha final en `YYYYMMDD`. |

### 13.2 `RunManifest`

Entidad implementada para auditar ejecuciones. Antes de persistirse ejecuta
`validate()` para evitar manifests incompletos:

| Campo | Tipo | Descripcion |
|---|---|---|
| `run_id` | string | Identificador unico de la corrida. |
| `pipeline_name` | string | Nombre del pipeline que ejecuto la corrida. |
| `provider` | string | Provider utilizado. |
| `status` | string | `success`, `failed` o `partial`. |
| `started_at_utc` | datetime | Inicio de ejecucion. |
| `finished_at_utc` | datetime | Fin de ejecucion. |
| `tickers` | list | Tickers solicitados. |
| `params` | dict | Parametros de descarga. |
| `rows_written` | dict | Filas guardadas por dataset. |
| `results` | dict | Resultado por ticker. |
| `errors` | list | Errores capturados si existen. |

### 13.3 Tablas SQLite

La Fase 4 implemento tres tablas:

| Tabla | Llave primaria | Proposito |
|---|---|---|
| `stock_eod` | `ticker`, `date`, `source` | Historico EOD de subyacentes. |
| `option_eod` | `ticker`, `date`, `expiration_date`, `option_type`, `strike`, `source` | Observaciones EOD de contratos de opcion. |
| `run_manifests` | `run_id` | Auditoria completa de corridas. |

Las tablas contienen constraints para precios positivos, volumen no negativo,
`ask >= bid`, `mid` entre `bid` y `ask`, `expiration_date >= date` y status de
manifest controlado.

### 13.4 `MarketDataIngestionResult`

Resultado devuelto por el pipeline:

| Campo | Tipo | Descripcion |
|---|---|---|
| `run_id` | string | Identificador persistido en `run_manifests`. |
| `status` | string | Estado final de la corrida. |
| `results` | dict | Resultado por ticker. |
| `errors` | list | Errores por ticker. |
| `rows_written` | dict | Filas enviadas a SQLite. |

## 14. Flujos De Trabajo De La Aplicacion

### 14.1 Ingesta De Un Ticker

```text
Usuario define ticker y rango
  -> pipeline valida parametros
  -> descarga StockEOD
  -> guarda StockEOD
  -> descarga OptionEOD con subyacente ya disponible
  -> guarda OptionEOD
  -> escribe manifest
  -> devuelve resumen
```

### 14.2 Ingesta Multi-Ticker

```text
Usuario define lista de tickers
  -> pipeline itera ticker por ticker
  -> cada ticker descarga stock y opciones
  -> cada resultado se guarda en SQLite
  -> los errores se capturan por ticker
  -> se persiste manifest completo
  -> se devuelve status success, partial o failed
```

### 14.3 Reutilizacion En Notebook

```text
Notebook
  -> carga repository
  -> lee OptionEOD desde storage
  -> filtra contratos de interes
  -> analiza resultados
```

El notebook no debe volver a llamar ThetaData si el dataset ya existe y cumple
la ventana necesaria.

### 14.4 Reingesta Controlada

```text
Usuario solicita refresh
  -> pipeline descarga nueva ventana
  -> valida llaves naturales
  -> combina con datos existentes
  -> elimina duplicados por llave natural
  -> guarda version actualizada
```

## 15. Diseno De Implementacion

### 15.1 Storage Inicial

Decision ejecutada:

```text
SQLite inicial en data/sqlite/backtesting.sqlite.
```

Razon: el objetivo inmediato es reproducibilidad con integridad. SQLite permite
llaves primarias, constraints, lecturas por rango y una base natural para
backtesting sin depender de archivos CSV dispersos.

### 15.2 Interfaz Del Repository

Interfaz implementada:

```python
repository = SQLiteMarketDataRepository("data/sqlite/backtesting.sqlite")

repository.save_stock_eod(stock_eod)
repository.save_option_eod(option_eod)

stock_eod = repository.load_stock_eod("AAPL", start_date="20260518", end_date="20260520")
option_eod = repository.load_option_eod("AAPL", start_date="20260518", end_date="20260520")
manifest = repository.load_run_manifest("<run_id>")
```

Reglas:

- recibir solo DataFrames ya validados;
- validar de nuevo antes de guardar si el costo es bajo;
- deduplicar por llave natural con `ON CONFLICT`;
- ordenar por fecha, vencimiento, tipo y strike;
- no llamar providers;
- no escribir dentro de `src`.

### 15.3 Pipeline De Ingesta

Interfaz implementada:

```python
pipeline = MarketDataIngestionPipeline(
    stock_provider=ThetaDataStocks(client),
    option_provider=ThetaDataOptions(client),
    repository=SQLiteMarketDataRepository("data/sqlite/backtesting.sqlite"),
)

result = pipeline.run_option_eod_ingestion(
    tickers=["AAPL", "MSFT"],
    start_date="20260518",
    end_date="20260520",
    strike_range=1,
    max_dte=30,
)
```

Salida esperada:

```python
{
    "run_id": "...",
    "rows_written": {"stock_eod": 6, "option_eod": 232},
    "results": {"AAPL": {"status": "success"}, "MSFT": {"status": "success"}},
    "status": "success",
}
```

### 15.4 Configuracion

Archivo implementado:

```text
configs/market_data_ingestion.json
```

Contenido conceptual:

```json
{
  "provider": "ThetaData",
  "database_path": "data/sqlite/backtesting.sqlite",
  "tickers": ["AAPL"],
  "start_date": "20260518",
  "end_date": "20260520"
}
```

La configuracion se mantiene como ejemplo operativo pequeno. No se agrego un
framework de configuracion para evitar complejidad innecesaria.

## 16. Estructura De Archivos Implementada

```text
configs/
  market_data_ingestion.json

data/
  sqlite/
    backtesting.sqlite

src/modulos/
  storage/
    base.py
    sqlite_market_data_repository.py
  pipelines/
    market_data_ingestion.py

docs/workflows/
  local_market_data_pipeline.md

tests/
  test_sqlite_market_data_repository.py
  test_market_data_ingestion_pipeline.py
```

Nota: los archivos SQLite generados dentro de `data/sqlite/` quedan fuera del
commit normal mediante `.gitignore`.

## 17. Pruebas Ejecutadas

### 17.1 Repository

- guarda `StockEOD` y puede cargarlo de vuelta;
- guarda `OptionEOD` y puede cargarlo de vuelta;
- deduplica por llave natural;
- filtra por rango de fechas;
- rechaza DataFrames invalidos;
- rechaza manifests incompletos o sin resultado por cada ticker solicitado;
- carga manifests completos desde SQLite.

### 17.2 Pipeline

- descarga stock y opciones usando providers simulados;
- guarda ambos datasets;
- crea manifest con `status=success`;
- registra errores con `status=partial` cuando falla solo una parte;
- soporta multiples tickers;
- deduplica tickers repetidos despues de normalizarlos;
- no llama ThetaData real durante pruebas unitarias.

### 17.3 Integracion Local Opcional

Validacion ejecutada:

```bash
python3 -m unittest discover -s tests
```

Resultado observado:

```text
Ran 31 tests
OK
```

### 17.4 Smoke Test Real De Fase 4

Con Theta Terminal activo se ejecuto el pipeline real contra SQLite:

```text
ticker: AAPL
start_date: 20260518
end_date: 20260520
strike_range: 1
max_dte: 30
```

Resultado observado:

```text
status success
rows_written {'stock_eod': 3, 'option_eod': 116}
manifest_status success
manifest_tickers ['AAPL']
stock_rows_loaded 3
option_rows_loaded 116
```

## 18. Criterios De Exito

La Fase 4 se considera exitosa porque:

- `StockEOD` y `OptionEOD` pueden guardarse y cargarse desde storage local.
- El pipeline puede ejecutar ingesta para multiples tickers.
- Cada corrida genera un manifest auditable en SQLite.
- Los notebooks pueden consumir datos desde storage sin llamar ThetaData
  directamente.
- Las pruebas pasan sin Theta Terminal activo.
- La estructura no introduce dependencias fuertes entre providers, storage y
  estrategias.
- La API publica sigue siendo simple y legible.

## 19. Riesgos Y Mitigaciones

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Storage crece demasiado rapido | Lentitud y archivos pesados | Usar SQLite en `data/sqlite`, ventanas controladas y filtros `strike_range`/`max_dte`. |
| Consultas de backtesting requieren integridad | Resultados inconsistentes | Llaves primarias naturales y constraints en SQLite. |
| Notebooks vuelven a descargar datos directamente | Se pierde reproducibilidad | Documentar y usar pipeline como punto unico de ingesta. |
| Manifest incompleto | No se puede auditar una corrida | `RunManifest.validate()`, campos `NOT NULL` y tabla `run_manifests`. |
| Datos generados entran al repo | Repositorio pesado | Ignorar `data/sqlite/*.sqlite` y archivos WAL/SHM. |
| Pipeline demasiado abstracto | Dificil de mantener | Implementar una sola ruta clara: stock + option EOD. |

## 20. Decisiones Abiertas

| Decision | Opciones | Decision Ejecutada |
|---|---|---|
| Backend principal | CSV, SQLite, Parquet | SQLite desde Fase 4. |
| Ubicacion final de datos | `data/processed`, `data/sqlite` | `data/sqlite/backtesting.sqlite`. |
| Granularidad de persistencia | Por ticker, por fecha, por corrida | Tablas normalizadas por contrato. |
| Manifest | JSON por corrida o tabla SQLite | Tabla SQLite `run_manifests`. |
| Configuracion | YAML, JSON, Python dict | JSON para evitar dependencia nueva. |
| Refresh | Sobrescribir o mergear | Upsert con deduplicacion por llave natural. |

## 21. Orden De Implementacion Ejecutado

1. Crear `src/modulos/storage/base.py` con `StorageKey`, `RunManifest` y helpers.
2. Crear `SQLiteMarketDataRepository` con tablas `stock_eod`, `option_eod` y `run_manifests`.
3. Exponer API publica desde `modulos.storage`.
4. Crear `MarketDataIngestionPipeline` y `MarketDataIngestionResult`.
5. Exponer API publica desde `modulos.pipelines`.
6. Agregar pruebas de repository con SQLite temporal.
7. Agregar pruebas de pipeline multi-ticker con providers simulados.
8. Crear `configs/market_data_ingestion.json`.
9. Documentar `docs/workflows/local_market_data_pipeline.md`.
10. Ajustar `.gitignore` para excluir bases SQLite locales.

## 22. Decision Final Ejecutada

La Fase 4 se implemento como una expansion natural de Fase 3. El provider
ThetaData ya funcionaba; ahora los datos descargados pueden persistirse,
auditarse y reutilizarse.

La decision ejecutada es:

```text
Repository SQLite + manifest SQLite + pipeline multi-ticker simple de ingesta.
```

Este diseno respeta las tres leyes de arquitectura:

- **Todo es trade-off:** se asume SQLite como backend local para ganar integridad
  y consultas reproducibles sin saltar todavia a infraestructura distribuida.
- **Por que > como:** el objetivo no es guardar archivos, sino construir una
  base confiable para evaluar estrategias con opciones.
- **Espectro de decisiones:** se fija SQLite como backend inicial y se deja
  abierta la evolucion futura hacia Parquet, cloud o schedulers.

Con esto el proyecto quedara listo para una Fase 5 enfocada en datasets de
hedging y primeras estrategias reproducibles.
