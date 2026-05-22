# Fase 3 y Preparacion De Fase 4: Storage Y Pipeline Local

## 1. Portada

**Titulo:** Resumen De Fase 3 Y Diseno De Implementacion Para Fase 4  
**Subtitulo:** De provider ThetaData validado a pipeline local reproducible  
**Proyecto:** Infraestructura cuantitativa para estrategias con opciones  
**Fecha De Creacion:** 2026-05-21  
**Autores:** Rigo / Codex  

### Resumen

Este documento cierra la Fase 3 del proyecto y prepara la Fase 4. La Fase 3
construyo la capa formal de conexion con ThetaData: cliente HTTP, providers para
stock y opciones, mappers hacia contratos internos, pruebas unitarias, smoke
test real y reglas operativas de importacion desde notebooks.

La Fase 4 debe convertir esa capacidad de descarga en un flujo reproducible:
descargar datos, validarlos, persistirlos localmente, registrar metadata de la
corrida y entregar datasets limpios a notebooks o pipelines posteriores. El
objetivo no es construir todavia una plataforma compleja, sino una base local,
simple y auditable que pueda evolucionar hacia produccion sin reescribir el
nucleo.

## 2. Indice

1. Portada  
2. Indice  
3. Proposito Del Documento  
4. Resumen Ejecutivo De Fase 3  
5. Cambios Realizados En Fase 3  
6. Validaciones Y Evidencia Operativa  
7. Decisiones De Arquitectura Que Quedan Fijas  
8. Problemas Detectados Y Correcciones Aplicadas  
9. Objetivo De Fase 4  
10. Frontera De Fase 4  
11. Arquitectura Conceptual De Fase 4  
12. Componentes Logicos Propuestos  
13. Contratos Y Entidades De Persistencia  
14. Flujos De Trabajo De La Aplicacion  
15. Diseno De Implementacion  
16. Estructura De Archivos Esperada  
17. Pruebas Esperadas  
18. Criterios De Exito  
19. Riesgos Y Mitigaciones  
20. Decisiones Abiertas  
21. Orden Recomendado De Implementacion  

## 3. Proposito Del Documento

El proposito de este documento es dejar trazabilidad entre dos etapas:

- La Fase 3, donde se logro consultar ThetaData y normalizar datos bajo
  contratos internos.
- La Fase 4, donde esos datos deben empezar a almacenarse y circular por un
  pipeline local reproducible.

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

## 9. Objetivo De Fase 4

La Fase 4 debe construir el primer flujo local reproducible para datos de
mercado:

```text
provider validado
  -> pipeline de ingesta
  -> storage local
  -> metadata de corrida
  -> datasets reutilizables por notebooks y estrategias
```

El objetivo no es crear un sistema distribuido ni un data lake complejo. El
objetivo es que cada descarga pueda repetirse, auditarse y reutilizarse sin
volver a consultar ThetaData manualmente desde notebooks.

## 10. Frontera De Fase 4

### 10.1 Incluye

- Crear repositorios locales de datos para `StockEOD` y `OptionEOD`.
- Definir una interfaz simple para guardar y cargar DataFrames validados.
- Crear un pipeline de ingesta que coordine provider, validacion y storage.
- Registrar metadata de cada corrida.
- Crear estructura local bajo `data/`.
- Crear configuraciones pequenas para tickers y parametros de descarga.
- Agregar pruebas unitarias con storage temporal.
- Documentar como correr el pipeline local.

### 10.2 No Incluye

- Backtesting completo.
- Estrategias de hedging.
- Optimizacion de portafolios de opciones.
- Scheduler automatico.
- Descargas masivas multi-proceso.
- Modelos Heston o jump diffusion.
- Persistencia cloud.
- API externa.

## 11. Arquitectura Conceptual De Fase 4

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
  -> LocalMarketDataRepository
  -> RunManifest
  -> Dataset listo para investigacion
```

La Fase 4 introduce dos conceptos nuevos:

- **Repository:** objeto encargado de guardar y cargar datos.
- **Pipeline:** orquestador pequeno que ejecuta una tarea de principio a fin.

## 12. Componentes Logicos Propuestos

### 12.1 `storage`

Responsabilidad:

- guardar DataFrames validados;
- cargar datasets por contrato, ticker y rango de fechas;
- evitar duplicados por llave natural;
- mantener una estructura local simple;
- no descargar datos;
- no calcular estrategias.

### 12.2 `pipelines`

Responsabilidad:

- recibir parametros de una corrida;
- llamar providers;
- mandar resultados a storage;
- construir metadata de ejecucion;
- devolver resumen de la corrida.

### 12.3 `configs`

Responsabilidad:

- definir tickers;
- definir ventanas de fechas;
- definir parametros conservadores de opciones;
- separar configuracion de codigo.

### 12.4 `data`

Responsabilidad:

- almacenar datos descargados y procesados;
- mantener archivos fuera de `src`;
- facilitar auditoria local;
- evitar subir snapshots pesados al repositorio.

## 13. Contratos Y Entidades De Persistencia

### 13.1 `StorageKey`

Entidad propuesta para ubicar un dataset:

| Campo | Tipo | Descripcion |
|---|---|---|
| `contract_name` | string | Nombre logico: `StockEOD` u `OptionEOD`. |
| `source` | string | Fuente de datos, por ejemplo `ThetaData`. |
| `ticker` | string | Simbolo del subyacente. |
| `start_date` | string | Fecha inicial en `YYYYMMDD`. |
| `end_date` | string | Fecha final en `YYYYMMDD`. |

### 13.2 `RunManifest`

Entidad propuesta para auditar ejecuciones:

| Campo | Tipo | Descripcion |
|---|---|---|
| `run_id` | string | Identificador unico de la corrida. |
| `started_at_utc` | datetime | Inicio de ejecucion. |
| `finished_at_utc` | datetime | Fin de ejecucion. |
| `status` | string | `success`, `failed` o `partial`. |
| `provider` | string | Provider utilizado. |
| `tickers` | list | Tickers solicitados. |
| `params` | dict | Parametros de descarga. |
| `rows_written` | dict | Filas guardadas por dataset. |
| `errors` | list | Errores capturados si existen. |

### 13.3 `MarketDataSnapshot`

No tiene que ser una clase pesada al inicio. Puede ser un resultado simple del
pipeline:

| Campo | Tipo | Descripcion |
|---|---|---|
| `stock_eod` | DataFrame | Datos de subyacente validados. |
| `option_eod` | DataFrame | Datos de opciones validados. |
| `manifest` | dict | Metadata de corrida. |

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

### 14.2 Reutilizacion En Notebook

```text
Notebook
  -> carga repository
  -> lee OptionEOD desde storage
  -> filtra contratos de interes
  -> analiza resultados
```

El notebook no debe volver a llamar ThetaData si el dataset ya existe y cumple
la ventana necesaria.

### 14.3 Reingesta Controlada

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

La opcion mas practica para Fase 4 es empezar con archivos locales por contrato
y ticker. Dos alternativas razonables:

| Alternativa | Ventaja | Desventaja | Recomendacion |
|---|---|---|---|
| CSV particionado | Muy simple, legible, facil de revisar. | Menos eficiente en volumen alto. | Buena para el primer corte. |
| SQLite | Mejor para consultas y deduplicacion. | Requiere mas diseno inicial. | Buena si ya se quiere preparar backtesting. |

Decision recomendada:

```text
Implementar repository con interfaz comun y backend CSV inicial.
Dejar SQLite como backend futuro o segundo corte de Fase 4.
```

Razon: el objetivo inmediato es reproducibilidad y claridad. CSV permite
inspeccionar resultados facilmente durante investigacion. La interfaz del
repository debe permitir migrar despues a SQLite sin cambiar notebooks.

### 15.2 Interfaz Del Repository

Interfaz sugerida:

```python
repository = LocalMarketDataRepository(base_path="data/processed")

repository.save_stock_eod(stock_eod)
repository.save_option_eod(option_eod)

stock_eod = repository.load_stock_eod("AAPL", start_date="20260518", end_date="20260520")
option_eod = repository.load_option_eod("AAPL", start_date="20260518", end_date="20260520")
```

Reglas:

- recibir solo DataFrames ya validados;
- validar de nuevo antes de guardar si el costo es bajo;
- deduplicar por llave natural;
- ordenar por fecha, vencimiento, tipo y strike;
- no llamar providers;
- no escribir dentro de `src`.

### 15.3 Pipeline De Ingesta

Interfaz sugerida:

```python
pipeline = MarketDataIngestionPipeline(
    stock_provider=ThetaDataStocks(client),
    option_provider=ThetaDataOptions(client),
    repository=LocalMarketDataRepository("data/processed"),
)

result = pipeline.run_option_eod_ingestion(
    ticker="AAPL",
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
    "ticker": "AAPL",
    "stock_rows": 3,
    "option_rows": 116,
    "status": "success",
}
```

### 15.4 Configuracion

Archivo sugerido:

```text
configs/market_data_ingestion.yaml
```

Contenido conceptual:

```yaml
provider: ThetaData
tickers:
  - AAPL
start_date: "20260518"
end_date: "20260520"
option_filters:
  right: both
  expiration: "*"
  strike: "*"
  strike_range: 1
  max_dte: 30
```

La configuracion debe ser pequena. No conviene crear un framework complejo de
configuracion todavia.

## 16. Estructura De Archivos Esperada

```text
configs/
  market_data_ingestion.yaml

data/
  processed/
    stock_eod/
      source=ThetaData/
        ticker=AAPL.csv
    option_eod/
      source=ThetaData/
        ticker=AAPL.csv
  manifests/
    market_data_ingestion/
      <run_id>.json

src/modulos/
  storage/
    base.py
    local_market_data_repository.py
  pipelines/
    market_data_ingestion.py

docs/workflows/
  local_market_data_pipeline.md

tests/
  test_local_market_data_repository.py
  test_market_data_ingestion_pipeline.py
```

Nota: los archivos generados dentro de `data/` deben quedar fuera del commit
normal salvo fixtures pequenos o muestras intencionales.

## 17. Pruebas Esperadas

### 17.1 Repository

- guarda `StockEOD` y puede cargarlo de vuelta;
- guarda `OptionEOD` y puede cargarlo de vuelta;
- deduplica por llave natural;
- filtra por rango de fechas;
- rechaza DataFrames invalidos;
- no modifica el DataFrame original.

### 17.2 Pipeline

- descarga stock y opciones usando providers simulados;
- guarda ambos datasets;
- crea manifest con `status=success`;
- registra errores con `status=failed`;
- no llama ThetaData real durante pruebas unitarias.

### 17.3 Integracion Local Opcional

Con Theta Terminal activo:

```bash
python3 -m unittest discover -s tests
```

y un smoke test manual del pipeline:

```bash
python3 - <<'PY'
from modulos.pipelines.market_data_ingestion import MarketDataIngestionPipeline

# Construccion del pipeline segun la interfaz implementada en Fase 4.
PY
```

El smoke test real debe mantenerse separado de las pruebas unitarias para no
depender de licencia, terminal local o disponibilidad de datos.

## 18. Criterios De Exito

La Fase 4 se considera exitosa cuando:

- `StockEOD` y `OptionEOD` pueden guardarse y cargarse desde storage local.
- El pipeline puede ejecutar una ingesta completa para al menos un ticker.
- Cada corrida genera un manifest auditable.
- Los notebooks pueden consumir datos desde storage sin llamar ThetaData
  directamente.
- Las pruebas pasan sin Theta Terminal activo.
- La estructura no introduce dependencias fuertes entre providers, storage y
  estrategias.
- La API publica sigue siendo simple y legible.

## 19. Riesgos Y Mitigaciones

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Storage crece demasiado rapido | Lentitud y archivos pesados | Empezar con ventanas cortas y particion por ticker. |
| CSV no escala para consultas complejas | Dificulta backtesting amplio | Mantener interfaz de repository para migrar a SQLite. |
| Notebooks vuelven a descargar datos directamente | Se pierde reproducibilidad | Documentar y usar pipeline como punto unico de ingesta. |
| Manifest incompleto | No se puede auditar una corrida | Definir campos minimos desde el primer corte. |
| Datos generados entran al repo | Repositorio pesado | Ignorar `data/processed` y versionar solo fixtures pequenos. |
| Pipeline demasiado abstracto | Dificil de mantener | Implementar una sola ruta clara: stock + option EOD. |

## 20. Decisiones Abiertas

| Decision | Opciones | Recomendacion Inicial |
|---|---|---|
| Backend principal | CSV, SQLite, Parquet | CSV primero; SQLite despues si el volumen lo exige. |
| Ubicacion final de datos | `data/processed`, `data/sqlite` | `data/processed` para Fase 4. |
| Granularidad de archivos | Por ticker, por fecha, por corrida | Por contrato + source + ticker. |
| Manifest | JSON por corrida o tabla SQLite | JSON por corrida en Fase 4. |
| Configuracion | YAML, JSON, Python dict | YAML si ya esta disponible; JSON si se prefiere cero dependencia nueva. |
| Refresh | Sobrescribir o mergear | Merge con deduplicacion por llave natural. |

## 21. Orden Recomendado De Implementacion

1. Crear `src/modulos/storage/base.py` con entidades ligeras de storage.
2. Crear `LocalMarketDataRepository` con backend CSV.
3. Agregar pruebas de save/load/deduplicacion para `StockEOD`.
4. Agregar pruebas de save/load/deduplicacion para `OptionEOD`.
5. Crear `MarketDataIngestionPipeline` como orquestador pequeno.
6. Crear manifest JSON por corrida.
7. Agregar pruebas del pipeline con providers simulados.
8. Crear `configs/market_data_ingestion.yaml` o equivalente JSON.
9. Documentar `docs/workflows/local_market_data_pipeline.md`.
10. Ejecutar smoke test real con Theta Terminal activo.

## 22. Decision Recomendada

La Fase 4 debe implementarse como una expansion natural de Fase 3, no como una
reescritura. El provider ThetaData ya funciona; ahora la prioridad es que los
datos descargados puedan persistirse, auditarse y reutilizarse.

La decision recomendada es:

```text
Repository local CSV + manifest JSON + pipeline simple de ingesta.
```

Este diseno respeta las tres leyes de arquitectura:

- **Todo es trade-off:** se sacrifica eficiencia de alto volumen para ganar
  claridad, trazabilidad y velocidad de investigacion.
- **Por que > como:** el objetivo no es guardar archivos, sino construir una
  base confiable para evaluar estrategias con opciones.
- **Espectro de decisiones:** se fija una interfaz estable de storage, pero se
  deja abierto el backend futuro hacia SQLite o Parquet.

Con esto el proyecto quedara listo para una Fase 5 enfocada en datasets de
hedging y primeras estrategias reproducibles.
