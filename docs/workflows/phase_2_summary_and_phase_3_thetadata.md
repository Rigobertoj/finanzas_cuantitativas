# Fase 2 y Preparacion De Fase 3: Provider ThetaData

## 1. Proposito Del Documento

Este documento cierra la Fase 2 del proyecto de infraestructura cuantitativa y
prepara la Fase 3, enfocada en construir el provider formal de ThetaData.

La Fase 2 fijo el lenguaje interno del sistema: contratos de datos, validadores,
fixtures y pruebas. La Fase 3 debe usar esa base para convertir respuestas de
ThetaData en DataFrames normalizados, validados y listos para pipelines futuros.

## 2. Resumen Ejecutivo De Fase 2

La Fase 2 implemento una capa ligera de contratos basada en pandas. La decision
principal fue mantener los schemas como objetos declarativos, no como clases de
negocio pesadas. Esto permite que cada modulo se pueda modificar con rapidez sin
acoplar ingestion, storage, modelos o estrategias.

El resultado es una base simple:

```text
DataFrame crudo
  -> contrato declarativo
  -> validador composable
  -> DataFrame normalizado
```

Esta estructura permite que ThetaData, Yahoo u otra fuente puedan mapear sus
respuestas hacia el mismo lenguaje interno.

## 3. Cambios Realizados En Fase 2

### 3.1 Serie De Commits

| Commit | Tipo | Proposito |
|---|---|---|
| `94305fb` | Implementacion | Definio contratos ligeros en `src/modulos/schemas`. |
| `d39963c` | Implementacion | Agrego validadores composables en `src/modulos/validation`. |
| `ff9a603` | Test | Cubrio contratos con fixtures pequenos y pruebas unitarias. |
| `768adc5` | Documentacion | Explico contratos y reglas de calidad en `docs/datasets`. |

### 3.2 Modulos Implementados

| Modulo | Responsabilidad |
|---|---|
| `src/modulos/schemas/base.py` | Define `ColumnSpec` y `DataContract`. |
| `src/modulos/schemas/market_data.py` | Define `StockEOD`, `OptionEOD` y `OptionGreeks`. |
| `src/modulos/schemas/strategy_data.py` | Define `HedgingDataset` y `StrategyResult`. |
| `src/modulos/validation/base.py` | Valida columnas, tipos, nulos y llaves naturales. |
| `src/modulos/validation/market_data_checks.py` | Aplica reglas financieras para datos de mercado. |
| `src/modulos/validation/strategy_data_checks.py` | Aplica reglas para datasets y resultados de estrategia. |

### 3.3 Documentacion Y Pruebas

| Artefacto | Proposito |
|---|---|
| `docs/datasets/option_contracts.md` | Referencia humana de contratos. |
| `docs/datasets/data_quality_rules.md` | Reglas transversales de calidad de datos. |
| `docs/datasets/phase_0_1_summary_and_phase_2_contracts.md` | Puente entre Fase 0/1 y Fase 2. |
| `tests/fixtures/*.csv` | Datos minimos para validar contratos sin ThetaData. |
| `tests/test_market_data_contracts.py` | Pruebas para contratos de mercado. |
| `tests/test_strategy_data_contracts.py` | Pruebas para contratos de estrategia. |

### 3.4 Validacion

La validacion de Fase 2 se realizo con:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Resultado esperado:

```text
Ran 12 tests
OK
```

Tambien se valido que los exports publicos de `modulos.schemas` y
`modulos.validation` sean importables desde `PYTHONPATH=src`.

## 4. Criterios De Diseno Que Quedaron Fijados

### 4.1 Simplicidad

Los schemas no ejecutan logica de negocio. Solo declaran estructura. Los
validadores reciben DataFrames y devuelven copias normalizadas. Esto mantiene el
codigo entendible y facil de revisar.

### 4.2 Bajo acoplamiento

Los contratos no conocen ThetaData, Yahoo, SQLite ni estrategias. Los
validadores tampoco descargan datos ni persisten archivos. Cada modulo cumple
una funcion pequena.

### 4.3 Composicion

El flujo esta pensado para componerse:

```text
provider -> normalizador -> validador -> storage/pipeline
```

Cada pieza puede reemplazarse sin reescribir todo el sistema.

### 4.4 Fallar temprano

Los validadores levantan `ValueError` con mensajes claros cuando faltan
columnas, hay errores de tipo, existen duplicados o se violan reglas financieras
como `ask < bid` o `expiration_date < date`. Las opciones 0DTE son validas
porque pueden vencer el mismo dia de observacion.

## 5. Que Desbloquea La Fase 2

La Fase 2 permite implementar ThetaData sin contaminar el resto del proyecto con
detalles propios del provider. A partir de ahora, ThetaData no debe definir la
forma interna del sistema; solo debe adaptarse a los contratos internos.

Esto habilita:

- provider formal de ThetaData;
- ingesta multi-ticker;
- validacion inmediata de datos descargados;
- storage local en fases posteriores;
- pipelines de hedging;
- notebooks publicables que consuman datasets limpios.

## 6. Objetivo De Fase 3

La Fase 3 debe implementar el provider formal de ThetaData como una capa de
adaptacion entre la API local de Theta Terminal y los contratos internos.

Objetivo central:

```text
Consultar ThetaData localmente y devolver DataFrames normalizados bajo
contratos `StockEOD` y `OptionEOD`, sin que notebooks o estrategias llamen la
API directamente.
```

## 7. Frontera De Fase 3

### 7.1 Incluye

- Crear cliente HTTP local para ThetaData.
- Verificar si Theta Terminal esta activo.
- Consultar endpoints de stock EOD y option EOD.
- Soportar parametros basicos: ticker, fechas, expiracion, right, strike,
  `strike_range`, `max_dte`.
- Convertir respuestas de ThetaData a contratos internos.
- Validar los DataFrames usando los validadores de Fase 2.
- Crear pruebas con respuestas simuladas, sin llamar a ThetaData real.
- Documentar el flujo de ingesta.

### 7.2 No Incluye

- Persistencia SQLite definitiva.
- Pipelines completos de hedging.
- Backtesting.
- Estrategias.
- Heston o jump diffusion.
- Scheduler o automatizacion.
- Descargas masivas sin controles de tamano.

## 8. Arquitectura Conceptual De Fase 3

La Fase 3 debe seguir esta topologia:

```text
Theta Terminal local
  -> ThetaDataClient
  -> ThetaDataOptions / ThetaDataStocks
  -> mapper a contratos internos
  -> validate_option_eod / validate_stock_eod
  -> DataFrame normalizado
```

### 8.1 Responsabilidades

| Componente | Responsabilidad | No debe hacer |
|---|---|---|
| `ThetaDataClient` | Construir requests, manejar base URL, timeouts y errores HTTP. | Interpretar estrategias o guardar datasets. |
| `ThetaDataOptions` | Consultar opciones y mapear columnas a `OptionEOD`. | Calcular P&L o Greeks internos. |
| `ThetaDataStocks` | Consultar precios del subyacente y mapear a `StockEOD`. | Ejecutar modelos de retornos. |
| Validadores | Verificar contratos y reglas financieras. | Descargar datos. |
| Tests | Probar mapeos y errores con fixtures simulados. | Depender de Theta Terminal real. |

## 9. Estructura De Archivos Propuesta

```text
src/modulos/
  data_sources/
    base.py
    thetadata_client.py
    thetadata_options.py
    thetadata_stocks.py

docs/workflows/
  market_data_ingestion.md
  phase_2_summary_and_phase_3_thetadata.md

tests/
  fixtures/
    thetadata_option_eod_response.csv
    thetadata_stock_eod_response.csv
  test_thetadata_client.py
  test_thetadata_mappers.py
```

## 10. Diseno De Implementacion

### 10.1 `data_sources/base.py`

Debe contener piezas comunes y simples:

```python
class DataSourceError(RuntimeError):
    pass

class DataSourceUnavailable(DataSourceError):
    pass
```

Tambien puede incluir una funcion pequena para normalizar tickers si resulta
util, pero debe evitar convertirse en un modulo utilitario gigante.

### 10.2 `thetadata_client.py`

Responsabilidad:

- guardar `base_url`;
- aplicar `timeout`;
- construir requests GET;
- agregar `format=csv` o `format=json`;
- convertir errores HTTP en errores propios;
- exponer `health_check()`.

Interfaz sugerida:

```python
client = ThetaDataClient(base_url="http://127.0.0.1:25503/v3")
client.health_check(symbol="AAPL")
frame = client.get_csv("/option/history/eod", params={...})
```

Reglas:

- no debe conocer contratos como `OptionEOD`;
- no debe validar reglas financieras;
- no debe guardar archivos;
- no debe importar estrategias.

### 10.3 `thetadata_options.py`

Responsabilidad:

- construir parametros para endpoints de opciones;
- llamar al cliente;
- mapear columnas ThetaData a `OptionEOD`;
- calcular `mid` si existen `bid` y `ask`;
- agregar `source="ThetaData"`;
- agregar `downloaded_at_utc`;
- validar con `validate_option_eod`.

Interfaz sugerida:

```python
provider = ThetaDataOptions(client)

option_eod = provider.get_option_eod(
    ticker="AMZN",
    start_date="20260401",
    end_date="20260430",
    expiration="*",
    right="both",
    strike="*",
    strike_range=30,
    max_dte=120,
)
```

Salida esperada:

```text
DataFrame que cumple `OptionEOD`.
```

### 10.4 `thetadata_stocks.py`

Responsabilidad:

- consultar historico EOD del subyacente;
- mapear columnas a `StockEOD`;
- agregar `source="ThetaData"`;
- agregar `downloaded_at_utc`;
- validar con `validate_stock_eod`.

Interfaz sugerida:

```python
provider = ThetaDataStocks(client)

stock_eod = provider.get_stock_eod(
    ticker="AMZN",
    start_date="20260401",
    end_date="20260430",
)
```

Salida esperada:

```text
DataFrame que cumple `StockEOD`.
```

## 11. Mapeo Conceptual De Columnas

### 11.1 Opciones EOD

| ThetaData | Contrato interno | Comentario |
|---|---|---|
| `symbol` | `ticker` | Normalizar a mayusculas. |
| `created` | `date` | Convertir a fecha de observacion. |
| `expiration` | `expiration_date` | Convertir a fecha. |
| `right` | `option_type` | Normalizar a `call` o `put`. |
| `strike` | `strike` | Numerico positivo. |
| `bid` | `bid` | Opcional pero validable. |
| `ask` | `ask` | Opcional pero validable. |
| `close` | `last_price` | Usar como ultimo precio EOD si aplica. |
| calculado | `mid` | `(bid + ask) / 2` cuando existan ambos. |
| stock EOD mergeado | `underlying_price` | Puede venir de `ThetaDataStocks` o merge posterior. |
| constante | `source` | `ThetaData`. |
| runtime | `downloaded_at_utc` | Timestamp UTC. |

### 11.2 Stock EOD

| ThetaData | Contrato interno | Comentario |
|---|---|---|
| `symbol` | `ticker` | Normalizar a mayusculas. |
| `created` | `date` | Convertir a fecha. |
| `open` | `open` | Opcional. |
| `high` | `high` | Opcional. |
| `low` | `low` | Opcional. |
| `close` | `close` | Requerido. |
| `volume` | `volume` | Opcional. |
| constante | `source` | `ThetaData`. |
| runtime | `downloaded_at_utc` | Timestamp UTC. |

## 12. Manejo De Fechas

ThetaData usa parametros tipo `YYYYMMDD` en varios endpoints. Internamente el
sistema debe normalizar fechas a columnas parseables por pandas y luego a
contratos `date`.

Reglas propuestas:

- Las funciones publicas aceptan strings `YYYYMMDD` para hablar naturalmente con
  ThetaData.
- Los DataFrames internos usan columnas `date` y `expiration_date`.
- El provider valida `start_date <= end_date`.
- El provider no debe extender automaticamente rangos de fecha; solo puede
  recortar `end_date` a ayer si esta regla se documenta explicitamente.

## 13. Control De Tamano Y Riesgo Operativo

El endpoint de opciones puede devolver muchos datos si se usa
`expiration="*"` y `strike="*"`. La Fase 3 debe proteger al usuario con
parametros conservadores.

Recomendaciones:

- soportar `strike_range`;
- soportar `max_dte`;
- permitir ventanas de fechas cortas;
- documentar que descargas amplias pueden ser pesadas;
- no implementar descargas masivas en paralelo todavia.

## 14. Pruebas Esperadas

Las pruebas no deben depender de que Theta Terminal este encendido. Deben usar
fixtures locales y clientes simulados.

### 14.1 Pruebas Del Cliente

- construye URL correcta;
- agrega `format=csv`;
- convierte errores de conexion en `DataSourceUnavailable`;
- convierte errores HTTP en `DataSourceError`.

### 14.2 Pruebas De Mappers

- `ThetaDataOptions` convierte respuesta simulada a `OptionEOD`;
- calcula `mid`;
- normaliza `option_type`;
- rechaza `ask < bid`;
- valida expiraciones vencidas.

### 14.3 Pruebas De Stocks

- `ThetaDataStocks` convierte respuesta simulada a `StockEOD`;
- normaliza ticker;
- rechaza `close <= 0`;
- rechaza duplicados.

## 15. Criterios De Aceptacion De Fase 3

La Fase 3 se considera lista cuando:

- existe `ThetaDataClient`;
- existen providers `ThetaDataOptions` y `ThetaDataStocks`;
- ambos providers devuelven DataFrames que cumplen contratos internos;
- las pruebas pasan sin internet y sin Theta Terminal real;
- la documentacion de ingesta explica como usar el provider con Theta Terminal;
- no hay dependencias desde providers hacia estrategias, modelos o notebooks;
- errores comunes tienen mensajes claros.

## 16. Riesgos Y Mitigaciones

| Riesgo | Impacto | Mitigacion |
|---|---|---|
| Theta Terminal apagado | No hay datos reales | `health_check()` y error `DataSourceUnavailable`. |
| Columnas ThetaData cambian o varian | Mappers fallan | Mapeo centralizado y pruebas con fixtures. |
| Descargas demasiado grandes | Lentitud o archivos enormes | `strike_range`, `max_dte` y ventanas cortas. |
| Provider demasiado inteligente | Acoplamiento fuerte | Mantenerlo limitado a consulta, mapeo y validacion. |
| Tests dependen del servidor local | Pruebas fragiles | Usar cliente simulado y fixtures. |

## 17. Orden Recomendado De Implementacion

1. Crear `data_sources/base.py` con errores propios.
2. Crear `thetadata_client.py` con `get_csv`, `get_json` y `health_check`.
3. Crear fixtures simulados de respuestas ThetaData.
4. Crear `thetadata_stocks.py` y mapear a `StockEOD`.
5. Crear `thetadata_options.py` y mapear a `OptionEOD`.
6. Agregar pruebas unitarias de cliente y mappers.
7. Crear `docs/workflows/market_data_ingestion.md`.
8. Ejecutar pruebas completas.

## 18. Decision Recomendada

Implementar Fase 3 con un provider sencillo, sin abstracciones excesivas:

- clases pequenas;
- funciones de mapeo explicitas;
- errores propios minimos;
- validadores existentes como frontera de calidad;
- pruebas con fixtures locales;
- cero dependencia hacia estrategias.

Esta decision mantiene el sistema practico y revisable, pero deja la puerta
abierta para que en fases futuras el provider alimente storage, pipelines y
estrategias sin reescribir la base.
