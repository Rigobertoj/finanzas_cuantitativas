# Market Data Ingestion Con ThetaData

## Proposito

Este workflow explica como usar los providers de ThetaData implementados en
`src/modulos/data_sources` para convertir respuestas de Theta Terminal en
DataFrames normalizados bajo los contratos internos del proyecto.

## Requisitos

- Theta Terminal v3 debe estar activo localmente.
- La API local debe responder en `http://127.0.0.1:25503/v3`.
- Las credenciales y licencia de ThetaData deben estar configuradas fuera del
  repositorio.

## Health Check

```python
from modulos.data_sources import ThetaDataClient

client = ThetaDataClient()
client.health_check("AAPL")
```

Si Theta Terminal no esta disponible, el cliente levanta
`DataSourceUnavailable`.

## Descargar Stock EOD

```python
from modulos.data_sources import ThetaDataClient, ThetaDataStocks

client = ThetaDataClient()
stocks = ThetaDataStocks(client)

stock_eod = stocks.get_stock_eod(
    ticker="AMZN",
    start_date="20260401",
    end_date="20260417",
)
```

La salida cumple el contrato `StockEOD` y ya paso por `validate_stock_eod`.

## Descargar Option EOD

```python
from modulos.data_sources import ThetaDataClient, ThetaDataOptions

client = ThetaDataClient()
options = ThetaDataOptions(client)

option_eod = options.get_option_eod(
    ticker="AMZN",
    start_date="20260401",
    end_date="20260417",
    expiration="*",
    right="both",
    strike="*",
    strike_range=8,
    max_dte=120,
)
```

La salida cumple el contrato `OptionEOD`. Si no se pasa un DataFrame de
subyacente, el provider descarga `StockEOD` y lo mergea por fecha para agregar
`underlying_price`.

## Uso Con Subyacente Ya Descargado

```python
stock_eod = stocks.get_stock_eod("AMZN", "20260401", "20260417")

option_eod = options.get_option_eod(
    ticker="AMZN",
    start_date="20260401",
    end_date="20260417",
    underlying_prices=stock_eod,
)
```

Este flujo evita descargar dos veces el mismo historico de stock cuando se
trabaja con varios filtros de opciones.

## Reglas Operativas

- Usar ventanas de fechas pequenas al inicio.
- Usar `strike_range` y `max_dte` para controlar el tamano de la descarga.
- No llamar ThetaData directamente desde notebooks de investigacion.
- Descargar, validar y luego pasar DataFrames limpios a pipelines.

## Pruebas Sin Theta Terminal

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Estas pruebas usan fixtures simulados y no dependen de internet ni de Theta
Terminal.

## Smoke Test Real

Con Theta Terminal activo:

```bash
PYTHONPATH=src python3 - <<'PY'
from modulos.data_sources import ThetaDataClient

client = ThetaDataClient()
print(client.health_check("AAPL"))
PY
```

Para probar descarga real:

```bash
PYTHONPATH=src python3 - <<'PY'
from modulos.data_sources import ThetaDataClient, ThetaDataOptions, ThetaDataStocks

client = ThetaDataClient(timeout=120)
stocks = ThetaDataStocks(client)
options = ThetaDataOptions(client)

stock_eod = stocks.get_stock_eod(
    ticker="AAPL",
    start_date="20260518",
    end_date="20260520",
)

option_eod = options.get_option_eod(
    ticker="AAPL",
    start_date="20260518",
    end_date="20260520",
    strike_range=1,
    max_dte=30,
    underlying_prices=stock_eod,
)
print(stock_eod.shape)
print(option_eod.shape)
PY
```

Si no hay datos para el rango elegido, probar con un ticker liquido y una
ventana reciente disponible para la licencia activa.

Resultado observado el 2026-05-21 con Theta Terminal activo:

```text
stock (3, 9)
option (116, 14)
```

Durante la prueba aparecieron opciones 0DTE. Por eso la regla de calidad permite
`expiration_date == date` y solo rechaza contratos cuyo vencimiento sea anterior
a la fecha de observacion.
