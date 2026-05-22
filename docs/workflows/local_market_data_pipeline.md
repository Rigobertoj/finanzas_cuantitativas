# Pipeline Local De Market Data Con SQLite

## Proposito

Este workflow explica como ejecutar la ingesta local de datos de mercado usando
ThetaData, validadores internos y SQLite como storage inicial.

La base local por defecto es:

```text
data/sqlite/backtesting.sqlite
```

Esta ubicacion prepara el camino para que fases posteriores de backtesting y
estrategias consuman datos ya validados sin volver a descargar desde notebooks.

## Componentes

| Componente | Responsabilidad |
|---|---|
| `ThetaDataStocks` | Descargar y validar `StockEOD`. |
| `ThetaDataOptions` | Descargar y validar `OptionEOD`. |
| `SQLiteMarketDataRepository` | Guardar datos y manifests con constraints de integridad. |
| `MarketDataIngestionPipeline` | Coordinar multiples tickers, capturar errores y persistir manifest. |

## Ejemplo De Uso

```python
from modulos.data_sources import ThetaDataClient, ThetaDataOptions, ThetaDataStocks
from modulos.pipelines import MarketDataIngestionPipeline
from modulos.storage import SQLiteMarketDataRepository

client = ThetaDataClient(timeout=120)
stocks = ThetaDataStocks(client)
options = ThetaDataOptions(client, stocks)
repository = SQLiteMarketDataRepository("data/sqlite/backtesting.sqlite")

pipeline = MarketDataIngestionPipeline(
    stock_provider=stocks,
    option_provider=options,
    repository=repository,
)

result = pipeline.run_option_eod_ingestion(
    tickers=["AAPL", "MSFT"],
    start_date="20260518",
    end_date="20260520",
    strike_range=1,
    max_dte=30,
)

print(result.status)
print(result.rows_written)
```

## Cargar Datos Desde SQLite

```python
from modulos.storage import SQLiteMarketDataRepository

repository = SQLiteMarketDataRepository("data/sqlite/backtesting.sqlite")

stock_eod = repository.load_stock_eod("AAPL", "20260518", "20260520")
option_eod = repository.load_option_eod("AAPL", "20260518", "20260520")
manifest = repository.load_run_manifest("<run_id>")
```

## Integridad De Datos

SQLite aplica llaves primarias naturales:

- `stock_eod`: `ticker`, `date`, `source`.
- `option_eod`: `ticker`, `date`, `expiration_date`, `option_type`, `strike`, `source`.
- `run_manifests`: `run_id`.

Tambien aplica reglas de calidad con `CHECK` constraints:

- precios positivos cuando son requeridos;
- volumen no negativo;
- `ask >= bid`;
- `mid` entre `bid` y `ask`;
- `expiration_date >= date`;
- `status` del manifest limitado a `success`, `partial` o `failed`.

## Manifest

Cada corrida escribe un registro completo en `run_manifests`. El manifest
incluye:

- `run_id`;
- nombre del pipeline;
- provider;
- status final;
- timestamps de inicio y fin;
- tickers solicitados;
- parametros de descarga;
- filas escritas;
- resultados por ticker;
- errores capturados.

Si un ticker falla y otro se guarda correctamente, el status es `partial`. El
manifest se conserva para auditar la corrida.

## Pruebas

Las pruebas unitarias no dependen de Theta Terminal:

```bash
python3 -m unittest discover -s tests
```

Usan providers simulados y bases SQLite temporales.
