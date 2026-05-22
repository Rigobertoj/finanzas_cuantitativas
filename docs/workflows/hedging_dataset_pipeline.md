# Pipeline De HedgingDataset

## Proposito

El `HedgingDatasetPipeline` transforma datos de mercado ya persistidos en
SQLite en un dataset listo para estrategias de cobertura. Esta capa evita que
los notebooks mezclen seleccion de contratos, supuestos, calculo de volatilidad
y griegas.

## Flujo

```text
Market data SQLite
  -> HedgingDatasetPipeline
  -> RebalanceCalendar
  -> contract selection
  -> realized volatility
  -> Black-Scholes implied volatility and Greeks
  -> validate_hedging_dataset
  -> hedging_datasets + hedging_dataset_manifests
```

## Ejemplo

```python
from modulos.pipelines import HedgingDatasetPipeline
from modulos.storage import SQLiteMarketDataRepository

repository = SQLiteMarketDataRepository("data/sqlite/backtesting.sqlite")
pipeline = HedgingDatasetPipeline(repository)

result = pipeline.build(
    tickers=["AAPL"],
    start_date="20260518",
    end_date="20260520",
    option_type="call",
    min_dte=30,
    max_dte=60,
    target_moneyness=1.0,
    risk_free_rate=0.045,
    day_count="ACT/365",
    realized_volatility_window=20,
    rebalance_frequency="daily",
)

hedging_dataset = result.frame
```

## Defaults De Investigacion

| Parametro | Default | Motivo |
|---|---:|---|
| `option_type` | `call` | Primer benchmark de cobertura. |
| `target_moneyness` | `1.0` | Contratos ATM como punto inicial defendible. |
| `min_dte` | `30` | Evita vencimientos demasiado cortos. |
| `max_dte` | `60` | Mantiene contratos comparables. |
| `day_count` | `ACT/365` | Convencion de tiempo en anos para Black-Scholes. |
| `risk_free_rate` | configurable | Fase 5 usa tasa fija por corrida. |
| `realized_volatility_window` | `20` | Ventana rolling simple y reproducible. |
| `rebalance_frequency` | `daily` | Usa todas las fechas disponibles. |

## Seleccion De Contratos

Para cada ticker y fecha:

1. Filtra por `option_type`.
2. Filtra por `min_dte <= dte <= max_dte`.
3. Calcula `moneyness = strike / underlying_price`.
4. Ordena por menor distancia al `target_moneyness`.
5. Usa mayor volumen como desempate.
6. Usa menor spread relativo como siguiente desempate.
7. Usa menor DTE y menor strike como desempates finales.

No se filtra por liquidez en esta fase. El volumen y el spread solo sirven para
resolver empates de forma deterministica.

## Feature Engineering

El pipeline calcula:

- `dte`;
- `time_to_maturity`;
- `moneyness`;
- `relative_spread`;
- `realized_volatility`;
- `implied_volatility`;
- `model_volatility`;
- `delta`;
- `gamma`;
- `vega`;
- `theta`;
- `rho`.

La volatilidad implicita se resuelve con Black-Scholes sobre `option_mid`. Si el
solver no converge, el valor queda nulo y las griegas asociadas quedan nulas.
La fila se conserva para auditoria.

## Storage

El dataset se guarda en:

```text
hedging_datasets
```

La metadata se guarda en:

```text
hedging_dataset_manifests
```

Esto permite reconstruir los supuestos de una corrida:

- tickers;
- fechas;
- tipo de opcion;
- rango DTE;
- moneyness objetivo;
- tasa libre de riesgo;
- day count;
- ventana de volatilidad realizada;
- frecuencia de rebalanceo.

## Pruebas

```bash
python -m unittest discover -s tests
```

Las pruebas usan SQLite temporal y no dependen de ThetaData.
