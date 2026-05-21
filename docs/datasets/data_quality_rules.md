# Reglas De Calidad De Datos

## Proposito

Las reglas de calidad protegen los pipelines de errores silenciosos. En
estrategias con opciones, un precio negativo, una expiracion vencida o un spread
invertido puede contaminar todo el resultado de P&L.

## Reglas Genericas

| Regla | Motivo |
|---|---|
| Columnas requeridas presentes | Evita fallas tardias en modelos y estrategias. |
| Fechas parseables | Permite calcular vencimientos y rebalanceos. |
| Numericos parseables | Evita operaciones sobre strings. |
| Llave natural sin duplicados | Evita contar dos veces el mismo contrato o resultado. |
| Ticker normalizado a mayusculas | Mantiene joins consistentes. |
| Fuente preservada | Permite auditar origen de datos. |

## Reglas De Mercado

| Dataset | Regla |
|---|---|
| `StockEOD` | `close`, `open`, `high`, `low` deben ser positivos cuando existan. |
| `StockEOD` | `high >= low` cuando ambos existan. |
| `OptionEOD` | `expiration_date > date`. |
| `OptionEOD` | `strike`, `mid`, `underlying_price` deben ser positivos. |
| `OptionEOD` | `ask >= bid` cuando ambos existan. |
| `OptionEOD` | `mid` debe estar entre `bid` y `ask` cuando los tres existan. |
| `OptionGreeks` | `delta` debe estar entre `-1` y `1`. |
| `OptionGreeks` | `gamma` y `vega` no deben ser negativos cuando existan. |

## Reglas De Estrategia

| Dataset | Regla |
|---|---|
| `HedgingDataset` | `expiration_date > date`. |
| `HedgingDataset` | `option_mid`, `underlying_price` y `time_to_maturity` deben ser positivos. |
| `StrategyResult` | `transaction_cost` no debe ser negativo cuando exista. |

## Criterio De Error

Los validadores deben fallar temprano con `ValueError`. La intencion es que el
usuario pueda corregir la fuente del problema antes de ejecutar modelos o
estrategias.

## Ejecucion De Pruebas

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

Las pruebas usan fixtures pequenos y no requieren ThetaData, internet ni
notebooks.
