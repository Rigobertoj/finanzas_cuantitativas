# Contratos De Datos Para Estrategias Con Opciones

## Proposito

Este documento es la referencia humana para los contratos implementados en
`src/modulos/schemas`. Los contratos definen la forma minima que deben cumplir
los DataFrames usados por providers, pipelines, modelos, estrategias y reportes.

La decision de diseno es mantener contratos ligeros:

- un contrato declara columnas, tipos logicos, columnas requeridas y llave
  natural;
- un validador revisa reglas genericas y financieras;
- ninguna schema descarga datos, persiste archivos o ejecuta estrategias.

## Logica De Definicion

Los schemas se definen desde el flujo de negocio, no desde la forma cruda de un
provider. ThetaData, Yahoo u otra fuente pueden tener nombres distintos, pero
todos deben mapearse a los mismos contratos internos.

La logica es:

1. **Entidad financiera:** que objeto representa el dataset.
2. **Productor:** que modulo lo genera.
3. **Consumidor:** que modulo lo necesita.
4. **Campos requeridos:** minimo necesario para no romper el consumidor.
5. **Campos opcionales:** informacion util pero no obligatoria.
6. **Llave natural:** granularidad que evita duplicados.
7. **Reglas de calidad:** validaciones antes de avanzar en el pipeline.

## Contratos Implementados

| Contrato | Modulo | Proposito |
|---|---|---|
| `StockEOD` | `modulos.schemas.market_data` | Observacion diaria del subyacente. |
| `OptionEOD` | `modulos.schemas.market_data` | Observacion diaria de un contrato de opcion. |
| `OptionGreeks` | `modulos.schemas.market_data` | Sensibilidades de una opcion desde provider o modelo. |
| `HedgingDataset` | `modulos.schemas.strategy_data` | Dataset listo para estrategias de cobertura. |
| `StrategyResult` | `modulos.schemas.strategy_data` | Salida comparable de una estrategia. |

## Ejemplo De Uso

```python
import pandas as pd

from modulos.validation import validate_option_eod

df = pd.read_csv("tests/fixtures/option_eod_sample.csv")
option_eod = validate_option_eod(df)
```

El resultado es una copia normalizada del DataFrame. Si falta una columna
requerida o una regla financiera falla, el validador lanza `ValueError`.

## Relacion Con Fases Futuras

Estos contratos desbloquean:

- provider formal de ThetaData;
- storage local reproducible;
- pipelines de hedging;
- estrategias comparables;
- notebooks publicables que consuman datasets ya preparados.

Los contratos no intentan resolver toda la arquitectura. Solo fijan el lenguaje
comun entre modulos.
