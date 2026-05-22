# Imports Y Entorno De Desarrollo

## Proposito

El codigo modular del proyecto vive dentro de `src/`, pero `src` no debe formar
parte del import publico. La carpeta `src` solo es una convencion de layout para
separar codigo fuente, notebooks, documentacion y artefactos del proyecto.

## Import Correcto

Usar:

```python
from modulos.data_sources import ThetaDataOptions
```

Evitar:

```python
from src.modulos.data_sources import ThetaDataOptions
```

El segundo import falla desde notebooks o carpetas externas porque `src` no es
el paquete del proyecto. El paquete real es `modulos`.

## Instalacion Editable

Desde la raiz del repositorio:

```bash
python3 -m pip install -e .
```

Despues de esa instalacion, los modulos se pueden importar desde cualquier
carpeta o notebook que use el mismo interprete de Python:

```python
from modulos.data_sources import ThetaDataClient, ThetaDataOptions, ThetaDataStocks

client = ThetaDataClient()
options = ThetaDataOptions(client)
```

## Verificacion Rapida

Desde cualquier carpeta:

```bash
python3 - <<'PY'
from modulos.data_sources import ThetaDataOptions
print(ThetaDataOptions.__name__)
PY
```

Salida esperada:

```text
ThetaDataOptions
```

## Notebooks

En Jupyter, el kernel debe usar el mismo ambiente donde se ejecuto:

```bash
python3 -m pip install -e .
```

Si un notebook sigue mostrando `ModuleNotFoundError`, normalmente significa que
el kernel activo no corresponde al ambiente instalado. En ese caso, cambiar el
kernel o reinstalar el paquete editable desde el ambiente que usa Jupyter.

Para este proyecto se puede registrar el ambiente local del repositorio como
kernel de Jupyter:

```bash
./env/bin/python -m ipykernel install --user \
  --name cuantitativas-env \
  --display-name "cuantitativas (env)"
```

Despues, seleccionar el kernel `cuantitativas (env)` dentro del notebook.
