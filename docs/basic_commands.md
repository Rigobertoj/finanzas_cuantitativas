# Latex

scripts/render_tex.sh course_notes/01/notes/02.valuation.tex

scripts/render_tex.sh course_notes/01/notes/02.valuation.tex xelatex

scripts/render_tex.sh course_notes/01/notes/02.valuation.tex lualatex


## Comandos para renderizar

scripts/render_tex.sh --publish-parent course_notes/01/notes/02.valuation.tex

# Comandos ThetaData

ThetaData v3 requiere que Theta Terminal este corriendo en la maquina local. El
notebook legado `course_notes/legacy_projects/option_hedging_original/clean_data.ipynb`
ya esta preparado para consumir la API HTTP local en:

```bash
http://127.0.0.1:25503/v3
```

## 1. Verificar Java

Theta Terminal v3 requiere Java 21 o superior:

```bash
java -version
```

## 2. Preparar Theta Terminal

Crear un directorio local para el JAR y las credenciales:

```bash
mkdir -p ~/ThetaTerminal
cd ~/ThetaTerminal
```

Descargar `ThetaTerminalv3.jar` desde la documentacion oficial de ThetaData y
guardarlo en ese directorio.

Crear `creds.txt` en el mismo directorio que el JAR:

```text
correo@ejemplo.com
password
```

No guardar `creds.txt` dentro del repositorio.

## 3. Activar el servidor local

Desde el directorio donde estan `ThetaTerminalv3.jar` y `creds.txt`:

```bash
java -jar ThetaTerminalv3.jar
```

Mientras este proceso siga corriendo, la API local queda disponible en el puerto
`25503`.

## 4. Probar que el servidor responde

En otra terminal:

```bash
curl "http://127.0.0.1:25503/v3/option/list/expirations?symbol=AAPL&format=json"
```

Si el servidor esta activo, debe regresar una lista de expiraciones. Si aparece
`Couldn't connect to server`, Theta Terminal no esta corriendo o esta usando otro
puerto.

## 5. Consultas utiles de opciones

Lista de expiraciones:

```bash
curl "http://127.0.0.1:25503/v3/option/list/expirations?symbol=AAPL&format=json"
```

Snapshot de quotes para toda la cadena:

```bash
curl "http://127.0.0.1:25503/v3/option/snapshot/quote?symbol=AAPL&expiration=*&format=csv"
```

Historico EOD de opciones:

```bash
curl "http://127.0.0.1:25503/v3/option/history/eod?symbol=AAPL&expiration=*&start_date=20260401&end_date=20260417&right=both&strike=*&format=csv"
```

Historico intradia de quotes con intervalo de 1 minuto:

```bash
curl "http://127.0.0.1:25503/v3/option/history/quote?symbol=AAPL&expiration=*&date=20260417&interval=1m&right=both&strike=*&format=csv"
```

Greeks del snapshot para toda la cadena, si la suscripcion lo permite:

```bash
curl "http://127.0.0.1:25503/v3/option/snapshot/greeks/all?symbol=AAPL&expiration=*&format=csv"
```

## 6. Descargar varios subyacentes desde Python

Patron compatible con las funciones del notebook legado
`course_notes/legacy_projects/option_hedging_original/clean_data.ipynb`:

```python
tickers = ["AAPL", "AMZN", "CVX"]

for ticker in tickers:
    df = build_base_options_dataset(
        ticker=ticker,
        start_date="20260401",
        end_date="20260417",
        strike_range=8,
        max_dte=120,
    )
    df.to_csv(f"{ticker}_options_base_3m.csv", index=False)
```

Para evitar descargas demasiado pesadas, usar ventanas de fechas chicas,
`strike_range` y `max_dte`. El endpoint EOD con `expiration=*` puede devolver
muchas filas.
