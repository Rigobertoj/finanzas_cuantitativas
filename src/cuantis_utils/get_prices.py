import httpx
import pandas as pd
from datetime import datetime, timezone

BASE = "https://query1.finance.yahoo.com/v7/finance/options"

def _to_dt_estimate(unix_ts: int | None) -> str | None:
    if unix_ts is None:
        return None
    # Yahoo manda timestamps en segundos (UTC). La imagen dice "EST", pero Yahoo no siempre da TZ explícita.
    # Guardamos ISO UTC para consistencia.
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()

def fetch_option_chain_snapshot(ticker: str) -> pd.DataFrame:
    """
    Descarga un snapshot de la option chain (todas las expiraciones disponibles)
    y lo devuelve en un DataFrame con columnas estilo Yahoo (como en tu imagen).
    """
    with httpx.Client(http2=True, timeout=30.0) as client:
        # 1) primer request para traer expiraciones y quote del subyacente
        r0 = client.get(f"{BASE}/{ticker}")
        r0.raise_for_status()
        j0 = r0.json()

        result0 = j0["optionChain"]["result"][0]
        expirations = result0.get("expirationDates", [])
        quote = result0.get("quote", {}) or {}
        underlying_close = quote.get("regularMarketPreviousClose") or quote.get("regularMarketPrice")

        asof_utc = datetime.now(timezone.utc).isoformat()

        rows = []
        for exp_ts in expirations:
            r = client.get(f"{BASE}/{ticker}", params={"date": exp_ts})
            r.raise_for_status()
            jr = r.json()
            res = jr["optionChain"]["result"][0]
            opt = res["options"][0]

            exp_iso = _to_dt_estimate(opt.get("expirationDate"))
            for side_name, side_label in [("calls", "Call"), ("puts", "Put")]:
                for c in opt.get(side_name, []):
                    rows.append({
                        "Contract Name": c.get("contractSymbol"),                 # contractSymbol
                        "Last Trade Date (EST)": _to_dt_estimate(c.get("lastTradeDate")),
                        "Strike": c.get("strike"),
                        "Last Price": c.get("lastPrice"),
                        "Bid": c.get("bid"),
                        "Ask": c.get("ask"),
                        "Change": c.get("change"),
                        "% Change": c.get("percentChange"),
                        "Volume": c.get("volume"),
                        "Open Interest": c.get("openInterest"),
                        "Implied Volatility": c.get("impliedVolatility"),
                        "Option": side_label,                                     # Call/Put
                        "Date": exp_iso,                                           # expiración
                        "Close price": underlying_close,                           # close del subyacente (proxy)
                        "asof_utc": asof_utc,                                      # timestamp del snapshot
                        "ticker": ticker
                    })

    df = pd.DataFrame(rows)

    # Orden y limpieza ligera
    desired_cols = [
        "Contract Name", "Last Trade Date (EST)", "Strike", "Last Price", "Bid", "Ask",
        "Change", "% Change", "Volume", "Open Interest", "Implied Volatility",
        "Option", "Date", "Close price", "asof_utc", "ticker"
    ]
    df = df[desired_cols].sort_values(["Date", "Option", "Strike"]).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = fetch_option_chain_snapshot("CVX")
    df.to_csv("cvx_option_chain_snapshot.csv", index=False, encoding="utf-8")
    print(df.head(10))
    print(f"\nFilas: {len(df)}  |  Guardado: cvx_option_chain_snapshot.csv")
