from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd
import yfinance as yf

YAHOO_OPTIONS_BASE = "https://query1.finance.yahoo.com/v7/finance/options"
DEFAULT_DB_PATH = Path("../data/options/option_chain_history.sqlite")

TABLE_NAME = "option_chain_snapshots"

SNAPSHOT_COLUMNS = [
    "ticker",
    "snapshot_utc",
    "expiration_date",
    "option_type",
    "contract_symbol",
    "last_trade_date_utc",
    "strike",
    "last_price",
    "bid",
    "ask",
    "change",
    "percent_change",
    "volume",
    "open_interest",
    "implied_volatility",
    "in_the_money",
    "contract_size",
    "currency",
    "underlying_price",
    "source",
    "downloaded_at_utc",
]

NUMERIC_COLUMNS = [
    "strike",
    "last_price",
    "bid",
    "ask",
    "change",
    "percent_change",
    "volume",
    "open_interest",
    "implied_volatility",
    "underlying_price",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_iso_utc_from_unix(unix_ts: int | float | None) -> str | None:
    if unix_ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(unix_ts), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return None


def _to_iso_utc_from_any(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _normalize_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    df = pd.DataFrame(rows)
    for column in SNAPSHOT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[SNAPSHOT_COLUMNS]

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "in_the_money" in df.columns:
        df["in_the_money"] = df["in_the_money"].apply(
            lambda value: None if pd.isna(value) else int(bool(value))
        )

    return df.sort_values(["expiration_date", "option_type", "strike"]).reset_index(drop=True)


@dataclass
class OptionChainDownloader:
    timeout_seconds: float = 30.0
    yfinance_cache_dir: Path | str = Path("../data/options/.yfinance_cache")

    def __post_init__(self) -> None:
        cache_dir = Path(self.yfinance_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            yf.cache.set_cache_location(str(cache_dir))
            yf.set_tz_cache_location(str(cache_dir))
        except Exception:
            # Si no se puede configurar cache, continuamos y usamos fallback HTTP.
            pass

    def fetch_snapshot(self, ticker: str) -> pd.DataFrame:
        ticker = ticker.strip().upper()
        if not ticker:
            raise ValueError("ticker no puede estar vacio")

        yfinance_error: Exception | None = None
        try:
            df = self._fetch_with_yfinance(ticker)
            if not df.empty:
                return df
        except Exception as exc:
            yfinance_error = exc

        try:
            df = self._fetch_with_http_yahoo(ticker)
            if not df.empty:
                return df
        except Exception as http_exc:
            if yfinance_error is not None:
                raise RuntimeError(
                    f"No fue posible descargar opciones para {ticker}. "
                    f"yfinance error={yfinance_error!r}; http error={http_exc!r}"
                ) from http_exc
            raise

        if yfinance_error is not None:
            raise RuntimeError(
                f"Se intentaron dos fuentes para {ticker}, pero no se obtuvo informacion valida. "
                f"yfinance error={yfinance_error!r}"
            )
        raise RuntimeError(f"No se encontro informacion de opciones para {ticker}")

    def _fetch_with_yfinance(self, ticker: str) -> pd.DataFrame:
        snapshot_utc = _utc_now_iso()
        downloaded_at_utc = snapshot_utc
        tk = yf.Ticker(ticker)
        expirations = list(tk.options or [])
        if not expirations:
            raise ValueError(f"Yahoo no reporta expiraciones de opciones para {ticker}")

        underlying_price = self._resolve_underlying_price(tk)
        rows: list[dict] = []

        for expiration_date in expirations:
            chain = tk.option_chain(expiration_date)
            rows.extend(
                self._rows_from_yfinance_side(
                    side_df=chain.calls,
                    ticker=ticker,
                    expiration_date=expiration_date,
                    option_type="call",
                    snapshot_utc=snapshot_utc,
                    downloaded_at_utc=downloaded_at_utc,
                    underlying_price=underlying_price,
                )
            )
            rows.extend(
                self._rows_from_yfinance_side(
                    side_df=chain.puts,
                    ticker=ticker,
                    expiration_date=expiration_date,
                    option_type="put",
                    snapshot_utc=snapshot_utc,
                    downloaded_at_utc=downloaded_at_utc,
                    underlying_price=underlying_price,
                )
            )

        return _normalize_rows(rows)

    def _rows_from_yfinance_side(
        self,
        side_df: pd.DataFrame,
        ticker: str,
        expiration_date: str,
        option_type: str,
        snapshot_utc: str,
        downloaded_at_utc: str,
        underlying_price: float | None,
    ) -> list[dict]:
        if side_df is None or side_df.empty:
            return []

        rows: list[dict] = []
        for row in side_df.to_dict(orient="records"):
            rows.append(
                {
                    "ticker": ticker,
                    "snapshot_utc": snapshot_utc,
                    "expiration_date": str(expiration_date),
                    "option_type": option_type,
                    "contract_symbol": row.get("contractSymbol"),
                    "last_trade_date_utc": _to_iso_utc_from_any(row.get("lastTradeDate")),
                    "strike": row.get("strike"),
                    "last_price": row.get("lastPrice"),
                    "bid": row.get("bid"),
                    "ask": row.get("ask"),
                    "change": row.get("change"),
                    "percent_change": row.get("percentChange"),
                    "volume": row.get("volume"),
                    "open_interest": row.get("openInterest"),
                    "implied_volatility": row.get("impliedVolatility"),
                    "in_the_money": row.get("inTheMoney"),
                    "contract_size": row.get("contractSize"),
                    "currency": row.get("currency"),
                    "underlying_price": underlying_price,
                    "source": "yfinance",
                    "downloaded_at_utc": downloaded_at_utc,
                }
            )
        return rows

    def _resolve_underlying_price(self, tk: yf.Ticker) -> float | None:
        try:
            fast_info = tk.fast_info or {}
            for key in ("last_price", "previous_close", "regular_market_previous_close"):
                value = fast_info.get(key)
                if value is not None and not pd.isna(value):
                    return float(value)
        except Exception:
            pass

        try:
            info = tk.info or {}
            for key in ("regularMarketPrice", "regularMarketPreviousClose", "previousClose"):
                value = info.get(key)
                if value is not None and not pd.isna(value):
                    return float(value)
        except Exception:
            pass

        return None

    def _fetch_with_http_yahoo(self, ticker: str) -> pd.DataFrame:
        snapshot_utc = _utc_now_iso()
        downloaded_at_utc = snapshot_utc

        with httpx.Client(http2=True, timeout=self.timeout_seconds) as client:
            r0 = client.get(f"{YAHOO_OPTIONS_BASE}/{ticker}")
            r0.raise_for_status()
            j0 = r0.json()

            result0 = (j0.get("optionChain", {}).get("result") or [])
            if not result0:
                raise ValueError(f"Yahoo no entrego resultado para ticker={ticker}")

            meta = result0[0]
            expirations = meta.get("expirationDates", []) or []
            if not expirations:
                raise ValueError(f"Yahoo no entrego expiraciones para ticker={ticker}")

            quote = meta.get("quote", {}) or {}
            underlying_price = quote.get("regularMarketPrice") or quote.get("regularMarketPreviousClose")

            rows: list[dict] = []
            for expiration_unix in expirations:
                r = client.get(f"{YAHOO_OPTIONS_BASE}/{ticker}", params={"date": expiration_unix})
                r.raise_for_status()
                jr = r.json()
                result = (jr.get("optionChain", {}).get("result") or [])
                if not result:
                    continue

                options_list = result[0].get("options", []) or []
                if not options_list:
                    continue
                option_bundle = options_list[0]

                exp_iso = _to_iso_utc_from_unix(option_bundle.get("expirationDate") or expiration_unix)
                expiration_date = exp_iso[:10] if exp_iso else None

                for side_name, option_type in (("calls", "call"), ("puts", "put")):
                    for contract in option_bundle.get(side_name, []) or []:
                        rows.append(
                            {
                                "ticker": ticker,
                                "snapshot_utc": snapshot_utc,
                                "expiration_date": expiration_date,
                                "option_type": option_type,
                                "contract_symbol": contract.get("contractSymbol"),
                                "last_trade_date_utc": _to_iso_utc_from_unix(contract.get("lastTradeDate")),
                                "strike": contract.get("strike"),
                                "last_price": contract.get("lastPrice"),
                                "bid": contract.get("bid"),
                                "ask": contract.get("ask"),
                                "change": contract.get("change"),
                                "percent_change": contract.get("percentChange"),
                                "volume": contract.get("volume"),
                                "open_interest": contract.get("openInterest"),
                                "implied_volatility": contract.get("impliedVolatility"),
                                "in_the_money": contract.get("inTheMoney"),
                                "contract_size": contract.get("contractSize"),
                                "currency": contract.get("currency"),
                                "underlying_price": underlying_price,
                                "source": "yahoo_http",
                                "downloaded_at_utc": downloaded_at_utc,
                            }
                        )

        return _normalize_rows(rows)


@dataclass
class OptionChainRepository:
    db_path: Path | str = DEFAULT_DB_PATH
    table_name: str = TABLE_NAME

    def __post_init__(self) -> None:
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def ensure_schema(self) -> None:
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            ticker TEXT NOT NULL,
            snapshot_utc TEXT NOT NULL,
            expiration_date TEXT NOT NULL,
            option_type TEXT NOT NULL,
            contract_symbol TEXT NOT NULL,
            last_trade_date_utc TEXT,
            strike REAL,
            last_price REAL,
            bid REAL,
            ask REAL,
            change REAL,
            percent_change REAL,
            volume REAL,
            open_interest REAL,
            implied_volatility REAL,
            in_the_money INTEGER,
            contract_size TEXT,
            currency TEXT,
            underlying_price REAL,
            source TEXT NOT NULL,
            downloaded_at_utc TEXT NOT NULL,
            PRIMARY KEY (ticker, snapshot_utc, contract_symbol, option_type)
        );
        """
        with self._connect() as conn:
            conn.execute(create_table_sql)
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_ticker_expiration "
                f"ON {self.table_name} (ticker, expiration_date);"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_snapshot "
                f"ON {self.table_name} (snapshot_utc);"
            )
            conn.commit()

    def insert_snapshot(self, df_snapshot: pd.DataFrame) -> int:
        if df_snapshot.empty:
            return 0

        self.ensure_schema()
        df_to_insert = df_snapshot.copy()
        for column in SNAPSHOT_COLUMNS:
            if column not in df_to_insert.columns:
                df_to_insert[column] = None
        df_to_insert = df_to_insert[SNAPSHOT_COLUMNS].where(pd.notna(df_to_insert), None)
        rows = [tuple(row) for row in df_to_insert.itertuples(index=False, name=None)]

        placeholders = ", ".join(["?"] * len(SNAPSHOT_COLUMNS))
        columns_sql = ", ".join(SNAPSHOT_COLUMNS)
        insert_sql = (
            f"INSERT OR IGNORE INTO {self.table_name} ({columns_sql}) "
            f"VALUES ({placeholders});"
        )

        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany(insert_sql, rows)
            conn.commit()
            inserted_rows = conn.total_changes - before
        return inserted_rows


def fetch_option_chain_snapshot(ticker: str) -> pd.DataFrame:
    downloader = OptionChainDownloader()
    return downloader.fetch_snapshot(ticker=ticker)


def update_option_chain_history(
    ticker: str,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> tuple[pd.DataFrame, int]:
    downloader = OptionChainDownloader()
    repository = OptionChainRepository(db_path=db_path)
    snapshot = downloader.fetch_snapshot(ticker=ticker)
    inserted = repository.insert_snapshot(snapshot)
    return snapshot, inserted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Descarga toda la option chain de un ticker y la guarda en SQLite historico."
    )
    parser.add_argument("ticker", nargs="?", default="CVX", help="Ticker de Yahoo, por ejemplo CVX o AAPL.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help="Ruta del archivo SQLite de salida.",
    )
    args = parser.parse_args()

    df_snapshot, inserted_count = update_option_chain_history(
        ticker=args.ticker,
        db_path=args.db_path,
    )

    print(df_snapshot.head(10))
    print(
        f"\nTicker: {args.ticker.upper()} | filas snapshot: {len(df_snapshot)} "
        f"| filas insertadas: {inserted_count} | db: {args.db_path}"
    )
