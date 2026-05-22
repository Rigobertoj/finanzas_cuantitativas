"""SQLite repository for validated market data and run manifests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

from modulos.schemas.market_data import OPTION_EOD_CONTRACT, STOCK_EOD_CONTRACT
from modulos.validation import validate_option_eod, validate_stock_eod

from .base import RunManifest


class SQLiteMarketDataRepository:
    """Persist market data and manifests in a local SQLite database.

    Parameters
    ----------
    database_path:
        SQLite file path. The default points to
        ``data/sqlite/backtesting.sqlite`` so later backtesting phases can reuse
        the same local database.

    Notes
    -----
    The repository validates DataFrames before writing and uses natural-key
    primary keys to prevent duplicate market observations.
    """

    def __init__(self, database_path: str | Path = "data/sqlite/backtesting.sqlite") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def save_stock_eod(self, frame: pd.DataFrame) -> int:
        """Validate and upsert ``StockEOD`` rows.

        Parameters
        ----------
        frame:
            DataFrame satisfying, or mappable to, the ``StockEOD`` contract.

        Returns
        -------
        int
            Number of validated rows submitted to SQLite.
        """

        data = validate_stock_eod(frame)
        rows = [_record_to_sqlite(row) for row in data.to_dict("records")]
        sql = """
            INSERT INTO stock_eod (
                ticker, date, open, high, low, close, volume, source, downloaded_at_utc
            )
            VALUES (
                :ticker, :date, :open, :high, :low, :close, :volume, :source, :downloaded_at_utc
            )
            ON CONFLICT(ticker, date, source) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume,
                downloaded_at_utc = excluded.downloaded_at_utc
        """
        with self._connect() as connection:
            connection.executemany(sql, rows)
        return len(rows)

    def save_option_eod(self, frame: pd.DataFrame) -> int:
        """Validate and upsert ``OptionEOD`` rows.

        Parameters
        ----------
        frame:
            DataFrame satisfying, or mappable to, the ``OptionEOD`` contract.

        Returns
        -------
        int
            Number of validated rows submitted to SQLite.
        """

        data = validate_option_eod(frame)
        rows = [_record_to_sqlite(row) for row in data.to_dict("records")]
        sql = """
            INSERT INTO option_eod (
                ticker, date, expiration_date, option_type, strike, bid, ask,
                mid, last_price, volume, open_interest, underlying_price,
                source, downloaded_at_utc
            )
            VALUES (
                :ticker, :date, :expiration_date, :option_type, :strike, :bid, :ask,
                :mid, :last_price, :volume, :open_interest, :underlying_price,
                :source, :downloaded_at_utc
            )
            ON CONFLICT(ticker, date, expiration_date, option_type, strike, source) DO UPDATE SET
                bid = excluded.bid,
                ask = excluded.ask,
                mid = excluded.mid,
                last_price = excluded.last_price,
                volume = excluded.volume,
                open_interest = excluded.open_interest,
                underlying_price = excluded.underlying_price,
                downloaded_at_utc = excluded.downloaded_at_utc
        """
        with self._connect() as connection:
            connection.executemany(sql, rows)
        return len(rows)

    def load_stock_eod(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        source: str = "ThetaData",
    ) -> pd.DataFrame:
        """Load stock EOD rows by ticker, source and optional date window."""

        return self._load_table(
            table="stock_eod",
            contract_columns=STOCK_EOD_CONTRACT.column_names,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            source=source,
            order_by="date",
        )

    def load_option_eod(
        self,
        ticker: str,
        start_date: str | None = None,
        end_date: str | None = None,
        source: str = "ThetaData",
    ) -> pd.DataFrame:
        """Load option EOD rows by ticker, source and optional date window."""

        return self._load_table(
            table="option_eod",
            contract_columns=OPTION_EOD_CONTRACT.column_names,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            source=source,
            order_by="date, expiration_date, option_type, strike",
        )

    def save_run_manifest(self, manifest: RunManifest) -> None:
        """Persist a complete pipeline manifest.

        Raises
        ------
        ValueError
            If the manifest is incomplete.
        """

        manifest.validate()
        row = {
            "run_id": manifest.run_id,
            "pipeline_name": manifest.pipeline_name,
            "provider": manifest.provider,
            "status": manifest.status,
            "started_at_utc": manifest.started_at_utc,
            "finished_at_utc": manifest.finished_at_utc,
            "tickers_json": json.dumps(list(manifest.tickers), sort_keys=True),
            "params_json": json.dumps(manifest.params, sort_keys=True),
            "rows_written_json": json.dumps(manifest.rows_written, sort_keys=True),
            "results_json": json.dumps(manifest.results, sort_keys=True),
            "errors_json": json.dumps(manifest.errors, sort_keys=True),
        }
        sql = """
            INSERT INTO run_manifests (
                run_id, pipeline_name, provider, status, started_at_utc,
                finished_at_utc, tickers_json, params_json, rows_written_json,
                results_json, errors_json
            )
            VALUES (
                :run_id, :pipeline_name, :provider, :status, :started_at_utc,
                :finished_at_utc, :tickers_json, :params_json, :rows_written_json,
                :results_json, :errors_json
            )
        """
        with self._connect() as connection:
            connection.execute(sql, row)

    def load_run_manifest(self, run_id: str) -> dict[str, Any]:
        """Return one persisted run manifest as a dictionary."""

        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM run_manifests WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Run manifest {run_id!r} was not found.")
        result = dict(row)
        for column in (
            "tickers_json",
            "params_json",
            "rows_written_json",
            "results_json",
            "errors_json",
        ):
            result[column[:-5]] = json.loads(result.pop(column))
        return result

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS stock_eod (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL NOT NULL CHECK(close > 0),
                    volume REAL CHECK(volume IS NULL OR volume >= 0),
                    source TEXT NOT NULL,
                    downloaded_at_utc TEXT NOT NULL,
                    PRIMARY KEY (ticker, date, source),
                    CHECK(open IS NULL OR open > 0),
                    CHECK(high IS NULL OR high > 0),
                    CHECK(low IS NULL OR low > 0),
                    CHECK(high IS NULL OR low IS NULL OR high >= low)
                );

                CREATE TABLE IF NOT EXISTS option_eod (
                    ticker TEXT NOT NULL,
                    date TEXT NOT NULL,
                    expiration_date TEXT NOT NULL,
                    option_type TEXT NOT NULL CHECK(option_type IN ('call', 'put')),
                    strike REAL NOT NULL CHECK(strike > 0),
                    bid REAL CHECK(bid IS NULL OR bid >= 0),
                    ask REAL CHECK(ask IS NULL OR ask >= 0),
                    mid REAL NOT NULL CHECK(mid > 0),
                    last_price REAL CHECK(last_price IS NULL OR last_price >= 0),
                    volume REAL CHECK(volume IS NULL OR volume >= 0),
                    open_interest REAL CHECK(open_interest IS NULL OR open_interest >= 0),
                    underlying_price REAL NOT NULL CHECK(underlying_price > 0),
                    source TEXT NOT NULL,
                    downloaded_at_utc TEXT NOT NULL,
                    PRIMARY KEY (ticker, date, expiration_date, option_type, strike, source),
                    CHECK(expiration_date >= date),
                    CHECK(ask IS NULL OR bid IS NULL OR ask >= bid),
                    CHECK(bid IS NULL OR ask IS NULL OR (mid >= bid AND mid <= ask))
                );

                CREATE TABLE IF NOT EXISTS run_manifests (
                    run_id TEXT PRIMARY KEY,
                    pipeline_name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('success', 'partial', 'failed')),
                    started_at_utc TEXT NOT NULL,
                    finished_at_utc TEXT NOT NULL,
                    tickers_json TEXT NOT NULL,
                    params_json TEXT NOT NULL,
                    rows_written_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL
                );
                """
            )

    def _load_table(
        self,
        table: str,
        contract_columns: tuple[str, ...],
        ticker: str,
        start_date: str | None,
        end_date: str | None,
        source: str,
        order_by: str,
    ) -> pd.DataFrame:
        clauses = ["ticker = ?", "source = ?"]
        params: list[Any] = [ticker.strip().upper(), source]
        if start_date is not None:
            clauses.append("date >= ?")
            params.append(_yyyymmdd_to_sql_date(start_date))
        if end_date is not None:
            clauses.append("date <= ?")
            params.append(_yyyymmdd_to_sql_date(end_date))

        sql = f"""
            SELECT {', '.join(contract_columns)}
            FROM {table}
            WHERE {' AND '.join(clauses)}
            ORDER BY {order_by}
        """
        with self._connect() as connection:
            return pd.read_sql_query(sql, connection, params=params)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _record_to_sqlite(record: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if pd.isna(value):
            clean[key] = None
        elif hasattr(value, "isoformat"):
            clean[key] = value.isoformat()
        else:
            clean[key] = value
    if "ticker" in clean and clean["ticker"] is not None:
        clean["ticker"] = str(clean["ticker"]).strip().upper()
    if "option_type" in clean and clean["option_type"] is not None:
        clean["option_type"] = str(clean["option_type"]).strip().lower()
    return clean


def _yyyymmdd_to_sql_date(value: str) -> str:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return text
