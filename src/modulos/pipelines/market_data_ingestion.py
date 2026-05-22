"""Pipeline for reproducible market-data ingestion into SQLite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from modulos.data_sources import ThetaDataClient, ThetaDataOptions, ThetaDataStocks
from modulos.storage import RunManifest, SQLiteMarketDataRepository, new_run_id, utc_timestamp


@dataclass(frozen=True)
class MarketDataIngestionResult:
    """Result summary returned by ``MarketDataIngestionPipeline``.

    Attributes
    ----------
    run_id:
        Identifier persisted in the manifest table.
    status:
        Final status: ``"success"``, ``"partial"`` or ``"failed"``.
    results:
        Per-ticker row counts and status details.
    errors:
        Per-ticker errors captured during the run.
    rows_written:
        Aggregated row counts submitted to storage.
    """

    run_id: str
    status: str
    results: dict[str, Any]
    errors: list[dict[str, str]]
    rows_written: dict[str, int]


class MarketDataIngestionPipeline:
    """Orchestrate provider downloads, SQLite storage and run manifests.

    The pipeline is the multi-ticker layer. Providers remain focused on one
    ticker per request; this class coordinates several tickers, captures partial
    failures and persists a complete manifest at the end of every run.

    Attributes
    ----------
    stock_provider:
        Provider used to download ``StockEOD`` data.
    option_provider:
        Provider used to download ``OptionEOD`` data.
    repository:
        SQLite repository used for market data and manifests.
    provider_name:
        Name written to manifests.
    """

    def __init__(
        self,
        stock_provider: ThetaDataStocks | None = None,
        option_provider: ThetaDataOptions | None = None,
        repository: SQLiteMarketDataRepository | None = None,
        provider_name: str = "ThetaData",
    ) -> None:
        client = ThetaDataClient()
        self.stock_provider = stock_provider or ThetaDataStocks(client)
        self.option_provider = option_provider or ThetaDataOptions(client, self.stock_provider)
        self.repository = repository or SQLiteMarketDataRepository()
        self.provider_name = provider_name

    def run_option_eod_ingestion(
        self,
        tickers: str | Sequence[str],
        start_date: str,
        end_date: str,
        expiration: str = "*",
        right: str = "both",
        strike: str = "*",
        strike_range: int | None = 8,
        max_dte: int | None = 120,
    ) -> MarketDataIngestionResult:
        """Download stock and option EOD data for one or more tickers.

        Parameters
        ----------
        tickers:
            One ticker or a sequence of tickers.
        start_date, end_date:
            Inclusive date window in ``YYYYMMDD`` format.
        expiration, right, strike, strike_range, max_dte:
            Filters forwarded to ``ThetaDataOptions.get_option_eod``.

        Returns
        -------
        MarketDataIngestionResult
            Run identifier, status, row counts and captured errors.

        Notes
        -----
        A manifest is always written when this method returns normally. If one
        ticker fails and another succeeds, the final status is ``"partial"``.
        """

        normalized_tickers = _normalize_tickers(tickers)
        run_id = new_run_id()
        started_at_utc = utc_timestamp()
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "expiration": expiration,
            "right": right,
            "strike": strike,
            "strike_range": strike_range,
            "max_dte": max_dte,
        }

        results: dict[str, Any] = {}
        errors: list[dict[str, str]] = []
        rows_written = {"stock_eod": 0, "option_eod": 0}

        for ticker in normalized_tickers:
            stock_rows = 0
            option_rows = 0
            try:
                stock_eod = self.stock_provider.get_stock_eod(ticker, start_date, end_date)
                stock_rows = self.repository.save_stock_eod(stock_eod)
                rows_written["stock_eod"] += stock_rows

                option_eod = self.option_provider.get_option_eod(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    expiration=expiration,
                    right=right,
                    strike=strike,
                    strike_range=strike_range,
                    max_dte=max_dte,
                    underlying_prices=stock_eod,
                )
                option_rows = self.repository.save_option_eod(option_eod)
                rows_written["option_eod"] += option_rows

                results[ticker] = {
                    "status": "success",
                    "stock_rows": stock_rows,
                    "option_rows": option_rows,
                }
            except Exception as exc:  # noqa: BLE001 - manifest must capture provider/storage failures.
                message = str(exc) or exc.__class__.__name__
                errors.append(
                    {
                        "ticker": ticker,
                        "error_type": exc.__class__.__name__,
                        "message": message,
                    }
                )
                results[ticker] = {"status": "failed", "error": message}
                if stock_rows or option_rows:
                    results[ticker]["stock_rows"] = stock_rows
                    results[ticker]["option_rows"] = option_rows

        status = _final_status(results)
        manifest = RunManifest(
            run_id=run_id,
            pipeline_name="market_data_ingestion",
            provider=self.provider_name,
            status=status,
            started_at_utc=started_at_utc,
            finished_at_utc=utc_timestamp(),
            tickers=tuple(normalized_tickers),
            params=params,
            rows_written=rows_written,
            results=results,
            errors=errors,
        )
        self.repository.save_run_manifest(manifest)

        return MarketDataIngestionResult(
            run_id=run_id,
            status=status,
            results=results,
            errors=errors,
            rows_written=rows_written,
        )


def _normalize_tickers(tickers: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(tickers, str):
        values = (tickers,)
    else:
        values = tuple(tickers)
    normalized_list: list[str] = []
    seen: set[str] = set()
    for ticker in values:
        normalized_ticker = str(ticker).strip().upper()
        if normalized_ticker and normalized_ticker not in seen:
            normalized_list.append(normalized_ticker)
            seen.add(normalized_ticker)
    normalized = tuple(normalized_list)
    if not normalized:
        raise ValueError("tickers cannot be empty.")
    return normalized


def _final_status(results: dict[str, Any]) -> str:
    statuses = {value["status"] for value in results.values()}
    if statuses == {"success"}:
        return "success"
    if "success" in statuses:
        return "partial"
    return "failed"
