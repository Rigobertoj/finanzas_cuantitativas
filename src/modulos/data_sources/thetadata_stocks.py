"""ThetaData stock endpoints mapped to internal contracts."""

from __future__ import annotations

import pandas as pd

from modulos.validation import validate_stock_eod

from .base import normalize_ticker, require_yyyymmdd, utc_now
from .thetadata_client import ThetaDataClient


class ThetaDataStocks:
    """Provider for stock end-of-day data from ThetaData.

    ``ThetaDataStocks`` converts the provider-specific stock EOD endpoint into
    the internal ``StockEOD`` contract. It owns the provider request parameters
    and mapping rules, but delegates HTTP behavior to ``ThetaDataClient`` and
    data-quality checks to ``validate_stock_eod``.

    Attributes
    ----------
    client:
        HTTP client used to query the local Theta Terminal API.

    Examples
    --------
    >>> provider = ThetaDataStocks(ThetaDataClient())
    >>> isinstance(provider.client, ThetaDataClient)
    True
    """

    def __init__(self, client: ThetaDataClient | None = None) -> None:
        """Create a stock provider.

        Parameters
        ----------
        client:
            Optional shared ``ThetaDataClient``. Passing a client lets several
            providers reuse the same base URL and timeout configuration.
        """

        self.client = client or ThetaDataClient()

    def get_stock_eod(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Download and validate stock EOD data.

        The method requests ThetaData's stock EOD endpoint, maps the response to
        the internal ``StockEOD`` columns and validates the resulting DataFrame
        before returning it.

        Parameters
        ----------
        ticker:
            Underlying symbol. It is normalized to uppercase.
        start_date, end_date:
            Inclusive dates in ``YYYYMMDD`` format.

        Returns
        -------
        pandas.DataFrame
            DataFrame that satisfies the ``StockEOD`` contract.

        Raises
        ------
        ValueError
            If dates are malformed or ``start_date`` is after ``end_date``.
        DataSourceError
            If ThetaData cannot complete the request.
        """

        ticker = normalize_ticker(ticker)
        start_date = require_yyyymmdd(start_date, "start_date")
        end_date = require_yyyymmdd(end_date, "end_date")
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date.")

        raw = self.client.get_csv(
            "/stock/history/eod",
            {"symbol": ticker, "start_date": start_date, "end_date": end_date},
        )
        mapped = map_stock_eod(raw, ticker=ticker, downloaded_at_utc=utc_now())
        return validate_stock_eod(mapped)


def map_stock_eod(
    raw: pd.DataFrame,
    ticker: str,
    downloaded_at_utc,
) -> pd.DataFrame:
    """Map a ThetaData stock EOD response to the ``StockEOD`` contract.

    Parameters
    ----------
    raw:
        Raw DataFrame returned by ``ThetaDataClient.get_csv``.
    ticker:
        Fallback ticker used when the provider response does not include a
        symbol column.
    downloaded_at_utc:
        Timestamp assigned to all mapped rows.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the columns required by ``StockEOD``. The returned frame
        is mapped but not validated; callers should pass it to
        ``validate_stock_eod``.

    Raises
    ------
    ValueError
        If the response does not include a usable date or close/price column.
    """

    ticker = normalize_ticker(ticker)
    if raw.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "source",
                "downloaded_at_utc",
            ]
        )

    frame = raw.copy()
    created_col = _first_existing(frame, ("created", "date", "timestamp"))
    close_col = _first_existing(frame, ("close", "price"))

    mapped = pd.DataFrame(
        {
            "ticker": _series_or_value(frame, ("symbol", "ticker"), ticker),
            "date": frame[created_col],
            "open": _optional_series(frame, "open"),
            "high": _optional_series(frame, "high"),
            "low": _optional_series(frame, "low"),
            "close": frame[close_col],
            "volume": _optional_series(frame, "volume"),
            "source": "ThetaData",
            "downloaded_at_utc": downloaded_at_utc,
        }
    )
    return mapped


def _first_existing(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise ValueError(f"ThetaData response is missing one of {candidates}.")


def _optional_series(frame: pd.DataFrame, column: str):
    if column in frame.columns:
        return frame[column]
    return pd.NA


def _series_or_value(frame: pd.DataFrame, columns: tuple[str, ...], value: str):
    for column in columns:
        if column in frame.columns:
            return frame[column]
    return value
