"""ThetaData option endpoints mapped to internal contracts."""

from __future__ import annotations

import pandas as pd

from modulos.validation import validate_option_eod

from .base import normalize_ticker, require_yyyymmdd, utc_now
from .thetadata_client import ThetaDataClient
from .thetadata_stocks import ThetaDataStocks


class ThetaDataOptions:
    """Provider for option end-of-day data from ThetaData.

    ``ThetaDataOptions`` is the provider adapter for option chains. It downloads
    raw option EOD rows, maps provider-specific columns to the internal
    ``OptionEOD`` contract, computes ``mid`` from bid/ask, attaches the
    underlying stock price and validates the result.

    The provider is deliberately limited to market-data ingestion. It should not
    price options, calculate Greeks, run hedging logic or persist data.

    Attributes
    ----------
    client:
        HTTP client used to query Theta Terminal.
    stock_provider:
        Stock provider used to download underlying prices when they are not
        supplied by the caller.

    Examples
    --------
    >>> client = ThetaDataClient()
    >>> provider = ThetaDataOptions(client)
    >>> provider.client is client
    True
    """

    def __init__(
        self,
        client: ThetaDataClient | None = None,
        stock_provider: ThetaDataStocks | None = None,
    ) -> None:
        """Create an option provider.

        Parameters
        ----------
        client:
            Optional shared ``ThetaDataClient``. If omitted, a default client is
            created.
        stock_provider:
            Optional ``ThetaDataStocks`` instance used to fetch underlying EOD
            prices when ``get_option_eod`` does not receive
            ``underlying_prices``.
        """

        self.client = client or ThetaDataClient()
        self.stock_provider = stock_provider or ThetaDataStocks(self.client)

    def get_option_eod(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        expiration: str = "*",
        right: str = "both",
        strike: str = "*",
        strike_range: int | None = 8,
        max_dte: int | None = 120,
        underlying_prices: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Download and validate option EOD data.

        If ``underlying_prices`` is not supplied, stock EOD data is downloaded
        with the same date range and merged by date.

        Parameters
        ----------
        ticker:
            Underlying symbol. It is normalized to uppercase.
        start_date, end_date:
            Inclusive observation window in ``YYYYMMDD`` format.
        expiration:
            ThetaData expiration filter. ``"*"`` requests all expirations
            allowed by the other filters.
        right:
            Option side filter accepted by ThetaData, commonly ``"both"``,
            ``"call"`` or ``"put"`` depending on endpoint support.
        strike:
            ThetaData strike filter. ``"*"`` requests strikes around the
            provider's selected range.
        strike_range:
            Optional range control used to limit the number of strikes returned.
            ``None`` omits the parameter.
        max_dte:
            Optional maximum days-to-expiration filter. ``None`` omits the
            parameter.
        underlying_prices:
            Optional DataFrame containing either ``close`` or
            ``underlying_price`` by date. Passing it avoids a second stock EOD
            request.

        Returns
        -------
        pandas.DataFrame
            DataFrame that satisfies the ``OptionEOD`` contract.

        Raises
        ------
        ValueError
            If dates are malformed, the date window is invalid, required
            provider columns are missing, or validation rules fail.
        DataSourceError
            If ThetaData cannot complete the request.
        """

        ticker = normalize_ticker(ticker)
        start_date = require_yyyymmdd(start_date, "start_date")
        end_date = require_yyyymmdd(end_date, "end_date")
        if start_date > end_date:
            raise ValueError("start_date must be before or equal to end_date.")

        params: dict[str, object] = {
            "symbol": ticker,
            "expiration": expiration,
            "start_date": start_date,
            "end_date": end_date,
            "right": right,
            "strike": strike,
        }
        if strike_range is not None:
            params["strike_range"] = strike_range
        if max_dte is not None:
            params["max_dte"] = max_dte

        raw = self.client.get_csv("/option/history/eod", params)
        stock_eod = underlying_prices
        if stock_eod is None:
            stock_eod = self.stock_provider.get_stock_eod(ticker, start_date, end_date)

        mapped = map_option_eod(
            raw,
            ticker=ticker,
            underlying_prices=stock_eod,
            downloaded_at_utc=utc_now(),
        )
        return validate_option_eod(mapped)


def map_option_eod(
    raw: pd.DataFrame,
    ticker: str,
    underlying_prices: pd.DataFrame,
    downloaded_at_utc,
) -> pd.DataFrame:
    """Map a ThetaData option EOD response to the ``OptionEOD`` contract.

    Parameters
    ----------
    raw:
        Raw option EOD DataFrame returned by ``ThetaDataClient.get_csv``.
    ticker:
        Fallback underlying symbol used when the provider response does not
        include a symbol column.
    underlying_prices:
        Stock EOD DataFrame containing ``date`` plus either ``close`` or
        ``underlying_price``. It is merged into option rows by observation date.
    downloaded_at_utc:
        Timestamp assigned to all mapped option rows.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the columns required by ``OptionEOD``. The returned frame
        is mapped but not validated; callers should pass it to
        ``validate_option_eod``.

    Raises
    ------
    ValueError
        If the option response is missing required provider columns or the
        underlying price DataFrame does not include ``close`` or
        ``underlying_price``.
    """

    ticker = normalize_ticker(ticker)
    columns = [
        "ticker",
        "date",
        "expiration_date",
        "option_type",
        "strike",
        "bid",
        "ask",
        "mid",
        "last_price",
        "volume",
        "open_interest",
        "underlying_price",
        "source",
        "downloaded_at_utc",
    ]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    frame = raw.copy()
    created_col = _first_existing(frame, ("created", "date", "timestamp"))
    expiration_col = _first_existing(frame, ("expiration", "expiration_date"))
    right_col = _first_existing(frame, ("right", "option_type"))
    close_col = _first_existing(frame, ("close", "last_price", "price"))

    mapped = pd.DataFrame(
        {
            "ticker": _series_or_value(frame, ("symbol", "ticker"), ticker),
            "date": pd.to_datetime(frame[created_col], errors="coerce").dt.date,
            "expiration_date": frame[expiration_col],
            "option_type": frame[right_col].map(_normalize_right),
            "strike": frame["strike"],
            "bid": _optional_series(frame, "bid"),
            "ask": _optional_series(frame, "ask"),
            "last_price": frame[close_col],
            "volume": _optional_series(frame, "volume"),
            "open_interest": _optional_series(frame, "open_interest"),
            "source": "ThetaData",
            "downloaded_at_utc": downloaded_at_utc,
        }
    )
    mapped["mid"] = _build_mid(mapped)
    return _merge_underlying_price(mapped, underlying_prices)


def _merge_underlying_price(options: pd.DataFrame, stock_eod: pd.DataFrame) -> pd.DataFrame:
    if "underlying_price" in stock_eod.columns:
        stock_col = "underlying_price"
    elif "close" in stock_eod.columns:
        stock_col = "close"
    else:
        raise ValueError("underlying_prices must include `close` or `underlying_price`.")

    stock = stock_eod.copy()
    stock["date"] = pd.to_datetime(stock["date"], errors="coerce").dt.date
    stock = stock[["date", stock_col]].rename(columns={stock_col: "underlying_price"})
    return options.merge(stock, on="date", how="left")


def _build_mid(frame: pd.DataFrame) -> pd.Series:
    if {"bid", "ask"}.issubset(frame.columns):
        bid = pd.to_numeric(frame["bid"], errors="coerce")
        ask = pd.to_numeric(frame["ask"], errors="coerce")
        return (bid + ask) / 2
    return pd.to_numeric(frame["last_price"], errors="coerce")


def _normalize_right(value) -> str:
    text = str(value).strip().lower()
    if text in {"c", "call"}:
        return "call"
    if text in {"p", "put"}:
        return "put"
    return text


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
