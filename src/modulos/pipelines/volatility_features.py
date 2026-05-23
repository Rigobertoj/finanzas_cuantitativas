"""Volatility feature engineering for hedging datasets."""

from __future__ import annotations

from math import sqrt

import pandas as pd


def add_realized_volatility(
    stock_eod: pd.DataFrame,
    window: int = 20,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """Add rolling realized volatility from log returns.

    The feature uses the simple Phase 5 definition requested for the first
    hedging dataset: rolling standard deviation of log returns, annualized by a
    configurable factor. It does not estimate expected return, subtract alpha,
    fit a volatility model or backfill missing values.

    Parameters
    ----------
    stock_eod:
        Stock EOD DataFrame with ``ticker``, ``date`` and ``close``. Rows are
        sorted by ticker and date before returns are computed.
    window:
        Rolling window in observations. The default is 20.
    annualization_factor:
        Annualization factor applied to the rolling standard deviation.

    Returns
    -------
    pandas.DataFrame
        Frame with ``ticker``, ``date`` and ``realized_volatility``.

    Raises
    ------
    ValueError
        If ``window`` is less than or equal to 1.

    Notes
    -----
    The first observations for each ticker naturally return null volatility
    until enough returns exist to fill the rolling window. The pipeline keeps
    those null values instead of imputing them.

    Examples
    --------
    >>> realized = add_realized_volatility(stock_eod, window=20)  # doctest: +SKIP
    >>> realized.columns.tolist()  # doctest: +SKIP
    ['ticker', 'date', 'realized_volatility']
    """

    if window <= 1:
        raise ValueError("realized volatility window must be greater than 1.")

    frame = stock_eod.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["ticker", "date"])
    frame["log_return"] = frame.groupby("ticker")["close"].transform(
        lambda series: (series.astype(float) / series.astype(float).shift(1)).apply(
            _safe_log
        )
    )
    frame["realized_volatility"] = (
        frame.groupby("ticker")["log_return"]
        .transform(lambda series: series.rolling(window=window).std())
        * sqrt(annualization_factor)
    )
    return frame[["ticker", "date", "realized_volatility"]]


def _safe_log(value: float) -> float:
    """Return ``log(value)`` and keep invalid ratios as ``NaN``."""

    if pd.isna(value) or value <= 0:
        return float("nan")
    from math import log

    return log(value)
