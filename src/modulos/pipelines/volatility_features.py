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

    Parameters
    ----------
    stock_eod:
        Stock EOD DataFrame with ``ticker``, ``date`` and ``close``.
    window:
        Rolling window in observations. The default is 20.
    annualization_factor:
        Annualization factor applied to the rolling standard deviation.

    Returns
    -------
    pandas.DataFrame
        Frame with ``date`` and ``realized_volatility``.
    """

    if window <= 1:
        raise ValueError("realized volatility window must be greater than 1.")

    frame = stock_eod.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.sort_values(["ticker", "date"])
    frame["log_return"] = frame.groupby("ticker")["close"].transform(
        lambda series: (series.astype(float) / series.astype(float).shift(1)).apply(_safe_log)
    )
    frame["realized_volatility"] = (
        frame.groupby("ticker")["log_return"]
        .transform(lambda series: series.rolling(window=window).std())
        * sqrt(annualization_factor)
    )
    return frame[["ticker", "date", "realized_volatility"]]


def _safe_log(value: float) -> float:
    if pd.isna(value) or value <= 0:
        return float("nan")
    from math import log

    return log(value)
