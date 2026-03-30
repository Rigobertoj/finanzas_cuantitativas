from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class AssetElementaryMetrics:
    """
    Research helper for asset prices and return metrics.

    This class centralizes a simple workflow:
    1) Download close prices from Yahoo Finance.
    2) Compute daily returns.
    3) Compute summary metrics (return, volatility, skew, intervals, etc.).

    Quick start:
        research = AssetElementaryMetrics(["JPM", "V"], start="2020-01-01")
        research.download_prices()
        metrics_df = research.metrics()

    Public API map:
    - Data loading: download_prices, compute_returns, get_prices, get_returns
    - Metrics: annual_return, annual_volatility, skew, vol_over_mean
    - Intervals/reporting: return_interval_pct, metrics
    - Statistics: describe_returns, covariance, correlation
    """

    tickers: InitVar[Iterable[str]]
    start: InitVar[str]
    end: InitVar[Optional[str]] = None
    price_field: InitVar[str] = "Close"

    __tickers: List[str] = field(init=False, repr=False, default_factory=list)
    __start: str = field(init=False, repr=False, default="")
    __end: Optional[str] = field(init=False, repr=False, default=None)
    __price_field: str = field(init=False, repr=False, default="Close")
    __prices: pd.DataFrame = field(init=False, repr=False, default_factory=pd.DataFrame)
    __returns: pd.DataFrame = field(
        init=False, repr=False, default_factory=pd.DataFrame
    )

    def __post_init__(
        self,
        tickers: Iterable[str],
        start: str,
        end: Optional[str],
        price_field: str,
    ) -> None:
        """Store validated inputs in private attributes."""
        self.tickers = tickers
        self.start = start
        self.end = end
        self.price_field = price_field

    def _reset_cache(self) -> None:
        """Reset derived cached data when source configuration changes."""
        self.__prices = pd.DataFrame()
        self.__returns = pd.DataFrame()

    def _get_tickers(self) -> List[str]:
        """Public read access to ticker symbols."""
        return list(self.__tickers)

    def _set_tickers(self, value: Iterable[str]) -> None:
        normalized = self._normalize_tickers(value)
        if not normalized:
            raise ValueError("tickers must contain at least one symbol.")
        self.__tickers = normalized
        self._reset_cache()

    def _get_start(self) -> str:
        """Public read access to start date."""
        return self.__start

    def _set_start(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("start must be a non-empty date string.")
        self.__start = value
        self._reset_cache()

    def _get_end(self) -> Optional[str]:
        """Public read access to end date."""
        return self.__end

    def _set_end(self, value: Optional[str]) -> None:
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError("end must be None or a non-empty date string.")
        self.__end = value
        self._reset_cache()

    def _get_price_field(self) -> str:
        """Public read access to selected price field."""
        return self.__price_field

    def _set_price_field(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("price_field must be a non-empty string.")
        self.__price_field = value
        self._reset_cache()

    def _get_prices(self) -> pd.DataFrame:
        """Read-only style accessor for cached prices."""
        return self.__prices.copy()

    def _set_prices_cache(self, value: pd.DataFrame) -> None:
        """Internal cache update for prices."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("prices cache must be a pandas DataFrame.")
        self.__prices = value.copy()

    def _get_returns(self) -> pd.DataFrame:
        """Read-only style accessor for cached returns."""
        return self.__returns.copy()

    def _set_returns_cache(self, value: pd.DataFrame) -> None:
        """Internal cache update for returns."""
        if not isinstance(value, pd.DataFrame):
            raise TypeError("returns cache must be a pandas DataFrame.")
        self.__returns = value.copy()

    @staticmethod
    def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
        """Normalize constructor tickers into a list."""
        if isinstance(tickers, str):
            return [tickers]
        return [str(t) for t in tickers]

    @staticmethod
    def _normalize_select(
        tickers: Optional[Union[str, Sequence[str]]],
    ) -> Optional[List[str]]:
        """Normalize optional ticker selection to list or None."""
        if tickers is None:
            return None
        if isinstance(tickers, str):
            return [tickers]
        return list(tickers)

    @staticmethod
    def _select_columns(df: pd.DataFrame, tickers: Optional[List[str]]) -> pd.DataFrame:
        """Return selected ticker columns, validating all requested names."""
        if tickers is None:
            return df
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise ValueError(f"Tickers not found in data: {missing}")
        return df[tickers]

    def download_prices(self) -> pd.DataFrame:
        """
        Download price data from Yahoo Finance and cache it in ``self.prices``.

        Returns:
            pd.DataFrame: Price table indexed by date, columns are tickers.
        """
        data = yf.download(
            self.tickers,
            start=self.start,
            end=self.end,
            progress=False,
        )

        if isinstance(data.columns, pd.MultiIndex):
            if self.price_field in data.columns.levels[0]:
                prices = data[self.price_field]
            else:
                raise ValueError(
                    f"price_field '{self.price_field}' not found in downloaded data."
                )
        else:
            if self.price_field in data.columns:
                prices = data[self.price_field]
            else:
                raise ValueError(
                    f"price_field '{self.price_field}' not found in downloaded data."
                )

        self._set_prices_cache(prices.dropna())
        return self.prices

    def compute_returns(self) -> pd.DataFrame:
        """
        Compute daily percentage returns from cached prices.

        If prices are not available, it triggers ``download_prices()`` first.

        Returns:
            pd.DataFrame: Daily returns by ticker.
        """
        if self.prices.empty:
            self.download_prices()

        self._set_returns_cache(self.prices.pct_change().dropna())
        return self.returns

    def get_prices(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Return cached prices for all tickers or a subset.

        Args:
            tickers: Optional ticker or list/tuple of tickers to filter.

        Returns:
            pd.DataFrame: Filtered price table.
        """
        if self.prices.empty:
            self.download_prices()

        select = self._normalize_select(tickers)
        return self._select_columns(self.prices, select)

    def get_returns(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Return cached returns for all tickers or a subset.

        Args:
            tickers: Optional ticker or list/tuple of tickers to filter.

        Returns:
            pd.DataFrame: Filtered return table.
        """
        if self.returns.empty:
            self.compute_returns()
        select = self._normalize_select(tickers)
        return self._select_columns(self.returns, select)

    def annual_return(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        """
        Compute annualized expected return using 252 trading days.

        Returns:
            pd.Series: Annualized mean return by ticker.
        """
        data = self.get_returns(tickers)
        return data.mean() * 252

    def annual_volatility(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        """
        Compute annualized volatility using 252 trading days.

        Returns:
            pd.Series: Annualized standard deviation by ticker.
        """
        data = self.get_returns(tickers)
        return data.std() * np.sqrt(252)

    def skew(self, tickers: Optional[Union[str, Sequence[str]]] = None) -> pd.Series:
        """
        Compute skewness of daily returns.

        Returns:
            pd.Series: Return skewness by ticker.
        """
        data = self.get_returns(tickers)
        return data.skew()

    def vol_over_mean(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.Series:
        """
        Compute volatility divided by annualized return.

        A lower value can indicate a better risk/return profile
        for this simple ratio.

        Returns:
            pd.Series: Volatility over annual return by ticker.
        """
        annual_ret = self.annual_return(tickers)
        annual_vol = self.annual_volatility(tickers)
        return annual_vol / annual_ret.replace(0, np.nan)

    def return_interval_pct(
        self, tickers: Optional[Union[str, Sequence[str]]] = None, z: float = 2.65
    ) -> pd.DataFrame:
        """
        Estimate a return interval in percentage terms.

        Interval is computed as:
            annual_return_pct +/- z * annual_volatility_pct

        Args:
            tickers: Optional ticker selection.
            z: Z-score multiplier (default 2.65).

        Returns:
            pd.DataFrame: DataFrame with ``low`` and ``high`` columns.
        """
        annual_ret_pct = self.annual_return(tickers) * 100
        annual_vol_pct = self.annual_volatility(tickers) * 100
        low = annual_ret_pct - z * annual_vol_pct
        high = annual_ret_pct + z * annual_vol_pct
        return pd.DataFrame({"low": low, "high": high})

    def metrics(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Build a consolidated metrics table per ticker.

        Columns:
        - annual_return
        - annual_volatility
        - annual_return_pct
        - annual_volatility_pct
        - skew
        - vol_over_mean
        - return_interval_low_pct
        - return_interval_high_pct

        Returns:
            pd.DataFrame: Multi-metric summary by ticker.
        """
        if self.returns.empty:
            self.compute_returns()

        data = self.get_returns(tickers)

        annual_return = data.mean() * 252
        annual_vol = data.std() * np.sqrt(252)

        annual_return_pct = annual_return * 100
        annual_vol_pct = annual_vol * 100

        vol_over_mean = annual_vol / annual_return.replace(0, np.nan)

        interval_low = annual_return_pct - 2.65 * annual_vol_pct
        interval_high = annual_return_pct + 2.65 * annual_vol_pct

        metrics_df = pd.DataFrame(
            {
                "annual_return": annual_return,
                "annual_volatility": annual_vol,
                "annual_return_pct": annual_return_pct,
                "annual_volatility_pct": annual_vol_pct,
                "skew": data.skew(),
                "vol_over_mean": vol_over_mean,
                "return_interval_low_pct": interval_low,
                "return_interval_high_pct": interval_high,
            }
        )

        return metrics_df

    def describe_returns(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Return descriptive statistics for daily returns.

        Returns:
            pd.DataFrame: ``pandas.DataFrame.describe()`` over returns.
        """
        data = self.get_returns(tickers)
        return data.describe()

    def covariance(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute covariance matrix of daily returns.

        Returns:
            pd.DataFrame: Covariance matrix.
        """
        data = self.get_returns(tickers)
        return data.cov()

    def correlation(
        self, tickers: Optional[Union[str, Sequence[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute correlation matrix of daily returns.

        Returns:
            pd.DataFrame: Correlation matrix.
        """
        data = self.get_returns(tickers)
        return data.corr()


# Bind properties after dataclass processing to keep constructor parameters
# (`tickers`, `start`, `end`, `price_field`) and still encapsulate state.
AssetElementaryMetrics.tickers = property(AssetElementaryMetrics._get_tickers, AssetElementaryMetrics._set_tickers)
AssetElementaryMetrics.start = property(AssetElementaryMetrics._get_start, AssetElementaryMetrics._set_start)
AssetElementaryMetrics.end = property(AssetElementaryMetrics._get_end, AssetElementaryMetrics._set_end)
AssetElementaryMetrics.price_field = property(
    AssetElementaryMetrics._get_price_field, AssetElementaryMetrics._set_price_field
)
AssetElementaryMetrics.prices = property(AssetElementaryMetrics._get_prices)
AssetElementaryMetrics.returns = property(AssetElementaryMetrics._get_returns)
