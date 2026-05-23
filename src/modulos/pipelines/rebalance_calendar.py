"""Rebalance calendar utilities for hedging datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RebalanceCalendar:
    """Select rebalance dates from available market observations.

    ``RebalanceCalendar`` does not generate market dates from an external
    exchange calendar. It filters the dates that already exist in a DataFrame,
    which keeps the first implementation simple and avoids inventing rows for
    dates where the repository has no market data.

    Parameters
    ----------
    frequency:
        ``"daily"``, ``"weekly"`` or ``"custom"``.
    custom_dates:
        Dates used when ``frequency="custom"``.

    Attributes
    ----------
    frequency:
        Rebalance rule. ``"daily"`` keeps all available dates, ``"weekly"``
        keeps the last available observation in each week and ``"custom"``
        keeps only dates listed in ``custom_dates``.
    custom_dates:
        Optional explicit rebalance dates accepted by ``pandas.to_datetime``.
    """

    frequency: str = "daily"
    custom_dates: tuple[str, ...] | None = None

    def filter(self, frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
        """Return rows whose date belongs to the rebalance calendar.

        Parameters
        ----------
        frame:
            DataFrame containing the date column to filter.
        date_column:
            Name of the column that stores observation dates.

        Returns
        -------
        pandas.DataFrame
            Filtered copy of ``frame``. The date column is converted to pandas
            datetime dtype in the returned frame.

        Raises
        ------
        ValueError
            If ``frequency`` is not ``"daily"``, ``"weekly"`` or ``"custom"``.

        Examples
        --------
        >>> calendar = RebalanceCalendar("weekly")
        >>> calendar.filter(frame)  # doctest: +SKIP
        """

        if frame.empty:
            return frame.copy()

        result = frame.copy()
        result[date_column] = pd.to_datetime(result[date_column], errors="coerce")
        frequency = self.frequency.strip().lower()

        if frequency == "daily":
            return result
        if frequency == "weekly":
            weekly_dates = (
                result[[date_column]]
                .drop_duplicates()
                .assign(week=lambda data: data[date_column].dt.to_period("W"))
                .sort_values(date_column)
                .groupby("week", as_index=False)
                .tail(1)[date_column]
            )
            return result[result[date_column].isin(set(weekly_dates))].copy()
        if frequency == "custom":
            dates = _normalize_dates(self.custom_dates or ())
            return result[result[date_column].dt.date.isin(dates)].copy()

        raise ValueError("rebalance frequency must be daily, weekly, or custom.")


def _normalize_dates(values: Iterable[str]) -> set:
    """Convert custom rebalance dates to ``datetime.date`` values."""

    return set(pd.to_datetime(list(values), errors="coerce").date)
