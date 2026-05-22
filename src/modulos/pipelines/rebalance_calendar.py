"""Rebalance calendar utilities for hedging datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class RebalanceCalendar:
    """Select rebalance dates from available market observations.

    Parameters
    ----------
    frequency:
        ``"daily"``, ``"weekly"`` or ``"custom"``.
    custom_dates:
        Dates used when ``frequency="custom"``.
    """

    frequency: str = "daily"
    custom_dates: tuple[str, ...] | None = None

    def filter(self, frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
        """Return rows whose date belongs to the rebalance calendar."""

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
    return set(pd.to_datetime(list(values), errors="coerce").date)
