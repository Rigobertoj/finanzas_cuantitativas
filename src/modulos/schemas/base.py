"""Data contract primitives used by the quantitative modules.

The schemas in this package are intentionally lightweight. They describe the
shape expected from pandas DataFrames without owning ingestion, storage, or
strategy logic. This keeps contracts easy to read, quick to change, and safe to
reuse from notebooks or pipelines.

Example
-------
>>> from modulos.schemas.market_data import STOCK_EOD_CONTRACT
>>> STOCK_EOD_CONTRACT.required_columns
('ticker', 'date', 'close', 'source', 'downloaded_at_utc')
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColumnKind = Literal["string", "date", "datetime", "float"]


@dataclass(frozen=True)
class ColumnSpec:
    """Definition for one DataFrame column.

    Parameters
    ----------
    name:
        Column name expected in the DataFrame.
    kind:
        Logical type used by validators.
    required:
        Whether the column must be present for the contract to be valid.
    description:
        Short business meaning of the column.
    """

    name: str
    kind: ColumnKind
    required: bool
    description: str


@dataclass(frozen=True)
class DataContract:
    """Declarative schema for a tabular dataset.

    A contract is deliberately passive: it stores column expectations and a
    natural key. Validation lives in ``modulos.validation`` so contracts remain
    dependency-light and composable.
    """

    name: str
    columns: tuple[ColumnSpec, ...]
    natural_key: tuple[str, ...]
    description: str

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return the required column names in declaration order."""

        return tuple(column.name for column in self.columns if column.required)

    @property
    def optional_columns(self) -> tuple[str, ...]:
        """Return the optional column names in declaration order."""

        return tuple(column.name for column in self.columns if not column.required)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return all declared column names in declaration order."""

        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnSpec:
        """Return a column definition by name.

        Raises
        ------
        KeyError
            If ``name`` is not part of the contract.
        """

        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"{name!r} is not declared in {self.name}.")
