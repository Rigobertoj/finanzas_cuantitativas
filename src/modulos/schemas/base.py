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
    """Describe one expected column inside a tabular data contract.

    ``ColumnSpec`` is the smallest schema primitive in the project. It does not
    validate data by itself; it only records the business meaning and logical
    type that validators should enforce later.

    Attributes
    ----------
    name:
        Column name expected in a pandas DataFrame.
    kind:
        Logical type used by ``modulos.validation``. Supported values are
        ``"string"``, ``"date"``, ``"datetime"`` and ``"float"``.
    required:
        Whether the column must be present and non-null for the contract to be
        valid.
    description:
        Human-readable business meaning of the column.

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

    Examples
    --------
    >>> spec = ColumnSpec("close", "float", True, "Closing price.")
    >>> spec.required
    True
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

    ``DataContract`` is the shared language between data providers, validators,
    storage and future strategy pipelines. Providers map external responses into
    DataFrames that match a contract; validators enforce the declared shape and
    business rules; downstream modules consume only validated frames.

    Attributes
    ----------
    name:
        Stable logical name of the dataset, for example ``"StockEOD"`` or
        ``"OptionEOD"``.
    columns:
        Ordered tuple of ``ColumnSpec`` definitions.
    natural_key:
        Columns that identify a unique row after normalization.
    description:
        Short explanation of the dataset's role in the system.

    Notes
    -----
    The class is frozen to make contract definitions immutable at runtime. This
    avoids accidental schema drift inside notebooks or pipelines.

    Examples
    --------
    >>> contract = DataContract(
    ...     name="Example",
    ...     columns=(ColumnSpec("ticker", "string", True, "Symbol."),),
    ...     natural_key=("ticker",),
    ...     description="Example contract.",
    ... )
    >>> contract.required_columns
    ('ticker',)
    """

    name: str
    columns: tuple[ColumnSpec, ...]
    natural_key: tuple[str, ...]
    description: str

    @property
    def required_columns(self) -> tuple[str, ...]:
        """Return required column names in declaration order.

        Returns
        -------
        tuple[str, ...]
            Names of columns declared with ``required=True``.
        """

        return tuple(column.name for column in self.columns if column.required)

    @property
    def optional_columns(self) -> tuple[str, ...]:
        """Return optional column names in declaration order.

        Returns
        -------
        tuple[str, ...]
            Names of columns declared with ``required=False``.
        """

        return tuple(column.name for column in self.columns if not column.required)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Return all declared column names in declaration order.

        Returns
        -------
        tuple[str, ...]
            Required and optional columns exactly as they were declared.
        """

        return tuple(column.name for column in self.columns)

    def column(self, name: str) -> ColumnSpec:
        """Return a column definition by name.

        Parameters
        ----------
        name:
            Column name to look up in the contract.

        Returns
        -------
        ColumnSpec
            Column definition associated with ``name``.

        Raises
        ------
        KeyError
            If ``name`` is not part of the contract.

        Examples
        --------
        >>> contract = DataContract(
        ...     name="Example",
        ...     columns=(ColumnSpec("ticker", "string", True, "Symbol."),),
        ...     natural_key=("ticker",),
        ...     description="Example contract.",
        ... )
        >>> contract.column("ticker").kind
        'string'
        """

        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"{name!r} is not declared in {self.name}.")
