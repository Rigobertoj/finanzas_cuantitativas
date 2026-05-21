"""Composable DataFrame validation helpers.

Validators return a copy of the received DataFrame with simple normalization
applied. They raise ``ValueError`` with clear messages when a contract rule is
broken. This keeps failures explicit in notebooks, pipelines, and tests.
"""

from __future__ import annotations

import pandas as pd

from modulos.schemas import DataContract


def validate_contract(df: pd.DataFrame, contract: DataContract) -> pd.DataFrame:
    """Validate generic contract rules and normalize basic column types.

    Parameters
    ----------
    df:
        DataFrame to validate.
    contract:
        Declarative contract with columns and natural key.

    Returns
    -------
    pandas.DataFrame
        A copy of ``df`` with basic parsing applied.

    Raises
    ------
    ValueError
        If required columns are missing, values cannot be parsed, or natural key
        duplicates exist.
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{contract.name} must be a pandas DataFrame.")

    result = df.copy()
    _require_columns(result, contract)
    _normalize_declared_columns(result, contract)
    _reject_required_nulls(result, contract)
    _reject_duplicate_key(result, contract)
    return result


def require_allowed_values(
    df: pd.DataFrame,
    column: str,
    allowed_values: set[str],
    contract_name: str,
) -> None:
    """Ensure non-null values in ``column`` belong to ``allowed_values``."""

    if column not in df.columns:
        return
    values = set(df[column].dropna().astype(str).str.lower().unique())
    invalid = sorted(values - allowed_values)
    if invalid:
        raise ValueError(f"{contract_name}.{column} has invalid values: {invalid}.")


def require_positive(df: pd.DataFrame, columns: tuple[str, ...], contract_name: str) -> None:
    """Ensure present columns contain strictly positive values when non-null."""

    for column in columns:
        if column in df.columns and (df[column].dropna() <= 0).any():
            raise ValueError(f"{contract_name}.{column} must be positive.")


def require_non_negative(df: pd.DataFrame, columns: tuple[str, ...], contract_name: str) -> None:
    """Ensure present columns contain non-negative values when non-null."""

    for column in columns:
        if column in df.columns and (df[column].dropna() < 0).any():
            raise ValueError(f"{contract_name}.{column} must be non-negative.")


def reject_duplicate_key(df: pd.DataFrame, contract: DataContract) -> None:
    """Reject duplicate rows using a contract natural key."""

    key_columns = [column for column in contract.natural_key if column in df.columns]
    if len(key_columns) != len(contract.natural_key):
        return
    duplicates = df.duplicated(subset=key_columns, keep=False)
    if duplicates.any():
        raise ValueError(f"{contract.name} has duplicate natural-key rows.")


def _require_columns(df: pd.DataFrame, contract: DataContract) -> None:
    missing = [column for column in contract.required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{contract.name} is missing required columns: {missing}.")


def _normalize_declared_columns(df: pd.DataFrame, contract: DataContract) -> None:
    for column in contract.columns:
        if column.name not in df.columns:
            continue
        if column.kind == "string":
            df[column.name] = df[column.name].astype("string").str.strip()
        elif column.kind == "date":
            parsed = pd.to_datetime(df[column.name], errors="coerce")
            _reject_parse_failures(df[column.name], parsed, contract.name, column.name)
            df[column.name] = parsed.dt.date
        elif column.kind == "datetime":
            parsed = pd.to_datetime(df[column.name], errors="coerce", utc=True)
            _reject_parse_failures(df[column.name], parsed, contract.name, column.name)
            df[column.name] = parsed
        elif column.kind == "float":
            parsed = pd.to_numeric(df[column.name], errors="coerce")
            _reject_parse_failures(df[column.name], parsed, contract.name, column.name)
            df[column.name] = parsed


def _reject_parse_failures(
    original: pd.Series,
    parsed: pd.Series,
    contract_name: str,
    column_name: str,
) -> None:
    failed = parsed.isna() & original.notna()
    if failed.any():
        raise ValueError(f"{contract_name}.{column_name} has values that cannot be parsed.")


def _reject_required_nulls(df: pd.DataFrame, contract: DataContract) -> None:
    null_columns = [
        column for column in contract.required_columns if df[column].isna().any()
    ]
    if null_columns:
        raise ValueError(f"{contract.name} has nulls in required columns: {null_columns}.")


def _reject_duplicate_key(df: pd.DataFrame, contract: DataContract) -> None:
    reject_duplicate_key(df, contract)
