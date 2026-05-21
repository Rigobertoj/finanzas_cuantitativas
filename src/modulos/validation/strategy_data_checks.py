"""Validation functions for strategy datasets and results."""

from __future__ import annotations

import pandas as pd

from modulos.schemas.strategy_data import (
    HEDGING_DATASET_CONTRACT,
    STRATEGY_RESULT_CONTRACT,
)
from modulos.validation.base import (
    reject_duplicate_key,
    require_allowed_values,
    require_non_negative,
    require_positive,
    validate_contract,
)


def validate_hedging_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a ``HedgingDataset`` DataFrame.

    Example
    -------
    >>> import pandas as pd
    >>> from modulos.validation.strategy_data_checks import validate_hedging_dataset
    >>> df = pd.DataFrame({
    ...     "ticker": ["AMZN"],
    ...     "date": ["2026-04-01"],
    ...     "expiration_date": ["2026-06-19"],
    ...     "option_type": ["call"],
    ...     "strike": [190],
    ...     "option_mid": [4.2],
    ...     "underlying_price": [185.2],
    ...     "time_to_maturity": [0.22],
    ...     "risk_free_rate": [0.045],
    ... })
    >>> validate_hedging_dataset(df).shape[0]
    1
    """

    result = validate_contract(df, HEDGING_DATASET_CONTRACT)
    _normalize_ticker(result)
    _normalize_option_type(result, HEDGING_DATASET_CONTRACT.name)
    reject_duplicate_key(result, HEDGING_DATASET_CONTRACT)
    require_positive(
        result,
        (
            "strike",
            "option_mid",
            "underlying_price",
            "time_to_maturity",
            "implied_volatility",
            "model_volatility",
        ),
        HEDGING_DATASET_CONTRACT.name,
    )
    _require_expiration_after_date(result, HEDGING_DATASET_CONTRACT.name)
    return result


def validate_strategy_result(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a ``StrategyResult`` DataFrame."""

    result = validate_contract(df, STRATEGY_RESULT_CONTRACT)
    _normalize_ticker(result)
    reject_duplicate_key(result, STRATEGY_RESULT_CONTRACT)
    require_non_negative(result, ("transaction_cost",), STRATEGY_RESULT_CONTRACT.name)
    return result


def _normalize_ticker(df: pd.DataFrame) -> None:
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()


def _normalize_option_type(df: pd.DataFrame, contract_name: str) -> None:
    df["option_type"] = df["option_type"].astype("string").str.strip().str.lower()
    require_allowed_values(df, "option_type", {"call", "put"}, contract_name)


def _require_expiration_after_date(df: pd.DataFrame, contract_name: str) -> None:
    rows = df["expiration_date"] <= df["date"]
    if rows.any():
        raise ValueError(f"{contract_name}.expiration_date must be after date.")
