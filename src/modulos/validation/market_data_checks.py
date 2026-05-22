"""Validation functions for market data contracts.

Each validator accepts a pandas DataFrame, applies generic contract checks, and
then enforces a small set of finance-specific rules. The returned DataFrame is a
normalized copy that can be passed to pipelines or stored locally.

Example
-------
>>> import pandas as pd
>>> from modulos.validation.market_data_checks import validate_stock_eod
>>> df = pd.DataFrame({
...     "ticker": ["amzn"],
...     "date": ["2026-04-01"],
...     "close": [185.2],
...     "source": ["ThetaData"],
...     "downloaded_at_utc": ["2026-04-02T00:00:00Z"],
... })
>>> validate_stock_eod(df)["ticker"].iloc[0]
'AMZN'
"""

from __future__ import annotations

import pandas as pd

from modulos.schemas.market_data import (
    OPTION_EOD_CONTRACT,
    OPTION_GREEKS_CONTRACT,
    STOCK_EOD_CONTRACT,
)
from modulos.validation.base import (
    reject_duplicate_key,
    require_allowed_values,
    require_non_negative,
    require_positive,
    validate_contract,
)


def validate_stock_eod(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a ``StockEOD`` DataFrame."""

    result = validate_contract(df, STOCK_EOD_CONTRACT)
    _normalize_ticker(result)
    reject_duplicate_key(result, STOCK_EOD_CONTRACT)
    require_positive(result, ("close", "open", "high", "low"), STOCK_EOD_CONTRACT.name)
    require_non_negative(result, ("volume",), STOCK_EOD_CONTRACT.name)
    _require_high_greater_or_equal_low(result)
    return result


def validate_option_eod(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an ``OptionEOD`` DataFrame."""

    result = validate_contract(df, OPTION_EOD_CONTRACT)
    _normalize_ticker(result)
    _normalize_option_type(result, OPTION_EOD_CONTRACT.name)
    reject_duplicate_key(result, OPTION_EOD_CONTRACT)
    require_positive(result, ("strike", "mid", "underlying_price"), OPTION_EOD_CONTRACT.name)
    require_non_negative(
        result,
        ("bid", "ask", "last_price", "volume", "open_interest"),
        OPTION_EOD_CONTRACT.name,
    )
    _require_expiration_not_before_date(result, OPTION_EOD_CONTRACT.name)
    _require_ask_greater_or_equal_bid(result)
    _require_mid_between_bid_ask(result)
    return result


def validate_option_greeks(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an ``OptionGreeks`` DataFrame."""

    result = validate_contract(df, OPTION_GREEKS_CONTRACT)
    _normalize_ticker(result)
    _normalize_option_type(result, OPTION_GREEKS_CONTRACT.name)
    reject_duplicate_key(result, OPTION_GREEKS_CONTRACT)
    require_positive(result, ("strike", "implied_volatility"), OPTION_GREEKS_CONTRACT.name)
    require_non_negative(result, ("gamma", "vega"), OPTION_GREEKS_CONTRACT.name)
    _require_expiration_after_date(result, OPTION_GREEKS_CONTRACT.name)
    _require_delta_range(result)
    return result


def _normalize_ticker(df: pd.DataFrame) -> None:
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()


def _normalize_option_type(df: pd.DataFrame, contract_name: str) -> None:
    df["option_type"] = df["option_type"].astype("string").str.strip().str.lower()
    require_allowed_values(df, "option_type", {"call", "put"}, contract_name)


def _require_high_greater_or_equal_low(df: pd.DataFrame) -> None:
    if {"high", "low"}.issubset(df.columns):
        rows = df["high"].notna() & df["low"].notna() & (df["high"] < df["low"])
        if rows.any():
            raise ValueError("StockEOD.high must be greater than or equal to low.")


def _require_expiration_not_before_date(df: pd.DataFrame, contract_name: str) -> None:
    rows = df["expiration_date"] < df["date"]
    if rows.any():
        raise ValueError(f"{contract_name}.expiration_date must not be before date.")


def _require_ask_greater_or_equal_bid(df: pd.DataFrame) -> None:
    if {"ask", "bid"}.issubset(df.columns):
        rows = df["ask"].notna() & df["bid"].notna() & (df["ask"] < df["bid"])
        if rows.any():
            raise ValueError("OptionEOD.ask must be greater than or equal to bid.")


def _require_mid_between_bid_ask(df: pd.DataFrame) -> None:
    if not {"mid", "bid", "ask"}.issubset(df.columns):
        return
    rows = (
        df["mid"].notna()
        & df["bid"].notna()
        & df["ask"].notna()
        & ((df["mid"] < df["bid"]) | (df["mid"] > df["ask"]))
    )
    if rows.any():
        raise ValueError("OptionEOD.mid must be between bid and ask.")


def _require_delta_range(df: pd.DataFrame) -> None:
    rows = df["delta"].notna() & ((df["delta"] < -1.0) | (df["delta"] > 1.0))
    if rows.any():
        raise ValueError("OptionGreeks.delta must be between -1 and 1.")
