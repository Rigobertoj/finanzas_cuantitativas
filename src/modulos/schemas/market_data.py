"""Market data contracts for option strategy research.

These contracts describe normalized market datasets after provider-specific
fields have been mapped into project conventions. They do not call ThetaData,
Yahoo, storage, or models.

Example
-------
>>> from modulos.schemas.market_data import OPTION_EOD_CONTRACT
>>> "mid" in OPTION_EOD_CONTRACT.required_columns
True
"""

from __future__ import annotations

from .base import ColumnSpec, DataContract


STOCK_EOD_CONTRACT = DataContract(
    name="StockEOD",
    description="Daily end-of-day observation for an underlying asset.",
    natural_key=("ticker", "date", "source"),
    columns=(
        ColumnSpec("ticker", "string", True, "Underlying symbol."),
        ColumnSpec("date", "date", True, "Market date."),
        ColumnSpec("open", "float", False, "Opening price."),
        ColumnSpec("high", "float", False, "Daily high price."),
        ColumnSpec("low", "float", False, "Daily low price."),
        ColumnSpec("close", "float", True, "Closing price."),
        ColumnSpec("volume", "float", False, "Daily traded volume."),
        ColumnSpec("source", "string", True, "Data provider."),
        ColumnSpec("downloaded_at_utc", "datetime", True, "Download timestamp in UTC."),
    ),
)

OPTION_EOD_CONTRACT = DataContract(
    name="OptionEOD",
    description="Daily end-of-day observation for one option contract.",
    natural_key=("ticker", "date", "expiration_date", "option_type", "strike", "source"),
    columns=(
        ColumnSpec("ticker", "string", True, "Underlying symbol."),
        ColumnSpec("date", "date", True, "Observation date."),
        ColumnSpec("expiration_date", "date", True, "Option expiration date."),
        ColumnSpec("option_type", "string", True, "Option side: call or put."),
        ColumnSpec("strike", "float", True, "Option strike."),
        ColumnSpec("bid", "float", False, "Best bid price."),
        ColumnSpec("ask", "float", False, "Best ask price."),
        ColumnSpec("mid", "float", True, "Mid price used by downstream models."),
        ColumnSpec("last_price", "float", False, "Last traded price."),
        ColumnSpec("volume", "float", False, "Daily option volume."),
        ColumnSpec("open_interest", "float", False, "Open interest."),
        ColumnSpec("underlying_price", "float", True, "Underlying price on observation date."),
        ColumnSpec("source", "string", True, "Data provider."),
        ColumnSpec("downloaded_at_utc", "datetime", True, "Download timestamp in UTC."),
    ),
)

OPTION_GREEKS_CONTRACT = DataContract(
    name="OptionGreeks",
    description="Sensitivities for one option contract from a provider or model.",
    natural_key=("ticker", "date", "expiration_date", "option_type", "strike", "source"),
    columns=(
        ColumnSpec("ticker", "string", True, "Underlying symbol."),
        ColumnSpec("date", "date", True, "Observation date."),
        ColumnSpec("expiration_date", "date", True, "Option expiration date."),
        ColumnSpec("option_type", "string", True, "Option side: call or put."),
        ColumnSpec("strike", "float", True, "Option strike."),
        ColumnSpec("delta", "float", True, "First-order sensitivity to underlying price."),
        ColumnSpec("gamma", "float", False, "Second-order sensitivity to underlying price."),
        ColumnSpec("vega", "float", False, "Sensitivity to volatility."),
        ColumnSpec("theta", "float", False, "Sensitivity to time decay."),
        ColumnSpec("rho", "float", False, "Sensitivity to interest rates."),
        ColumnSpec("implied_volatility", "float", False, "Implied volatility."),
        ColumnSpec("source", "string", True, "Provider or model source."),
    ),
)

MARKET_DATA_CONTRACTS = (
    STOCK_EOD_CONTRACT,
    OPTION_EOD_CONTRACT,
    OPTION_GREEKS_CONTRACT,
)
