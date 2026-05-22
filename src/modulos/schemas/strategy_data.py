"""Strategy input and output contracts.

The contracts in this module sit after market-data normalization. They define
what strategy modules consume and what they must return so strategy results can
be compared under common assumptions.

Example
-------
>>> from modulos.schemas.strategy_data import HEDGING_DATASET_CONTRACT
>>> HEDGING_DATASET_CONTRACT.natural_key
('ticker', 'date', 'expiration_date', 'option_type', 'strike')
"""

from __future__ import annotations

from .base import ColumnSpec, DataContract


HEDGING_DATASET_CONTRACT = DataContract(
    name="HedgingDataset",
    description="Processed option dataset ready for hedging strategies.",
    natural_key=("ticker", "date", "expiration_date", "option_type", "strike"),
    columns=(
        ColumnSpec("ticker", "string", True, "Underlying symbol."),
        ColumnSpec("date", "date", True, "Rebalance date."),
        ColumnSpec("expiration_date", "date", True, "Option expiration date."),
        ColumnSpec("option_type", "string", True, "Option side: call or put."),
        ColumnSpec("strike", "float", True, "Option strike."),
        ColumnSpec("option_mid", "float", True, "Option price used for trade or valuation."),
        ColumnSpec("underlying_price", "float", True, "Underlying spot price."),
        ColumnSpec("time_to_maturity", "float", True, "Time to maturity in years."),
        ColumnSpec("risk_free_rate", "float", True, "Risk-free rate in decimal form."),
        ColumnSpec("implied_volatility", "float", False, "Market implied volatility."),
        ColumnSpec("realized_volatility", "float", False, "Rolling realized volatility."),
        ColumnSpec("model_volatility", "float", False, "Model volatility assumption."),
        ColumnSpec("delta", "float", False, "Delta used by hedging strategies."),
        ColumnSpec("gamma", "float", False, "Gamma used by delta-gamma hedging."),
        ColumnSpec("vega", "float", False, "Sensitivity to volatility."),
        ColumnSpec("theta", "float", False, "Sensitivity to time decay."),
        ColumnSpec("rho", "float", False, "Sensitivity to interest rates."),
        ColumnSpec("dte", "float", False, "Calendar days to expiration."),
        ColumnSpec("moneyness", "float", False, "Strike divided by underlying price."),
        ColumnSpec("relative_spread", "float", False, "Bid-ask spread divided by mid price."),
    ),
)

STRATEGY_RESULT_CONTRACT = DataContract(
    name="StrategyResult",
    description="Comparable output produced by one strategy run.",
    natural_key=("run_id", "strategy_name", "ticker", "date"),
    columns=(
        ColumnSpec("run_id", "string", True, "Unique run identifier."),
        ColumnSpec("strategy_name", "string", True, "Canonical strategy name."),
        ColumnSpec("ticker", "string", True, "Underlying symbol."),
        ColumnSpec("date", "date", True, "Evaluation date."),
        ColumnSpec("portfolio_value", "float", True, "Total portfolio value."),
        ColumnSpec("cash", "float", False, "Cash balance."),
        ColumnSpec("underlying_position", "float", False, "Underlying position."),
        ColumnSpec("option_position", "float", False, "Option position."),
        ColumnSpec("pnl", "float", True, "Profit and loss."),
        ColumnSpec("transaction_cost", "float", False, "Estimated transaction cost."),
        ColumnSpec("delta_exposure", "float", False, "Net delta exposure."),
        ColumnSpec("gamma_exposure", "float", False, "Net gamma exposure."),
    ),
)

STRATEGY_DATA_CONTRACTS = (
    HEDGING_DATASET_CONTRACT,
    STRATEGY_RESULT_CONTRACT,
)
