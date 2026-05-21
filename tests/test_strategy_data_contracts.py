from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from modulos.schemas import HEDGING_DATASET_CONTRACT, STRATEGY_RESULT_CONTRACT
from modulos.validation import validate_hedging_dataset, validate_strategy_result


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class StrategyDataContractTests(unittest.TestCase):
    def test_hedging_contract_key_is_strategy_granularity(self) -> None:
        self.assertEqual(
            HEDGING_DATASET_CONTRACT.natural_key,
            ("ticker", "date", "expiration_date", "option_type", "strike"),
        )

    def test_hedging_dataset_fixture_validates(self) -> None:
        frame = pd.read_csv(FIXTURES / "hedging_dataset_sample.csv")
        result = validate_hedging_dataset(frame)
        self.assertEqual(result["ticker"].unique().tolist(), ["AMZN"])

    def test_hedging_dataset_rejects_expired_option(self) -> None:
        frame = pd.read_csv(FIXTURES / "hedging_dataset_sample.csv")
        frame.loc[0, "expiration_date"] = "2026-03-01"
        with self.assertRaisesRegex(ValueError, "expiration_date"):
            validate_hedging_dataset(frame)

    def test_strategy_result_contract_requires_pnl(self) -> None:
        self.assertIn("pnl", STRATEGY_RESULT_CONTRACT.required_columns)

    def test_strategy_result_validates_minimal_frame(self) -> None:
        frame = pd.DataFrame(
            {
                "run_id": ["run-001"],
                "strategy_name": ["delta_hedging"],
                "ticker": ["amzn"],
                "date": ["2026-04-01"],
                "portfolio_value": [1000.0],
                "pnl": [12.5],
            }
        )
        result = validate_strategy_result(frame)
        self.assertEqual(result["ticker"].iloc[0], "AMZN")


if __name__ == "__main__":
    unittest.main()
