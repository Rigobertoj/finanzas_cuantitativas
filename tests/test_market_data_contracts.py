from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from modulos.schemas import OPTION_EOD_CONTRACT, STOCK_EOD_CONTRACT
from modulos.validation import validate_option_eod, validate_stock_eod


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class MarketDataContractTests(unittest.TestCase):
    def test_stock_contract_exposes_required_columns(self) -> None:
        self.assertIn("ticker", STOCK_EOD_CONTRACT.required_columns)
        self.assertIn("close", STOCK_EOD_CONTRACT.required_columns)
        self.assertEqual(STOCK_EOD_CONTRACT.natural_key, ("ticker", "date", "source"))

    def test_stock_eod_fixture_validates_and_normalizes_ticker(self) -> None:
        frame = pd.read_csv(FIXTURES / "stock_eod_sample.csv")
        result = validate_stock_eod(frame)
        self.assertEqual(result["ticker"].tolist(), ["AMZN", "CVX"])

    def test_option_contract_exposes_mid_price(self) -> None:
        self.assertIn("mid", OPTION_EOD_CONTRACT.required_columns)

    def test_option_eod_fixture_validates(self) -> None:
        frame = pd.read_csv(FIXTURES / "option_eod_sample.csv")
        result = validate_option_eod(frame)
        self.assertEqual(result["option_type"].tolist(), ["call", "put"])

    def test_option_eod_allows_zero_dte_contracts(self) -> None:
        frame = pd.read_csv(FIXTURES / "option_eod_sample.csv").iloc[[0]].copy()
        frame["date"] = "2026-04-01"
        frame["expiration_date"] = "2026-04-01"
        result = validate_option_eod(frame)
        self.assertEqual(result["expiration_date"].iloc[0], result["date"].iloc[0])

    def test_option_eod_rejects_bad_spread(self) -> None:
        frame = pd.read_csv(FIXTURES / "option_eod_sample.csv")
        frame.loc[0, "ask"] = 3.0
        with self.assertRaisesRegex(ValueError, "ask"):
            validate_option_eod(frame)

    def test_option_eod_rejects_expired_contract_rows(self) -> None:
        frame = pd.read_csv(FIXTURES / "option_eod_sample.csv").iloc[[0]].copy()
        frame["date"] = "2026-04-02"
        frame["expiration_date"] = "2026-04-01"
        with self.assertRaisesRegex(ValueError, "expiration_date"):
            validate_option_eod(frame)

    def test_stock_eod_rejects_missing_required_column(self) -> None:
        frame = pd.read_csv(FIXTURES / "stock_eod_sample.csv").drop(columns=["close"])
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            validate_stock_eod(frame)

    def test_stock_eod_rejects_duplicates_after_ticker_normalization(self) -> None:
        frame = pd.read_csv(FIXTURES / "stock_eod_sample.csv").iloc[[0, 0]].copy()
        frame.iloc[0, frame.columns.get_loc("ticker")] = "amzn"
        frame.iloc[1, frame.columns.get_loc("ticker")] = "AMZN"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_stock_eod(frame)


if __name__ == "__main__":
    unittest.main()
