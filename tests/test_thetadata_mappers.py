from __future__ import annotations

import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from modulos.data_sources import map_option_eod, map_stock_eod
from modulos.validation import validate_option_eod, validate_stock_eod


FIXTURES = Path(__file__).resolve().parent / "fixtures"
DOWNLOADED_AT = datetime(2026, 4, 3, tzinfo=timezone.utc)


class ThetaDataMapperTests(unittest.TestCase):
    def test_stock_mapper_returns_stock_eod_contract(self) -> None:
        raw = pd.read_csv(FIXTURES / "thetadata_stock_eod_response.csv")
        mapped = map_stock_eod(raw, ticker="amzn", downloaded_at_utc=DOWNLOADED_AT)
        result = validate_stock_eod(mapped)
        self.assertEqual(result["ticker"].tolist(), ["AMZN", "AMZN"])
        self.assertEqual(result["close"].iloc[0], 185.20)

    def test_option_mapper_merges_underlying_and_returns_option_eod_contract(self) -> None:
        raw_options = pd.read_csv(FIXTURES / "thetadata_option_eod_response.csv")
        raw_stocks = pd.read_csv(FIXTURES / "thetadata_stock_eod_response.csv")
        stock_eod = validate_stock_eod(
            map_stock_eod(raw_stocks, ticker="AMZN", downloaded_at_utc=DOWNLOADED_AT)
        )

        mapped = map_option_eod(
            raw_options,
            ticker="AMZN",
            underlying_prices=stock_eod,
            downloaded_at_utc=DOWNLOADED_AT,
        )
        result = validate_option_eod(mapped)

        self.assertEqual(result["option_type"].unique().tolist(), ["call"])
        self.assertEqual(result["underlying_price"].iloc[0], 185.20)
        self.assertAlmostEqual(result["mid"].iloc[0], 4.20)

    def test_option_mapper_rejects_missing_underlying_price_after_validation(self) -> None:
        raw_options = pd.read_csv(FIXTURES / "thetadata_option_eod_response.csv")
        stock_eod = pd.DataFrame(
            {
                "ticker": ["AMZN"],
                "date": ["2026-04-10"],
                "close": [200.0],
                "source": ["ThetaData"],
                "downloaded_at_utc": [DOWNLOADED_AT],
            }
        )
        mapped = map_option_eod(
            raw_options,
            ticker="AMZN",
            underlying_prices=stock_eod,
            downloaded_at_utc=DOWNLOADED_AT,
        )
        with self.assertRaisesRegex(ValueError, "underlying_price"):
            validate_option_eod(mapped)


if __name__ == "__main__":
    unittest.main()
