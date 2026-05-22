from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from modulos.pipelines import HedgingDatasetPipeline
from modulos.storage import SQLiteMarketDataRepository


DOWNLOADED_AT = datetime(2026, 1, 10, tzinfo=timezone.utc)


class HedgingDatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repository = SQLiteMarketDataRepository(
            Path(self.tempdir.name) / "backtesting.sqlite"
        )
        self.repository.save_stock_eod(_stock_frame())
        self.repository.save_option_eod(_option_frame())

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_build_creates_valid_hedging_dataset_and_manifest(self) -> None:
        pipeline = HedgingDatasetPipeline(self.repository)

        result = pipeline.build(
            tickers=["amzn"],
            start_date="20260105",
            end_date="20260107",
            option_type="call",
            min_dte=30,
            max_dte=60,
            target_moneyness=1.0,
            risk_free_rate=0.04,
            realized_volatility_window=2,
        )

        self.assertEqual(result.rows_written, 3)
        self.assertEqual(result.frame["ticker"].unique().tolist(), ["AMZN"])
        self.assertIn("implied_volatility", result.frame.columns)
        self.assertIn("realized_volatility", result.frame.columns)
        self.assertIn("delta", result.frame.columns)
        self.assertTrue(result.frame["time_to_maturity"].gt(0).all())

        stored = self.repository.load_hedging_dataset(result.dataset_id)
        manifest = self.repository.load_hedging_dataset_manifest(result.dataset_id)

        self.assertEqual(len(stored), 3)
        self.assertEqual(manifest["rows_written"], 3)
        self.assertEqual(manifest["params"]["option_type"], "call")
        self.assertEqual(manifest["params"]["min_dte"], 30)
        self.assertEqual(manifest["params"]["max_dte"], 60)

    def test_build_raises_when_no_contract_is_eligible(self) -> None:
        pipeline = HedgingDatasetPipeline(self.repository)

        with self.assertRaisesRegex(ValueError, "No eligible"):
            pipeline.build(
                tickers=["AMZN"],
                start_date="20260105",
                end_date="20260107",
                min_dte=1,
                max_dte=5,
                persist=False,
            )


def _stock_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AMZN", "AMZN", "AMZN"],
            "date": ["2026-01-05", "2026-01-06", "2026-01-07"],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
            "source": ["ThetaData", "ThetaData", "ThetaData"],
            "downloaded_at_utc": [DOWNLOADED_AT, DOWNLOADED_AT, DOWNLOADED_AT],
        }
    )


def _option_frame() -> pd.DataFrame:
    rows = []
    for date, underlying in (
        ("2026-01-05", 100.0),
        ("2026-01-06", 101.0),
        ("2026-01-07", 102.0),
    ):
        rows.extend(
            [
                {
                    "ticker": "AMZN",
                    "date": date,
                    "expiration_date": "2026-02-20",
                    "option_type": "call",
                    "strike": 100.0,
                    "bid": 4.8,
                    "ask": 5.2,
                    "mid": 5.0,
                    "last_price": 5.0,
                    "volume": 100,
                    "open_interest": 500,
                    "underlying_price": underlying,
                    "source": "ThetaData",
                    "downloaded_at_utc": DOWNLOADED_AT,
                },
                {
                    "ticker": "AMZN",
                    "date": date,
                    "expiration_date": "2026-02-20",
                    "option_type": "call",
                    "strike": 110.0,
                    "bid": 1.8,
                    "ask": 2.2,
                    "mid": 2.0,
                    "last_price": 2.0,
                    "volume": 1000,
                    "open_interest": 800,
                    "underlying_price": underlying,
                    "source": "ThetaData",
                    "downloaded_at_utc": DOWNLOADED_AT,
                },
            ]
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
