from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from modulos.pipelines import MarketDataIngestionPipeline
from modulos.storage import SQLiteMarketDataRepository
from modulos.validation import validate_option_eod, validate_stock_eod


DOWNLOADED_AT = datetime(2026, 5, 21, tzinfo=timezone.utc)


class FakeStockProvider:
    def get_stock_eod(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        if ticker == "FAIL":
            raise RuntimeError("stock unavailable")
        return validate_stock_eod(
            pd.DataFrame(
                {
                    "ticker": [ticker],
                    "date": ["2026-05-18"],
                    "open": [100.0],
                    "high": [102.0],
                    "low": [99.0],
                    "close": [101.0],
                    "volume": [1000],
                    "source": ["ThetaData"],
                    "downloaded_at_utc": [DOWNLOADED_AT],
                }
            )
        )


class FakeOptionProvider:
    def get_option_eod(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        expiration: str = "*",
        right: str = "both",
        strike: str = "*",
        strike_range: int | None = 8,
        max_dte: int | None = 120,
        underlying_prices: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if ticker == "FAIL":
            raise RuntimeError("options unavailable")
        return validate_option_eod(
            pd.DataFrame(
                {
                    "ticker": [ticker],
                    "date": ["2026-05-18"],
                    "expiration_date": ["2026-06-19"],
                    "option_type": ["call"],
                    "strike": [100.0],
                    "bid": [4.0],
                    "ask": [4.4],
                    "mid": [4.2],
                    "last_price": [4.1],
                    "volume": [10],
                    "open_interest": [100],
                    "underlying_price": [101.0],
                    "source": ["ThetaData"],
                    "downloaded_at_utc": [DOWNLOADED_AT],
                }
            )
        )


class FailingOptionProvider(FakeOptionProvider):
    def get_option_eod(self, *args, **kwargs) -> pd.DataFrame:
        raise RuntimeError("option chain unavailable")


class MarketDataIngestionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repository = SQLiteMarketDataRepository(
            Path(self.tempdir.name) / "backtesting.sqlite"
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_pipeline_ingests_multiple_tickers_and_writes_manifest(self) -> None:
        pipeline = MarketDataIngestionPipeline(
            stock_provider=FakeStockProvider(),
            option_provider=FakeOptionProvider(),
            repository=self.repository,
        )

        result = pipeline.run_option_eod_ingestion(
            tickers=["AMZN", "MSFT"],
            start_date="20260518",
            end_date="20260518",
            strike_range=1,
            max_dte=30,
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(result.rows_written, {"stock_eod": 2, "option_eod": 2})
        self.assertEqual(len(self.repository.load_stock_eod("AMZN")), 1)
        self.assertEqual(len(self.repository.load_option_eod("MSFT")), 1)

        manifest = self.repository.load_run_manifest(result.run_id)
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(manifest["tickers"], ["AMZN", "MSFT"])
        self.assertEqual(manifest["rows_written"]["option_eod"], 2)

    def test_pipeline_records_partial_failures_without_losing_manifest(self) -> None:
        pipeline = MarketDataIngestionPipeline(
            stock_provider=FakeStockProvider(),
            option_provider=FakeOptionProvider(),
            repository=self.repository,
        )

        result = pipeline.run_option_eod_ingestion(
            tickers=["AMZN", "FAIL"],
            start_date="20260518",
            end_date="20260518",
        )

        self.assertEqual(result.status, "partial")
        self.assertEqual(result.results["AMZN"]["status"], "success")
        self.assertEqual(result.results["FAIL"]["status"], "failed")
        self.assertEqual(result.errors[0]["ticker"], "FAIL")

        manifest = self.repository.load_run_manifest(result.run_id)
        self.assertEqual(manifest["status"], "partial")
        self.assertEqual(len(manifest["errors"]), 1)

    def test_pipeline_deduplicates_repeated_tickers_in_manifest(self) -> None:
        pipeline = MarketDataIngestionPipeline(
            stock_provider=FakeStockProvider(),
            option_provider=FakeOptionProvider(),
            repository=self.repository,
        )

        result = pipeline.run_option_eod_ingestion(
            tickers=["amzn", "AMZN"],
            start_date="20260518",
            end_date="20260518",
        )

        manifest = self.repository.load_run_manifest(result.run_id)
        self.assertEqual(manifest["tickers"], ["AMZN"])
        self.assertEqual(result.rows_written, {"stock_eod": 1, "option_eod": 1})

    def test_pipeline_manifest_counts_stock_rows_when_options_fail(self) -> None:
        pipeline = MarketDataIngestionPipeline(
            stock_provider=FakeStockProvider(),
            option_provider=FailingOptionProvider(),
            repository=self.repository,
        )

        result = pipeline.run_option_eod_ingestion(
            tickers=["AMZN"],
            start_date="20260518",
            end_date="20260518",
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.rows_written, {"stock_eod": 1, "option_eod": 0})
        self.assertEqual(result.results["AMZN"]["stock_rows"], 1)

        manifest = self.repository.load_run_manifest(result.run_id)
        self.assertEqual(manifest["rows_written"], {"option_eod": 0, "stock_eod": 1})


if __name__ == "__main__":
    unittest.main()
