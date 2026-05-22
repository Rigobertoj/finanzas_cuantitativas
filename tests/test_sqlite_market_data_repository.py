from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from modulos.storage import RunManifest, SQLiteMarketDataRepository


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class SQLiteMarketDataRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.tempdir.name) / "backtesting.sqlite"
        self.repository = SQLiteMarketDataRepository(self.database_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_save_and_load_stock_eod_deduplicates_natural_key(self) -> None:
        frame = pd.read_csv(FIXTURES / "stock_eod_sample.csv")

        self.assertEqual(self.repository.save_stock_eod(frame), 2)
        self.assertEqual(self.repository.save_stock_eod(frame), 2)

        result = self.repository.load_stock_eod("AMZN", "20260401", "20260430")
        self.assertEqual(len(result), 1)
        self.assertEqual(result["ticker"].iloc[0], "AMZN")

    def test_save_and_load_option_eod_deduplicates_natural_key(self) -> None:
        frame = pd.read_csv(FIXTURES / "option_eod_sample.csv")

        self.assertEqual(self.repository.save_option_eod(frame), 2)
        self.repository.save_option_eod(frame)

        result = self.repository.load_option_eod("AMZN", "20260401", "20260430")
        self.assertEqual(len(result), 2)
        self.assertEqual(result["option_type"].tolist(), ["call", "put"])

    def test_save_option_eod_rejects_invalid_rows(self) -> None:
        frame = pd.read_csv(FIXTURES / "option_eod_sample.csv")
        frame.loc[0, "ask"] = 1.0

        with self.assertRaisesRegex(ValueError, "ask"):
            self.repository.save_option_eod(frame)

    def test_save_and_load_complete_manifest(self) -> None:
        manifest = RunManifest(
            run_id="run-1",
            pipeline_name="market_data_ingestion",
            provider="ThetaData",
            status="success",
            started_at_utc="2026-05-21T00:00:00+00:00",
            finished_at_utc="2026-05-21T00:01:00+00:00",
            tickers=("AMZN",),
            params={"start_date": "20260401", "end_date": "20260402"},
            rows_written={"stock_eod": 1, "option_eod": 2},
            results={"AMZN": {"status": "success"}},
            errors=[],
        )

        self.repository.save_run_manifest(manifest)
        result = self.repository.load_run_manifest("run-1")

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["tickers"], ["AMZN"])
        self.assertEqual(result["rows_written"]["option_eod"], 2)

    def test_manifest_must_be_complete_before_persistence(self) -> None:
        manifest = RunManifest(
            run_id="",
            pipeline_name="market_data_ingestion",
            provider="ThetaData",
            status="success",
            started_at_utc="2026-05-21T00:00:00+00:00",
            finished_at_utc="2026-05-21T00:01:00+00:00",
            tickers=("AMZN",),
            params={},
            rows_written={},
            results={},
            errors=[],
        )

        with self.assertRaisesRegex(ValueError, "missing required fields"):
            self.repository.save_run_manifest(manifest)

    def test_manifest_must_include_results_for_every_ticker(self) -> None:
        manifest = RunManifest(
            run_id="run-missing-result",
            pipeline_name="market_data_ingestion",
            provider="ThetaData",
            status="partial",
            started_at_utc="2026-05-21T00:00:00+00:00",
            finished_at_utc="2026-05-21T00:01:00+00:00",
            tickers=("AMZN", "MSFT"),
            params={},
            rows_written={},
            results={"AMZN": {"status": "success"}},
            errors=[],
        )

        with self.assertRaisesRegex(ValueError, "missing tickers"):
            self.repository.save_run_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
