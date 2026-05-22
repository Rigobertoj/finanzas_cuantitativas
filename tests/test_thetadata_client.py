from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

from modulos.data_sources import DataSourceUnavailable, ThetaDataClient, ThetaDataStocks


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class ThetaDataClientTests(unittest.TestCase):
    def test_get_csv_adds_format_parameter(self) -> None:
        response = Mock()
        response.text = "symbol,close\nAMZN,185.2\n"
        response.raise_for_status.return_value = None

        with patch("modulos.data_sources.thetadata_client.requests.get", return_value=response) as get:
            client = ThetaDataClient(base_url="http://localhost:25503/v3")
            frame = client.get_csv("/stock/history/eod", {"symbol": "AMZN"})

        self.assertEqual(frame["symbol"].iloc[0], "AMZN")
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["format"], "csv")

    def test_health_check_uses_expirations_endpoint(self) -> None:
        response = Mock()
        response.json.return_value = [{"expiration": "2026-06-19"}]
        response.raise_for_status.return_value = None

        with patch("modulos.data_sources.thetadata_client.requests.get", return_value=response):
            client = ThetaDataClient()
            self.assertTrue(client.health_check("aapl"))

    def test_connection_error_becomes_unavailable(self) -> None:
        with patch(
            "modulos.data_sources.thetadata_client.requests.get",
            side_effect=requests.ConnectionError("down"),
        ):
            client = ThetaDataClient()
            with self.assertRaises(DataSourceUnavailable):
                client.get_json("/option/list/expirations", {"symbol": "AAPL"})

    def test_stock_provider_accepts_iso_dates_and_sends_compact_dates(self) -> None:
        client = Mock()
        client.get_csv.return_value = pd.read_csv(FIXTURES / "thetadata_stock_eod_response.csv")
        provider = ThetaDataStocks(client)

        provider.get_stock_eod("amzn", start_date="2026-04-01", end_date="2026-04-02")

        params = client.get_csv.call_args.args[1]
        self.assertEqual(params["start_date"], "20260401")
        self.assertEqual(params["end_date"], "20260402")


if __name__ == "__main__":
    unittest.main()
