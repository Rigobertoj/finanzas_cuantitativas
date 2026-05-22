from __future__ import annotations

import unittest

import pandas as pd

from modulos.pipelines import OptionSelectionConfig, RebalanceCalendar, select_contracts


class OptionSelectionPolicyTests(unittest.TestCase):
    def test_select_contracts_uses_moneyness_and_volume_tie_breaker(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["AMZN", "AMZN", "AMZN"],
                "date": ["2026-01-05", "2026-01-05", "2026-01-05"],
                "expiration_date": ["2026-02-20", "2026-02-20", "2026-02-20"],
                "option_type": ["call", "call", "put"],
                "strike": [99.0, 101.0, 100.0],
                "bid": [4.0, 4.0, 3.0],
                "ask": [4.4, 4.4, 3.4],
                "mid": [4.2, 4.2, 3.2],
                "volume": [10, 100, 1000],
                "underlying_price": [100.0, 100.0, 100.0],
            }
        )

        selected = select_contracts(
            frame,
            OptionSelectionConfig(
                option_type="call",
                min_dte=30,
                max_dte=60,
                target_moneyness=1.0,
            ),
        )

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected["strike"].iloc[0], 101.0)

    def test_rebalance_calendar_weekly_keeps_last_available_date(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2026-01-05", "2026-01-06", "2026-01-12"],
                "value": [1, 2, 3],
            }
        )

        result = RebalanceCalendar("weekly").filter(frame)

        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d").tolist(), ["2026-01-06", "2026-01-12"])


if __name__ == "__main__":
    unittest.main()
