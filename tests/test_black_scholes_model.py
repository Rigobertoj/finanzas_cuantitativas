from __future__ import annotations

import unittest

from modulos.models import (
    BlackScholesInputs,
    black_scholes_greeks,
    black_scholes_price,
    implied_volatility,
)


class BlackScholesModelTests(unittest.TestCase):
    def test_call_price_and_greeks_are_reasonable(self) -> None:
        inputs = BlackScholesInputs(
            option_type="call",
            underlying_price=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
        )

        price = black_scholes_price(inputs)
        greeks = black_scholes_greeks(inputs)

        self.assertAlmostEqual(price, 10.4506, places=3)
        self.assertGreater(greeks.delta, 0.0)
        self.assertLess(greeks.delta, 1.0)
        self.assertGreater(greeks.gamma, 0.0)
        self.assertGreater(greeks.vega, 0.0)

    def test_implied_volatility_recovers_input_volatility(self) -> None:
        inputs = BlackScholesInputs(
            option_type="call",
            underlying_price=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
        )

        price = black_scholes_price(inputs)
        result = implied_volatility(
            option_price=price,
            option_type="call",
            underlying_price=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.2, places=4)

    def test_implied_volatility_returns_none_when_price_cannot_be_bracketed(self) -> None:
        result = implied_volatility(
            option_price=500.0,
            option_type="call",
            underlying_price=100.0,
            strike=100.0,
            time_to_maturity=1.0,
            risk_free_rate=0.05,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
