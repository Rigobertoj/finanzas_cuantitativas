"""Black-Scholes pricing, implied volatility and Greeks."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, isfinite, log, pi, sqrt


@dataclass(frozen=True)
class BlackScholesInputs:
    """Inputs required by Black-Scholes formulas."""

    option_type: str
    underlying_price: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float


@dataclass(frozen=True)
class BlackScholesGreeks:
    """First-order Black-Scholes sensitivities and gamma."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def black_scholes_price(inputs: BlackScholesInputs) -> float:
    """Return the Black-Scholes price for a European call or put."""

    d1, d2 = _d1_d2(inputs)
    side = _normalize_option_type(inputs.option_type)
    if side == "call":
        return inputs.underlying_price * _norm_cdf(d1) - inputs.strike * exp(
            -inputs.risk_free_rate * inputs.time_to_maturity
        ) * _norm_cdf(d2)
    return inputs.strike * exp(-inputs.risk_free_rate * inputs.time_to_maturity) * _norm_cdf(
        -d2
    ) - inputs.underlying_price * _norm_cdf(-d1)


def black_scholes_greeks(inputs: BlackScholesInputs) -> BlackScholesGreeks:
    """Return delta, gamma, vega, theta and rho."""

    d1, d2 = _d1_d2(inputs)
    side = _normalize_option_type(inputs.option_type)
    pdf = _norm_pdf(d1)
    discount = exp(-inputs.risk_free_rate * inputs.time_to_maturity)

    gamma = pdf / (
        inputs.underlying_price * inputs.volatility * sqrt(inputs.time_to_maturity)
    )
    vega = inputs.underlying_price * pdf * sqrt(inputs.time_to_maturity)

    if side == "call":
        delta = _norm_cdf(d1)
        time_decay = -(
            inputs.underlying_price * pdf * inputs.volatility
        ) / (2 * sqrt(inputs.time_to_maturity))
        theta = (
            time_decay
            - inputs.risk_free_rate * inputs.strike * discount * _norm_cdf(d2)
        )
        rho = inputs.strike * inputs.time_to_maturity * discount * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1
        time_decay = -(
            inputs.underlying_price * pdf * inputs.volatility
        ) / (2 * sqrt(inputs.time_to_maturity))
        theta = (
            time_decay
            + inputs.risk_free_rate * inputs.strike * discount * _norm_cdf(-d2)
        )
        rho = -inputs.strike * inputs.time_to_maturity * discount * _norm_cdf(-d2)

    return BlackScholesGreeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def implied_volatility(
    option_price: float,
    option_type: str,
    underlying_price: float,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    min_volatility: float = 0.0001,
    max_volatility: float = 5.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> float | None:
    """Solve Black-Scholes implied volatility with bisection.

    ``None`` is returned when the solver cannot bracket or converge to a
    plausible volatility. The pipeline stores that case as ``NaN`` so the row
    remains auditable without pretending the Greek calculation succeeded.
    """

    if min(option_price, underlying_price, strike, time_to_maturity) <= 0:
        return None

    low = min_volatility
    high = max_volatility
    low_price = black_scholes_price(
        BlackScholesInputs(
            option_type, underlying_price, strike, time_to_maturity, risk_free_rate, low
        )
    )
    high_price = black_scholes_price(
        BlackScholesInputs(
            option_type, underlying_price, strike, time_to_maturity, risk_free_rate, high
        )
    )

    if option_price < low_price - tolerance or option_price > high_price + tolerance:
        return None

    for _ in range(max_iterations):
        mid = (low + high) / 2
        price = black_scholes_price(
            BlackScholesInputs(
                option_type, underlying_price, strike, time_to_maturity, risk_free_rate, mid
            )
        )
        diff = price - option_price
        if abs(diff) <= tolerance and isfinite(mid):
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid

    return None


def _d1_d2(inputs: BlackScholesInputs) -> tuple[float, float]:
    if (
        min(
            inputs.underlying_price,
            inputs.strike,
            inputs.time_to_maturity,
            inputs.volatility,
        )
        <= 0
    ):
        raise ValueError(
            "Black-Scholes inputs require positive price, strike, maturity and volatility."
        )
    d1 = (
        log(inputs.underlying_price / inputs.strike)
        + (inputs.risk_free_rate + 0.5 * inputs.volatility**2) * inputs.time_to_maturity
    ) / (inputs.volatility * sqrt(inputs.time_to_maturity))
    d2 = d1 - inputs.volatility * sqrt(inputs.time_to_maturity)
    return d1, d2


def _normalize_option_type(option_type: str) -> str:
    side = str(option_type).strip().lower()
    if side not in {"call", "put"}:
        raise ValueError("option_type must be call or put.")
    return side


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _norm_pdf(value: float) -> float:
    return exp(-0.5 * value * value) / sqrt(2.0 * pi)
