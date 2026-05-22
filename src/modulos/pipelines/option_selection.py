"""Deterministic option-contract selection for hedging datasets."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OptionSelectionConfig:
    """Configuration for selecting one option contract per date."""

    option_type: str = "call"
    min_dte: int = 30
    max_dte: int = 60
    target_moneyness: float = 1.0


def select_contracts(
    option_eod: pd.DataFrame,
    config: OptionSelectionConfig,
) -> pd.DataFrame:
    """Select one deterministic contract for each ticker and date.

    The ranking avoids ambiguous results by sorting with explicit tie-breakers:
    moneyness distance, volume, relative spread, DTE and strike.
    """

    if config.min_dte < 0 or config.max_dte < config.min_dte:
        raise ValueError("DTE bounds are invalid.")
    if config.target_moneyness <= 0:
        raise ValueError("target_moneyness must be positive.")

    frame = option_eod.copy()
    if frame.empty:
        return frame

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["expiration_date"] = pd.to_datetime(frame["expiration_date"], errors="coerce")
    frame["option_type"] = frame["option_type"].astype("string").str.lower()
    frame["dte"] = (frame["expiration_date"] - frame["date"]).dt.days
    frame["moneyness"] = frame["strike"] / frame["underlying_price"]
    frame["moneyness_distance"] = (frame["moneyness"] - config.target_moneyness).abs()
    frame["relative_spread"] = _relative_spread(frame)

    candidates = frame[
        (frame["option_type"] == config.option_type.lower())
        & (frame["dte"] >= config.min_dte)
        & (frame["dte"] <= config.max_dte)
    ].copy()
    if candidates.empty:
        return candidates

    candidates["_volume_rank"] = pd.to_numeric(
        candidates.get("volume"),
        errors="coerce",
    ).fillna(-1)
    candidates["_spread_rank"] = candidates["relative_spread"].fillna(float("inf"))
    selected = (
        candidates.sort_values(
            [
                "ticker",
                "date",
                "moneyness_distance",
                "_volume_rank",
                "_spread_rank",
                "dte",
                "strike",
            ],
            ascending=[True, True, True, False, True, True, True],
        )
        .groupby(["ticker", "date"], as_index=False)
        .head(1)
        .drop(columns=["_volume_rank", "_spread_rank"])
    )
    selected["selection_rank"] = 1.0
    return selected.reset_index(drop=True)


def _relative_spread(frame: pd.DataFrame) -> pd.Series:
    if not {"bid", "ask", "mid"}.issubset(frame.columns):
        return pd.Series(pd.NA, index=frame.index)
    bid = pd.to_numeric(frame["bid"], errors="coerce")
    ask = pd.to_numeric(frame["ask"], errors="coerce")
    mid = pd.to_numeric(frame["mid"], errors="coerce")
    spread = (ask - bid) / mid
    return spread.where(mid > 0)
