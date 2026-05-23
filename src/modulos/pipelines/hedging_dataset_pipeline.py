"""Build reproducible hedging datasets from SQLite market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from modulos.models import (
    BlackScholesInputs,
    black_scholes_greeks,
    implied_volatility,
)
from modulos.storage import (
    HedgingDatasetManifest,
    SQLiteMarketDataRepository,
    new_run_id,
    utc_timestamp,
)
from modulos.validation import validate_hedging_dataset

from .option_selection import OptionSelectionConfig, select_contracts
from .rebalance_calendar import RebalanceCalendar
from .volatility_features import add_realized_volatility


@dataclass(frozen=True)
class HedgingDatasetResult:
    """Result returned by ``HedgingDatasetPipeline.build``.

    Attributes
    ----------
    dataset_id:
        Unique identifier assigned to the generated dataset. When
        ``persist=True`` this value is used as the primary dataset key in
        SQLite.
    frame:
        Validated ``HedgingDataset`` DataFrame ready to be consumed by future
        strategy modules.
    rows_written:
        Number of rows submitted to SQLite. The value is ``0`` when the build
        runs with ``persist=False``.
    """

    dataset_id: str
    frame: pd.DataFrame
    rows_written: int


@dataclass(frozen=True)
class HedgingDatasetAssumptions:
    """Configurable research assumptions used during dataset construction.

    The dataclass keeps model and rebalance assumptions out of notebooks. This
    makes a run easier to reproduce because the same values are written to the
    dataset manifest when the pipeline persists the result.

    Attributes
    ----------
    risk_free_rate:
        Fixed risk-free rate used by Black-Scholes calculations, expressed in
        decimal form.
    day_count:
        Convention used to transform DTE into year fraction. Supported values
        are ``"ACT/365"``, ``"ACT/360"`` and ``"ACT/252"``.
    realized_volatility_window:
        Rolling window, in observations, used to estimate realized volatility
        from stock log returns.
    annualization_factor:
        Factor used to annualize realized volatility. The default ``252``
        treats observations as trading days.
    rebalance_frequency:
        Calendar rule used by ``RebalanceCalendar``. Supported values are
        ``"daily"``, ``"weekly"`` and ``"custom"``.
    custom_rebalance_dates:
        Explicit dates used only when ``rebalance_frequency="custom"``.
    """

    risk_free_rate: float
    day_count: str = "ACT/365"
    realized_volatility_window: int = 20
    annualization_factor: int = 252
    rebalance_frequency: str = "daily"
    custom_rebalance_dates: tuple[str, ...] | None = None


class HedgingDatasetPipeline:
    """Transform stored market data into a validated hedging dataset.

    ``HedgingDatasetPipeline`` is the bridge between raw market data and future
    option strategies. It loads ``StockEOD`` and ``OptionEOD`` rows from
    ``SQLiteMarketDataRepository``, applies a deterministic contract selection
    policy, computes volatility and Black-Scholes features, validates the
    output contract and optionally persists both dataset and manifest.

    The pipeline does not call ThetaData directly. Market data must already be
    available in SQLite, which keeps strategy research reproducible and
    independent from provider availability.

    Attributes
    ----------
    repository:
        Storage adapter used to load market data and persist generated datasets.
    """

    def __init__(self, repository: SQLiteMarketDataRepository | None = None) -> None:
        """Create a hedging dataset pipeline.

        Parameters
        ----------
        repository:
            Optional repository instance. Passing an explicit repository is
            useful for tests, temporary SQLite databases and notebooks that
            should target a specific local database file.
        """

        self.repository = repository or SQLiteMarketDataRepository()

    def build(
        self,
        tickers: str | Sequence[str],
        start_date: str,
        end_date: str,
        option_type: str = "call",
        min_dte: int = 30,
        max_dte: int = 60,
        target_moneyness: float = 1.0,
        risk_free_rate: float = 0.045,
        day_count: str = "ACT/365",
        realized_volatility_window: int = 20,
        annualization_factor: int = 252,
        rebalance_frequency: str = "daily",
        custom_rebalance_dates: Sequence[str] | None = None,
        source: str = "ThetaData",
        source_run_id: str | None = None,
        persist: bool = True,
    ) -> HedgingDatasetResult:
        """Build, validate and optionally persist a ``HedgingDataset``.

        Parameters
        ----------
        tickers:
            One ticker or a sequence of tickers. Values are normalized to
            uppercase and duplicates are removed while preserving order.
        start_date, end_date:
            Inclusive market-data window. ``YYYYMMDD`` and ISO date strings are
            accepted by the repository loaders.
        option_type:
            Option side selected for the first research dataset. The default is
            ``"call"`` but ``"put"`` is supported by the selection policy.
        min_dte, max_dte:
            Inclusive days-to-expiration bounds used to keep selected contracts
            comparable. Defaults are 30 and 60.
        target_moneyness:
            Target ``strike / underlying_price``. The default ``1.0`` selects
            contracts closest to ATM.
        risk_free_rate:
            Fixed rate used by Black-Scholes calculations for this run.
        day_count:
            Convention used to compute ``time_to_maturity`` from DTE. Supported
            values are ``"ACT/365"``, ``"ACT/360"`` and ``"ACT/252"``.
        realized_volatility_window:
            Rolling window used by ``add_realized_volatility``.
        annualization_factor:
            Factor used to annualize realized volatility.
        rebalance_frequency:
            Calendar rule used to keep observation dates. Supported values are
            ``"daily"``, ``"weekly"`` and ``"custom"``.
        custom_rebalance_dates:
            Explicit dates used when ``rebalance_frequency="custom"``.
        source:
            Market-data source label used by repository loaders.
        source_run_id:
            Optional ingestion run identifier linking the dataset to a prior
            market-data manifest.
        persist:
            When ``True``, write the generated dataset and manifest to SQLite.
            When ``False``, return the DataFrame without writing storage rows.

        Returns
        -------
        HedgingDatasetResult
            Dataset identifier, validated DataFrame and number of rows written.

        Raises
        ------
        ValueError
            If tickers are empty, DTE or moneyness configuration is invalid,
            the day-count convention is unsupported, no eligible contracts are
            found, or the output fails ``validate_hedging_dataset``.

        Notes
        -----
        Implied volatility is solved with Black-Scholes bisection. If the solver
        does not converge for one row, the row is kept and implied volatility
        plus Greeks are stored as null values.

        Examples
        --------
        >>> from modulos.storage import SQLiteMarketDataRepository
        >>> repo = SQLiteMarketDataRepository(":memory:")  # doctest: +SKIP
        >>> pipeline = HedgingDatasetPipeline(repo)  # doctest: +SKIP
        >>> result = pipeline.build(["AAPL"], "20260518", "20260520")  # doctest: +SKIP
        >>> result.frame.columns  # doctest: +SKIP
        Index([...], dtype='object')
        """

        normalized_tickers = _normalize_tickers(tickers)
        dataset_id = new_run_id("hedging-dataset")
        assumptions = HedgingDatasetAssumptions(
            risk_free_rate=risk_free_rate,
            day_count=day_count,
            realized_volatility_window=realized_volatility_window,
            annualization_factor=annualization_factor,
            rebalance_frequency=rebalance_frequency,
            custom_rebalance_dates=tuple(custom_rebalance_dates or ()) or None,
        )
        selection_config = OptionSelectionConfig(
            option_type=option_type,
            min_dte=min_dte,
            max_dte=max_dte,
            target_moneyness=target_moneyness,
        )

        frames: list[pd.DataFrame] = []
        for ticker in normalized_tickers:
            stock_eod = self.repository.load_stock_eod(
                ticker, start_date, end_date, source
            )
            option_eod = self.repository.load_option_eod(
                ticker, start_date, end_date, source
            )
            frames.append(
                _build_for_ticker(
                    stock_eod=stock_eod,
                    option_eod=option_eod,
                    selection_config=selection_config,
                    assumptions=assumptions,
                )
            )

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if combined.empty:
            raise ValueError(
                "No eligible option contracts were found for the hedging dataset."
            )

        validated = validate_hedging_dataset(combined)
        rows_written = 0
        if persist:
            rows_written = self.repository.save_hedging_dataset(
                dataset_id,
                validated,
                source_run_id=source_run_id,
            )
            manifest = HedgingDatasetManifest(
                dataset_id=dataset_id,
                source_run_id=source_run_id,
                pipeline_name="hedging_dataset_pipeline",
                status="success",
                created_at_utc=utc_timestamp(),
                tickers=tuple(normalized_tickers),
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "option_type": option_type,
                    "min_dte": min_dte,
                    "max_dte": max_dte,
                    "target_moneyness": target_moneyness,
                    "risk_free_rate": risk_free_rate,
                    "day_count": day_count,
                    "realized_volatility_window": realized_volatility_window,
                    "annualization_factor": annualization_factor,
                    "rebalance_frequency": rebalance_frequency,
                    "source": source,
                },
                rows_written=rows_written,
                errors=[],
            )
            self.repository.save_hedging_dataset_manifest(manifest)

        return HedgingDatasetResult(
            dataset_id=dataset_id,
            frame=validated,
            rows_written=rows_written,
        )


def _build_for_ticker(
    stock_eod: pd.DataFrame,
    option_eod: pd.DataFrame,
    selection_config: OptionSelectionConfig,
    assumptions: HedgingDatasetAssumptions,
) -> pd.DataFrame:
    """Build the hedging dataset slice for one ticker.

    This helper keeps per-ticker processing isolated so the public ``build``
    method can focus on orchestration and persistence. Empty inputs return an
    empty frame, allowing the caller to detect whether any ticker produced an
    eligible contract.
    """

    if stock_eod.empty or option_eod.empty:
        return pd.DataFrame()

    calendar = RebalanceCalendar(
        frequency=assumptions.rebalance_frequency,
        custom_dates=assumptions.custom_rebalance_dates,
    )
    option_frame = calendar.filter(option_eod)
    selected = select_contracts(option_frame, selection_config)
    if selected.empty:
        return selected

    realized = add_realized_volatility(
        stock_eod,
        window=assumptions.realized_volatility_window,
        annualization_factor=assumptions.annualization_factor,
    )
    selected = selected.merge(
        realized,
        on=["ticker", "date"],
        how="left",
    )
    selected["time_to_maturity"] = selected["dte"] / _day_count_denominator(
        assumptions.day_count
    )
    selected["risk_free_rate"] = assumptions.risk_free_rate
    selected["option_mid"] = selected["mid"]
    selected = _add_black_scholes_features(selected)

    columns = [
        "ticker",
        "date",
        "expiration_date",
        "option_type",
        "strike",
        "option_mid",
        "underlying_price",
        "time_to_maturity",
        "risk_free_rate",
        "implied_volatility",
        "realized_volatility",
        "model_volatility",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
        "dte",
        "moneyness",
        "relative_spread",
    ]
    return selected[columns]


def _add_black_scholes_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add implied volatility, model volatility and Greeks to selected rows.

    Rows whose implied volatility cannot be solved are preserved with null
    feature values. This behavior supports auditability: a strategy can decide
    later whether to drop, impute or report those rows.
    """

    result = frame.copy()
    implied_values: list[float | None] = []
    greeks_rows: list[dict[str, float | None]] = []

    for row in result.to_dict("records"):
        iv = implied_volatility(
            option_price=float(row["option_mid"]),
            option_type=str(row["option_type"]),
            underlying_price=float(row["underlying_price"]),
            strike=float(row["strike"]),
            time_to_maturity=float(row["time_to_maturity"]),
            risk_free_rate=float(row["risk_free_rate"]),
        )
        implied_values.append(iv)
        if iv is None:
            greeks_rows.append(
                {
                    "delta": None,
                    "gamma": None,
                    "vega": None,
                    "theta": None,
                    "rho": None,
                }
            )
            continue
        greeks = black_scholes_greeks(
            BlackScholesInputs(
                option_type=str(row["option_type"]),
                underlying_price=float(row["underlying_price"]),
                strike=float(row["strike"]),
                time_to_maturity=float(row["time_to_maturity"]),
                risk_free_rate=float(row["risk_free_rate"]),
                volatility=iv,
            )
        )
        greeks_rows.append(
            {
                "delta": greeks.delta,
                "gamma": greeks.gamma,
                "vega": greeks.vega,
                "theta": greeks.theta,
                "rho": greeks.rho,
            }
        )

    result["implied_volatility"] = implied_values
    result["model_volatility"] = implied_values
    for column in ("delta", "gamma", "vega", "theta", "rho"):
        result[column] = [values[column] for values in greeks_rows]
    return result


def _day_count_denominator(day_count: str) -> float:
    """Return the denominator used to convert DTE into years.

    Parameters
    ----------
    day_count:
        Day-count convention. Supported values are ``"ACT/365"``,
        ``"ACT/360"`` and ``"ACT/252"``.

    Returns
    -------
    float
        Denominator used in ``time_to_maturity = dte / denominator``.

    Raises
    ------
    ValueError
        If the convention is not supported.
    """

    conventions = {
        "ACT/365": 365.0,
        "ACT/360": 360.0,
        "ACT/252": 252.0,
    }
    key = day_count.strip().upper()
    if key not in conventions:
        raise ValueError("day_count must be ACT/365, ACT/360, or ACT/252.")
    return conventions[key]


def _normalize_tickers(tickers: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize ticker input into an ordered tuple of unique symbols."""

    if isinstance(tickers, str):
        values = (tickers,)
    else:
        values = tuple(tickers)
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in values:
        value = str(ticker).strip().upper()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    if not normalized:
        raise ValueError("tickers cannot be empty.")
    return tuple(normalized)
