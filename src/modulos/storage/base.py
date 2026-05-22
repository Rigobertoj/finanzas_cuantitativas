"""Storage primitives for local market-data repositories.

The storage layer owns persistence concerns only. It should receive validated
DataFrames, write them to a local backend, and expose simple read methods for
notebooks and pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def new_run_id(prefix: str = "market-data") -> str:
    """Return a unique run identifier for pipeline manifests."""

    return f"{prefix}-{uuid4().hex}"


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StorageKey:
    """Identify a market-data slice inside local storage.

    Attributes
    ----------
    contract_name:
        Logical dataset name, for example ``"StockEOD"`` or ``"OptionEOD"``.
    source:
        Data source name, for example ``"ThetaData"``.
    ticker:
        Underlying ticker.
    start_date:
        Inclusive start date in ``YYYYMMDD`` format.
    end_date:
        Inclusive end date in ``YYYYMMDD`` format.
    """

    contract_name: str
    source: str
    ticker: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class RunManifest:
    """Complete audit record for one pipeline execution.

    The manifest is intentionally strict: every field is required before it is
    persisted. This avoids incomplete run records and keeps later backtesting
    results traceable to their source data.

    Attributes
    ----------
    run_id:
        Unique identifier for the pipeline execution.
    pipeline_name:
        Name of the pipeline that created the manifest.
    provider:
        Provider used by the run, for example ``"ThetaData"``.
    status:
        Final run status: ``"success"``, ``"partial"`` or ``"failed"``.
    started_at_utc, finished_at_utc:
        ISO-8601 UTC timestamps for the run window.
    tickers:
        Tickers requested by the run.
    params:
        Download and pipeline parameters.
    rows_written:
        Number of rows written by dataset and ticker.
    results:
        Per-ticker status and row counts.
    errors:
        Per-ticker or run-level errors.
    """

    run_id: str
    pipeline_name: str
    provider: str
    status: str
    started_at_utc: str
    finished_at_utc: str
    tickers: tuple[str, ...]
    params: dict[str, Any]
    rows_written: dict[str, Any]
    results: dict[str, Any]
    errors: list[dict[str, str]]

    def validate(self) -> None:
        """Validate manifest completeness before persistence.

        Raises
        ------
        ValueError
            If any required field is missing or ``status`` is not valid.
        """

        required_text = {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "provider": self.provider,
            "status": self.status,
            "started_at_utc": self.started_at_utc,
            "finished_at_utc": self.finished_at_utc,
        }
        missing = [name for name, value in required_text.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"RunManifest is missing required fields: {missing}.")
        if self.status not in {"success", "partial", "failed"}:
            raise ValueError("RunManifest.status must be success, partial, or failed.")
        if not self.tickers:
            raise ValueError("RunManifest.tickers cannot be empty.")
        if self.params is None or self.rows_written is None or self.results is None:
            raise ValueError("RunManifest dictionaries cannot be None.")
        if self.errors is None:
            raise ValueError("RunManifest.errors cannot be None.")
        missing_results = sorted(set(self.tickers) - set(self.results))
        if missing_results:
            raise ValueError(
                f"RunManifest.results is missing tickers: {missing_results}."
            )
