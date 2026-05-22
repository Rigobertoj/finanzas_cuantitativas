"""Shared primitives for market data providers.

The data-source layer should stay small. It translates external provider
details into project contracts, but it does not run strategies or persist data.
"""

from __future__ import annotations

from datetime import datetime, timezone


class DataSourceError(RuntimeError):
    """Base exception for provider-layer failures.

    The data-source layer raises this exception, or a subclass of it, when a
    provider cannot complete a request or returns data that cannot be consumed
    safely. It gives notebooks, pipelines and tests one stable error family to
    catch without depending on provider-specific exceptions such as
    ``requests.HTTPError``.

    Examples
    --------
    >>> isinstance(DataSourceError("bad response"), RuntimeError)
    True
    """


class DataSourceUnavailable(DataSourceError):
    """Raised when a provider cannot be reached or times out.

    This exception is intended for operational failures: Theta Terminal is not
    running, the local API port is unavailable, or a request exceeds the
    configured timeout. It is separate from ``DataSourceError`` so callers can
    distinguish infrastructure availability from other provider problems.

    Examples
    --------
    >>> isinstance(DataSourceUnavailable("terminal down"), DataSourceError)
    True
    """


def normalize_ticker(ticker: str) -> str:
    """Return an uppercase ticker or raise ``ValueError`` for empty input."""

    clean = str(ticker).strip().upper()
    if not clean:
        raise ValueError("ticker cannot be empty.")
    return clean


def require_yyyymmdd(value: str, field_name: str) -> str:
    """Validate a date string and return it in ``YYYYMMDD`` format."""

    text = str(value).strip()
    for date_format in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, date_format).strftime("%Y%m%d")
        except ValueError:
            pass
    raise ValueError(f"{field_name} must use YYYYMMDD or YYYY-MM-DD format.")


def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)
