"""HTTP client for the local ThetaData Terminal API.

The client is intentionally provider-only: it knows how to call the local REST
API and parse CSV/JSON responses, but it does not know about option strategy
contracts.

Example
-------
>>> client = ThetaDataClient()
>>> isinstance(client.base_url, str)
True
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from .base import DataSourceError, DataSourceUnavailable, normalize_ticker


@dataclass(frozen=True)
class ThetaDataClient:
    """HTTP client for the local Theta Terminal v3 API.

    The client is intentionally narrow: it builds GET requests, adds the
    requested output format, parses CSV/JSON responses and converts connection
    failures into project-specific exceptions. It does not know about internal
    data contracts, storage, strategies or notebooks.

    Attributes
    ----------
    base_url:
        Base URL of the local Theta Terminal API. The default points to the
        standard v3 local endpoint.
    timeout:
        Maximum request time in seconds before raising
        ``DataSourceUnavailable``.

    Examples
    --------
    >>> client = ThetaDataClient(base_url="http://127.0.0.1:25503/v3")
    >>> client.timeout
    120
    """

    base_url: str = "http://127.0.0.1:25503/v3"
    timeout: int = 120

    def get_csv(self, endpoint: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Request a CSV endpoint and return the response as a DataFrame.

        Parameters
        ----------
        endpoint:
            ThetaData endpoint path, with or without a leading slash.
        params:
            Query parameters accepted by the endpoint. The client adds
            ``format=csv`` automatically.

        Returns
        -------
        pandas.DataFrame
            Parsed CSV response. Empty responses become empty DataFrames.

        Raises
        ------
        DataSourceUnavailable
            If Theta Terminal cannot be reached or times out.
        DataSourceError
            If ThetaData returns an HTTP or request-level error.
        """

        response = self._get(endpoint, params=params, output_format="csv")
        text = response.text.strip()
        if not text:
            return pd.DataFrame()
        return pd.read_csv(io.StringIO(text))

    def get_json(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """Request a JSON endpoint and return decoded JSON.

        Parameters
        ----------
        endpoint:
            ThetaData endpoint path, with or without a leading slash.
        params:
            Query parameters accepted by the endpoint. The client adds
            ``format=json`` automatically.

        Returns
        -------
        Any
            Decoded JSON payload returned by ThetaData.

        Raises
        ------
        DataSourceUnavailable
            If Theta Terminal cannot be reached or times out.
        DataSourceError
            If the request fails or the response is not valid JSON.
        """

        response = self._get(endpoint, params=params, output_format="json")
        try:
            return response.json()
        except ValueError as exc:
            raise DataSourceError("ThetaData returned invalid JSON.") from exc

    def health_check(self, symbol: str = "AAPL") -> bool:
        """Return whether Theta Terminal responds to a lightweight query.

        Parameters
        ----------
        symbol:
            Liquid ticker used to query option expirations.

        Returns
        -------
        bool
            ``True`` when the local API returns a response.

        Raises
        ------
        DataSourceUnavailable
            If Theta Terminal is not running or cannot be reached.
        DataSourceError
            If the API responds with an unexpected provider error.
        """

        symbol = normalize_ticker(symbol)
        data = self.get_json("/option/list/expirations", {"symbol": symbol})
        return data is not None

    def _get(
        self,
        endpoint: str,
        params: dict[str, Any] | None,
        output_format: str,
    ) -> requests.Response:
        clean_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        request_params = dict(params or {})
        request_params["format"] = output_format
        url = f"{self.base_url.rstrip('/')}{clean_endpoint}"

        try:
            response = requests.get(url, params=request_params, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.ConnectionError as exc:
            raise DataSourceUnavailable(
                f"ThetaData Terminal is not available at {self.base_url}."
            ) from exc
        except requests.Timeout as exc:
            raise DataSourceUnavailable(
                f"ThetaData request timed out after {self.timeout} seconds."
            ) from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise DataSourceError(f"ThetaData HTTP error: {status}.") from exc
        except requests.RequestException as exc:
            raise DataSourceError(f"ThetaData request failed: {exc}.") from exc
