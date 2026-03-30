from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError:
    arch_model = None

if __package__:
    from .AssetElementaryMetrics import AssetElementaryMetrics
    from .Model2HypothesisTester import Model2HypothesisTester
else:
    from AssetElementaryMetrics import AssetElementaryMetrics
    from Model2HypothesisTester import Model2HypothesisTester


def _prepare_numeric_series(values: pd.Series, name: str) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    clean_series = pd.to_numeric(series, errors="coerce").astype(float)
    clean_series = clean_series[np.isfinite(clean_series)].dropna()
    if clean_series.size < 8:
        raise ValueError(f"`{name}` must contain at least 8 finite numeric values.")
    return clean_series


@dataclass(frozen=True)
class VolatilityFitResult:
    model_name: str
    params: Dict[str, float]
    conditional_volatility: pd.Series
    conditional_variance: pd.Series
    residuals: pd.Series
    standardized_residuals: pd.Series
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "params": dict(self.params),
            "conditional_volatility": self.conditional_volatility.copy(),
            "conditional_variance": self.conditional_variance.copy(),
            "residuals": self.residuals.copy(),
            "standardized_residuals": self.standardized_residuals.copy(),
            "log_likelihood": self.log_likelihood,
            "aic": self.aic,
            "bic": self.bic,
            "metadata": dict(self.metadata),
        }


class EWMAVolatilityModel:
    """EWMA volatility model:
    sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * eps_{t-1}^2
    """

    def __init__(self, decay: float = 0.94, use_sample_mean: bool = True) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("`decay` must satisfy 0 < decay < 1.")
        self.decay = float(decay)
        self.use_sample_mean = bool(use_sample_mean)
        self._last_fit: Optional[VolatilityFitResult] = None

    @property
    def name(self) -> str:
        return "ewma"

    def fit(self, returns: pd.Series) -> VolatilityFitResult:
        clean_returns = _prepare_numeric_series(returns, name="returns")
        mean_return = float(clean_returns.mean()) if self.use_sample_mean else 0.0
        residuals = clean_returns - mean_return

        variance_values = np.empty(clean_returns.size, dtype=float)
        variance_values[0] = float(np.var(residuals.to_numpy(dtype=float), ddof=0))
        variance_values[0] = max(variance_values[0], 1e-12)

        residual_values = residuals.to_numpy(dtype=float)
        for idx in range(1, clean_returns.size):
            prev_var = variance_values[idx - 1]
            prev_resid = residual_values[idx - 1]
            variance_values[idx] = self.decay * prev_var + (1.0 - self.decay) * (prev_resid ** 2.0)
            variance_values[idx] = max(variance_values[idx], 1e-12)

        variance = pd.Series(
            variance_values,
            index=clean_returns.index,
            name="conditional_variance",
        )
        volatility = np.sqrt(variance).rename("conditional_volatility")
        standardized = (residuals / volatility).replace([np.inf, -np.inf], np.nan).dropna()

        aligned_index = standardized.index
        fit_result = VolatilityFitResult(
            model_name=self.name,
            params={"lambda": self.decay, "mu": mean_return},
            conditional_volatility=volatility.loc[aligned_index].copy(),
            conditional_variance=variance.loc[aligned_index].copy(),
            residuals=residuals.loc[aligned_index].copy(),
            standardized_residuals=standardized.copy(),
            metadata={"use_sample_mean": self.use_sample_mean},
        )
        self._last_fit = fit_result
        return fit_result

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._last_fit is None:
            raise RuntimeError("Fit the EWMA model before calling `forecast_variance`.")
        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")
        last_var = float(self._last_fit.conditional_variance.iloc[-1])
        return np.full(horizon, last_var, dtype=float)


class _ArchPackageVolatilityModel:
    def __init__(
        self,
        *,
        vol: str,
        p: int,
        q: int,
        o: int = 0,
        dist: str = "normal",
        mean: str = "Zero",
        rescale: bool = True,
    ) -> None:
        if p < 1:
            raise ValueError("`p` must be >= 1.")
        if q < 0:
            raise ValueError("`q` must be >= 0.")
        if o < 0:
            raise ValueError("`o` must be >= 0.")

        self.vol = vol
        self.p = int(p)
        self.q = int(q)
        self.o = int(o)
        self.dist = dist
        self.mean = mean
        self.rescale = bool(rescale)
        self._last_fit = None
        self._last_result: Optional[VolatilityFitResult] = None

    @property
    def name(self) -> str:
        if self.vol == "ARCH":
            return f"arch({self.p})"
        return f"garch({self.p},{self.q})"

    @staticmethod
    def _require_arch_package() -> None:
        if arch_model is None:
            raise ImportError(
                "Package `arch` is required for ARCH/GARCH models. "
                "Install it with `pip install arch`."
            )

    def fit(self, returns: pd.Series) -> VolatilityFitResult:
        self._require_arch_package()
        clean_returns = _prepare_numeric_series(returns, name="returns")

        model = arch_model(
            clean_returns,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
            rescale=self.rescale,
        )
        fitted = model.fit(disp="off", update_freq=0)

        raw = pd.DataFrame(
            {
                "residuals": np.asarray(fitted.resid, dtype=float),
                "standardized_residuals": np.asarray(fitted.std_resid, dtype=float),
                "conditional_volatility": np.asarray(fitted.conditional_volatility, dtype=float),
            },
            index=clean_returns.index,
        )
        raw["conditional_variance"] = raw["conditional_volatility"] ** 2.0
        clean = raw.replace([np.inf, -np.inf], np.nan).dropna()

        params = {str(key): float(value) for key, value in fitted.params.items()}
        fit_result = VolatilityFitResult(
            model_name=self.name,
            params=params,
            conditional_volatility=clean["conditional_volatility"].copy(),
            conditional_variance=clean["conditional_variance"].copy(),
            residuals=clean["residuals"].copy(),
            standardized_residuals=clean["standardized_residuals"].copy(),
            log_likelihood=float(fitted.loglikelihood),
            aic=float(fitted.aic),
            bic=float(fitted.bic),
            metadata={
                "distribution": self.dist,
                "mean_model": self.mean,
                "convergence_flag": int(getattr(fitted, "convergence_flag", 0)),
            },
        )

        self._last_fit = fitted
        self._last_result = fit_result
        return fit_result

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self._last_fit is None:
            raise RuntimeError(f"Fit `{self.name}` before calling `forecast_variance`.")
        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")

        forecast = self._last_fit.forecast(horizon=horizon, reindex=False)
        values = forecast.variance.iloc[-1].to_numpy(dtype=float)
        return np.asarray(values, dtype=float)


class ARCHVolatilityModel(_ArchPackageVolatilityModel):
    def __init__(
        self,
        p: int = 1,
        dist: str = "normal",
        mean: str = "Zero",
        rescale: bool = True,
    ) -> None:
        super().__init__(vol="ARCH", p=p, q=0, o=0, dist=dist, mean=mean, rescale=rescale)


class GARCHVolatilityModel(_ArchPackageVolatilityModel):
    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = "normal",
        mean: str = "Zero",
        rescale: bool = True,
    ) -> None:
        super().__init__(vol="GARCH", p=p, q=q, o=0, dist=dist, mean=mean, rescale=rescale)


class AssetVolatilityAnalysis:
    """Workflow coordinator for volatility fitting + hypothesis diagnostics."""

    def __init__(
        self,
        *,
        tickers: Optional[Sequence[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        assets: Optional[AssetElementaryMetrics] = None,
        prices: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        alpha: float = 0.05,
        default_lags: int = 10,
        ewma_decay: float = 0.94,
    ) -> None:
        self.assets = self._resolve_assets(
            assets=assets,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            prices_provided=prices is not None,
            returns_provided=returns is not None,
        )
        self.prices, self.returns = self._resolve_market_data(
            assets=self.assets,
            prices=prices,
            returns=returns,
        )
        self.primary_returns = _prepare_numeric_series(self.returns.iloc[:, 0], name="returns")

        self.ModelHypothesis2 = Model2HypothesisTester(
            series=self.primary_returns,
            alpha=alpha,
            default_lags=default_lags,
        )

        self.models: Dict[str, Any] = {
            "ewma": EWMAVolatilityModel(decay=ewma_decay),
            "arch": ARCHVolatilityModel(p=1),
            "garch": GARCHVolatilityModel(p=1, q=1),
        }

        self.last_model_name: Optional[str] = None
        self.last_fit_result: Optional[VolatilityFitResult] = None
        self.last_diagnostics: Optional[Dict[str, Any]] = None

    @staticmethod
    def _resolve_assets(
        *,
        assets: Optional[AssetElementaryMetrics],
        tickers: Optional[Sequence[str]],
        start_date: Optional[str],
        end_date: Optional[str],
        prices_provided: bool,
        returns_provided: bool,
    ) -> Optional[AssetElementaryMetrics]:
        if assets is not None:
            return assets
        if tickers:
            if start_date is None:
                raise ValueError("`start_date` is required when `tickers` are provided.")
            return AssetElementaryMetrics(tickers=tickers, start=start_date, end=end_date)
        if prices_provided and returns_provided:
            return None
        raise ValueError(
            "Provide `assets`, or (`tickers` + `start_date`), or both `prices` and `returns`."
        )

    @staticmethod
    def _resolve_market_data(
        *,
        assets: Optional[AssetElementaryMetrics],
        prices: Optional[pd.DataFrame],
        returns: Optional[pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if prices is None:
            if assets is None:
                raise ValueError("`prices` is required when `assets` is not provided.")
            prices_data = assets.get_prices()
        else:
            prices_data = prices

        if returns is None:
            if assets is None:
                raise ValueError("`returns` is required when `assets` is not provided.")
            returns_data = assets.get_returns()
        else:
            returns_data = returns

        if not isinstance(prices_data, pd.DataFrame) or prices_data.empty:
            raise ValueError("`prices` must be a non-empty pandas DataFrame.")
        if not isinstance(returns_data, pd.DataFrame) or returns_data.empty:
            raise ValueError("`returns` must be a non-empty pandas DataFrame.")

        return prices_data.copy(), returns_data.copy()

    def register_model(self, name: str, model: Any) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("`name` must be a non-empty string.")
        if not hasattr(model, "fit"):
            raise TypeError("`model` must expose a callable `fit` method.")
        self.models[name.strip().lower()] = model

    def available_models(self) -> tuple[str, ...]:
        return tuple(sorted(self.models.keys()))

    def fit_model(self, model_name: str = "ewma") -> VolatilityFitResult:
        normalized_name = model_name.strip().lower()
        model = self.models.get(normalized_name)
        if model is None:
            raise ValueError(
                f"`model_name` must be one of {list(self.available_models())}. "
                f"Received `{model_name}`."
            )

        fit_result = model.fit(self.primary_returns)
        self.last_model_name = normalized_name
        self.last_fit_result = fit_result
        return fit_result

    def run_diagnostics(
        self,
        fit_result: Optional[VolatilityFitResult] = None,
        lags: Optional[int] = None,
    ) -> Dict[str, Any]:
        selected_fit = fit_result if fit_result is not None else self.last_fit_result
        if selected_fit is None:
            raise RuntimeError("Fit a volatility model before running diagnostics.")

        pre_fit = self.ModelHypothesis2.run_pre_fit_tests(lags=lags)
        residual = self.ModelHypothesis2.run_residual_tests(
            residuals=selected_fit.residuals,
            standardized_residuals=selected_fit.standardized_residuals,
            lags=lags,
        )

        pre_summary = pre_fit["summary"].copy()
        pre_summary["stage"] = "pre_fit"
        residual_summary = residual["summary"].copy()
        residual_summary["stage"] = "residual"

        diagnostics = {
            "pre_fit": pre_fit,
            "residual": residual,
            "summary": pd.concat([pre_summary, residual_summary], ignore_index=True),
        }
        self.last_diagnostics = diagnostics
        return diagnostics

    def fit_and_diagnose(
        self,
        model_name: str = "ewma",
        lags: Optional[int] = None,
    ) -> Dict[str, Any]:
        fit_result = self.fit_model(model_name=model_name)
        diagnostics = self.run_diagnostics(fit_result=fit_result, lags=lags)
        return {
            "fit": fit_result,
            "diagnostics": diagnostics,
        }

    def forecast_variance(self, horizon: int = 1) -> np.ndarray:
        if self.last_model_name is None:
            raise RuntimeError("Fit a model before requesting variance forecasts.")
        model = self.models[self.last_model_name]
        if not hasattr(model, "forecast_variance"):
            raise RuntimeError(f"Model `{self.last_model_name}` does not support forecasting.")
        return np.asarray(model.forecast_variance(horizon=horizon), dtype=float)


AssetsVolatilityAnalysis = AssetVolatilityAnalysis


def _main_():
    ticket = ["CVX"]
    start_date = "2022-01-01"
    end_date = "2026-03-03"
    
    Volatily = AssetVolatilityAnalysis(tickers=ticket, start_date=start_date, end_date=end_date)
    result = Volatily.fit_model()
    print(result)
    return

if __name__ == "__main__":
    _main_()