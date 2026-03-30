from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

if __package__:
    from .Model1HypothesisTester import HypothesisTestResult
else:
    from Model1HypothesisTester import HypothesisTestResult


class Model2HypothesisTester:
    """Hypothesis tests for volatility model assumptions and diagnostics."""

    def __init__(
        self,
        series: pd.Series,
        alpha: float = 0.05,
        default_lags: int = 10,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("`alpha` must satisfy 0 < alpha < 1.")
        if default_lags < 1:
            raise ValueError("`default_lags` must be >= 1.")

        self.series = self._prepare_series(series, name="series")
        self.alpha = float(alpha)
        self.default_lags = int(default_lags)

    @staticmethod
    def _prepare_series(values: pd.Series, name: str) -> pd.Series:
        series = values if isinstance(values, pd.Series) else pd.Series(values)
        clean_series = pd.to_numeric(series, errors="coerce").astype(float)
        clean_series = clean_series[np.isfinite(clean_series)].dropna()

        if clean_series.size < 8:
            raise ValueError(f"`{name}` must contain at least 8 finite numeric values.")

        return clean_series

    @staticmethod
    def _decision_text(reject_null: bool, equal_text: str, different_text: str) -> str:
        return different_text if reject_null else equal_text

    @staticmethod
    def _max_supported_lag(values: pd.Series) -> int:
        return max(1, min(50, values.size // 5))

    def _resolve_lag(self, values: pd.Series, lags: Optional[int]) -> int:
        lag_value = self.default_lags if lags is None else int(lags)
        if lag_value < 1:
            raise ValueError("`lags` must be >= 1.")
        return min(lag_value, self._max_supported_lag(values))

    def _build_test_result(
        self,
        name: str,
        statistic: float,
        p_value: float,
        equal_text: str,
        different_text: str,
    ) -> HypothesisTestResult:
        reject_null = bool(p_value < self.alpha)
        return HypothesisTestResult(
            name=name,
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=reject_null,
            decision=self._decision_text(reject_null, equal_text, different_text),
        )

    @staticmethod
    def _results_to_frame(results: Dict[str, HypothesisTestResult]) -> pd.DataFrame:
        return pd.DataFrame([result.as_dict() for result in results.values()])

    def test_arch_lm(
        self,
        values: Optional[pd.Series] = None,
        lags: Optional[int] = None,
        name: str = "arch_lm",
    ) -> HypothesisTestResult:
        series = self.series if values is None else self._prepare_series(values, name="values")
        lag = self._resolve_lag(series, lags)

        lm_stat, lm_pvalue, _, _ = het_arch(series.to_numpy(dtype=float), nlags=lag)

        return self._build_test_result(
            name=name,
            statistic=lm_stat,
            p_value=lm_pvalue,
            equal_text="Sin efecto ARCH",
            different_text="Con efecto ARCH",
        )

    def test_ljung_box(
        self,
        values: Optional[pd.Series] = None,
        lags: Optional[int] = None,
        squared: bool = False,
        name: Optional[str] = None,
    ) -> HypothesisTestResult:
        series = self.series if values is None else self._prepare_series(values, name="values")
        lag = self._resolve_lag(series, lags)
        test_series = series.pow(2.0) if squared else series

        lb_result = acorr_ljungbox(test_series.to_numpy(dtype=float), lags=[lag], return_df=True)
        statistic = float(lb_result["lb_stat"].iloc[-1])
        p_value = float(lb_result["lb_pvalue"].iloc[-1])

        if squared:
            result_name = name or "ljung_box_squared"
            equal_text = "Sin autocorrelacion en cuadrados"
            different_text = "Con autocorrelacion en cuadrados"
        else:
            result_name = name or "ljung_box"
            equal_text = "Sin autocorrelacion"
            different_text = "Con autocorrelacion"

        return self._build_test_result(
            name=result_name,
            statistic=statistic,
            p_value=p_value,
            equal_text=equal_text,
            different_text=different_text,
        )

    def test_jarque_bera(
        self,
        values: Optional[pd.Series] = None,
        name: str = "jarque_bera",
    ) -> HypothesisTestResult:
        series = self.series if values is None else self._prepare_series(values, name="values")
        jb_result = stats.jarque_bera(series.to_numpy(dtype=float))

        return self._build_test_result(
            name=name,
            statistic=float(jb_result.statistic),
            p_value=float(jb_result.pvalue),
            equal_text="Normalidad no rechazada",
            different_text="Normalidad rechazada",
        )

    def run_pre_fit_tests(self, lags: Optional[int] = None) -> Dict[str, Any]:
        tests = {
            "arch_lm_returns": self.test_arch_lm(lags=lags, name="arch_lm_returns"),
            "ljung_box_returns": self.test_ljung_box(lags=lags, name="ljung_box_returns"),
            "ljung_box_squared_returns": self.test_ljung_box(
                lags=lags,
                squared=True,
                name="ljung_box_squared_returns",
            ),
        }
        return {
            "tests": tests,
            "summary": self._results_to_frame(tests),
        }

    def run_residual_tests(
        self,
        residuals: pd.Series,
        standardized_residuals: Optional[pd.Series] = None,
        lags: Optional[int] = None,
    ) -> Dict[str, Any]:
        residual_series = self._prepare_series(residuals, name="residuals")

        if standardized_residuals is None:
            scale = float(residual_series.std(ddof=0))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("Cannot standardize residuals with a non-positive scale.")
            std_residuals = residual_series / scale
        else:
            std_residuals = self._prepare_series(standardized_residuals, name="standardized_residuals")

        tests = {
            "ljung_box_residuals": self.test_ljung_box(
                values=std_residuals,
                lags=lags,
                name="ljung_box_residuals",
            ),
            "ljung_box_squared_residuals": self.test_ljung_box(
                values=std_residuals,
                lags=lags,
                squared=True,
                name="ljung_box_squared_residuals",
            ),
            "arch_lm_residuals": self.test_arch_lm(
                values=std_residuals,
                lags=lags,
                name="arch_lm_residuals",
            ),
            "jarque_bera_residuals": self.test_jarque_bera(
                values=std_residuals,
                name="jarque_bera_residuals",
            ),
        }

        return {
            "tests": tests,
            "summary": self._results_to_frame(tests),
        }
