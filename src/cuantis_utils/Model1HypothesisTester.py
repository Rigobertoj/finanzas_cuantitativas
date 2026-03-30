from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .TestDistribuitions import best_fit_distribution, make_pdf, DEFAULT_DISTRIBUTIONS
else:
    from TestDistribuitions import best_fit_distribution, make_pdf, DEFAULT_DISTRIBUTIONS


@dataclass(frozen=True)
class HypothesisTestResult:
    """Container for a hypothesis-test output."""

    name: str
    statistic: float
    p_value: float
    alpha: float
    reject_null: bool
    decision: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "alpha": self.alpha,
            "reject_null": self.reject_null,
            "decision": self.decision,
        }


class Model1HypothesisTester:
    """Generalize the workflow from 04H over a price time series.

    Given a price series, it computes:
    - Arithmetic returns: ``R_t = P_t / P_{t-1} - 1``
    - Log returns: ``r_t = log(P_t) - log(P_{t-1})``

    Then it runs the same statistical checks used in notebook 04H.
    """
    
    def __init__(
        self,
        price_series: pd.Series,
        alpha: float = 0.05,
        bins: int = 200,
        pdf_size: int = 4000,
        distributions: Optional[Sequence[Any]] = None,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("`alpha` must satisfy 0 < alpha < 1.")
        if bins < 2:
            raise ValueError("`bins` must be >= 2.")
        if pdf_size < 2:
            raise ValueError("`pdf_size` must be >= 2.")

        self.price_series = self._prepare_price_series(price_series)
        self.alpha = float(alpha)
        self.bins = int(bins)
        self.pdf_size = int(pdf_size)
        self.distributions = tuple(distributions) if distributions is not None else None

        self.returns: Optional[pd.DataFrame] = None
        self.distribution_results: Optional[Dict[str, Any]] = None
        self.DEFAULT_DISTRIBUTIONS = DEFAULT_DISTRIBUTIONS
    
    @property
    def get_distributions(self):
        return self.distributions
    
    @staticmethod
    def _prepare_price_series(price_series: pd.Series) -> pd.Series:
        if not isinstance(price_series, pd.Series):
            raise TypeError("`price_series` must be a pandas Series.")

        clean_prices = pd.to_numeric(price_series, errors="coerce").astype(float)
        clean_prices = clean_prices[np.isfinite(clean_prices)]
        clean_prices = clean_prices[clean_prices > 0.0]
        clean_prices = clean_prices.dropna().sort_index()

        if clean_prices.size < 3:
            raise ValueError("`price_series` must contain at least 3 positive finite values.")

        return clean_prices

    def compute_returns(self) -> pd.DataFrame:
        """Build arithmetic and log return series from the input prices."""
        log_prices = np.log(self.price_series)

        returns = pd.DataFrame(index=self.price_series.index.copy())
        returns["R"] = self.price_series.pct_change()
        returns["r"] = log_prices - log_prices.shift(1)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

        if returns.shape[0] < 2:
            raise ValueError("Not enough observations after return computation.")

        self.returns = returns
        return returns.copy()

    def _get_returns(self) -> pd.DataFrame:
        if self.returns is None:
            return self.compute_returns()
        return self.returns

    @staticmethod
    def _decision_text(reject_null: bool, equal_text: str, different_text: str) -> str:
        return different_text if reject_null else equal_text

    def test_variance_equality(self) -> HypothesisTestResult:
        """Levene test for equality of variance between R and r."""
        returns = self._get_returns()
        statistic, p_value = stats.levene(returns["R"], returns["r"])
        reject_null = bool(p_value < self.alpha)

        return HypothesisTestResult(
            name="levene_variance_R_vs_r",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=reject_null,
            decision=self._decision_text(reject_null, "Varianzas iguales", "Varianzas diferentes"),
        )

    def test_mean_difference(self) -> HypothesisTestResult:
        """Test mean(R-r) against zero, equivalent to the mean check in 04H."""
        returns = self._get_returns()
        delta_returns = returns["R"] - returns["r"]
        statistic, p_value = stats.ttest_1samp(delta_returns, popmean=0.0, nan_policy="omit")
        reject_null = bool(p_value < self.alpha)

        return HypothesisTestResult(
            name="mean_difference_R_minus_r",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=reject_null,
            decision=self._decision_text(reject_null, "Medias iguales", "Medias diferentes"),
        )

    def compute_small_return_approximation(self) -> pd.Series:
        """Approximation used in 04H: r_t ~= R_t - R_t^2 / 2."""
        returns = self._get_returns()
        return returns["R"] - ((returns["R"] ** 2.0) / 2.0)

    def test_small_return_approximation(self) -> HypothesisTestResult:
        """Compare r_t approximation against R as done in 04H."""
        returns = self._get_returns()
        approx_log_returns = self.compute_small_return_approximation()
        statistic, p_value = stats.ttest_ind(approx_log_returns, returns["R"], nan_policy="omit")
        reject_null = bool(p_value < self.alpha)

        return HypothesisTestResult(
            name="mean_difference_rt_approx_vs_R",
            statistic=float(statistic),
            p_value=float(p_value),
            alpha=self.alpha,
            reject_null=reject_null,
            decision=self._decision_text(reject_null, "Medias iguales", "Medias diferentes"),
        )

    def fit_distributions(self) -> Dict[str, Any]:
        """Fit best distributions for arithmetic and log returns."""        
        returns = self._get_returns()

        name_best_distribution_R, best_params_dist_R = best_fit_distribution(
            returns["R"],
            bins=self.bins,
            distributions=self.distributions,
        )
        name_best_distribution_r, best_params_dist_r = best_fit_distribution(
            returns["r"],
            bins=self.bins,
            distributions=self.distributions,
        )

        best_dist_R = getattr(stats, name_best_distribution_R)
        best_dist_r = getattr(stats, name_best_distribution_r)
        pdf_R = make_pdf(best_dist_R, best_params_dist_R, size=self.pdf_size)
        pdf_r = make_pdf(best_dist_r, best_params_dist_r, size=self.pdf_size)

        results = {
            "name_best_distribution_R": name_best_distribution_R,
            "best_params_dist_R": tuple(float(value) for value in best_params_dist_R),
            "name_best_distribution_r": name_best_distribution_r,
            "best_params_dist_r": tuple(float(value) for value in best_params_dist_r),
            "pdf_R": pdf_R,
            "pdf_r": pdf_r,
        }
        
        self.distribution_results = results
        return {
            **results,
            "pdf_R": results["pdf_R"].copy(),
            "pdf_r": results["pdf_r"].copy(),
        }

    def plot_fitted_distributions(
        self,
        bins: int = 60,
        alpha_hist: float = 0.4,
    ) -> tuple[Any, Any]:
        """Plot histogram + fitted PDF for both return series."""
        if bins < 2:
            raise ValueError("`bins` must be >= 2.")
        if not 0.0 < alpha_hist <= 1.0:
            raise ValueError("`alpha_hist` must satisfy 0 < alpha_hist <= 1.")

        returns = self._get_returns()
        if self.distribution_results is None:
            self.fit_distributions()
        assert self.distribution_results is not None

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        returns["R"].plot(
            kind="hist",
            bins=bins,
            density=True,
            alpha=alpha_hist,
            label="Returns",
            ax=axes[0],
        )
        self.distribution_results["pdf_R"].plot(
            ax=axes[0],
            lw=2,
            label=f"PDF ajustada: {self.distribution_results['name_best_distribution_R']}",
        )
        axes[0].legend()
        axes[0].set_title("R: Histograma y PDF ajustada")

        returns["r"].plot(
            kind="hist",
            bins=bins,
            density=True,
            alpha=alpha_hist,
            label="Log Returns",
            ax=axes[1],
        )
        self.distribution_results["pdf_r"].plot(
            ax=axes[1],
            lw=2,
            label=f"PDF ajustada: {self.distribution_results['name_best_distribution_r']}",
        )
        axes[1].legend()
        axes[1].set_title("r: Histograma y PDF ajustada")

        fig.tight_layout()
        return fig, axes

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return a compact summary."""
        returns = self._get_returns().copy()
        variance_result = self.test_variance_equality()
        mean_result = self.test_mean_difference()
        approximation_result = self.test_small_return_approximation()
        distributions = self.fit_distributions()

        summary = pd.DataFrame(
            [
                variance_result.as_dict(),
                mean_result.as_dict(),
                approximation_result.as_dict(),
            ]
        )

        return {
            "returns": returns,
            "r_t_approximation": self.compute_small_return_approximation().copy(),
            "variance_test": variance_result,
            "mean_test": mean_result,
            "approximation_test": approximation_result,
            "distribution_fit": distributions,
            "summary": summary,
        }


if __name__ == "__main__":
    pass

    ticket = "CVX"
    
    CXV = Model1HypothesisTester()
