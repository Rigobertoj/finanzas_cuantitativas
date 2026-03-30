import hashlib
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .Model1HypothesisTester import Model1HypothesisTester
    from .AssetVolatilityAnalysis import AssetVolatilityAnalysis, VolatilityFitResult
    from .AssetElementaryMetrics import AssetElementaryMetrics
    from .TestDistribuitions import (
        best_fit_distribution,
        DEFAULT_DISTRIBUTIONS,
        DEFAULT_DISTRIBUTION_NAMES,
    )
else:
    from Model1HypothesisTester import Model1HypothesisTester
    from AssetVolatilityAnalysis import AssetVolatilityAnalysis, VolatilityFitResult
    from AssetElementaryMetrics import AssetElementaryMetrics
    from TestDistribuitions import (
        best_fit_distribution, 
        DEFAULT_DISTRIBUTIONS, 
        DEFAULT_DISTRIBUTION_NAMES
    )


class AssetBehaveSimulation:
    tickets: list[str]
    start_date: str
    end_date: str
    num_distributions: int = 5

    def __init__(
        self,
        tickets: list[str],
        start_date: str,
        end_date: str = None,
        num_distributions: int = 5,
        fit_bins: int = 80,
    ) -> None:
        
        if num_distributions < 1:
            raise ValueError("`num_distributions` debe ser >= 1.")
        if fit_bins < 2:
            raise ValueError("`fit_bins` debe ser >= 2.")

        self.tickets = tickets
        self.start_date = start_date
        self.end_date = end_date
        self.num_distributions = int(num_distributions)
        self.fit_bins = int(fit_bins)

        self.Assets = AssetElementaryMetrics(tickers=tickets, start=start_date, end=end_date)

        self.prices = self.Assets.get_prices()
        self.returns = self.Assets.get_returns()
        self.primary_returns = pd.to_numeric(self.returns.iloc[:, 0], errors="coerce").dropna()

        self.mu = float(self.primary_returns.mean())
        self.sigma = float(self.primary_returns.std())

        self.try_distributions = tuple(DEFAULT_DISTRIBUTIONS[: self.num_distributions])
        self._distribution_fit_cache: dict[tuple[Any, ...], dict[str, Any]] = {}

        self.distributions_name: Optional[str] = None
        self.distributions_params: Optional[tuple[float, ...]] = None
        self.distributions = None

        self.AssetsModel1 = Model1HypothesisTester(
            self.prices.iloc[:, 0],
            bins=400,
            distributions=self.try_distributions,
        )
        self.AssetsVolatility = AssetVolatilityAnalysis(
            assets=self.Assets,
            prices=self.prices,
            returns=self.returns,
        )
        self.volatility_fit_result: Optional[VolatilityFitResult] = None

    def back_simulation(self, simulations : int = 1000, steps : int = 1000) -> np.ndarray:
        S0 = float(self.prices.iloc[0, 0])
        
        matrix_simulation = self._simulation_step(S0=S0, simulations=simulations, steps=steps)
        
        matriz_dX = [self.get_random_walk(size=steps) for simulation in range(simulations)]
        
        model = self.set_model()
        
        for simulation in range(simulations):
            for step in range(1, steps):
                W = matriz_dX[simulation][step - 1]
                Sn = matrix_simulation[simulation][step - 1]
                step_value = model(S0=Sn, mu=self.mu, sigma=self.sigma, W=W)
                
                matrix_simulation[simulation][step] = step_value
        
        return np.array(matrix_simulation)

    def fit_volatility_model(self, model_name: str = "ewma", lags: Optional[int] = None):
        result = self.AssetsVolatility.fit_and_diagnose(model_name=model_name, lags=lags)
        self.volatility_fit_result = result["fit"]
        return result

    def back_simulation_with_conditional_volatility(
        self,
        simulations: int = 1000,
        model_name: str = "ewma",
        lags: Optional[int] = None,
    ) -> dict[str, Any]:
        if simulations < 1:
            raise ValueError("`simulations` must be >= 1.")

        volatility_output = self.fit_volatility_model(model_name=model_name, lags=lags)
        fit_result = volatility_output["fit"]

        steps = len(self.prices)
        S0 = float(self.prices.iloc[0, 0])
        matrix_simulation = self._simulation_step(S0=S0, simulations=simulations, steps=steps)
        matrix_random_walk = [self.get_random_walk(size=steps) for _ in range(simulations)]

        sigma_path = fit_result.conditional_volatility.to_numpy(dtype=float)
        if sigma_path.size < steps:
            last_sigma = float(sigma_path[-1])
            sigma_path = np.pad(sigma_path, (0, steps - sigma_path.size), mode="constant", constant_values=last_sigma)
        elif sigma_path.size > steps:
            sigma_path = sigma_path[-steps:]

        model = self.set_model()

        for simulation in range(simulations):
            for step in range(1, steps):
                W = float(matrix_random_walk[simulation][step - 1])
                Sn = float(matrix_simulation[simulation][step - 1])
                sigma_t = float(sigma_path[step - 1])
                matrix_simulation[simulation][step] = model(S0=Sn, mu=self.mu, sigma=sigma_t, W=W)

        return {
            "simulation": np.array(matrix_simulation),
            "volatility": volatility_output,
        }
    
    def forward_simulation(self, simulations : int = 1000):
        return
    
    @staticmethod
    def _build_returns_signature(values: pd.Series) -> str:
        clean_values = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if clean_values.size == 0:
            raise ValueError("No hay retornos validos para ajustar una distribucion.")
        contiguous_values = np.ascontiguousarray(clean_values)
        return hashlib.sha1(contiguous_values.tobytes()).hexdigest()

    def _get_primary_returns(self) -> pd.Series:
        if self.returns.empty:
            self.returns = self.Assets.get_returns()
        return pd.to_numeric(self.returns.iloc[:, 0], errors="coerce").dropna()

    def _resolve_candidate_distributions(self, distribution_name: Optional[str]) -> tuple[Any, ...]:
        if distribution_name is None:
            if not self.try_distributions:
                raise ValueError("No hay distribuciones candidatas para ajustar.")
            return self.try_distributions

        dist = getattr(stats, distribution_name, None)
        if dist is None or not hasattr(dist, "fit") or not hasattr(dist, "pdf"):
            raise ValueError(f"`{distribution_name}` no es una distribucion valida de scipy.stats.")
        return (dist,)

    @staticmethod
    def _step_stock_price(steps : int, S0 : float):
        steps_arry = np.zeros(steps)
        steps_arry[0] = S0
        return steps_arry
    
    def _simulation_step(self, S0 : float, simulations : int = 1000, steps : int = 1000 ):
        return [self._step_stock_price(steps=steps, S0=S0) for simulation in range(simulations)]

    def clear_distribution_cache(self) -> None:
        self._distribution_fit_cache.clear()

    def get_distribution(self):
        return DEFAULT_DISTRIBUTION_NAMES

    def set_distribution_returns(
        self,
        distributions: Optional[str] = None,
        bins: Optional[int] = None,
        force_refit: bool = False,
    ):
        bins_to_use = self.fit_bins if bins is None else int(bins)
        if bins_to_use < 2:
            raise ValueError("`bins` debe ser >= 2.")

        returns = self._get_primary_returns()
        candidate_distributions = self._resolve_candidate_distributions(distributions)
        candidate_names = tuple(dist.name for dist in candidate_distributions)

        cache_key = (
            self._build_returns_signature(returns),
            bins_to_use,
            candidate_names,
        )

        if not force_refit and cache_key in self._distribution_fit_cache:
            cached_fit = self._distribution_fit_cache[cache_key]
            self.distributions_name = cached_fit["name"]
            self.distributions_params = cached_fit["params"]
            self.distributions = cached_fit["frozen"]
            return self.distributions

        name, params = best_fit_distribution(
            returns,
            bins=bins_to_use,
            distributions=candidate_distributions,
        )
        params = tuple(float(value) for value in params)

        dist = getattr(stats, name)
        shape_arg = params[:-2]
        loc = params[-2]
        scale = params[-1]
        frozen_dist = dist(*shape_arg, loc=loc, scale=scale)

        self.distributions_name = name
        self.distributions_params = params
        self.distributions = frozen_dist

        self._distribution_fit_cache[cache_key] = {
            "name": name,
            "params": params,
            "frozen": frozen_dist,
        }

        return self.distributions
    
    def get_random_walk(self, size : int = None):
        if self.distributions is None:
            self.set_distribution_returns()
        
        if size is None:
            size = len(self.prices)
        
        return self.distributions.rvs(size = size)
    
    def _model1(self, S0: float, mu : float, sigma: float, W : list[float]):
        mu_model = mu - (0.5 * sigma**2)
        sigma_model = sigma * W
        return S0 * np.exp(mu_model + sigma_model)
    
    def _model2(self):
        return
    
    def set_model(self):
        return self._model1



def _main_():
    ticket = ["CVX"]
    start_date = "2022-01-01"
    end_date = "2026-03-03"

    simulation = AssetBehaveSimulation(tickets=ticket, start_date=start_date, end_date=end_date)
    print(simulation.sigma)
    #print(simulation.set_distribution_returns())
    #print(simulation.distributions_name)
    
    #print(simulation.get_random_walk(size = 100))

    #simulation.run_simulation()
    return


if __name__ == "__main__":
    _main_()
