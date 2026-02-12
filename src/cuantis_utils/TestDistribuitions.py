# -*- coding: utf-8 -*-
"""
Created on Sun May 24 19:29:04 2020

fuente : https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python?lq=1
@author: zaratejo
"""

#https://pypi.org/project/fitter/

import warnings
from typing import Any, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from matplotlib.axes import Axes


DEFAULT_DISTRIBUTION_NAMES: Tuple[str, ...] = (
    "alpha",
    "anglit",
    "arcsine",
    "beta",
    "betaprime",
    "bradford",
    "burr",
    "cauchy",
    "chi",
    "chi2",
    "cosine",
    "dgamma",
    "dweibull",
    "erlang",
    "expon",
    "exponnorm",
    "exponweib",
    "exponpow",
    "f",
    "fatiguelife",
    "fisk",
    "foldcauchy",
    "foldnorm",
    "frechet_r",
    "frechet_l",
    "genlogistic",
    "genpareto",
    "gennorm",
    "genexpon",
    "genextreme",
    "gausshyper",
    "gamma",
    "gengamma",
    "genhalflogistic",
    "gilbrat",
    "gompertz",
    "gumbel_r",
    "gumbel_l",
    "halfcauchy",
    "halflogistic",
    "halfnorm",
    "halfgennorm",
    "hypsecant",
    "invgamma",
    "invgauss",
    "invweibull",
    "johnsonsb",
    "johnsonsu",
    "ksone",
    "kstwobign",
    "laplace",
    "levy",
    "levy_l",
    "levy_stable",
    "logistic",
    "loggamma",
    "loglaplace",
    "lognorm",
    "lomax",
    "maxwell",
    "mielke",
    "nakagami",
    "ncx2",
    "ncf",
    "nct",
    "norm",
    "pareto",
    "pearson3",
    "powerlaw",
    "powerlognorm",
    "powernorm",
    "rdist",
    "reciprocal",
    "rayleigh",
    "rice",
    "recipinvgauss",
    "semicircular",
    "t",
    "triang",
    "truncexpon",
    "truncnorm",
    "tukeylambda",
    "uniform",
    "vonmises",
    "vonmises_line",
    "wald",
    "weibull_min",
    "weibull_max",
    "wrapcauchy",
)

_DISTRIBUTION_ALIASES = {
    # Legacy names from old scipy versions.
    "gilbrat": "gibrat",
    "frechet_r": "invweibull",
    "frechet_l": "weibull_max",
}


def _resolve_distribution(name: str) -> Optional[Any]:
    """Resolve a scipy distribution name, including legacy aliases."""
    dist = getattr(st, name, None)
    if dist is None:
        alias = _DISTRIBUTION_ALIASES.get(name)
        dist = getattr(st, alias, None) if alias else None

    if dist is None:
        return None
    if not hasattr(dist, "fit") or not hasattr(dist, "pdf"):
        return None

    return dist


def _build_default_distributions(names: Sequence[str]) -> Tuple[Any, ...]:
    """Build a tuple of unique scipy distributions available in this environment."""
    resolved = []
    seen = set()
    for name in names:
        dist = _resolve_distribution(name)
        if dist is None:
            continue

        dist_name = getattr(dist, "name", name)
        if dist_name in seen:
            continue

        seen.add(dist_name)
        resolved.append(dist)

    if not resolved:
        raise RuntimeError("No valid default distributions were resolved from scipy.stats.")

    return tuple(resolved)


DEFAULT_DISTRIBUTIONS: Tuple[Any, ...] = _build_default_distributions(DEFAULT_DISTRIBUTION_NAMES)


def _prepare_numeric_data(data: Union[pd.Series, np.ndarray, Sequence[float]]) -> np.ndarray:
    """Convert input data to a clean 1D numeric array."""
    raw_values = np.asarray(data).ravel()
    values = pd.to_numeric(pd.Series(raw_values), errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("`data` must contain at least two finite numeric values.")
    return values


def _distribution_pdf(distribution: Any, x: np.ndarray, params: Tuple[float, ...]) -> Optional[np.ndarray]:
    """Evaluate PDF values for a scipy continuous distribution."""
    if len(params) < 2 or not hasattr(distribution, "pdf"):
        return None

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    pdf = distribution.pdf(x, *arg, loc=loc, scale=scale)
    return np.asarray(pdf, dtype=float)


def _split_distribution_params(params: Sequence[float]) -> Tuple[Tuple[float, ...], float, float]:
    """Split scipy distribution parameters into shape args, loc and scale."""
    if len(params) < 2:
        raise ValueError("`params` must include at least loc and scale.")

    values = tuple(float(value) for value in params)
    arg = values[:-2]
    loc = values[-2]
    scale = values[-1]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("`scale` parameter must be a positive finite number.")

    return arg, loc, scale


def _resolve_pdf_interval(
    dist: Any,
    arg: Tuple[float, ...],
    loc: float,
    scale: float,
    lower_quantile: float,
    upper_quantile: float,
) -> Tuple[float, float]:
    """Return a finite interval where the PDF can be evaluated safely."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            start = float(dist.ppf(lower_quantile, *arg, loc=loc, scale=scale))
            end = float(dist.ppf(upper_quantile, *arg, loc=loc, scale=scale))
            if np.isfinite(start) and np.isfinite(end) and start < end:
                return start, end
        except Exception:
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            support_start, support_end = dist.support(*arg, loc=loc, scale=scale)
            support_start = float(support_start)
            support_end = float(support_end)
            if np.isfinite(support_start) and np.isfinite(support_end) and support_start < support_end:
                return support_start, support_end
        except Exception:
            pass

    spread = max(4.0 * scale, 1.0)
    return loc - spread, loc + spread


# Create models from data
def best_fit_distribution(
    data: Union[pd.Series, np.ndarray, Sequence[float]],
    bins: int = 200,
    ax: Optional[Axes] = None,
    distributions: Optional[Sequence[Any]] = None,
) -> Tuple[str, Tuple[float, ...]]:
    """Return the best fitted distribution name and parameters based on SSE.

    The function fits each candidate distribution to `data` and compares the
    fitted PDF against the empirical histogram density.
    """
    if bins < 2:
        raise ValueError("`bins` must be >= 2.")

    clean_data = _prepare_numeric_data(data)
    y_hist, bin_edges = np.histogram(clean_data, bins=bins, density=True)
    x_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    candidates = tuple(distributions) if distributions is not None else DEFAULT_DISTRIBUTIONS
    if not candidates:
        raise ValueError("`distributions` cannot be empty.")

    best_distribution_name: Optional[str] = None
    best_params: Tuple[float, ...] = ()
    best_sse = np.inf

    for distribution in candidates:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                params = tuple(distribution.fit(clean_data))
                pdf = _distribution_pdf(distribution, x_centers, params)
            except Exception:
                continue

        if pdf is None or not np.all(np.isfinite(pdf)):
            continue

        sse = float(np.sum((y_hist - pdf) ** 2.0))
        if not np.isfinite(sse):
            continue

        if ax is not None:
            pd.Series(pdf, index=x_centers).plot(ax=ax)

        if sse < best_sse:
            best_sse = sse
            best_distribution_name = distribution.name
            best_params = params

    if best_distribution_name is None:
        raise ValueError("No candidate distribution could be fitted to the provided data.")

    return best_distribution_name, best_params

def make_pdf(
    dist: Any,
    params: Sequence[float],
    size: int = 10000,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> pd.Series:
    """Generate a PDF series for a fitted scipy distribution.

    Args:
        dist: Scipy distribution object (e.g. ``scipy.stats.norm``).
        params: Fitted parameters ordered as ``(*shape_args, loc, scale)``.
        size: Number of points in the generated grid.
        lower_quantile: Lower percentile used to define the plotting interval.
        upper_quantile: Upper percentile used to define the plotting interval.

    Returns:
        A pandas ``Series`` where index is the x-grid and values are PDF values.
    """
    if size < 2:
        raise ValueError("`size` must be >= 2.")
    if not (0.0 < lower_quantile < upper_quantile < 1.0):
        raise ValueError("Quantiles must satisfy 0 < lower_quantile < upper_quantile < 1.")
    if not hasattr(dist, "pdf"):
        raise TypeError("`dist` must provide a `pdf` method.")

    arg, loc, scale = _split_distribution_params(params)
    start, end = _resolve_pdf_interval(dist, arg, loc, scale, lower_quantile, upper_quantile)
    x = np.linspace(start, end, size, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = np.asarray(dist.pdf(x, *arg, loc=loc, scale=scale), dtype=float)

    if y.shape != x.shape:
        raise ValueError("Generated PDF has an unexpected shape.")

    y[~np.isfinite(y)] = 0.0
    y = np.clip(y, a_min=0.0, a_max=None)
    if not np.any(y > 0):
        raise ValueError("Unable to generate a valid PDF with the provided distribution and parameters.")

    return pd.Series(y, index=x, name=f"{getattr(dist, 'name', 'distribution')}_pdf")

def main():
    # Load data from statsmodels datasets
    RawData = pd.read_csv('Data_OilCompany.csv')
    #RawData = pd.read_csv('Data_OilCompany10records.csv')


    #data=RawData

    #Index(['  Year 1  ', '  Year 2  ', '  Year 3  ', '  Year 4  ', '  Year 5  '], dtype='object')

    #Test Year 2
    data=RawData.iloc[:,1]
    #data=float(data.replace("-",0))

    #data=pd.DataFrame(b[:,7])

    # Plot for comparison
    plt.figure(figsize=(12,8))
    ax = data.plot(kind='hist', bins=200, density=True, alpha=0.5)

    # Save plot limits
    dataYLim = ax.get_ylim()

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200, ax)
    best_dist = getattr(st, best_fit_name)

    # Update plots
    ax.set_ylim(dataYLim)
    ax.set_title(u'Data\n All Fitted Distributions')
    ax.set_xlabel(u'%')
    ax.set_ylabel('Frequency')

    # Make PDF with best params
    pdf = make_pdf(best_dist, best_fit_params)

    # Display
    plt.figure(figsize=(12,8))
    ax = pdf.plot(lw=2, label='PDF', legend=True)
    data.plot(kind='hist', bins=50, density=True, alpha=0.5, label='Data', legend=True, ax=ax)

    param_names = (best_dist.shapes + ', loc, scale').split(', ') if best_dist.shapes else ['loc', 'scale']
    param_str = ', '.join(['{}={:0.2f}'.format(k,v) for k,v in zip(param_names, best_fit_params)])
    dist_str = '{}({})'.format(best_fit_name, param_str)

    ax.set_title(u'Data with best fit distribution \n' + dist_str)
    ax.set_xlabel(u'%')
    ax.set_ylabel('Frequency')


if __name__ == "__main__":
    # main()
    print(type(st.norm))
    
    pass

