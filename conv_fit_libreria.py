"""
conv_fit_libreria.py

Libreria per l'analisi di dati di fotoluminescenza risolta nel tempo (TRPL)
mediante convoluzione tra IRF sperimentale e decadimenti esponenziali.

Funzionalità principali:
- caricamento di dati PL e IRF da file CSV;
- normalizzazione dell'IRF e stima del passo temporale (dt);
- costruzione di modelli convoluti mono- e bi-esponenziali;
- fit non lineare (SciPy least_squares) con stima di errori e intervalli di confidenza;
- strutture dataclass per raccogliere in modo ordinato i risultati dei fit;
- funzione di supporto per il plotting dei dati e della curva fittata.

La libreria è pensata per l'elaborazione dei dati della tesi (misure PL),
ma può essere riutilizzata anche in altri esperimenti TRPL con struttura simile.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import LinAlgError, inv, pinv
from scipy.integrate import simpson
from scipy.optimize import OptimizeResult, least_squares
from scipy.stats import t as student_t
from scipy.signal import fftconvolve

# Try to enable the scienceplots style if available, otherwise fall back gracefully.
try: 
    import scienceplots 

    plt.style.use(["science", "grid"])
except ModuleNotFoundError: 
    plt.style.use("default")


DEFAULT_DT = 1e-3 # default time bin size (for example in ns)


@dataclass
class MonoExpFitResult:
    """Container for mono-exponential convolution fit results."""

    tau: float
    amplitude: float
    offset: float
    time: np.ndarray
    model: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    optimizer: OptimizeResult
    statistics: Dict[str, Any]
    sse: float
    sst: float
    r_squared: float


@dataclass
class BiExpFitResult:
    """Container for bi-exponential convolution fit results."""

    tau1: float
    tau2: float
    alpha: float
    amplitude: float
    offset: float
    time: np.ndarray
    model: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    optimizer: OptimizeResult
    statistics: Dict[str, Any]
    sse: float
    sst: float
    r_squared: float


def load_time_resolved_csv(
    filepath: str | Path,
    *,
    delimiter: str = ",",
    skip_header: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load time-resolved PL data and IRF from a CSV file.

    The expected columns are:
        0 -> PL time axis
        1 -> PL intensity
        4 -> IRF time axis
        5 -> IRF intensity

    Parameters
    ----------
    filepath:
        Path to the CSV file.
    delimiter:
        Column separator used within the file.
    skip_header:
        Number of header rows to skip while loading.

    Returns
    -------
    Tuple of numpy arrays: (t_pl, pl_signal, t_irf, irf_signal).
    """
    filepath = Path(filepath)
    data = np.genfromtxt(
        filepath,
        delimiter=delimiter,
        usecols=(0, 1, 4, 5),
        skip_header=skip_header,
    )
    if data.ndim == 1:
        data = data[np.newaxis, :]

    data = np.nan_to_num(data)
    t_pl = data[:, 0]
    pl_signal = data[:, 1]
    t_irf = data[:, 2]
    irf_signal = data[:, 3]
    return t_pl, pl_signal, t_irf, irf_signal


def normalize_irf(
    t_irf: Sequence[float],
    irf_counts: Sequence[float],
    *,
    add_point: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalise the IRF counts and optionally prepend an extra sample.

    Parameters
    ----------
    t_irf, irf_counts:
        Time axis and counts of the measured IRF.
    add_point:
        Optional tuple (time, counts) to prepend before normalisation.

    Returns
    -------
    Tuple (t_irf_out, irf_normalised, irf_area).
    """
    t_irf_arr = np.asarray(t_irf, dtype=float)
    counts = np.asarray(irf_counts, dtype=float)

    if add_point is not None:
        time_pt, count_pt = add_point
        t_irf_arr = np.insert(t_irf_arr, 0, time_pt)
        counts = np.insert(counts, 0, count_pt)

    area_irf = simpson(counts, t_irf_arr)
    if not np.isfinite(area_irf) or area_irf <= 0:
        dt_fallback = estimate_sampling_interval(t_irf_arr)
        area_irf = float(counts.sum() * dt_fallback)

    max_count = float(np.max(counts)) if counts.size else 1.0
    irf_normalised = counts / max(max_count, np.finfo(float).eps)
    return t_irf_arr, irf_normalised, float(area_irf)


def estimate_sampling_interval(
    t_axis: Sequence[float],
    fallback: float = DEFAULT_DT,
) -> float:
    """Estimate the sampling interval (dt) using the median of successive differences."""
    t_arr = np.asarray(t_axis, dtype=float)
    if t_arr.size < 2:
        return fallback
    diffs = np.diff(t_arr)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return fallback
    return float(np.median(diffs))


def best_affine_scaling(y: np.ndarray, model: np.ndarray) -> Tuple[float, float]:
    """
    Compute the optimal affine parameters y ≈ A * model + B in closed form.
    Returns (A, B).
    """
    y = np.asarray(y, dtype=float)
    model = np.asarray(model, dtype=float)
    if y.size != model.size:
        raise ValueError("Observed data and model must have the same length.")

    y_mean = float(np.mean(y))
    m_mean = float(np.mean(model))
    y_centered = y - y_mean
    m_centered = model - m_mean

    denom = float(np.dot(m_centered, m_centered))
    if denom <= 0:
        return 0.0, y_mean

    A = float(np.dot(y_centered, m_centered) / denom)
    B = y_mean - A * m_mean
    return A, B


def _causal_exponential_kernel(
    t_rel: np.ndarray,
    tau: float,
    shift: float = 0.0,
) -> np.ndarray:
    """Causal exponential kernel exp(-(t - shift) / tau) / tau."""
    tau_safe = max(float(tau), np.finfo(float).tiny)
    kernel = np.where(
        t_rel >= shift,
        np.exp(-(t_rel - shift) / tau_safe) / tau_safe,
        0.0,
    )
    return kernel


def monoexp_convolution_model(
    t_irf: Sequence[float],
    irf_signal: Sequence[float],
    t_eval: Sequence[float],
    tau: float,
    *,
    dt: Optional[float] = None,
    shift: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the convolution of the IRF with a causal mono-exponential decay f(t) = θ(t - tshift) * exp(-(t - tshift) / τ)/τ.

    Parameters
    ----------
    t_irf, irf_signal:
        Time axis and values of the IRF.
    t_eval:
        Time axis where the model should be sampled.
    tau:
        Lifetime of the exponential tail.
    dt:
        Optional sampling interval; if omitted it is estimated from t_irf.
    shift:
        Time shift applied to the exponential kernel.

    Returns
    -------
    np.ndarray with the model evaluated on t_eval.
    """
    t_irf = np.asarray(t_irf, dtype=float)
    irf_signal = np.asarray(irf_signal, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    if t_irf.size == 0 or irf_signal.size == 0:
        return np.zeros_like(t_eval)

    dt_eff = float(dt) if dt is not None else estimate_sampling_interval(t_irf)
    t0 = float(t_irf[0])
    t_rel = t_irf - t0

    kernel = _causal_exponential_kernel(t_rel, tau, shift=shift)
    conv = fftconvolve(irf_signal, kernel, mode="full") * dt_eff
    t_conv = t0 + np.arange(conv.size) * dt_eff
    return np.interp(t_eval, t_conv, conv, left=0.0, right=0.0)     # map the convolution back onto the experimental time grid so model_opt and y_meas share the same length and sampling; linear interpolation is a simple, reliable choice.


def standard_errors_after_fit(
    res: OptimizeResult,
    y_meas: Sequence[float],
    model_values: Sequence[float],
    A_opt: float,
    B_opt: float,
    *,
    sigma: Optional[Sequence[float]] = None,
    p_total: int = 3,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Estimate parameter uncertainties for a mono-exponential fit, considering
    one non-linear parameter (tau) and two linear (A, B).

    Returns a dictionary containing variances, standard errors, and confidence
    intervals for tau, A, and B.
    """
    y_meas = np.asarray(y_meas, dtype=float)
    model_values = np.asarray(model_values, dtype=float)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)
    n_obs = y_meas.size
    dof = max(n_obs - p_total, 1)

    if sigma_arr is None:
        rss = float(np.sum(res.fun**2))
        s2 = rss / dof
    else:
        rss_white = float(np.sum(res.fun**2))
        s2 = rss_white / dof

    J = res.jac
    JTJ = J.T @ J       
    try:
        cov_nonlin = s2 * inv(JTJ)      # get the covariance matrix for the non-linear parameter (tau) as the inverse of the Jacobian transpose times Jacobian, scaled by the estimated variance s2 (Gauss-Newton approximation)
    except LinAlgError:
        cov_nonlin = s2 * pinv(JTJ)

    se_tau = float(np.sqrt(cov_nonlin[0, 0]))

    if sigma_arr is None:
        X = np.column_stack([model_values, np.ones_like(model_values)])
    else:
        w = 1.0 / sigma_arr
        X = np.column_stack([model_values, np.ones_like(model_values)]) * w[:, None]

    XtX = X.T @ X
    try:
        cov_AB = s2 * inv(XtX)
    except LinAlgError:
        cov_AB = s2 * pinv(XtX)

    se_A, se_B = np.sqrt(np.clip(np.diag(cov_AB), 0.0, np.inf))
    tcrit = float(student_t.ppf(1.0 - 0.5 * alpha, dof))
    tau_opt = float(res.x[0])

    ci_tau = (tau_opt - tcrit * se_tau, tau_opt + tcrit * se_tau)
    ci_A = (A_opt - tcrit * se_A, A_opt + tcrit * se_A)
    ci_B = (B_opt - tcrit * se_B, B_opt + tcrit * se_B)

    return {
        "dof": dof,
        "rss": float(np.sum(res.fun**2)),
        "s2": float(s2),
        "tcrit": tcrit,
        "tau": {"value": tau_opt, "se": se_tau, "ci": ci_tau},
        "A": {"value": A_opt, "se": float(se_A), "ci": ci_A},
        "B": {"value": B_opt, "se": float(se_B), "ci": ci_B},
        "cov_nonlin": cov_nonlin,
        "corr_nonlin": _corr_from_cov(cov_nonlin),
        "cov_AB": cov_AB,
        "corr_AB": _corr_from_cov(cov_AB),
    }


def standard_errors_after_fit_multi(
    res: OptimizeResult,
    y_meas: Sequence[float],
    model_values: Sequence[float],
    A_opt: float,
    B_opt: float,
    *,
    sigma: Optional[Sequence[float]] = None,
    p_total: Optional[int] = None,
    alpha: float = 0.05,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Generalisation of "standard_errors_after_fit" for multiple non-linear parameters.
    """
    y_meas = np.asarray(y_meas, dtype=float)
    model_values = np.asarray(model_values, dtype=float)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)
    n_obs = y_meas.size
    k_nonlin = len(res.x)
    if p_total is None:
        p_total = k_nonlin + 2  # include A and B
    dof = max(n_obs - p_total, 1)

    if sigma_arr is None:
        rss = float(np.sum(res.fun**2))
        s2 = rss / dof
    else:
        rss = float(np.sum(res.fun**2))
        s2 = rss / dof

    J = res.jac
    JTJ = J.T @ J
    try:
        cov_nonlin = s2 * inv(JTJ)
    except LinAlgError:
        cov_nonlin = s2 * pinv(JTJ)

    se_nonlin = np.sqrt(np.clip(np.diag(cov_nonlin), 0.0, np.inf))
    tcrit = float(student_t.ppf(1.0 - 0.5 * alpha, dof))

    if names is None:
        name_list = [f"p{i + 1}" for i in range(k_nonlin)]
    else:
        name_list = list(names)
        if len(name_list) != k_nonlin:
            raise ValueError(
                f"Expected {k_nonlin} parameter names, received {len(name_list)}."
            )

    if sigma_arr is None:
        X = np.column_stack([model_values, np.ones_like(model_values)])
    else:
        w = 1.0 / sigma_arr
        X = np.column_stack([model_values, np.ones_like(model_values)]) * w[:, None]

    XtX = X.T @ X
    try:
        cov_AB = s2 * inv(XtX)
    except LinAlgError:
        cov_AB = s2 * pinv(XtX)

    se_A, se_B = np.sqrt(np.clip(np.diag(cov_AB), 0.0, np.inf))
    ci_A = (A_opt - tcrit * se_A, A_opt + tcrit * se_A)
    ci_B = (B_opt - tcrit * se_B, B_opt + tcrit * se_B)

    params = {
        nm: {
            "value": float(val),
            "se": float(se),
            "ci": (float(val - tcrit * se), float(val + tcrit * se)),
        }
        for nm, val, se in zip(name_list, res.x, se_nonlin)
    }

    params.update(
        {
            "A": {"value": float(A_opt), "se": float(se_A), "ci": ci_A},
            "B": {"value": float(B_opt), "se": float(se_B), "ci": ci_B},
        }
    )

    return {
        "dof": dof,
        "rss": rss,
        "s2": float(s2),
        "tcrit": tcrit,
        **params,
        "cov_nonlin": cov_nonlin,
        "corr_nonlin": _corr_from_cov(cov_nonlin),
        "cov_AB": cov_AB,
        "corr_AB": _corr_from_cov(cov_AB),
    }


def fit_monoexponential_convolution(
    t_meas: Sequence[float],
    y_meas: Sequence[float],
    t_irf: Sequence[float],
    irf_signal: Sequence[float],
    *,      # remaining arguments are keyword-only
    tau_guess: float = 1.0,
    tau_bounds: Tuple[float, float] = (1e-6, 20.0),
    shift: float = 0.0,     # time shift between stimulus and response of the system, it's better to give it as a fixed parameter and usually is 0. We observe that if we give to a causal system tshift<0 the fit doesn't change and the effect is absorbed in the parameter A, for tshift>0 we get some artifacts, in particular it "advances" the fitted curve. That's because physically tshift represents the time delay between the "peak" of the IRF and the start of the response (decay for example), a positive tshift means that the system starts to decay after the IRF peaks which is not our case and the fit can't account for this with A as in the case with negative tshift.
    dt: Optional[float] = None,     # sampling size (time bin), will be the same for IRF and exp kernel sampling
    max_nfev: int = 20000,
    sigma: Optional[Sequence[float]] = None,        # for weighted least squares (optional)
    p_total: int = 3,       # number of total free parameters (for calculating the DOF)
    alpha: float = 0.05,    # here significance level for confidence intervals, not the same alpha as in biexponential below
) -> MonoExpFitResult:
    """
    Fit a mono-exponential convolution model to time-resolved PL data.
    """
    t_meas = np.asarray(t_meas, dtype=float)
    y_meas = np.asarray(y_meas, dtype=float)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)
    dt_eff = float(dt) if dt is not None else estimate_sampling_interval(t_irf)

    def model_eval(tau_val: float) -> np.ndarray:
        return monoexp_convolution_model(
            t_irf,
            irf_signal,
            t_meas,
            tau_val,
            dt=dt_eff,
            shift=shift,
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        tau_val = float(params[0])
        model_vals = model_eval(tau_val)
        A, B = best_affine_scaling(y_meas, model_vals)
        res_vec = y_meas - (A * model_vals + B)
        if sigma_arr is not None:
            return res_vec / sigma_arr
        return res_vec

    bounds = (
        np.array([tau_bounds[0]], dtype=float),
        np.array([tau_bounds[1]], dtype=float),
    )
    res = least_squares(
        residuals,
        x0=np.array([tau_guess], dtype=float),
        bounds=bounds,
        method="trf",
        max_nfev=max_nfev
    )

    tau_opt = float(res.x[0])
    model_opt = model_eval(tau_opt)
    A_opt, B_opt = best_affine_scaling(y_meas, model_opt)
    fitted = A_opt * model_opt + B_opt
    residual_vec = y_meas - fitted

    sse = float(np.sum(residual_vec**2))
    sst = float(np.sum((y_meas - np.mean(y_meas)) ** 2))
    r_squared = float(1.0 - sse / sst) if sst > 0 else float("nan")

    stats = standard_errors_after_fit(
        res,
        y_meas,
        model_opt,
        A_opt,
        B_opt,
        sigma=sigma_arr,
        p_total=p_total,
        alpha=alpha,
    )
    stats["sse"] = sse
    stats["sst"] = sst
    stats["r_squared"] = r_squared

    return MonoExpFitResult(
        tau=tau_opt,
        amplitude=A_opt,
        offset=B_opt,
        time=t_meas,
        model=model_opt,
        fitted=fitted,
        residuals=residual_vec,
        optimizer=res,
        statistics=stats,
        sse=sse,
        sst=sst,
        r_squared=r_squared,
    )


def biexponential_convolution_model(
    t_irf: Sequence[float],
    irf_signal: Sequence[float],
    t_eval: Sequence[float],
    tau1: float,
    tau2: float,
    alpha: float,
    *,
    dt: Optional[float] = None,
    shift: float = 0.0,
) -> np.ndarray:
    """
    Evaluate the convolution of the IRF with a causal bi-exponential decay.
    """
    t_irf = np.asarray(t_irf, dtype=float)
    irf_signal = np.asarray(irf_signal, dtype=float)
    t_eval = np.asarray(t_eval, dtype=float)

    if t_irf.size == 0 or irf_signal.size == 0:
        return np.zeros_like(t_eval)

    dt_eff = float(dt) if dt is not None else estimate_sampling_interval(t_irf)
    t0 = float(t_irf[0])
    t_rel = t_irf - t0

    tau1_safe = max(float(tau1), np.finfo(float).tiny)
    tau2_safe = max(float(tau2), np.finfo(float).tiny)
    alpha_clipped = float(np.clip(alpha, 0.0, 1.0))

    h1 = np.where(
        t_rel >= shift,
        np.exp(-(t_rel - shift) / tau1_safe) / tau1_safe,
        0.0,
    )
    h2 = np.where(
        t_rel >= shift,
        np.exp(-(t_rel - shift) / tau2_safe) / tau2_safe,
        0.0,
    )
    kernel = alpha_clipped * h1 + (1.0 - alpha_clipped) * h2

    conv = fftconvolve(irf_signal, kernel, mode="full") * dt_eff
    t_conv = t0 + np.arange(conv.size) * dt_eff
    return np.interp(t_eval, t_conv, conv, left=0.0, right=0.0)


def fit_biexponential_convolution(
    t_meas: Sequence[float],
    y_meas: Sequence[float],
    t_irf: Sequence[float],
    irf_signal: Sequence[float],
    *,
    initial: Tuple[float, float, float] = (0.7, 0.7, 0.01),     # initial guesses for (tau1, tau2, alpha)
    bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]] = (
        (1e-6, 1e-6, 0.0),
        (20.0, 20.0, 1.0),
    ),
    shift: float = 0.0,
    dt: Optional[float] = None,
    max_nfev: int = 20000,
    sigma: Optional[Sequence[float]] = None,
    p_total: Optional[int] = 5,
    alpha: float = 0.05,
    names: Optional[Iterable[str]] = ("tau1", "tau2", "alpha"),
) -> BiExpFitResult:
    """Fit a bi-exponential convolution model to TRPL data."""
    t_meas = np.asarray(t_meas, dtype=float)
    y_meas = np.asarray(y_meas, dtype=float)
    sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)
    dt_eff = float(dt) if dt is not None else estimate_sampling_interval(t_irf)

    def model_eval(params: Sequence[float]) -> np.ndarray:
        tau1_val, tau2_val, alpha_val = params
        return biexponential_convolution_model(
            t_irf,
            irf_signal,
            t_meas,
            tau1_val,
            tau2_val,
            alpha_val,
            dt=dt_eff,
            shift=shift,
        )

    def residuals(params: np.ndarray) -> np.ndarray:
        model_vals = model_eval(params)
        A, B = best_affine_scaling(y_meas, model_vals)
        res_vec = y_meas - (A * model_vals + B)
        if sigma_arr is not None:
            return res_vec / sigma_arr
        return res_vec

    lower_bounds = np.array(bounds[0], dtype=float)
    upper_bounds = np.array(bounds[1], dtype=float)

    res = least_squares(
        residuals,
        x0=np.array(initial, dtype=float),
        bounds=(lower_bounds, upper_bounds),
        method="trf",
        max_nfev=max_nfev,
    )

    tau1_opt, tau2_opt, alpha_opt = map(float, res.x)
    model_opt = model_eval(res.x)
    A_opt, B_opt = best_affine_scaling(y_meas, model_opt)
    fitted = A_opt * model_opt + B_opt
    residual_vec = y_meas - fitted

    sse = float(np.sum(residual_vec**2))
    sst = float(np.sum((y_meas - np.mean(y_meas)) ** 2))
    r_squared = float(1.0 - sse / sst) if sst > 0 else float("nan")

    stats = standard_errors_after_fit_multi(
        res,
        y_meas,
        model_opt,
        A_opt,
        B_opt,
        sigma=sigma_arr,
        p_total=p_total,
        alpha=alpha,
        names=names,
    )
    stats["sse"] = sse
    stats["sst"] = sst
    stats["r_squared"] = r_squared

    return BiExpFitResult(
        tau1=tau1_opt,
        tau2=tau2_opt,
        alpha=alpha_opt,
        amplitude=A_opt,
        offset=B_opt,
        time=t_meas,
        model=model_opt,
        fitted=fitted,
        residuals=residual_vec,
        optimizer=res,
        statistics=stats,
        sse=sse,
        sst=sst,
        r_squared=r_squared,
    )


def plot_fit(
    ax: plt.Axes,
    t_meas: Sequence[float],
    y_meas: Sequence[float],
    fitted: Sequence[float],
    *,
    data_label: str = "Dati",
    fit_label: str = "Fit",
    color_data: str = "#4470cd",
    color_fit: str = "r",
) -> None:
    """Utility to plot measured data and fitted curve on the supplied axis."""
    ax.scatter(t_meas, y_meas, s=0.5, color=color_data, label=data_label)
    ax.plot(t_meas, fitted, color=color_fit, linewidth=2, label=fit_label)
    ax.legend(fontsize=12, markerscale=5)


def _corr_from_cov(cov: np.ndarray) -> np.ndarray:
    """Compute correlation matrix from covariance matrix."""
    diag = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    denom = np.outer(diag, diag)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / denom
    np.fill_diagonal(corr, 1.0)
    corr[~np.isfinite(corr)] = 0.0
    return corr