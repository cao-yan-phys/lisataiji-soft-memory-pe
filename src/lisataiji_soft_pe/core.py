from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from . import response_generated


LABELS = ["log10H", "Lambda", "sinbeta", "Psi", "taus"]
PLOT_LABELS = [
    r"$\log_{10}H$",
    r"$\lambda$",
    r"$\sin\beta$",
    r"$\Psi$",
    r"$\tau_*$",
]
SHAPE_BOUNDS = np.asarray(
    [
        [0.0, 2.0 * np.pi],
        [-1.0, 1.0],
        [0.0, np.pi],
        [0.0, 1.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class SoftEvent:
    log10H: float = -19.0
    Lambda: float = math.pi / 4.0
    beta: float = math.pi / 3.0
    Psi: float = math.pi / 5.0
    taus: float = 0.3

    @property
    def sinbeta(self) -> float:
        return float(np.sin(self.beta))

    def vector(self) -> np.ndarray:
        return np.asarray([self.log10H, self.Lambda, self.sinbeta, self.Psi, self.taus], dtype=float)


def json_clean(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return json_clean(asdict(obj))
    if isinstance(obj, dict):
        return {str(key): json_clean(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, complex):
        return [float(obj.real), float(obj.imag)]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return str(obj)
    return obj


def ranges_from_event(event: SoftEvent) -> np.ndarray:
    x = event.vector()
    return np.asarray(
        [
            [x[0] - 2.0, x[0] + 2.0],
            [0.0, 2.0 * np.pi],
            [-1.0, 1.0],
            [0.0, np.pi],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def configure_context(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {} if config is None else config
    grid = cfg.get("grid", {})
    noise = cfg.get("noise", {})
    seeds = tuple(noise.get("seeds", [123, 124, 125, 126]))
    if len(seeds) != 4:
        raise ValueError("noise.seeds must contain four integers: Taiji A, LISA A, Taiji E, LISA E")
    return response_generated.configure(
        sampling_frequency_hz=float(grid.get("sampling_frequency_hz", 1.0)),
        duration_yr=float(grid.get("duration_yr", 1.0e-3)),
        eta_value=float(grid.get("eta", np.pi / 4.0)),
        noise_seeds=tuple(int(seed) for seed in seeds),
    )


def event_from_config(config: dict[str, Any]) -> SoftEvent:
    event = config.get("event", {})
    return SoftEvent(
        log10H=float(event.get("log10H", -19.0)),
        Lambda=float(event.get("Lambda", np.pi / 4.0)),
        beta=float(event.get("beta", np.pi / 3.0)),
        Psi=float(event.get("Psi", np.pi / 5.0)),
        taus=float(event.get("taus", 0.3)),
    )


def signal_snr(ctx: dict[str, Any], streams: dict[str, Any]) -> dict[str, float]:
    f = ctx["f"]
    lisa2 = ctx["innerA1lisa"](streams["A1lisa"], streams["A1lisa"], f) + ctx["innerA1lisa"](
        streams["E1lisa"], streams["E1lisa"], f
    )
    taiji2 = ctx["innerA1taiji"](streams["A1taiji"], streams["A1taiji"], f) + ctx["innerA1taiji"](
        streams["E1taiji"], streams["E1taiji"], f
    )
    lisa = float(np.asarray(jnp.sqrt(lisa2)))
    taiji = float(np.asarray(jnp.sqrt(taiji2)))
    return {"lisa": lisa, "taiji": taiji, "lisataiji": float(np.sqrt(lisa * lisa + taiji * taiji))}


def build_soft_signal(ctx: dict[str, Any], event: SoftEvent) -> dict[str, Any]:
    log10H, Lambda, sinbeta, Psi, taus = event.vector()
    beta = float(np.arcsin(np.clip(sinbeta, -1.0, 1.0)))
    H = 10.0**log10H
    ts = taus * ctx["T"]
    streams = {
        "A1lisa": ctx["A1lisa"](H, ts, Lambda, beta, Psi),
        "E1lisa": ctx["E1lisa"](H, ts, Lambda, beta, Psi),
        "A1taiji": ctx["A1taiji"](H, ts, Lambda, beta, Psi),
        "E1taiji": ctx["E1taiji"](H, ts, Lambda, beta, Psi),
    }
    return {
        **streams,
        "truths": jnp.asarray(event.vector(), dtype=jnp.float64),
        "ranges": jnp.asarray(ranges_from_event(event), dtype=jnp.float64),
        "event": event,
        "snr": signal_snr(ctx, streams),
    }


def build_soft_template_likelihood(ctx: dict[str, Any], signal: dict[str, Any], *, inject_noise: bool = True):
    f = ctx["f"]
    T = ctx["T"]
    ranges = signal["ranges"]
    inject = 1.0 if inject_noise else 0.0
    noise_taiji_A = jnp.asarray(ctx["tilde_noise_taiji_A"])
    noise_taiji_E = jnp.asarray(ctx["tilde_noise_taiji_E"])
    noise_lisa_A = jnp.asarray(ctx["tilde_noise_lisa_A"])
    noise_lisa_E = jnp.asarray(ctx["tilde_noise_lisa_E"])
    true_taiji_A = jnp.asarray(signal["A1taiji"])
    true_taiji_E = jnp.asarray(signal["E1taiji"])
    true_lisa_A = jnp.asarray(signal["A1lisa"])
    true_lisa_E = jnp.asarray(signal["E1lisa"])

    @jax.jit
    def lnprior(x):
        inside = jnp.all((x > ranges[:, 0]) & (x < ranges[:, 1]))
        return jnp.where(inside, 0.0, -jnp.inf)

    @jax.jit
    def lnlike(x):
        log10H, Lambda, sinbeta, Psi, taus = x
        H = 10.0**log10H
        beta = jnp.arcsin(sinbeta)
        ts = taus * T
        resid_taiji_A = -noise_taiji_A[1:-1] * inject + ctx["A1taiji"](H, ts, Lambda, beta, Psi) - true_taiji_A
        resid_taiji_E = -noise_taiji_E[1:-1] * inject + ctx["E1taiji"](H, ts, Lambda, beta, Psi) - true_taiji_E
        resid_lisa_A = -noise_lisa_A[1:-1] * inject + ctx["A1lisa"](H, ts, Lambda, beta, Psi) - true_lisa_A
        resid_lisa_E = -noise_lisa_E[1:-1] * inject + ctx["E1lisa"](H, ts, Lambda, beta, Psi) - true_lisa_E
        return -0.5 * (
            ctx["innerA1taiji"](resid_taiji_A, resid_taiji_A, f)
            + ctx["innerA1taiji"](resid_taiji_E, resid_taiji_E, f)
            + ctx["innerA1lisa"](resid_lisa_A, resid_lisa_A, f)
            + ctx["innerA1lisa"](resid_lisa_E, resid_lisa_E, f)
        )

    @jax.jit
    def lnprob(x):
        lp = lnprior(x)
        return jnp.where(jnp.isfinite(lp), lp + lnlike(x), -jnp.inf)

    return {"lnprior": lnprior, "lnlike": lnlike, "lnprob": lnprob, "inject_noise": bool(inject_noise)}


def stream_inner(ctx: dict[str, Any], left: dict[str, Any], right: dict[str, Any]) -> float:
    f = ctx["f"]
    value = (
        ctx["innerA1lisa"](left["A1lisa"], right["A1lisa"], f)
        + ctx["innerA1lisa"](left["E1lisa"], right["E1lisa"], f)
        + ctx["innerA1taiji"](left["A1taiji"], right["A1taiji"], f)
        + ctx["innerA1taiji"](left["E1taiji"], right["E1taiji"], f)
    )
    return float(np.asarray(value))


def data_streams(ctx: dict[str, Any], signal: dict[str, Any], *, inject_noise: bool = True) -> dict[str, Any]:
    scale = 1.0 if inject_noise else 0.0
    return {
        "A1lisa": jnp.asarray(signal["A1lisa"]) + scale * jnp.asarray(ctx["tilde_noise_lisa_A"])[1:-1],
        "E1lisa": jnp.asarray(signal["E1lisa"]) + scale * jnp.asarray(ctx["tilde_noise_lisa_E"])[1:-1],
        "A1taiji": jnp.asarray(signal["A1taiji"]) + scale * jnp.asarray(ctx["tilde_noise_taiji_A"])[1:-1],
        "E1taiji": jnp.asarray(signal["E1taiji"]) + scale * jnp.asarray(ctx["tilde_noise_taiji_E"])[1:-1],
    }


def template_streams(ctx: dict[str, Any], x: np.ndarray) -> dict[str, Any]:
    log10H, Lambda, sinbeta, Psi, taus = [float(value) for value in np.asarray(x, dtype=float)]
    H = 10.0**log10H
    beta = np.arcsin(np.clip(sinbeta, -1.0, 1.0))
    ts = taus * ctx["T"]
    return {
        "A1lisa": ctx["A1lisa"](H, ts, Lambda, beta, Psi),
        "E1lisa": ctx["E1lisa"](H, ts, Lambda, beta, Psi),
        "A1taiji": ctx["A1taiji"](H, ts, Lambda, beta, Psi),
        "E1taiji": ctx["E1taiji"](H, ts, Lambda, beta, Psi),
    }


def unit_template_streams(ctx: dict[str, Any], shape: np.ndarray) -> dict[str, Any]:
    Lambda, sinbeta, Psi, taus = [float(value) for value in np.asarray(shape, dtype=float)]
    beta = np.arcsin(np.clip(sinbeta, -1.0, 1.0))
    ts = taus * ctx["T"]
    return {
        "A1lisa": ctx["A1lisa"](1.0, ts, Lambda, beta, Psi),
        "E1lisa": ctx["E1lisa"](1.0, ts, Lambda, beta, Psi),
        "A1taiji": ctx["A1taiji"](1.0, ts, Lambda, beta, Psi),
        "E1taiji": ctx["E1taiji"](1.0, ts, Lambda, beta, Psi),
    }


def profile_at_shape(ctx: dict[str, Any], data: dict[str, Any], shape: np.ndarray) -> dict[str, float]:
    unit = unit_template_streams(ctx, shape)
    dh = stream_inner(ctx, data, unit)
    hh = stream_inner(ctx, unit, unit)
    dd = stream_inner(ctx, data, data)
    if not np.isfinite(hh) or hh <= 0.0:
        return {"rho": -math.inf, "snr2": -math.inf, "H_hat": math.nan, "log10H_hat": math.nan, "profile_loglike": -math.inf}
    H_hat = dh / hh
    if H_hat <= 0.0:
        return {
            "rho": -math.inf,
            "snr2": -math.inf,
            "H_hat": float(H_hat),
            "log10H_hat": math.nan,
            "profile_loglike": float(-0.5 * dd),
            "dh": float(dh),
            "hh": float(hh),
            "dd": float(dd),
        }
    rho = dh / np.sqrt(hh)
    snr2 = rho * rho
    return {
        "rho": float(rho),
        "snr2": float(snr2),
        "H_hat": float(H_hat),
        "log10H_hat": float(np.log10(H_hat)),
        "profile_loglike": float(-0.5 * (dd - snr2)),
        "dh": float(dh),
        "hh": float(hh),
        "dd": float(dd),
    }


def maximize_profile_snr(
    ctx: dict[str, Any],
    data: dict[str, Any],
    start_shape: np.ndarray,
    *,
    maxiter: int = 160,
) -> dict[str, Any]:
    start_shape = np.minimum(np.maximum(np.asarray(start_shape, dtype=float), SHAPE_BOUNDS[:, 0]), SHAPE_BOUNDS[:, 1])

    def clip(x: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(np.asarray(x, dtype=float), SHAPE_BOUNDS[:, 0]), SHAPE_BOUNDS[:, 1])

    def objective(x: np.ndarray) -> float:
        prof = profile_at_shape(ctx, data, clip(x))
        return 1.0e100 if not np.isfinite(prof["rho"]) else -prof["rho"]

    start_prof = profile_at_shape(ctx, data, start_shape)
    result = minimize(
        objective,
        start_shape,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 1.0e-6, "fatol": 1.0e-6, "disp": False},
    )
    candidate_shape = clip(result.x)
    candidate_prof = profile_at_shape(ctx, data, candidate_shape)
    accepted = bool(candidate_prof["rho"] >= start_prof["rho"])
    shape_hat = candidate_shape if accepted else start_shape
    profile = candidate_prof if accepted else start_prof
    return {
        "shape_hat": shape_hat,
        "profile": profile,
        "accepted_optimizer_result": accepted,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "optimizer_nit": int(result.nit),
        "optimizer_nfev": int(result.nfev),
        "candidate_rho": float(candidate_prof["rho"]),
        "start_rho": float(start_prof["rho"]),
    }


def mle_vector(profile_result: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [profile_result["profile"]["log10H_hat"], *np.asarray(profile_result["shape_hat"], dtype=float)],
        dtype=float,
    )


def _scale_stream(stream: dict[str, Any], scale: float) -> dict[str, Any]:
    return {key: jnp.asarray(value) * scale for key, value in stream.items()}


def _sub_streams(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {key: jnp.asarray(left[key]) - jnp.asarray(right[key]) for key in left}


def finite_difference_jacobian(
    ctx: dict[str, Any],
    x: np.ndarray,
    *,
    parameter_indices: list[int] | None = None,
    bounds: np.ndarray | None = None,
    rel_step: float = 1.0e-5,
    min_step: float = 1.0e-8,
) -> tuple[list[dict[str, Any]], np.ndarray, list[int]]:
    x = np.asarray(x, dtype=float)
    indices = list(range(x.size)) if parameter_indices is None else list(parameter_indices)
    bounds_arr = None if bounds is None else np.asarray(bounds, dtype=float)
    derivatives: list[dict[str, Any]] = []
    steps: list[float] = []

    for idx in indices:
        step = max(min_step, rel_step * max(abs(float(x[idx])), 1.0))
        if bounds_arr is not None:
            low, high = bounds_arr[idx]
            margin = min(float(x[idx] - low), float(high - x[idx]))
            if margin > 0.0:
                step = min(step, 0.45 * margin)
        xp = x.copy()
        xm = x.copy()
        xp[idx] += step
        xm[idx] -= step
        if bounds_arr is not None:
            low, high = bounds_arr[idx]
            xp[idx] = min(max(xp[idx], low + min_step), high - min_step)
            xm[idx] = min(max(xm[idx], low + min_step), high - min_step)
        actual_step = 0.5 * (xp[idx] - xm[idx])
        if actual_step <= 0.0 or not np.isfinite(actual_step):
            raise ValueError(f"cannot choose a finite-difference step for parameter {idx}")
        plus = template_streams(ctx, xp)
        minus = template_streams(ctx, xm)
        derivatives.append(_scale_stream(_sub_streams(plus, minus), 1.0 / (2.0 * actual_step)))
        steps.append(float(actual_step))
    return derivatives, np.asarray(steps, dtype=float), indices


def fisher_matrix(
    ctx: dict[str, Any],
    x: np.ndarray,
    *,
    labels: list[str] | None = None,
    bounds: np.ndarray | None = None,
    parameter_indices: list[int] | None = None,
    rel_step: float = 1.0e-5,
    rcond: float = 1.0e-12,
) -> dict[str, Any]:
    derivatives, steps, indices = finite_difference_jacobian(
        ctx,
        x,
        parameter_indices=parameter_indices,
        bounds=bounds,
        rel_step=rel_step,
    )
    npar = len(derivatives)
    gamma = np.empty((npar, npar), dtype=float)
    for i in range(npar):
        for j in range(i, npar):
            gamma[i, j] = stream_inner(ctx, derivatives[i], derivatives[j])
            gamma[j, i] = gamma[i, j]
    eigvals = np.linalg.eigvalsh(gamma)
    positive = eigvals[np.isfinite(eigvals) & (eigvals > 0.0)]
    condition = float(positive[-1] / positive[0]) if positive.size == npar else math.inf
    try:
        covariance = np.linalg.inv(gamma)
        inversion = "inv"
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(gamma, rcond=rcond)
        inversion = "pinv_linalg_error"
    if not np.all(np.isfinite(covariance)) or (np.isfinite(condition) and condition > 1.0 / rcond):
        covariance = np.linalg.pinv(gamma, rcond=rcond)
        inversion = "pinv_condition"
    diag = np.diag(covariance)
    sigma = np.full_like(diag, np.nan, dtype=float)
    ok = diag >= 0.0
    sigma[ok] = np.sqrt(diag[ok])
    return {
        "x": np.asarray(x, dtype=float),
        "parameter_indices": np.asarray(indices, dtype=int),
        "labels": None if labels is None else [labels[i] for i in indices],
        "finite_difference_steps": steps,
        "fisher": gamma,
        "covariance": covariance,
        "sigma": sigma,
        "eigenvalues": eigvals,
        "condition": condition,
        "inversion": inversion,
        "rel_step": rel_step,
        "rcond": rcond,
    }


def diagnostics(
    ctx: dict[str, Any],
    event: SoftEvent,
    *,
    inject_noise: bool = True,
    maxiter: int = 160,
    rel_step: float = 1.0e-5,
    compute_mle: bool = False,
    compute_fim: bool = False,
) -> dict[str, Any]:
    signal = build_soft_signal(ctx, event)
    likelihood = build_soft_template_likelihood(ctx, signal, inject_noise=inject_noise)
    truths = np.asarray(signal["truths"], dtype=float)
    ranges = np.asarray(signal["ranges"], dtype=float)
    lnprob_truth = float(np.asarray(likelihood["lnprob"](jnp.asarray(truths)).block_until_ready()))
    profile = None
    x_mle = None
    if compute_mle:
        data = data_streams(ctx, signal, inject_noise=inject_noise)
        profile = maximize_profile_snr(ctx, data, truths[1:], maxiter=maxiter)
        x_mle = mle_vector(profile)
        profile["x_mle"] = x_mle
    fisher_truth = None
    fisher_mle = None
    if compute_fim:
        fisher_truth = fisher_matrix(ctx, truths, labels=LABELS, bounds=ranges, rel_step=rel_step)
        if x_mle is not None and np.all(np.isfinite(x_mle)) and np.all((x_mle > ranges[:, 0]) & (x_mle < ranges[:, 1])):
            fisher_mle = fisher_matrix(ctx, x_mle, labels=LABELS, bounds=ranges, rel_step=rel_step)
    return {
        "labels": LABELS,
        "event": event,
        "truths": truths,
        "ranges": ranges,
        "snr": signal["snr"],
        "inject_noise": bool(inject_noise),
        "lnprob_at_truth": lnprob_truth,
        "matched_filter": profile,
        "fisher": {"truth": fisher_truth, "mle": fisher_mle},
    }


def flatten_chain(chain: np.ndarray, *, discard: int = 0, thin: int = 1) -> np.ndarray:
    if chain.ndim != 3:
        raise ValueError("chain must have shape (n_steps, n_walkers, n_dim)")
    return chain[discard::thin].reshape(-1, chain.shape[-1])


def parameter_stats(samples: np.ndarray) -> dict[str, np.ndarray]:
    q16, q50, q84 = np.percentile(samples, [16, 50, 84], axis=0)
    return {"q16": q16, "median": q50, "q84": q84, "err_minus": q50 - q16, "err_plus": q84 - q50}


def initial_walkers(center: np.ndarray, ranges: np.ndarray, *, n_walkers: int, seed: int, scatter: float) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    ndim = center.size
    if n_walkers < 2 * ndim:
        raise ValueError(f"n_walkers must be at least 2*ndim={2 * ndim}")
    rng = np.random.default_rng(seed)
    pos = center + rng.normal(size=(n_walkers, ndim)) * float(scatter)
    return np.clip(pos, ranges[:, 0] + 1.0e-10, ranges[:, 1] - 1.0e-10)


def emcee_moves(kind: str):
    import emcee

    if kind == "stretch":
        return emcee.moves.StretchMove(a=2.0)
    if kind == "mixed":
        return [(emcee.moves.StretchMove(a=2.0), 0.7), (emcee.moves.DEMove(), 0.3)]
    if kind == "robust":
        return [
            (emcee.moves.StretchMove(a=1.8), 0.50),
            (emcee.moves.DEMove(), 0.35),
            (emcee.moves.DESnookerMove(), 0.15),
        ]
    raise ValueError('moves must be "stretch", "mixed", or "robust"')


def estimate_tau(chain: np.ndarray) -> np.ndarray:
    import emcee

    try:
        return np.asarray(emcee.autocorr.integrated_time(chain, tol=0, quiet=True), dtype=float)
    except Exception:
        return np.full(chain.shape[-1], np.nan)


def run_mcmc(
    ctx: dict[str, Any],
    event: SoftEvent,
    *,
    inject_noise: bool = True,
    start: str = "mle",
    n_walkers: int = 48,
    n_burn: int = 300,
    max_steps: int = 12000,
    chunk_size: int = 500,
    seed: int = 0,
    scatter: float = 1.0e-3,
    moves: str = "robust",
    mf_maxiter: int = 160,
    progress: bool = False,
) -> dict[str, Any]:
    import emcee

    signal = build_soft_signal(ctx, event)
    likelihood = build_soft_template_likelihood(ctx, signal, inject_noise=inject_noise)
    truths = np.asarray(signal["truths"], dtype=float)
    ranges = np.asarray(signal["ranges"], dtype=float)
    if start == "truth":
        center = truths
        start_meta: dict[str, Any] = {"start": "truth", "x_start": truths}
    elif start == "mle":
        data = data_streams(ctx, signal, inject_noise=inject_noise)
        profile = maximize_profile_snr(ctx, data, truths[1:], maxiter=mf_maxiter)
        center = mle_vector(profile)
        if not (np.all(np.isfinite(center)) and np.all((center > ranges[:, 0]) & (center < ranges[:, 1]))):
            center = truths
            start_meta = {"start": "mle", "fallback": "truth", "x_start": truths, "profile": profile}
        else:
            start_meta = {"start": "mle", "x_start": center, "profile": profile}
    else:
        raise ValueError('start must be "mle" or "truth"')

    x0 = initial_walkers(center, ranges, n_walkers=n_walkers, seed=seed, scatter=scatter)
    lnprob_many = jax.jit(jax.vmap(likelihood["lnprob"]))
    _ = lnprob_many(jnp.asarray(x0)).block_until_ready()

    eval_batches = 0

    def lnprob_emcee(x_batch):
        nonlocal eval_batches
        eval_batches += 1
        return np.asarray(lnprob_many(jnp.asarray(x_batch)))

    sampler = emcee.EnsembleSampler(
        n_walkers,
        truths.size,
        lnprob_emcee,
        vectorize=True,
        moves=emcee_moves(moves),
    )

    started = time.perf_counter()
    state = sampler.run_mcmc(x0, n_burn, progress=progress)
    sampler.reset()
    history: list[dict[str, Any]] = []
    steps_done = 0
    while steps_done < max_steps:
        todo = min(chunk_size, max_steps - steps_done)
        state = sampler.run_mcmc(state, todo, progress=progress)
        steps_done += todo
        chain = sampler.get_chain()
        tau = estimate_tau(chain)
        finite = np.isfinite(tau) & (tau > 0.0)
        max_tau = float(np.max(tau[finite])) if np.any(finite) else math.inf
        n_over_tau = float(steps_done / max_tau) if np.isfinite(max_tau) and max_tau > 0.0 else 0.0
        history.append(
            {
                "steps": int(steps_done),
                "tau": tau,
                "max_tau": max_tau,
                "n_over_max_tau": n_over_tau,
                "reference_n_over_tau_50": bool(np.all(finite) and n_over_tau >= 50.0),
                "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
                "elapsed_sec": float(time.perf_counter() - started),
                "jax_eval_batches": int(eval_batches),
            }
        )

    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    tau = estimate_tau(chain)
    return {
        "chain": chain,
        "log_prob": log_prob,
        "acceptance_fraction": sampler.acceptance_fraction,
        "final_pos": np.asarray(state.coords),
        "final_prob": np.asarray(state.log_prob),
        "truths": truths,
        "ranges": ranges,
        "labels": np.asarray(LABELS),
        "snr": signal["snr"],
        "inject_noise": bool(inject_noise),
        "start": start_meta,
        "tau": tau,
        "history": history,
        "n_burn": int(n_burn),
        "steps_done": int(max_steps),
        "n_walkers": int(n_walkers),
        "seed": int(seed),
        "moves": moves,
        "elapsed_sec": float(time.perf_counter() - started),
    }


def save_mcmc_npz(path: str | Path, result: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        chain=result["chain"],
        log_prob=result["log_prob"],
        acceptance_fraction=result["acceptance_fraction"],
        final_pos=result["final_pos"],
        final_prob=result["final_prob"],
        truths=result["truths"],
        ranges=result["ranges"],
        labels=result["labels"],
        tau=result["tau"],
        snr_lisa=result["snr"]["lisa"],
        snr_taiji=result["snr"]["taiji"],
        snr_lisataiji=result["snr"]["lisataiji"],
        inject_noise=int(result["inject_noise"]),
    )
    return path


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "steps",
        "max_tau",
        "n_over_max_tau",
        "reference_n_over_tau_50",
        "mean_acceptance_fraction",
        "elapsed_sec",
        "jax_eval_batches",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fields})
    return path


def write_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_clean(obj), indent=2), encoding="utf-8")
    return path
