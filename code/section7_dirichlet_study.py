"""Shared numerical routines for the Section 7.2 simulation experiment.

The experiment uses q0=8 informed coordinates and n=16*r balanced logistic
observations. Independent stationary one-step pairs estimate the Dirichlet
forms, so no Markov-chain burn-in or time-series variance estimator is needed.
The parameter paths and output formatting are defined by the reader-facing
entry points in the package root.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


Q0 = 8


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def make_problem(k: int, n: int, xi: float) -> dict[str, float | int]:
    if k < Q0 or n % (2 * Q0) != 0 or xi <= 1:
        raise ValueError("Require k>=8, n divisible by 16, and xi>1")
    replication_factor = n // (2 * Q0)
    row_scale = np.sqrt(2.0 * (xi - 1.0) / replication_factor)

    # Each informed coordinate receives 2*r identical covariate entries.
    # Hence Xi^T Xi/4 contributes r*a^2/2=xi-1 on that coordinate.  The
    # following is the posterior's global upper-curvature diagonal, equal to
    # the Hessian diagonal at beta=0; the Hessian itself varies with beta.
    upper_curvature_diagonal = np.concatenate(
        [np.full(Q0, 1.0 + replication_factor * row_scale**2 / 2.0),
         np.ones(k - Q0)]
    )
    intended = np.concatenate([np.full(Q0, xi), np.ones(k - Q0)])
    if np.max(np.abs(upper_curvature_diagonal - intended)) >= 1e-12:
        raise RuntimeError(
            "Upper-curvature matrix does not have the intended diagonal"
        )

    return {
        "k": k,
        "n": n,
        "xi": float(xi),
        "q0": Q0,
        "replication_factor": replication_factor,
        "row_scale": float(row_scale),
    }


def sample_stiff_coordinates(
    rng: np.random.Generator,
    number_draws: int,
    row_scale: float,
    replication_factor: int,
    xi: float,
) -> np.ndarray:
    """Exact rejection sampler for the informed posterior marginal."""
    draws = np.empty(number_draws)
    filled = 0
    while filled < number_draws:
        remaining = number_draws - filled
        proposed = min(1_000_000, max(10_000, int(np.ceil(2 * np.sqrt(xi) * remaining))))
        candidates = rng.standard_normal(proposed)
        eta = row_scale * candidates
        log_acceptance = 2 * replication_factor * (
            np.log(2.0) + eta / 2.0 - softplus(eta)
        )
        accepted = candidates[np.log(rng.random(proposed)) < np.minimum(0.0, log_acceptance)]
        keep = min(accepted.size, remaining)
        if keep:
            draws[filled:filled + keep] = accepted[:keep]
            filled += keep
    return draws


def sample_from_pi(
    rng: np.random.Generator,
    problem: dict[str, float | int],
    number_draws: int,
) -> np.ndarray:
    k = int(problem["k"])
    beta = rng.standard_normal((number_draws, k))
    beta[:, :Q0] = sample_stiff_coordinates(
        rng,
        number_draws * Q0,
        float(problem["row_scale"]),
        int(problem["replication_factor"]),
        float(problem["xi"]),
    ).reshape(number_draws, Q0)
    return beta


def sample_directions(
    rng: np.random.Generator,
    number_draws: int,
    k: int,
) -> np.ndarray:
    w = rng.standard_normal((number_draws, k))
    return w / np.linalg.norm(w, axis=1, keepdims=True)


def line_log_density(
    u: np.ndarray,
    theta: np.ndarray,
    w: np.ndarray,
    row_scale: float,
    replication_factor: int,
) -> np.ndarray:
    eta = row_scale * (theta + w * u[:, None])
    return -0.5 * u**2 + replication_factor * np.sum(
        eta - 2 * softplus(eta), axis=1
    )


def line_log_gradient(
    u: np.ndarray,
    theta: np.ndarray,
    w: np.ndarray,
    row_scale: float,
    replication_factor: int,
) -> np.ndarray:
    eta = row_scale * (theta + w * u[:, None])
    return -u + replication_factor * np.sum(
        -row_scale * w * np.tanh(eta / 2), axis=1
    )


def find_line_modes(
    theta: np.ndarray,
    w: np.ndarray,
    row_scale: float,
    replication_factor: int,
) -> np.ndarray:
    bound = 1 + replication_factor * row_scale * np.sum(np.abs(w), axis=1)
    lower = -bound
    upper = bound
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        gradient = line_log_gradient(
            midpoint, theta, w, row_scale, replication_factor
        )
        to_right = gradient > 0
        lower[to_right] = midpoint[to_right]
        upper[~to_right] = midpoint[~to_right]
    return 0.5 * (lower + upper)


def sample_bar_s_line(
    rng: np.random.Generator,
    theta: np.ndarray,
    w: np.ndarray,
    row_scale: float,
    replication_factor: int,
) -> tuple[np.ndarray, int, float]:
    number_draws = theta.shape[0]
    modes = find_line_modes(theta, w, row_scale, replication_factor)
    log_at_mode = line_log_density(
        modes, theta, w, row_scale, replication_factor
    )
    new_u = np.empty(number_draws)
    remaining = np.arange(number_draws)
    proposal_count = 0
    maximum_error = -np.inf
    while remaining.size:
        candidates = rng.normal(modes[remaining], 1.0)
        differences = candidates - modes[remaining]
        log_acceptance = line_log_density(
            candidates,
            theta[remaining],
            w[remaining],
            row_scale,
            replication_factor,
        ) - log_at_mode[remaining] + 0.5 * differences**2
        maximum_error = max(maximum_error, float(np.max(log_acceptance)))
        proposal_count += remaining.size
        accepted = np.log(rng.random(remaining.size)) <= np.minimum(0.0, log_acceptance)
        new_u[remaining[accepted]] = candidates[accepted]
        remaining = remaining[~accepted]
    return new_u, proposal_count, maximum_error


def apply_s(
    rng: np.random.Generator,
    beta: np.ndarray,
    w: np.ndarray,
    problem: dict[str, float | int],
) -> tuple[np.ndarray, int]:
    old_u = np.sum(w * beta, axis=1)
    theta = beta[:, :Q0] - w[:, :Q0] * old_u[:, None]
    xi = float(problem["xi"])
    number_scales = 1 + int(np.ceil(np.log2(xi)))
    scale_indices = rng.integers(0, number_scales, beta.shape[0])
    proposal_sd = np.sqrt(2.0**scale_indices / xi)
    proposed_u = old_u + rng.normal(0.0, proposal_sd)
    log_acceptance = line_log_density(
        proposed_u,
        theta,
        w[:, :Q0],
        float(problem["row_scale"]),
        int(problem["replication_factor"]),
    ) - line_log_density(
        old_u,
        theta,
        w[:, :Q0],
        float(problem["row_scale"]),
        int(problem["replication_factor"]),
    )
    accepted = np.log(rng.random(beta.shape[0])) < log_acceptance
    change = np.where(accepted, proposed_u - old_u, 0.0)
    endpoint_w = w[:, [0, -1]]
    return 0.5 * (endpoint_w * change[:, None])**2, int(np.sum(accepted))


def apply_bar_s(
    rng: np.random.Generator,
    beta: np.ndarray,
    w: np.ndarray,
    problem: dict[str, float | int],
) -> tuple[np.ndarray, int, float]:
    old_u = np.sum(w * beta, axis=1)
    theta = beta[:, :Q0] - w[:, :Q0] * old_u[:, None]
    new_u, proposals, maximum_error = sample_bar_s_line(
        rng,
        theta,
        w[:, :Q0],
        float(problem["row_scale"]),
        int(problem["replication_factor"]),
    )
    change = new_u - old_u
    endpoint_w = w[:, [0, -1]]
    return 0.5 * (endpoint_w * change[:, None])**2, proposals, maximum_error


def estimate_ratio(numerator: np.ndarray, denominator: np.ndarray) -> dict[str, np.ndarray]:
    numerator_mean = numerator.mean(axis=0)
    denominator_mean = denominator.mean(axis=0)
    estimate = numerator_mean / denominator_mean
    influence = numerator / numerator_mean - denominator / denominator_mean
    log_se = influence.std(axis=0, ddof=1) / np.sqrt(numerator.shape[0])
    return {
        "estimate": estimate,
        "log_se": log_se,
        "lower": estimate * np.exp(-1.96 * log_se),
        "upper": estimate * np.exp(1.96 * log_se),
    }


def estimate_problem(
    rng: np.random.Generator,
    k: int,
    n: int,
    xi: float,
    number_draws: int,
    batch_size: int,
) -> pd.DataFrame:
    problem = make_problem(k, n, xi)
    s_contributions = np.empty((number_draws, 2))
    bar_contributions = np.empty((number_draws, 2))
    variance_contributions = np.empty((number_draws, 2))
    accepted_s = 0
    bar_proposals = 0
    maximum_error = -np.inf

    active_coordinates = sample_stiff_coordinates(
        rng,
        number_draws * Q0,
        float(problem["row_scale"]),
        int(problem["replication_factor"]),
        float(problem["xi"]),
    ).reshape(number_draws, Q0)

    for first in range(0, number_draws, batch_size):
        last = min(number_draws, first + batch_size)
        current = last - first
        beta = rng.standard_normal((current, k))
        beta[:, :Q0] = active_coordinates[first:last]
        w_s = sample_directions(rng, current, k)
        w_bar = sample_directions(rng, current, k)
        s_batch, accepted = apply_s(rng, beta, w_s, problem)
        bar_batch, proposals, error = apply_bar_s(rng, beta, w_bar, problem)
        s_contributions[first:last] = s_batch
        bar_contributions[first:last] = bar_batch
        variance_contributions[first:last] = beta[:, [0, -1]]**2
        accepted_s += accepted
        bar_proposals += proposals
        maximum_error = max(maximum_error, error)

    ratio_bar = estimate_ratio(s_contributions, bar_contributions)
    ratio_variance = estimate_ratio(s_contributions, variance_contributions)
    rows = []
    for column, coordinate, coordinate_type in [
        (0, 1, "large likelihood curvature"),
        (1, k, "prior-scale curvature"),
    ]:
        rows.append({
            "k": k,
            "n": n,
            "xi": xi,
            "q0": Q0,
            "replication_factor": int(problem["replication_factor"]),
            "row_scale": float(problem["row_scale"]),
            "coordinate": coordinate,
            "coordinate_type": coordinate_type,
            "E_S": s_contributions[:, column].mean(),
            "E_bar_S": bar_contributions[:, column].mean(),
            "posterior_variance": variance_contributions[:, column].mean(),
            "ratio_S_bar_S": ratio_bar["estimate"][column],
            "ratio_S_bar_S_log_se": ratio_bar["log_se"][column],
            "ratio_S_bar_S_lower_95": ratio_bar["lower"][column],
            "ratio_S_bar_S_upper_95": ratio_bar["upper"][column],
            "ratio_S_posterior_variance": ratio_variance["estimate"][column],
            "ratio_S_posterior_variance_log_se": ratio_variance["log_se"][column],
            "ratio_S_posterior_variance_lower_95": ratio_variance["lower"][column],
            "ratio_S_posterior_variance_upper_95": ratio_variance["upper"][column],
            "acceptance_S": accepted_s / number_draws,
            "acceptance_bar_S_envelope": number_draws / bar_proposals,
            "maximum_log_envelope_error": maximum_error,
        })
    return pd.DataFrame(rows)


def fit_power(data: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    log_x = np.log(data[x].to_numpy(dtype=float))
    log_y = np.log(data[y].to_numpy(dtype=float))
    power, intercept = np.polyfit(log_x, log_y, 1)
    fitted = intercept + power * log_x
    total = np.sum((log_y - log_y.mean())**2)
    r_squared = 1.0 - np.sum((log_y - fitted)**2) / total if total > 0 else np.nan
    return float(power), float(r_squared)
