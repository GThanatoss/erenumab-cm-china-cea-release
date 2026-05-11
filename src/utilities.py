from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.stats import beta as beta_dist
from scipy.stats import truncnorm


MMD_MIN = 0.0
MMD_MAX = 28.0


@lru_cache(maxsize=64)
def _truncated_normal_params_for_moments(
    mean: float,
    sd: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    if not lower < mean < upper:
        raise ValueError("Target truncated-normal mean must lie strictly inside the truncation bounds")
    if sd <= 0:
        raise ValueError("Target truncated-normal sd must be positive")

    max_var = (mean - lower) * (upper - mean)
    if sd**2 > max_var + 1e-10:
        raise ValueError(
            f"Target truncated-normal sd={sd} is infeasible for mean={mean} "
            f"within [{lower}, {upper}]. Maximum feasible sd={np.sqrt(max_var):.6f}."
        )

    def moments_at(loc: float, scale: float) -> tuple[float, float]:
        a = (lower - loc) / scale
        b = (upper - loc) / scale
        return (
            float(truncnorm.mean(a, b, loc=loc, scale=scale)),
            float(truncnorm.std(a, b, loc=loc, scale=scale)),
        )

    def equations(params: np.ndarray) -> np.ndarray:
        loc = float(params[0])
        scale = float(np.exp(params[1]))
        current_mean, current_sd = moments_at(loc, scale)
        return np.array([current_mean - mean, current_sd - sd], dtype=float)

    span = upper - lower
    guesses = [
        (mean, sd),
        (mean, max(sd, span / 3.0)),
        ((mean + upper) / 2.0, max(sd, span / 2.0)),
        ((mean + lower) / 2.0, max(sd, span / 2.0)),
        (mean, max(sd, span)),
    ]
    best_params: tuple[float, float] | None = None
    best_error = float("inf")
    for loc_guess, scale_guess in guesses:
        result = root(equations, np.array([loc_guess, np.log(scale_guess)], dtype=float))
        loc = float(result.x[0])
        scale = float(np.exp(result.x[1]))
        residual = equations(np.array([loc, np.log(scale)], dtype=float))
        error = float(np.max(np.abs(residual)))
        if error < best_error:
            best_error = error
            best_params = (loc, scale)
        if error < 1e-8:
            return loc, scale

    if best_params is not None and best_error < 1e-6:
        return best_params
    raise RuntimeError(
        "Could not solve truncated-normal location and scale for target mean and sd"
    )


def sample_truncated_normal(
    mean: float,
    sd: float,
    lower: float,
    upper: float,
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    loc, scale = _truncated_normal_params_for_moments(mean, sd, lower, upper)
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm.rvs(a, b, loc=loc, scale=scale, size=size, random_state=rng)


def state_indices_from_mmd(mmd: np.ndarray, states: pd.DataFrame) -> np.ndarray:
    values = np.asarray(mmd, dtype=float)
    ordered_states = states.sort_values("lower_bound").reset_index(drop=True)
    indices = np.full(values.shape, len(states) - 1, dtype=int)
    for idx, row in ordered_states.iterrows():
        lower = float(row["lower_bound"])
        if idx < len(ordered_states) - 1:
            upper = float(ordered_states.iloc[idx + 1]["lower_bound"])
            mask = (values >= lower) & (values < upper)
        else:
            upper = float(row["upper_bound"])
            mask = (values >= lower) & (values <= upper)
        indices[mask] = int(idx)
    below = values < float(ordered_states.iloc[0]["lower_bound"])
    above = values > float(ordered_states.iloc[-1]["upper_bound"])
    indices[below] = 0
    indices[above] = len(ordered_states) - 1
    return indices


def utilities_from_mmd(
    mmd: np.ndarray,
    states: pd.DataFrame,
    utility_overrides: dict[str, float] | None = None,
) -> np.ndarray:
    state_indices = state_indices_from_mmd(mmd, states)
    utilities = states["utility"].astype(float).to_numpy(copy=True)
    if utility_overrides:
        for idx, row in states.reset_index(drop=True).iterrows():
            state_id = str(row["state_id"])
            state_label = str(row["state_name"]).split()[0].replace("-", "_")
            aliases = [f"utility_{state_id}", f"utility_{state_label}"]
            if state_label == "24_28":
                aliases.append("utility_24_30")
            for key in aliases:
                if key in utility_overrides:
                    utilities[idx] = float(utility_overrides[key])
    return utilities[state_indices]


def state_share_table(mmd: np.ndarray, states: pd.DataFrame) -> dict[str, float]:
    indices = state_indices_from_mmd(mmd, states)
    total = float(len(indices))
    shares: dict[str, float] = {}
    for idx, row in states.reset_index(drop=True).iterrows():
        shares[f"share_{row['state_id']}"] = float(np.sum(indices == idx) / total)
    return shares


def beta_pert_sample(
    rng: np.random.Generator,
    low: float,
    mode: float,
    high: float,
    lamb: float = 4.0,
) -> float:
    if high <= low:
        return float(mode)
    mode = min(max(mode, low), high)
    alpha = 1.0 + lamb * (mode - low) / (high - low)
    beta_param = 1.0 + lamb * (high - mode) / (high - low)
    return float(low + beta_dist.rvs(alpha, beta_param, random_state=rng) * (high - low))


def clip_probability(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


def clip_nonnegative(value: float) -> float:
    return float(max(value, 0.0))


def probability_to_cycle_probability(probability: float, cycles: float) -> float:
    probability = clip_probability(probability)
    if cycles <= 0:
        raise ValueError("Cycle count must be positive")
    return float(1.0 - (1.0 - probability) ** (1.0 / cycles))
