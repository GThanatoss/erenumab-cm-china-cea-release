from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.optimize import minimize_scalar
from scipy.stats import truncnorm

from .data import ModelInputs, target_value, trajectory_value
from .utilities import MMD_MAX, MMD_MIN, state_indices_from_mmd


@dataclass(frozen=True)
class CalibrationTarget:
    arm: str
    mean_mmd_week12: float
    responder50_week12: float
    mmd_lt15_week12: float
    observed_mmd_change_week12: float


@dataclass(frozen=True)
class CalibrationResult:
    arm: str
    method: str
    parameters: dict[str, float]
    week12_mmd: np.ndarray
    mean_mmd_week12: float
    responder50_week12: float
    mmd_lt15_week12: float
    target_mean_mmd_week12: float
    target_responder50_week12: float
    target_mmd_lt15_week12: float
    objective_value: float

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "method": self.method,
            "target_mean_mmd_week12": self.target_mean_mmd_week12,
            "simulated_mean_mmd_week12": self.mean_mmd_week12,
            "residual_mean_mmd_week12": self.mean_mmd_week12 - self.target_mean_mmd_week12,
            "target_responder50_week12": self.target_responder50_week12,
            "simulated_responder50_week12": self.responder50_week12,
            "residual_responder50_week12": self.responder50_week12 - self.target_responder50_week12,
            "target_mmd_lt15_week12": self.target_mmd_lt15_week12,
            "simulated_mmd_lt15_week12": self.mmd_lt15_week12,
            "residual_mmd_lt15_week12": self.mmd_lt15_week12 - self.target_mmd_lt15_week12,
            "objective_value": self.objective_value,
            **self.parameters,
        }


def calibration_target(
    inputs: ModelInputs,
    arm: str,
    overrides: dict[str, Any] | None = None,
) -> CalibrationTarget:
    mean_mmd = target_value(inputs, arm, "mean_mmd_week12", overrides)
    mmd_change_key = f"mmd_change_w12_{arm}"
    if overrides and mmd_change_key in overrides:
        baseline_cfg = inputs.config["baseline_mmd_distribution"]
        baseline_mean = float(overrides.get("baseline_mmd_mean", baseline_cfg["mean"]))
        mean_mmd = baseline_mean + float(overrides[mmd_change_key])
    return CalibrationTarget(
        arm=arm,
        mean_mmd_week12=mean_mmd,
        responder50_week12=target_value(inputs, arm, "responder50_week12", overrides),
        mmd_lt15_week12=target_value(inputs, arm, "mmd_lt15_week12", overrides),
        observed_mmd_change_week12=trajectory_value(inputs, arm, "week12", "mmd_change", overrides),
    )


def _metrics(baseline_mmd: np.ndarray, week12_mmd: np.ndarray) -> tuple[float, float, float]:
    mean_mmd = float(np.mean(week12_mmd))
    responder50 = float(np.mean(week12_mmd <= 0.5 * baseline_mmd))
    mmd_lt15 = float(np.mean(week12_mmd < 15.0))
    return mean_mmd, responder50, mmd_lt15


def _objective(metrics: tuple[float, float, float], target: CalibrationTarget) -> float:
    mean_mmd, responder50, mmd_lt15 = metrics
    residuals = np.array(
        [
            (mean_mmd - target.mean_mmd_week12) / 0.25,
            (responder50 - target.responder50_week12) / 0.015,
            (mmd_lt15 - target.mmd_lt15_week12) / 0.015,
        ],
        dtype=float,
    )
    return float(np.sum(residuals**2))


def _calibration_start_shares(inputs: ModelInputs, arm: str) -> np.ndarray | None:
    if arm == inputs.intervention_arm:
        column = "calibration_start_erenumab"
    elif arm == inputs.comparator_arm:
        column = "calibration_start_bsc"
    else:
        return None
    if column not in inputs.mmd_states.columns:
        return None
    return inputs.mmd_states[column].astype(float).to_numpy()


def _target_state_shares(
    inputs: ModelInputs,
    arm: str,
    target: CalibrationTarget | None = None,
) -> np.ndarray | None:
    target_shares = _calibration_start_shares(inputs, arm)
    if target_shares is None:
        return None
    if target is None:
        return target_shares
    current_lt15 = float(np.sum(target_shares[:3]))
    desired_lt15 = min(max(float(target.mmd_lt15_week12), 0.0), 1.0)
    adjusted = target_shares.astype(float, copy=True)
    if current_lt15 > 0:
        adjusted[:3] *= desired_lt15 / current_lt15
    if current_lt15 < 1:
        adjusted[3:] *= (1.0 - desired_lt15) / (1.0 - current_lt15)
    total = float(np.sum(adjusted))
    if total > 0:
        adjusted /= total
    return adjusted


def _state_share_penalty(
    inputs: ModelInputs,
    arm: str,
    week12_mmd: np.ndarray,
    target: CalibrationTarget | None = None,
) -> float:
    target_shares = _target_state_shares(inputs, arm, target)
    if target_shares is None:
        return 0.0
    indices = state_indices_from_mmd(week12_mmd, inputs.mmd_states)
    simulated = np.array([np.mean(indices == idx) for idx in range(len(target_shares))], dtype=float)
    residuals = (simulated - target_shares) / 0.05
    return float(0.75 * np.sum(residuals**2))


def _single_normal_week12(
    baseline_mmd: np.ndarray,
    z: np.ndarray,
    mean_change: float,
    sd_change: float,
) -> np.ndarray:
    return np.clip(baseline_mmd + mean_change + sd_change * z, MMD_MIN, MMD_MAX)


def _mixture_week12(
    baseline_mmd: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    u: np.ndarray,
    params: np.ndarray,
) -> np.ndarray:
    weight, mean1, sd1, mean2, sd2 = params
    changes = np.where(u <= weight, mean1 + sd1 * z1, mean2 + sd2 * z2)
    return np.clip(baseline_mmd + changes, MMD_MIN, MMD_MAX)


def _continuous_random_draws(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.random(n),
        rng.random(n),
        rng.random(n),
        rng.standard_normal(n),
    )


def _truncated_normal_ppf(
    probabilities: np.ndarray,
    mean: float,
    sd: float,
    lower: float = MMD_MIN,
    upper: float = MMD_MAX,
) -> np.ndarray:
    sd = max(float(sd), 1e-6)
    a = (lower - mean) / sd
    b = (upper - mean) / sd
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return truncnorm.ppf(clipped, a, b, loc=mean, scale=sd)


def _rank_ordered_assignment(
    baseline_mmd: np.ndarray,
    values: np.ndarray,
    rank_weight: float,
    rank_noise: np.ndarray,
) -> np.ndarray:
    sorted_values = np.sort(values)
    baseline_rank = baseline_mmd.argsort().argsort().astype(float) / max(len(baseline_mmd) - 1, 1)
    rank_weight = float(np.clip(rank_weight, -1.0, 1.0))
    if abs(rank_weight) >= 0.999999:
        score = -rank_weight * baseline_rank
    else:
        score = (1.0 - abs(rank_weight)) * rank_noise - rank_weight * baseline_rank
    patient_order = np.argsort(score)
    candidate = np.empty_like(sorted_values)
    candidate[patient_order] = sorted_values
    return candidate


def _smooth_continuous_values_from_parameters(
    baseline_mmd: np.ndarray,
    params: dict[str, float],
    draw_seed: int,
) -> np.ndarray:
    u_mix, u_low, u_high, rank_noise = _continuous_random_draws(len(baseline_mmd), draw_seed)
    weight = float(np.clip(params["mix_weight_low"], 0.001, 0.999))
    low_values = _truncated_normal_ppf(
        u_low,
        float(params["component_mean_low"]),
        float(params["component_sd_low"]),
    )
    high_values = _truncated_normal_ppf(
        u_high,
        float(params["component_mean_high"]),
        float(params["component_sd_high"]),
    )
    marginal_values = np.where(u_mix <= weight, low_values, high_values)
    return _rank_ordered_assignment(
        baseline_mmd,
        marginal_values,
        float(params["rank_correlation_weight"]),
        rank_noise,
    )


def _histogram_smoothness_penalty(week12_mmd: np.ndarray, bin_width: float = 1.0) -> float:
    bins = np.arange(MMD_MIN, MMD_MAX + bin_width, bin_width)
    counts, _ = np.histogram(week12_mmd, bins=bins)
    probabilities = counts.astype(float) / max(len(week12_mmd), 1)
    if len(probabilities) < 3:
        return 0.0
    second_difference = probabilities[2:] - 2.0 * probabilities[1:-1] + probabilities[:-2]
    excess = np.maximum(np.abs(second_difference) - 0.004, 0.0)
    return float(np.sum(excess**2) * 100000.0)


def _local_continuity_penalty(week12_mmd: np.ndarray) -> float:
    counts, _ = np.histogram(week12_mmd, bins=np.arange(0.0, 31.0, 1.0))
    probabilities = counts.astype(float) / max(len(week12_mmd), 1)
    left_density = float(np.mean(probabilities[4:10]))
    middle_density = float(np.mean(probabilities[10:15]))
    right_density = float(np.mean(probabilities[15:20]))
    minimum_neighbor = min(left_density, right_density)
    target_middle_density = 0.45 * minimum_neighbor
    return float(max(0.0, target_middle_density - middle_density) ** 2 * 100000.0)


def _continuous_parameter_penalty(params: np.ndarray) -> float:
    _, mean_low, log_sd_low, mean_high, log_sd_high, _ = params
    sd_low = float(np.exp(log_sd_low))
    sd_high = float(np.exp(log_sd_high))
    separation = abs(float(mean_high) - float(mean_low))
    penalties = [
        (max(0.0, separation - 18.0) / 4.0) ** 2,
        (max(0.0, 1.5 - sd_low) / 1.0) ** 2,
        (max(0.0, 1.5 - sd_high) / 1.0) ** 2,
        (max(0.0, sd_low - 10.0) / 2.0) ** 2,
        (max(0.0, sd_high - 10.0) / 2.0) ** 2,
    ]
    return float(sum(penalties))


def _smooth_continuous_week12(
    baseline_mmd: np.ndarray,
    draw_seed: int,
    target: CalibrationTarget,
) -> tuple[np.ndarray, dict[str, float], float]:
    u_mix, u_low, u_high, rank_noise = _continuous_random_draws(len(baseline_mmd), draw_seed)

    def values_from_x(x: np.ndarray) -> np.ndarray:
        weight, mean_low, log_sd_low, mean_high, log_sd_high, rank_weight = x
        low_values = _truncated_normal_ppf(u_low, float(mean_low), float(np.exp(log_sd_low)))
        high_values = _truncated_normal_ppf(u_high, float(mean_high), float(np.exp(log_sd_high)))
        marginal_values = np.where(u_mix <= float(weight), low_values, high_values)
        return _rank_ordered_assignment(
            baseline_mmd,
            marginal_values,
            float(rank_weight),
            rank_noise,
        )

    def objective(x: np.ndarray) -> float:
        week12 = values_from_x(x)
        mean_mmd, responder50, mmd_lt15 = _metrics(baseline_mmd, week12)
        clinical_loss = (
            ((mean_mmd - target.mean_mmd_week12) / 0.03) ** 2
            + ((responder50 - target.responder50_week12) / 0.006) ** 2
            + 0.60 * ((mmd_lt15 - target.mmd_lt15_week12) / 0.03) ** 2
        )
        return float(
            clinical_loss
            + 0.05 * _histogram_smoothness_penalty(week12)
            + 0.20 * _local_continuity_penalty(week12)
            + 0.05 * _continuous_parameter_penalty(x)
        )

    bounds = [
        (0.25, 0.85),
        (4.0, 12.0),
        (np.log(2.0), np.log(7.0)),
        (13.0, 21.0),
        (np.log(2.0), np.log(7.0)),
        (-1.0, 1.0),
    ]
    seed = int(draw_seed % (2**31 - 1))
    de_fit = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        maxiter=60,
        popsize=7,
        polish=False,
        tol=1e-4,
        updating="immediate",
    )
    local_fit = minimize(
        objective,
        np.asarray(de_fit.x, dtype=float),
        method="Nelder-Mead",
        options={"maxiter": 900},
    )
    de_x = np.asarray(de_fit.x, dtype=float)
    local_x = np.asarray(local_fit.x, dtype=float)
    for candidate in (de_x, local_x):
        for idx, (lower, upper) in enumerate(bounds):
            candidate[idx] = min(max(float(candidate[idx]), lower), upper)
    best_x = local_x if objective(local_x) <= objective(de_x) else de_x
    week12 = values_from_x(best_x)
    params = {
        "mix_weight_low": float(best_x[0]),
        "component_mean_low": float(best_x[1]),
        "component_sd_low": float(np.exp(best_x[2])),
        "component_mean_high": float(best_x[3]),
        "component_sd_high": float(np.exp(best_x[4])),
        "rank_correlation_weight": float(best_x[5]),
        "continuous_draw_seed": float(draw_seed),
        "smoothness_penalty": _histogram_smoothness_penalty(week12),
        "local_continuity_penalty": _local_continuity_penalty(week12),
    }
    return week12, params, objective(best_x)


def _state_counts(n: int, shares: np.ndarray) -> np.ndarray:
    raw = shares * n
    counts = np.floor(raw).astype(int)
    remainder = int(n - counts.sum())
    if remainder > 0:
        order = np.argsort(raw - counts)[::-1]
        counts[order[:remainder]] += 1
    elif remainder < 0:
        order = np.argsort(raw - counts)
        for idx in order[: abs(remainder)]:
            if counts[idx] > 0:
                counts[idx] -= 1
    return counts


def _adjust_within_bounds(values: np.ndarray, lower: float, upper: float, target_mean: float) -> np.ndarray:
    adjusted = values.astype(float, copy=True)
    for _ in range(30):
        diff = target_mean - float(np.mean(adjusted))
        if abs(diff) < 1e-8:
            break
        adjusted = np.clip(adjusted + diff, lower, upper)
    return adjusted


def _state_informed_week12(
    inputs: ModelInputs,
    arm: str,
    baseline_mmd: np.ndarray,
    rng: np.random.Generator,
    target: CalibrationTarget,
) -> tuple[np.ndarray, dict[str, float], float]:
    shares = _target_state_shares(inputs, arm, target)
    if shares is None:
        raise ValueError(f"Missing calibration start shares for {arm}")

    counts = _state_counts(len(baseline_mmd), shares)
    state_values: list[np.ndarray] = []
    state_bounds: list[tuple[float, float]] = []
    for count, (_, row) in zip(counts, inputs.mmd_states.reset_index(drop=True).iterrows()):
        if count <= 0:
            continue
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        representative = float(row["representative_mmd"])
        sd = max((upper - lower) / 8.0, 0.05)
        values = rng.normal(representative, sd, count)
        values = np.clip(values, lower, upper)
        values = _adjust_within_bounds(values, lower, upper, representative)
        state_values.append(values)
        state_bounds.extend([(lower, upper)] * count)

    values = np.concatenate(state_values)
    bounds = np.array(state_bounds, dtype=float)
    for _ in range(30):
        diff = target.mean_mmd_week12 - float(np.mean(values))
        if abs(diff) < 1e-8:
            break
        values = np.clip(values + diff, bounds[:, 0], bounds[:, 1])

    sorted_values = np.sort(values)
    baseline_rank = baseline_mmd.argsort().argsort().astype(float) / max(len(baseline_mmd) - 1, 1)
    random_score = rng.standard_normal(len(baseline_mmd))
    best_week12: np.ndarray | None = None
    best_obj = float("inf")
    best_rho = 0.0
    for rho in np.linspace(-1.0, 1.0, 101):
        score = (1.0 - abs(float(rho))) * random_score - float(rho) * baseline_rank
        patient_order = np.argsort(score)
        candidate = np.empty_like(sorted_values)
        candidate[patient_order] = sorted_values
        obj = _objective(_metrics(baseline_mmd, candidate), target) + _state_share_penalty(
            inputs, arm, candidate, target
        )
        if obj < best_obj:
            best_obj = obj
            best_week12 = candidate
            best_rho = float(rho)

    assert best_week12 is not None
    params = {
        "rank_correlation_weight": best_rho,
        "uses_calibration_start_state_shares": 1.0,
    }
    return best_week12, params, best_obj


def frozen_calibration_available(inputs: ModelInputs) -> bool:
    if inputs.calibration_parameters.empty:
        return False
    columns = set(inputs.calibration_parameters.columns)
    common_required = {
        "arm",
        "method",
        "target_mean_mmd_week12",
        "target_responder50_week12",
        "target_mmd_lt15_week12",
        "rank_correlation_weight",
    }
    continuous_required = common_required | {
        "mix_weight_low",
        "component_mean_low",
        "component_sd_low",
        "component_mean_high",
        "component_sd_high",
        "continuous_draw_seed",
    }
    state_required = common_required | {
        "share_S1",
        "share_S2",
        "share_S3",
        "share_S4",
        "share_S5",
        "share_S6",
    }
    return continuous_required.issubset(columns) or state_required.issubset(columns)


def _frozen_row(inputs: ModelInputs, arm: str) -> dict[str, Any]:
    if not frozen_calibration_available(inputs):
        raise ValueError("Frozen calibration table is missing or incomplete")
    rows = inputs.calibration_parameters.loc[inputs.calibration_parameters["arm"] == arm]
    if rows.empty:
        raise KeyError(f"Frozen calibration parameters are missing for arm: {arm}")
    return rows.iloc[0].to_dict()


def _shares_from_frozen_row(
    row: dict[str, Any],
    target: CalibrationTarget,
    keep_frozen_mmd_lt15: bool = False,
) -> np.ndarray:
    shares = np.array([float(row[f"share_S{i}"]) for i in range(1, 7)], dtype=float)
    if keep_frozen_mmd_lt15:
        return shares / float(np.sum(shares))
    current_lt15 = float(np.sum(shares[:3]))
    desired_lt15 = min(max(target.mmd_lt15_week12, 0.0), 1.0)
    adjusted = shares.copy()
    if current_lt15 > 0:
        adjusted[:3] *= desired_lt15 / current_lt15
    if current_lt15 < 1:
        adjusted[3:] *= (1.0 - desired_lt15) / (1.0 - current_lt15)
    total = float(np.sum(adjusted))
    if total <= 0:
        raise ValueError("Frozen calibration shares sum to zero")
    return adjusted / total


def _state_values_for_counts(
    inputs: ModelInputs,
    counts: np.ndarray,
    target_mean: float,
    rng: np.random.Generator | None = None,
    stochastic: bool = False,
    sd_divisor: float = 8.0,
) -> np.ndarray:
    state_values: list[np.ndarray] = []
    bounds: list[tuple[float, float]] = []
    for count, (_, row) in zip(counts, inputs.mmd_states.reset_index(drop=True).iterrows()):
        if count <= 0:
            continue
        representative = float(row["representative_mmd"])
        lower = float(row["lower_bound"])
        upper = float(row["upper_bound"])
        if stochastic:
            if rng is None:
                raise ValueError("A random generator is required for stochastic state values")
            sd = max((upper - lower) / sd_divisor, 0.05)
            values = rng.normal(representative, sd, int(count))
            values = np.clip(values, lower, upper)
            values = _adjust_within_bounds(values, lower, upper, representative)
        else:
            values = np.full(int(count), representative, dtype=float)
        state_values.append(values)
        bounds.extend([(lower, upper)] * int(count))
    value_array = np.concatenate(state_values)
    bound_array = np.asarray(bounds, dtype=float)
    for _ in range(50):
        diff = target_mean - float(np.mean(value_array))
        if abs(diff) < 1e-10:
            break
        value_array = np.clip(value_array + diff, bound_array[:, 0], bound_array[:, 1])
    return value_array


def _shift_to_global_mean(values: np.ndarray, target_mean: float) -> np.ndarray:
    shifted = values.astype(float, copy=True)
    for _ in range(50):
        diff = target_mean - float(np.mean(shifted))
        if abs(diff) < 1e-10:
            break
        shifted = np.clip(shifted + diff, MMD_MIN, MMD_MAX)
    return shifted


def _is_continuous_frozen_row(row: dict[str, Any]) -> bool:
    return all(
        key in row and not pd.isna(row.get(key))
        for key in [
            "mix_weight_low",
            "component_mean_low",
            "component_sd_low",
            "component_mean_high",
            "component_sd_high",
            "continuous_draw_seed",
        ]
    )


def _scale_to_mmd_lt15_if_needed(
    values: np.ndarray,
    target: CalibrationTarget,
    row: dict[str, Any],
) -> np.ndarray:
    frozen_target = float(row["target_mmd_lt15_week12"])
    if abs(target.mmd_lt15_week12 - frozen_target) <= 1e-9:
        return values

    target_mean = target.mean_mmd_week12

    def transformed(scale: float) -> np.ndarray:
        scaled = target_mean + float(scale) * (values - target_mean)
        scaled = np.clip(scaled, MMD_MIN, MMD_MAX)
        return _shift_to_global_mean(scaled, target_mean)

    def objective(scale: float) -> float:
        candidate = transformed(scale)
        return float((np.mean(candidate < 15.0) - target.mmd_lt15_week12) ** 2 + 0.0025 * (scale - 1.0) ** 2)

    fit = minimize_scalar(objective, bounds=(0.45, 1.80), method="bounded")
    return transformed(float(fit.x))


def _optimize_rank_weight_if_needed(
    baseline_mmd: np.ndarray,
    values: np.ndarray,
    target: CalibrationTarget,
    row: dict[str, Any],
) -> tuple[np.ndarray, float]:
    base_rank_weight = float(row["rank_correlation_weight"])
    frozen_responder = float(row["target_responder50_week12"])
    if abs(target.responder50_week12 - frozen_responder) <= 1e-9:
        return values, base_rank_weight

    draw_seed = int(row.get("continuous_draw_seed") or row.get("calibration_seed") or 314159)
    _, _, _, rank_noise = _continuous_random_draws(len(baseline_mmd), draw_seed)
    best_week12: np.ndarray | None = None
    best_weight = base_rank_weight
    best_objective = float("inf")
    for rank_weight in np.linspace(-1.0, 1.0, 81):
        candidate = _rank_ordered_assignment(baseline_mmd, values, float(rank_weight), rank_noise)
        objective = _objective(_metrics(baseline_mmd, candidate), target)
        if objective < best_objective:
            best_objective = objective
            best_week12 = candidate
            best_weight = float(rank_weight)
    assert best_week12 is not None
    return best_week12, best_weight


def _apply_frozen_continuous_calibration(
    row: dict[str, Any],
    target: CalibrationTarget,
    baseline_mmd: np.ndarray,
    overrides: dict[str, Any] | None = None,
) -> CalibrationResult:
    params = {
        "mix_weight_low": float(row["mix_weight_low"]),
        "component_mean_low": float(row["component_mean_low"]),
        "component_sd_low": float(row["component_sd_low"]),
        "component_mean_high": float(row["component_mean_high"]),
        "component_sd_high": float(row["component_sd_high"]),
        "rank_correlation_weight": float(row["rank_correlation_weight"]),
    }
    draw_seed = int(row.get("continuous_draw_seed") or row.get("calibration_seed") or 314159)
    week12_mmd = _smooth_continuous_values_from_parameters(baseline_mmd, params, draw_seed)

    if abs(target.mean_mmd_week12 - float(row["target_mean_mmd_week12"])) > 1e-9:
        week12_mmd = _shift_to_global_mean(week12_mmd, target.mean_mmd_week12)

    mmd_change_only = bool(overrides and overrides.get("_mmd_change_only_recalibration"))
    if not mmd_change_only:
        week12_mmd = _scale_to_mmd_lt15_if_needed(week12_mmd, target, row)
        week12_mmd, params["rank_correlation_weight"] = _optimize_rank_weight_if_needed(
            baseline_mmd,
            week12_mmd,
            target,
            row,
        )

    metrics = _metrics(baseline_mmd, week12_mmd)
    objective_value = (
        _objective(metrics, target)
        + 0.05 * _histogram_smoothness_penalty(week12_mmd)
        + 0.20 * _local_continuity_penalty(week12_mmd)
    )
    parameters = {
        "uses_frozen_calibration_parameters": 1.0,
        "uses_continuous_week12_distribution": 1.0,
        "mix_weight_low": params["mix_weight_low"],
        "component_mean_low": params["component_mean_low"],
        "component_sd_low": params["component_sd_low"],
        "component_mean_high": params["component_mean_high"],
        "component_sd_high": params["component_sd_high"],
        "rank_correlation_weight": params["rank_correlation_weight"],
        "continuous_draw_seed": float(draw_seed),
        "smoothness_penalty": _histogram_smoothness_penalty(week12_mmd),
        "local_continuity_penalty": _local_continuity_penalty(week12_mmd),
    }
    return CalibrationResult(
        arm=target.arm,
        method=str(row["method"]),
        parameters=parameters,
        week12_mmd=week12_mmd,
        mean_mmd_week12=metrics[0],
        responder50_week12=metrics[1],
        mmd_lt15_week12=metrics[2],
        target_mean_mmd_week12=target.mean_mmd_week12,
        target_responder50_week12=target.responder50_week12,
        target_mmd_lt15_week12=target.mmd_lt15_week12,
        objective_value=objective_value,
    )


def apply_frozen_calibration(
    inputs: ModelInputs,
    arm: str,
    baseline_mmd: np.ndarray,
    overrides: dict[str, Any] | None = None,
) -> CalibrationResult:
    row = _frozen_row(inputs, arm)
    target = calibration_target(inputs, arm, overrides)
    if _is_continuous_frozen_row(row) and not (overrides and overrides.get("_use_state_calibration")):
        return _apply_frozen_continuous_calibration(row, target, baseline_mmd, overrides)

    mmd_change_only = bool(overrides and overrides.get("_mmd_change_only_recalibration"))
    shares = _shares_from_frozen_row(row, target, keep_frozen_mmd_lt15=mmd_change_only)
    counts = _state_counts(len(baseline_mmd), shares)
    value_target_mean = float(row["target_mean_mmd_week12"]) if mmd_change_only else target.mean_mmd_week12
    value_generation = str(row.get("value_generation") or "")
    stochastic_values = value_generation.startswith("stochastic")
    state_value_rng = None
    if stochastic_values:
        state_value_seed = int(row.get("state_value_seed") or row.get("calibration_seed") or 314159)
        state_value_rng = np.random.default_rng(state_value_seed)
    sd_divisor = float(row.get("within_state_sd_divisor") or 8.0)
    sorted_values = np.sort(
        _state_values_for_counts(
            inputs,
            counts,
            value_target_mean,
            rng=state_value_rng,
            stochastic=stochastic_values,
            sd_divisor=sd_divisor,
        )
    )

    baseline_rank = baseline_mmd.argsort().argsort().astype(float) / max(len(baseline_mmd) - 1, 1)
    base_rank_weight = float(row["rank_correlation_weight"])
    seed = int(row.get("calibration_seed") or 314159)
    random_score = np.random.default_rng(seed).standard_normal(len(baseline_mmd))

    def assign_with_rank(rank_weight: float) -> np.ndarray:
        if abs(rank_weight) >= 0.999999:
            score = -rank_weight * baseline_rank
        else:
            score = (1.0 - abs(rank_weight)) * random_score - rank_weight * baseline_rank
        patient_order = np.argsort(score)
        candidate = np.empty_like(sorted_values)
        candidate[patient_order] = sorted_values
        return candidate

    frozen_responder = float(row["target_responder50_week12"])
    if (not mmd_change_only) and abs(target.responder50_week12 - frozen_responder) > 1e-9:
        best_week12 = None
        best_rank_weight = base_rank_weight
        best_objective = float("inf")
        for candidate_rank in np.linspace(-1.0, 1.0, 101):
            candidate = assign_with_rank(float(candidate_rank))
            candidate_objective = _objective(_metrics(baseline_mmd, candidate), target)
            if candidate_objective < best_objective:
                best_objective = candidate_objective
                best_week12 = candidate
                best_rank_weight = float(candidate_rank)
        assert best_week12 is not None
        week12_mmd = best_week12
        rank_weight = best_rank_weight
    else:
        rank_weight = base_rank_weight
        week12_mmd = assign_with_rank(rank_weight)
    if mmd_change_only:
        week12_mmd = _shift_to_global_mean(week12_mmd, target.mean_mmd_week12)

    metrics = _metrics(baseline_mmd, week12_mmd)
    objective_value = _objective(metrics, target) + _state_share_penalty(inputs, arm, week12_mmd, target)
    parameters = {
        "rank_correlation_weight": rank_weight,
        "uses_frozen_calibration_parameters": 1.0,
        "calibration_seed": float(row.get("calibration_seed") or 314159),
        "state_value_seed": float(row.get("state_value_seed") or row.get("calibration_seed") or 314159),
    }
    for idx in range(1, 7):
        parameters[f"frozen_share_S{idx}"] = float(row[f"share_S{idx}"])
    return CalibrationResult(
        arm=arm,
        method=str(row["method"]),
        parameters=parameters,
        week12_mmd=week12_mmd,
        mean_mmd_week12=metrics[0],
        responder50_week12=metrics[1],
        mmd_lt15_week12=metrics[2],
        target_mean_mmd_week12=target.mean_mmd_week12,
        target_responder50_week12=target.responder50_week12,
        target_mmd_lt15_week12=target.mmd_lt15_week12,
        objective_value=objective_value,
    )


def frozen_calibration_rows(
    inputs: ModelInputs,
    n_patients: int,
    seed: int,
    calibration_method: str = "continuous",
) -> pd.DataFrame:
    from .simulation import SimulationSettings, _baseline_mmd

    settings = SimulationSettings(n_patients=n_patients, seed=seed, scenario_id="basecase")
    baseline = _baseline_mmd(inputs, settings, overrides=None)
    rows: list[dict[str, Any]] = []
    for idx, arm in enumerate([inputs.intervention_arm, inputs.comparator_arm]):
        draw_seed = seed + 1000 + idx
        rng = np.random.default_rng(draw_seed)
        overrides = {"_continuous_draw_seed": draw_seed}
        if calibration_method == "state":
            overrides["_use_state_calibration"] = True
        result = calibrate_week12_distribution(
            inputs,
            arm,
            baseline,
            rng,
            overrides=overrides,
        )
        target = calibration_target(inputs, arm, overrides=None)
        row: dict[str, Any] = {
            "parameter_id": f"cal_week12_{arm}",
            "arm": arm,
            "method": (
                "frozen_state_distribution_rank_calibration"
                if calibration_method == "state"
                else "frozen_smooth_continuous_mmd_calibration"
            ),
            "target_mean_mmd_week12": target.mean_mmd_week12,
            "target_responder50_week12": target.responder50_week12,
            "target_mmd_lt15_week12": target.mmd_lt15_week12,
            "rank_correlation_weight": float(result.parameters.get("rank_correlation_weight", np.nan)),
            "calibration_seed": seed,
            "continuous_draw_seed": draw_seed if calibration_method != "state" else np.nan,
            "calibration_n_patients": n_patients,
            "simulated_mean_mmd_week12": result.mean_mmd_week12,
            "simulated_responder50_week12": result.responder50_week12,
            "simulated_mmd_lt15_week12": result.mmd_lt15_week12,
            "objective_value": result.objective_value,
            "smoothness_penalty": float(result.parameters.get("smoothness_penalty", np.nan)),
            "local_continuity_penalty": float(result.parameters.get("local_continuity_penalty", np.nan)),
            "source_id": "S_MODEL_CALIBRATION",
            "status": "locked",
            "notes": (
                "Frozen continuous Week 12 MMD calibration parameters generated from current model inputs; "
                "MMD states are used only for utility mapping and diagnostics. Rerun calibrate_model.py "
                "after changing baseline MMD distribution or Week 12 calibration targets."
                if calibration_method != "state"
                else "Legacy frozen state-distribution calibration parameters; retained for validation scenario only."
            ),
        }
        if calibration_method == "state":
            shares = _target_state_shares(inputs, arm, target)
            if shares is None:
                raise ValueError(f"Cannot freeze calibration because state shares are missing for {arm}")
            row.update(
                {
                    "share_S1": float(shares[0]),
                    "share_S2": float(shares[1]),
                    "share_S3": float(shares[2]),
                    "share_S4": float(shares[3]),
                    "share_S5": float(shares[4]),
                    "share_S6": float(shares[5]),
                    "value_generation": "stochastic_within_state_normal_shifted_to_target_mean",
                    "state_value_seed": seed + 1000 + idx,
                    "within_state_sd_divisor": 8.0,
                }
            )
        else:
            row.update(
                {
                    "distribution_family": "two_component_truncated_normal_marginal",
                    "mix_weight_low": float(result.parameters["mix_weight_low"]),
                    "component_mean_low": float(result.parameters["component_mean_low"]),
                    "component_sd_low": float(result.parameters["component_sd_low"]),
                    "component_mean_high": float(result.parameters["component_mean_high"]),
                    "component_sd_high": float(result.parameters["component_sd_high"]),
                }
            )
        indices = state_indices_from_mmd(result.week12_mmd, inputs.mmd_states)
        for state_idx, state in inputs.mmd_states.reset_index(drop=True).iterrows():
            row[f"diagnostic_share_{state['state_id']}"] = float(np.mean(indices == state_idx))
        rows.append(row)
    rows_df = pd.DataFrame(rows)
    frozen_inputs = replace(inputs, calibration_parameters=rows_df)
    for row_idx, row in rows_df.iterrows():
        applied = apply_frozen_calibration(
            frozen_inputs,
            str(row["arm"]),
            baseline,
            overrides=None,
        )
        rows_df.loc[row_idx, "simulated_mean_mmd_week12"] = applied.mean_mmd_week12
        rows_df.loc[row_idx, "simulated_responder50_week12"] = applied.responder50_week12
        rows_df.loc[row_idx, "simulated_mmd_lt15_week12"] = applied.mmd_lt15_week12
        rows_df.loc[row_idx, "objective_value"] = applied.objective_value
    return rows_df


def calibrate_week12_distribution(
    inputs: ModelInputs,
    arm: str,
    baseline_mmd: np.ndarray,
    rng: np.random.Generator,
    overrides: dict[str, Any] | None = None,
    allow_mixture: bool = True,
) -> CalibrationResult:
    target = calibration_target(inputs, arm, overrides)
    if overrides and overrides.get("_mmd_change_only_recalibration") and frozen_calibration_available(inputs):
        return apply_frozen_calibration(inputs, arm, baseline_mmd, overrides)
    if overrides and overrides.get("_use_state_calibration") and _target_state_shares(inputs, arm, target) is not None:
        state_week12, state_params, state_obj = _state_informed_week12(inputs, arm, baseline_mmd, rng, target)
        state_metrics = _metrics(baseline_mmd, state_week12)
        return CalibrationResult(
            arm=arm,
            method="state_distribution_rank_calibration",
            parameters=state_params,
            week12_mmd=state_week12,
            mean_mmd_week12=state_metrics[0],
            responder50_week12=state_metrics[1],
            mmd_lt15_week12=state_metrics[2],
            target_mean_mmd_week12=target.mean_mmd_week12,
            target_responder50_week12=target.responder50_week12,
            target_mmd_lt15_week12=target.mmd_lt15_week12,
            objective_value=state_obj,
        )

    draw_seed = (
        int(overrides["_continuous_draw_seed"])
        if overrides and "_continuous_draw_seed" in overrides
        else int(rng.integers(0, 2**31 - 1))
    )
    continuous_week12, continuous_params, continuous_obj = _smooth_continuous_week12(
        baseline_mmd,
        draw_seed,
        target,
    )
    continuous_metrics = _metrics(baseline_mmd, continuous_week12)
    return CalibrationResult(
        arm=arm,
        method="smooth_continuous_mmd_calibration",
        parameters=continuous_params,
        week12_mmd=continuous_week12,
        mean_mmd_week12=continuous_metrics[0],
        responder50_week12=continuous_metrics[1],
        mmd_lt15_week12=continuous_metrics[2],
        target_mean_mmd_week12=target.mean_mmd_week12,
        target_responder50_week12=target.responder50_week12,
        target_mmd_lt15_week12=target.mmd_lt15_week12,
        objective_value=continuous_obj,
    )
    n = len(baseline_mmd)
    z_single = rng.standard_normal(n)

    def single_objective(x: np.ndarray) -> float:
        mean_change, log_sd = x
        week12 = _single_normal_week12(baseline_mmd, z_single, mean_change, np.exp(log_sd))
        return _objective(_metrics(baseline_mmd, week12), target) + _state_share_penalty(
            inputs, arm, week12, target
        )

    start = np.array([target.observed_mmd_change_week12, np.log(4.0)], dtype=float)
    single_bounds = [(-25.0, 5.0), (np.log(0.15), np.log(15.0))]
    single_fit = minimize(single_objective, start, method="Nelder-Mead", options={"maxiter": 600})
    if not single_fit.success:
        single_fit = minimize(single_objective, start, method="L-BFGS-B", bounds=single_bounds)
    single_params = np.array(single_fit.x, dtype=float)
    single_params[0] = min(max(single_params[0], single_bounds[0][0]), single_bounds[0][1])
    single_params[1] = min(max(single_params[1], single_bounds[1][0]), single_bounds[1][1])
    single_week12 = _single_normal_week12(
        baseline_mmd,
        z_single,
        float(single_params[0]),
        float(np.exp(single_params[1])),
    )
    single_metrics = _metrics(baseline_mmd, single_week12)
    single_obj = _objective(single_metrics, target) + _state_share_penalty(
        inputs, arm, single_week12, target
    )
    if single_obj <= 4.0 or not allow_mixture:
        return CalibrationResult(
            arm=arm,
            method="single_normal_change",
            parameters={
                "mean_change": float(single_params[0]),
                "sd_change": float(np.exp(single_params[1])),
            },
            week12_mmd=single_week12,
            mean_mmd_week12=single_metrics[0],
            responder50_week12=single_metrics[1],
            mmd_lt15_week12=single_metrics[2],
            target_mean_mmd_week12=target.mean_mmd_week12,
            target_responder50_week12=target.responder50_week12,
            target_mmd_lt15_week12=target.mmd_lt15_week12,
            objective_value=single_obj,
        )

    if _target_state_shares(inputs, arm, target) is not None:
        state_week12, state_params, state_obj = _state_informed_week12(inputs, arm, baseline_mmd, rng, target)
        state_metrics = _metrics(baseline_mmd, state_week12)
        return CalibrationResult(
            arm=arm,
            method="state_distribution_rank_calibration",
            parameters=state_params,
            week12_mmd=state_week12,
            mean_mmd_week12=state_metrics[0],
            responder50_week12=state_metrics[1],
            mmd_lt15_week12=state_metrics[2],
            target_mean_mmd_week12=target.mean_mmd_week12,
            target_responder50_week12=target.responder50_week12,
            target_mmd_lt15_week12=target.mmd_lt15_week12,
            objective_value=state_obj,
        )

    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    u = rng.random(n)

    def mixture_objective(x: np.ndarray) -> float:
        week12 = _mixture_week12(baseline_mmd, z1, z2, u, x)
        penalty = 0.0
        if x[2] <= 0 or x[4] <= 0:
            penalty += 1e6
        return _objective(_metrics(baseline_mmd, week12), target) + _state_share_penalty(
            inputs, arm, week12, target
        ) + penalty

    bounds = [
        (0.03, 0.97),
        (-25.0, 5.0),
        (0.15, 12.0),
        (-25.0, 5.0),
        (0.15, 12.0),
    ]
    initial_candidates = [
        np.array([0.55, target.observed_mmd_change_week12 - 3.0, 2.5, target.observed_mmd_change_week12 + 2.0, 4.0]),
        np.array([0.45, target.observed_mmd_change_week12 - 4.0, 2.0, target.observed_mmd_change_week12 + 1.0, 5.0]),
        np.array([0.65, target.observed_mmd_change_week12 - 2.5, 2.0, target.observed_mmd_change_week12 + 4.0, 3.5]),
    ]
    best_x = None
    best_obj = float("inf")
    for candidate in initial_candidates:
        fit = minimize(mixture_objective, candidate, method="Nelder-Mead", options={"maxiter": 900})
        x = np.array(fit.x, dtype=float)
        x = np.array([min(max(v, low), high) for v, (low, high) in zip(x, bounds)], dtype=float)
        obj = mixture_objective(x)
        if obj < best_obj:
            best_x = x
            best_obj = obj

    if best_obj > single_obj:
        de_fit = differential_evolution(
            mixture_objective,
            bounds=bounds,
            seed=int(rng.integers(0, 2**31 - 1)),
            maxiter=60,
            polish=True,
            updating="immediate",
        )
        if float(de_fit.fun) < best_obj:
            best_x = np.array(de_fit.x, dtype=float)
            best_obj = float(de_fit.fun)

    assert best_x is not None
    mixture_week12 = _mixture_week12(baseline_mmd, z1, z2, u, best_x)
    mixture_metrics = _metrics(baseline_mmd, mixture_week12)
    mixture_obj = _objective(mixture_metrics, target) + _state_share_penalty(
        inputs, arm, mixture_week12, target
    )

    if single_obj <= mixture_obj:
        return CalibrationResult(
            arm=arm,
            method="single_normal_change",
            parameters={
                "mean_change": float(single_params[0]),
                "sd_change": float(np.exp(single_params[1])),
            },
            week12_mmd=single_week12,
            mean_mmd_week12=single_metrics[0],
            responder50_week12=single_metrics[1],
            mmd_lt15_week12=single_metrics[2],
            target_mean_mmd_week12=target.mean_mmd_week12,
            target_responder50_week12=target.responder50_week12,
            target_mmd_lt15_week12=target.mmd_lt15_week12,
            objective_value=single_obj,
        )
    return CalibrationResult(
        arm=arm,
        method="two_component_normal_change",
        parameters={
            "mix_weight_component1": float(best_x[0]),
            "mean_change_component1": float(best_x[1]),
            "sd_change_component1": float(best_x[2]),
            "mean_change_component2": float(best_x[3]),
            "sd_change_component2": float(best_x[4]),
        },
        week12_mmd=mixture_week12,
        mean_mmd_week12=mixture_metrics[0],
        responder50_week12=mixture_metrics[1],
        mmd_lt15_week12=mixture_metrics[2],
        target_mean_mmd_week12=target.mean_mmd_week12,
        target_responder50_week12=target.responder50_week12,
        target_mmd_lt15_week12=target.mmd_lt15_week12,
        objective_value=mixture_obj,
    )
