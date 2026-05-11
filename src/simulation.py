from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .calibration import (
    CalibrationResult,
    apply_frozen_calibration,
    calibrate_week12_distribution,
    frozen_calibration_available,
)
from .costs import (
    acute_medication_cost_per_day,
    discontinuation_cycle_probability,
    erenumab_price_per_cycle,
    non_drug_medical_service_cost_per_cycle,
)
from .data import (
    ModelInputs,
    acute_medication_day_value,
    trajectory_value,
)
from .utilities import (
    MMD_MAX,
    MMD_MIN,
    sample_truncated_normal,
    state_share_table,
    utilities_from_mmd,
)


@dataclass(frozen=True)
class SimulationSettings:
    n_patients: int
    seed: int = 314159
    scenario_id: str = "basecase"
    calibration_mode: str = "frozen"
    total_cycles: int | None = None


@dataclass(frozen=True)
class SimulationOutput:
    results: pd.DataFrame
    cycle_trace: pd.DataFrame
    calibration_diagnostics: pd.DataFrame
    calibration_results: dict[str, CalibrationResult]
    baseline_summary: pd.DataFrame
    baseline_mmd: np.ndarray


def _utility_overrides(overrides: dict[str, Any] | None) -> dict[str, float]:
    if not overrides:
        return {}
    return {key: float(value) for key, value in overrides.items() if key.startswith("utility_")}


def _baseline_mmd(inputs: ModelInputs, settings: SimulationSettings, overrides: dict[str, Any] | None) -> np.ndarray:
    cfg = inputs.config["baseline_mmd_distribution"]
    mean = float(overrides.get("baseline_mmd_mean", cfg["mean"])) if overrides else float(cfg["mean"])
    sd = float(overrides.get("baseline_mmd_sd", cfg["sd"])) if overrides else float(cfg["sd"])
    lower = float(cfg["lower"])
    upper = float(cfg["upper"])
    rng = np.random.default_rng(settings.seed)
    return sample_truncated_normal(mean, sd, lower, upper, settings.n_patients, rng)


def _timepoint_mmd(
    inputs: ModelInputs,
    arm: str,
    baseline_mmd: np.ndarray,
    week12_mmd: np.ndarray,
    timepoint: str,
    overrides: dict[str, Any] | None,
) -> np.ndarray:
    if timepoint == "baseline":
        return baseline_mmd
    if timepoint == "week12":
        return week12_mmd
    change_t = trajectory_value(inputs, arm, timepoint, "mmd_change", None)
    base_change_12 = trajectory_value(inputs, arm, "week12", "mmd_change", None)
    fraction = 0.0 if base_change_12 == 0 else change_t / base_change_12
    return np.clip(baseline_mmd + fraction * (week12_mmd - baseline_mmd), MMD_MIN, MMD_MAX)


def _acute_days_for_cycle(inputs: ModelInputs, arm: str, cycle: int) -> float:
    baseline_days = acute_medication_day_value(inputs, arm, "baseline")
    week12_days = acute_medication_day_value(inputs, arm, "week12")
    if cycle <= 3:
        start_fraction = (cycle - 1) / 3.0
        end_fraction = cycle / 3.0
        start_days = baseline_days + start_fraction * (week12_days - baseline_days)
        end_days = baseline_days + end_fraction * (week12_days - baseline_days)
        return float((start_days + end_days) / 2.0)
    return float(week12_days)


def _simulation_total_cycles(inputs: ModelInputs, settings: SimulationSettings) -> int:
    return int(settings.total_cycles or inputs.total_cycles)


def _maintenance_decay_fraction(cycle_after_week12: int, p_short: float, p_long: float) -> float:
    if cycle_after_week12 <= 0:
        return 0.0
    d_short = 1.0 - (1.0 - p_short) ** cycle_after_week12
    d_long = 1.0 - (1.0 - p_long) ** cycle_after_week12
    return float(max(0.0, d_long - d_short))


def _acute_days_for_cycle_with_decay(
    inputs: ModelInputs,
    arm: str,
    cycle: int,
    p_short: float,
    p_long: float,
) -> float:
    if cycle <= inputs.decision_tree_cycles or arm != inputs.intervention_arm:
        return _acute_days_for_cycle(inputs, arm, cycle)
    erenumab_week12 = acute_medication_day_value(inputs, inputs.intervention_arm, "week12")
    bsc_week12 = acute_medication_day_value(inputs, inputs.comparator_arm, "week12")
    q = _maintenance_decay_fraction(cycle - inputs.decision_tree_cycles, p_short, p_long)
    return float(erenumab_week12 + q * (bsc_week12 - erenumab_week12))


def _cycle_mmd(
    inputs: ModelInputs,
    arm: str,
    cycle: int,
    baseline_mmd: np.ndarray,
    week12_mmd: np.ndarray,
    overrides: dict[str, Any] | None,
    comparator_week12_mmd: np.ndarray | None = None,
    p_short: float = 0.0,
    p_long: float = 0.0,
) -> np.ndarray:
    if cycle == 1:
        start = _timepoint_mmd(inputs, arm, baseline_mmd, week12_mmd, "baseline", overrides)
        end = _timepoint_mmd(inputs, arm, baseline_mmd, week12_mmd, "week4", overrides)
        return np.clip((start + end) / 2.0, MMD_MIN, MMD_MAX)
    if cycle == 2:
        start = _timepoint_mmd(inputs, arm, baseline_mmd, week12_mmd, "week4", overrides)
        end = _timepoint_mmd(inputs, arm, baseline_mmd, week12_mmd, "week8", overrides)
        return np.clip((start + end) / 2.0, MMD_MIN, MMD_MAX)
    if cycle == 3:
        start = _timepoint_mmd(inputs, arm, baseline_mmd, week12_mmd, "week8", overrides)
        end = week12_mmd
        return np.clip((start + end) / 2.0, MMD_MIN, MMD_MAX)
    if arm == inputs.intervention_arm and comparator_week12_mmd is not None:
        q = _maintenance_decay_fraction(cycle - inputs.decision_tree_cycles, p_short, p_long)
        return np.clip(week12_mmd + q * (comparator_week12_mmd - week12_mmd), MMD_MIN, MMD_MAX)
    return week12_mmd


def _simulate_arm(
    inputs: ModelInputs,
    arm: str,
    baseline_mmd: np.ndarray,
    calibration: CalibrationResult,
    settings: SimulationSettings,
    overrides: dict[str, Any] | None,
    comparator_week12_mmd: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float | str | int]]]:
    n = settings.n_patients
    week12_mmd = calibration.week12_mmd
    utility_overrides = _utility_overrides(overrides)
    rng = np.random.default_rng(settings.seed + (101 if arm == inputs.intervention_arm else 202))

    is_intervention = arm == inputs.intervention_arm
    on_treatment = np.ones(n, dtype=bool) if is_intervention else np.zeros(n, dtype=bool)
    price = erenumab_price_per_cycle(inputs, overrides)
    acute_cost_day = acute_medication_cost_per_day(inputs, overrides)
    non_drug_cost = non_drug_medical_service_cost_per_cycle(inputs, overrides)
    discontinuation_cfg = inputs.config["discontinuation"]
    p_short = discontinuation_cycle_probability(inputs, str(discontinuation_cfg["short_term_rule"]), overrides)
    p_long = discontinuation_cycle_probability(inputs, str(discontinuation_cfg["maintenance_rule"]), overrides)

    total_drug_cost = 0.0
    total_acute_cost = 0.0
    total_non_drug_cost = 0.0
    total_qaly = 0.0
    total_doses = 0.0
    trace: list[dict[str, float | str | int]] = []

    total_cycles = _simulation_total_cycles(inputs, settings)
    for cycle in range(1, total_cycles + 1):
        cycle_mmd = _cycle_mmd(
            inputs,
            arm,
            cycle,
            baseline_mmd,
            week12_mmd,
            overrides,
            comparator_week12_mmd=comparator_week12_mmd,
            p_short=p_short,
            p_long=p_long,
        )
        utilities = utilities_from_mmd(cycle_mmd, inputs.mmd_states, utility_overrides)
        qaly = float(np.mean(utilities) * inputs.cycle_length_weeks / 52.0)
        acute_days = _acute_days_for_cycle_with_decay(inputs, arm, cycle, p_short, p_long)
        acute_cost = float(acute_days * acute_cost_day)
        cycle_non_drug_cost = float(non_drug_cost)
        if is_intervention:
            dose_rate = float(np.mean(on_treatment))
            drug_cost = float(dose_rate * price)
            total_doses += dose_rate
        else:
            dose_rate = 0.0
            drug_cost = 0.0
        total_cycle_cost = drug_cost + acute_cost + cycle_non_drug_cost

        total_drug_cost += drug_cost
        total_acute_cost += acute_cost
        total_non_drug_cost += cycle_non_drug_cost
        total_qaly += qaly

        row: dict[str, float | str | int] = {
            "arm": arm,
            "cycle": cycle,
            "week_start": (cycle - 1) * inputs.cycle_length_weeks,
            "week_end": cycle * inputs.cycle_length_weeks,
            "mean_mmd": float(np.mean(cycle_mmd)),
            "mean_utility": float(np.mean(utilities)),
            "qaly": qaly,
            "acute_medication_days": acute_days,
            "erenumab_on_treatment_rate": dose_rate,
            "erenumab_drug_cost": drug_cost,
            "acute_rescue_medication_cost": acute_cost,
            "non_drug_medical_service_cost": cycle_non_drug_cost,
            "total_cost": total_cycle_cost,
        }
        row.update(state_share_table(cycle_mmd, inputs.mmd_states))
        trace.append(row)

        if is_intervention:
            p_disc = p_short if cycle <= inputs.decision_tree_cycles else p_long
            discontinuations = rng.random(n) < p_disc
            on_treatment = on_treatment & (~discontinuations)

    total_cost = total_drug_cost + total_acute_cost + total_non_drug_cost
    summary = {
        "arm": arm,
        "total_cost": total_cost,
        "total_qaly": total_qaly,
        "erenumab_drug_cost": total_drug_cost,
        "acute_rescue_medication_cost": total_acute_cost,
        "non_drug_medical_service_cost": total_non_drug_cost,
        "erenumab_cycles_or_doses_per_patient": total_doses,
        "erenumab_total_doses": total_doses * n,
        "week12_mean_mmd": calibration.mean_mmd_week12,
        "week12_responder50": calibration.responder50_week12,
        "week12_mmd_lt15": calibration.mmd_lt15_week12,
    }
    return summary, trace


def run_basecase_simulation(
    inputs: ModelInputs,
    settings: SimulationSettings,
    overrides: dict[str, Any] | None = None,
) -> SimulationOutput:
    baseline = _baseline_mmd(inputs, settings, overrides)
    calibration_results: dict[str, CalibrationResult] = {}
    for idx, arm in enumerate([inputs.intervention_arm, inputs.comparator_arm]):
        if settings.calibration_mode == "frozen" and frozen_calibration_available(inputs):
            calibration_results[arm] = apply_frozen_calibration(inputs, arm, baseline, overrides)
        elif settings.calibration_mode in {"recalibrate", "frozen"}:
            rng = np.random.default_rng(settings.seed + 1000 + idx)
            calibration_results[arm] = calibrate_week12_distribution(
                inputs,
                arm,
                baseline,
                rng,
                overrides,
            )
        else:
            raise ValueError(f"Unknown calibration_mode: {settings.calibration_mode}")

    summaries = []
    trace_rows: list[dict[str, float | str | int]] = []
    for arm in [inputs.intervention_arm, inputs.comparator_arm]:
        comparator_week12_mmd = (
            calibration_results[inputs.comparator_arm].week12_mmd
            if arm == inputs.intervention_arm
            else None
        )
        summary, trace = _simulate_arm(
            inputs,
            arm,
            baseline,
            calibration_results[arm],
            settings,
            overrides,
            comparator_week12_mmd=comparator_week12_mmd,
        )
        summaries.append(summary)
        trace_rows.extend(trace)

    results = pd.DataFrame(summaries)
    wtp_1x = float(inputs.config["model"]["wtp_1x_gdp_cny_per_qaly"])
    wtp_3x = float(inputs.config["model"]["wtp_3x_gdp_cny_per_qaly"])
    results["nmb_1x_gdp_cny"] = results["total_qaly"] * wtp_1x - results["total_cost"]
    results["nmb_3x_gdp_cny"] = results["total_qaly"] * wtp_3x - results["total_cost"]
    results["inmb_1x_gdp_cny"] = np.nan
    results["inmb_3x_gdp_cny"] = np.nan
    intervention = results.loc[results["arm"] == inputs.intervention_arm].iloc[0]
    comparator = results.loc[results["arm"] == inputs.comparator_arm].iloc[0]
    inc_cost = float(intervention["total_cost"] - comparator["total_cost"])
    inc_qaly = float(intervention["total_qaly"] - comparator["total_qaly"])
    icer = float(inc_cost / inc_qaly) if inc_qaly != 0 else np.inf
    incremental = {
        "arm": "incremental",
        "total_cost": inc_cost,
        "total_qaly": inc_qaly,
        "erenumab_drug_cost": float(intervention["erenumab_drug_cost"] - comparator["erenumab_drug_cost"]),
        "acute_rescue_medication_cost": float(
            intervention["acute_rescue_medication_cost"] - comparator["acute_rescue_medication_cost"]
        ),
        "non_drug_medical_service_cost": float(
            intervention["non_drug_medical_service_cost"] - comparator["non_drug_medical_service_cost"]
        ),
        "erenumab_cycles_or_doses_per_patient": float(intervention["erenumab_cycles_or_doses_per_patient"]),
        "erenumab_total_doses": float(intervention["erenumab_total_doses"]),
        "week12_mean_mmd": float(intervention["week12_mean_mmd"] - comparator["week12_mean_mmd"]),
        "week12_responder50": float(intervention["week12_responder50"] - comparator["week12_responder50"]),
        "week12_mmd_lt15": float(intervention["week12_mmd_lt15"] - comparator["week12_mmd_lt15"]),
        "icer_cny_per_qaly": icer,
        "nmb_1x_gdp_cny": np.nan,
        "nmb_3x_gdp_cny": np.nan,
        "inmb_1x_gdp_cny": inc_qaly * wtp_1x - inc_cost,
        "inmb_3x_gdp_cny": inc_qaly * wtp_3x - inc_cost,
    }
    results = pd.concat([results, pd.DataFrame([incremental])], ignore_index=True)
    if "icer_cny_per_qaly" not in results.columns:
        results["icer_cny_per_qaly"] = np.nan
    results.loc[results["arm"] != "incremental", "icer_cny_per_qaly"] = np.nan

    calibration_df = pd.DataFrame([value.diagnostics for value in calibration_results.values()])
    baseline_summary = pd.DataFrame(
        [
            {
                "n_patients": settings.n_patients,
                "seed": settings.seed,
                "scenario_id": settings.scenario_id,
                "total_cycles": _simulation_total_cycles(inputs, settings),
                "baseline_mean_mmd": float(np.mean(baseline)),
                "baseline_sd_mmd": float(np.std(baseline, ddof=1)),
                "baseline_min_mmd": float(np.min(baseline)),
                "baseline_max_mmd": float(np.max(baseline)),
            }
        ]
    )
    return SimulationOutput(
        results=results,
        cycle_trace=pd.DataFrame(trace_rows),
        calibration_diagnostics=calibration_df,
        calibration_results=calibration_results,
        baseline_summary=baseline_summary,
        baseline_mmd=baseline,
    )
