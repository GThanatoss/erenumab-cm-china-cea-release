from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist

from .data import ModelInputs, cost_range, cost_value
from .simulation import SimulationSettings, run_basecase_simulation
from .utilities import beta_pert_sample, clip_nonnegative, clip_probability


UTILITY_OVERRIDE_ORDER = (
    "utility_0_3",
    "utility_4_9",
    "utility_10_14",
    "utility_15_19",
    "utility_20_23",
    "utility_24_28",
)


@dataclass(frozen=True)
class OWSASpec:
    parameter_id: str
    name_cn: str
    base_value: float
    lower: float
    upper: float
    override_key: str
    value_type: str
    recalibrate_week12_mmd: bool = False


def incremental_summary(output) -> dict[str, float]:
    row = output.results.loc[output.results["arm"] == "incremental"].iloc[0]
    return {
        "incremental_cost": float(row["total_cost"]),
        "incremental_qaly": float(row["total_qaly"]),
        "icer_cny_per_qaly": float(row["icer_cny_per_qaly"]),
        "inmb_1x_gdp_cny": float(row.get("inmb_1x_gdp_cny", np.nan)),
        "inmb_3x_gdp_cny": float(row.get("inmb_3x_gdp_cny", np.nan)),
    }


def threshold_price_analysis(inputs: ModelInputs, base_output) -> pd.DataFrame:
    incremental = base_output.results.loc[base_output.results["arm"] == "incremental"].iloc[0]
    intervention = base_output.results.loc[base_output.results["arm"] == inputs.intervention_arm].iloc[0]
    current_price = cost_value(inputs, "erenumab_price_per_4w")
    doses_per_patient = float(intervention["erenumab_cycles_or_doses_per_patient"])
    incremental_cost = float(incremental["total_cost"])
    incremental_qaly = float(incremental["total_qaly"])
    current_drug_cost = current_price * doses_per_patient
    non_drug_incremental_cost = incremental_cost - current_drug_cost

    rows: list[dict[str, Any]] = []
    for threshold_label, wtp in [
        ("1x_gdp", float(inputs.config["model"]["wtp_1x_gdp_cny_per_qaly"])),
        ("3x_gdp", float(inputs.config["model"]["wtp_3x_gdp_cny_per_qaly"])),
    ]:
        allowable_drug_cost = wtp * incremental_qaly - non_drug_incremental_cost
        threshold_price = allowable_drug_cost / doses_per_patient if doses_per_patient else np.nan
        rows.append(
            {
                "scenario_id": f"threshold_price_{threshold_label}",
                "scenario_name": f"Threshold price at {threshold_label.replace('_', ' ')} WTP",
                "wtp_cny_per_qaly": wtp,
                "threshold_price_per_70mg_box_cny": threshold_price,
                "current_price_per_70mg_box_cny": current_price,
                "price_ratio_to_current": threshold_price / current_price if current_price else np.nan,
                "allowable_erenumab_drug_cost_cny": allowable_drug_cost,
                "current_erenumab_drug_cost_cny": current_drug_cost,
                "non_drug_incremental_cost_cny": non_drug_incremental_cost,
                "incremental_cost_at_current_price_cny": incremental_cost,
                "incremental_qaly": incremental_qaly,
                "erenumab_cycles_or_doses_per_patient": doses_per_patient,
            }
        )
    return pd.DataFrame(rows)


def run_time_horizon_scenarios(
    inputs: ModelInputs,
    n_patients: int,
    seed: int,
    calibration_mode: str = "frozen",
) -> pd.DataFrame:
    settings = SimulationSettings(
        n_patients=n_patients,
        seed=seed,
        scenario_id="12_week_time_horizon",
        calibration_mode=calibration_mode,
        total_cycles=inputs.decision_tree_cycles,
    )
    output = run_basecase_simulation(
        inputs,
        settings,
        overrides=None,
    )
    results = output.results.copy()
    results.insert(0, "scenario_id", "12_week_time_horizon")
    results.insert(1, "scenario_name", "12-week time horizon")
    results.insert(2, "time_horizon_weeks", inputs.decision_tree_cycles * inputs.cycle_length_weeks)
    results.insert(3, "total_cycles", inputs.decision_tree_cycles)

    legacy_settings = SimulationSettings(
        n_patients=n_patients,
        seed=seed,
        scenario_id="legacy_state_calibration",
        calibration_mode="recalibrate",
    )
    legacy_output = run_basecase_simulation(
        inputs,
        legacy_settings,
        overrides={"_use_state_calibration": True},
    )
    legacy_results = legacy_output.results.copy()
    legacy_results.insert(0, "scenario_id", "legacy_state_calibration")
    legacy_results.insert(1, "scenario_name", "Legacy state-distribution calibration")
    legacy_results.insert(2, "time_horizon_weeks", inputs.time_horizon_weeks)
    legacy_results.insert(3, "total_cycles", inputs.total_cycles)
    return pd.concat([results, legacy_results], ignore_index=True)


def _range_from_base(base: float, lower: float | None, upper: float | None, value_type: str) -> tuple[float, float]:
    if lower is None or upper is None:
        values = [base * 0.8, base * 1.2]
        lower_value, upper_value = min(values), max(values)
    else:
        lower_value, upper_value = float(lower), float(upper)
    if value_type in {"probability", "utility"}:
        lower_value = clip_probability(lower_value)
        upper_value = clip_probability(upper_value)
    if value_type == "cost":
        lower_value = clip_nonnegative(lower_value)
        upper_value = clip_nonnegative(upper_value)
    return lower_value, upper_value


def _psa_row(inputs: ModelInputs, parameter_id: str) -> dict[str, Any] | None:
    rows = inputs.psa_parameters.loc[inputs.psa_parameters["parameter_id"] == parameter_id]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _psa_bounds(inputs: ModelInputs, parameter_id: str) -> tuple[float | None, float | None]:
    row = _psa_row(inputs, parameter_id)
    if row is None:
        return None, None
    lower = None if pd.isna(row.get("lower")) else float(row["lower"])
    upper = None if pd.isna(row.get("upper")) else float(row["upper"])
    return lower, upper


def build_owsa_specs(inputs: ModelInputs) -> list[OWSASpec]:
    specs: list[OWSASpec] = []

    for parameter_id, name_cn, value_type in [
        ("erenumab_price_per_4w", "Cost of erenumab", "cost"),
        ("acute_medication_cost_per_day", "Weighted daily acute medication cost", "cost"),
        ("non_drug_medical_service_cost_per_4w_2025", "Non-drug medical service cost", "cost"),
    ]:
        base = cost_value(inputs, parameter_id)
        lower, upper = cost_range(inputs, parameter_id)
        low, high = _range_from_base(base, lower, upper, value_type)
        specs.append(OWSASpec(parameter_id, name_cn, base, low, high, parameter_id, value_type))

    for _, state in inputs.mmd_states.iterrows():
        state_label = str(state["state_name"]).split()[0].replace("-", "_")
        key = f"utility_{state_label}"
        psa_id = key
        base = float(state["utility"])
        lower, upper = _psa_bounds(inputs, psa_id)
        low, high = _range_from_base(base, lower, upper, "utility")
        specs.append(
            OWSASpec(psa_id, f"Utility of {state['state_name']}", base, low, high, key, "utility")
        )

    arm_aliases = [
        ("erenumab", inputs.intervention_arm, "Erenumab preventive treatment"),
        ("bsc", inputs.comparator_arm, "No preventive treatment"),
    ]
    for alias, arm, arm_cn in arm_aliases:
        psa_id = f"mmd_change_w12_{alias}"
        row = _psa_row(inputs, psa_id)
        if row is None:
            continue
        base = float(row["base_value"])
        lower = None if pd.isna(row.get("lower")) else float(row["lower"])
        upper = None if pd.isna(row.get("upper")) else float(row["upper"])
        low, high = _range_from_base(base, lower, upper, "effect")
        specs.append(
            OWSASpec(
                psa_id,
                f"Week 12 MMD change ({arm_cn})",
                base,
                low,
                high,
                f"mmd_change_w12_{arm}",
                "effect",
                recalibrate_week12_mmd=True,
            )
        )

    baseline_cfg = inputs.config["baseline_mmd_distribution"]
    for key, name_cn in [("mean", "Baseline MMD mean"), ("sd", "Baseline MMD standard deviation")]:
        base = float(baseline_cfg[key])
        if key == "mean" and "owsa_mean_lower" in baseline_cfg and "owsa_mean_upper" in baseline_cfg:
            low, high = float(baseline_cfg["owsa_mean_lower"]), float(baseline_cfg["owsa_mean_upper"])
        elif key == "sd" and "owsa_sd_lower" in baseline_cfg and "owsa_sd_upper" in baseline_cfg:
            low, high = float(baseline_cfg["owsa_sd_lower"]), float(baseline_cfg["owsa_sd_upper"])
        else:
            low, high = _range_from_base(base, None, None, "effect")
        specs.append(OWSASpec(f"baseline_mmd_{key}", name_cn, base, low, high, f"baseline_mmd_{key}", "effect"))

    discontinuation_cfg = inputs.config["discontinuation"]
    for parameter_id, name_cn in [
        (str(discontinuation_cfg["short_term_rule"]), "0-12 week double-blind all-cause discontinuation rate"),
        (str(discontinuation_cfg["maintenance_rule"]), "Long-term all-cause discontinuation rate"),
    ]:
        row = _psa_row(inputs, parameter_id)
        if row is None:
            continue
        base = float(row["base_value"])
        lower = None if pd.isna(row.get("lower")) else float(row["lower"])
        upper = None if pd.isna(row.get("upper")) else float(row["upper"])
        low, high = _range_from_base(base, lower, upper, "probability")
        specs.append(OWSASpec(parameter_id, name_cn, base, low, high, parameter_id, "probability"))

    return specs


def run_one_way_sensitivity(
    inputs: ModelInputs,
    base_output,
    n_patients: int,
    seed: int,
    calibration_mode: str = "frozen",
) -> pd.DataFrame:
    wtp = float(inputs.config["model"]["wtp_3x_gdp_cny_per_qaly"])
    base_summary = incremental_summary(base_output)
    base_icer = base_summary["icer_cny_per_qaly"]
    base_inmb = base_summary["incremental_qaly"] * wtp - base_summary["incremental_cost"]
    rows: list[dict[str, Any]] = []
    for spec in build_owsa_specs(inputs):
        values = [("lower", spec.lower), ("upper", spec.upper)]
        run_results: dict[str, float] = {}
        for label, value in values:
            overrides: dict[str, Any] = {spec.override_key: value}
            if spec.recalibrate_week12_mmd:
                overrides["_mmd_change_only_recalibration"] = True
            settings = SimulationSettings(
                n_patients=n_patients,
                seed=seed,
                scenario_id="basecase",
                calibration_mode=calibration_mode,
            )
            output = run_basecase_simulation(
                inputs,
                settings,
                overrides=overrides,
            )
            summary = incremental_summary(output)
            run_results[f"{label}_icer_cny_per_qaly"] = summary["icer_cny_per_qaly"]
            run_results[f"{label}_inmb_cny"] = summary["incremental_qaly"] * wtp - summary["incremental_cost"]
        low_icer = run_results["lower_icer_cny_per_qaly"]
        high_icer = run_results["upper_icer_cny_per_qaly"]
        low_inmb = run_results["lower_inmb_cny"]
        high_inmb = run_results["upper_inmb_cny"]
        rows.append(
            {
                "parameter_id": spec.parameter_id,
                "name_cn": spec.name_cn,
                "base_value": spec.base_value,
                "lower_value": spec.lower,
                "upper_value": spec.upper,
                "wtp_cny_per_qaly": wtp,
                "base_inmb_cny": base_inmb,
                "lower_inmb_cny": low_inmb,
                "upper_inmb_cny": high_inmb,
                "min_inmb_cny": min(low_inmb, high_inmb),
                "max_inmb_cny": max(low_inmb, high_inmb),
                "inmb_range_width": abs(high_inmb - low_inmb),
                "base_icer_cny_per_qaly": base_icer,
                "lower_icer_cny_per_qaly": low_icer,
                "upper_icer_cny_per_qaly": high_icer,
                "min_icer_cny_per_qaly": min(low_icer, high_icer),
                "max_icer_cny_per_qaly": max(low_icer, high_icer),
                "icer_range_width": abs(high_icer - low_icer),
                "range_width": abs(high_inmb - low_inmb),
            }
        )
    return pd.DataFrame(rows).sort_values("range_width", ascending=False).reset_index(drop=True)


def _sample_from_row(row: dict[str, Any], rng: np.random.Generator) -> float:
    distribution = str(row["distribution"])
    base = float(row["base_value"])
    lower = None if pd.isna(row.get("lower")) else float(row["lower"])
    upper = None if pd.isna(row.get("upper")) else float(row["upper"])

    if distribution == "normal":
        sd = float(row["se_sd"])
        return float(rng.normal(base, sd))
    if distribution == "beta":
        p1 = row.get("param1")
        p2 = row.get("param2")
        if not pd.isna(p1) and not pd.isna(p2):
            return float(beta_dist.rvs(float(p1), float(p2), random_state=rng))
        if lower is not None and upper is not None:
            return beta_pert_sample(rng, lower, base, upper)
        return clip_probability(base)
    if distribution == "gamma":
        if lower is not None and upper is not None:
            sd = max((upper - lower) / 3.92, base * 0.05)
            shape = (base / sd) ** 2
            scale = sd**2 / base
            return float(gamma_dist.rvs(shape, scale=scale, random_state=rng))
        return clip_nonnegative(base)
    return base


def _gamma_sample_from_mean_sd(rng: np.random.Generator, mean: float, sd: float) -> float:
    mean = clip_nonnegative(mean)
    if mean <= 0:
        return 0.0
    sd = max(float(sd), mean * 0.001)
    shape = (mean / sd) ** 2
    scale = sd**2 / mean
    return float(gamma_dist.rvs(shape, scale=scale, random_state=rng))


def _gamma_sample_from_range(
    rng: np.random.Generator,
    mean: float,
    lower: float,
    upper: float,
) -> float:
    if upper <= lower:
        return clip_nonnegative(mean)
    sd = max((upper - lower) / 3.92, mean * 0.05)
    return _gamma_sample_from_mean_sd(rng, mean, sd)


def _psa_parameter_mapping(inputs: ModelInputs) -> dict[str, str | None]:
    return {
        "mmd_change_w12_erenumab": f"mmd_change_w12_{inputs.intervention_arm}",
        "mmd_change_w12_bsc": f"mmd_change_w12_{inputs.comparator_arm}",
        "responder50_w12_erenumab": f"responder50_week12_{inputs.intervention_arm}",
        "responder50_w12_bsc": f"responder50_week12_{inputs.comparator_arm}",
        "mmd_lt15_w12_erenumab": f"mmd_lt15_week12_{inputs.intervention_arm}",
        "mmd_lt15_w12_bsc": f"mmd_lt15_week12_{inputs.comparator_arm}",
        "disc_allcause_12w": "disc_allcause_12w",
        "disc_allcause_64w": "disc_allcause_64w",
        "disc_lack_efficacy_64w": None,
        "disc_ae_64w": None,
        "erenumab_price_per_4w": None,
        "utility_0_3": "utility_0_3",
        "utility_4_9": "utility_4_9",
        "utility_10_14": "utility_10_14",
        "utility_15_19": "utility_15_19",
        "utility_20_23": "utility_20_23",
        "utility_24_28": "utility_24_28",
        "bsc_outpatient_cost_annual": None,
    }


def _project_nonincreasing(values: np.ndarray) -> np.ndarray:
    """Closest L2 projection onto a non-increasing sequence."""
    transformed = -np.asarray(values, dtype=float)
    levels: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []

    for idx, value in enumerate(transformed):
        levels.append(float(value))
        weights.append(1.0)
        starts.append(idx)
        ends.append(idx)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            pooled_level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2] = pooled_level
            weights[-2] = total_weight
            ends[-2] = ends[-1]
            levels.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    projected = np.empty_like(transformed)
    for level, start, end in zip(levels, starts, ends):
        projected[start : end + 1] = level
    return -projected


def _enforce_monotone_utility_overrides(overrides: dict[str, float]) -> None:
    if not all(key in overrides for key in UTILITY_OVERRIDE_ORDER):
        return
    sampled = np.array([clip_probability(overrides[key]) for key in UTILITY_OVERRIDE_ORDER], dtype=float)
    monotone = np.clip(_project_nonincreasing(sampled), 0.0, 1.0)
    for key, value in zip(UTILITY_OVERRIDE_ORDER, monotone):
        overrides[key] = float(value)


def sample_psa_overrides(inputs: ModelInputs, rng: np.random.Generator) -> dict[str, float]:
    overrides: dict[str, float] = {}
    mapping = _psa_parameter_mapping(inputs)
    for _, row_series in inputs.psa_parameters.iterrows():
        row = row_series.to_dict()
        parameter_id = str(row["parameter_id"])
        target_key = mapping.get(parameter_id)
        if target_key is None:
            continue
        value = _sample_from_row(row, rng)
        if target_key.startswith(("responder50_", "mmd_lt15_", "disc_", "utility_")):
            value = clip_probability(value)
        overrides[target_key] = value

    _enforce_monotone_utility_overrides(overrides)

    acute_base = cost_value(inputs, "acute_medication_cost_per_day")
    acute_lower, acute_upper = cost_range(inputs, "acute_medication_cost_per_day")
    if acute_lower is not None and acute_upper is not None:
        overrides["acute_medication_cost_per_day"] = _gamma_sample_from_range(
            rng, acute_base, acute_lower, acute_upper
        )

    non_drug_base = cost_value(inputs, "non_drug_medical_service_cost_per_4w_2025")
    overrides["non_drug_medical_service_cost_per_4w_2025"] = _gamma_sample_from_range(
        rng, non_drug_base, non_drug_base * 0.8, non_drug_base * 1.2
    )
    return overrides


def _psa_iteration_row(
    inputs: ModelInputs,
    iteration: int,
    overrides: dict[str, float],
    n_patients: int,
    seed: int,
    calibration_mode: str,
) -> dict[str, float]:
    settings = SimulationSettings(
        n_patients=n_patients,
        seed=seed + iteration,
        scenario_id="basecase",
        calibration_mode=calibration_mode,
    )
    output = run_basecase_simulation(inputs, settings, overrides=dict(overrides))
    inc = incremental_summary(output)
    intervention = output.results.loc[output.results["arm"] == inputs.intervention_arm].iloc[0]
    comparator = output.results.loc[output.results["arm"] == inputs.comparator_arm].iloc[0]
    row = {
        "iteration": iteration,
        "n_patients": n_patients,
        **inc,
        "intervention_total_cost": float(intervention["total_cost"]),
        "intervention_total_qaly": float(intervention["total_qaly"]),
        "comparator_total_cost": float(comparator["total_cost"]),
        "comparator_total_qaly": float(comparator["total_qaly"]),
    }
    for key, value in sorted(overrides.items()):
        row[f"sampled_{key}"] = value
    return row


def _run_psa_chunk(
    args: tuple[
        ModelInputs,
        list[tuple[int, dict[str, float]]],
        int,
        int,
        str,
    ],
) -> list[dict[str, float]]:
    inputs, chunk, n_patients, seed, calibration_mode = args
    return [
        _psa_iteration_row(inputs, iteration, overrides, n_patients, seed, calibration_mode)
        for iteration, overrides in chunk
    ]


def _chunk_psa_tasks(
    tasks: list[tuple[int, dict[str, float]]],
    workers: int,
) -> list[list[tuple[int, dict[str, float]]]]:
    if not tasks:
        return []
    chunk_count = min(len(tasks), max(workers * 4, workers))
    chunk_size = int(np.ceil(len(tasks) / chunk_count))
    return [tasks[start : start + chunk_size] for start in range(0, len(tasks), chunk_size)]


def run_probabilistic_sensitivity(
    inputs: ModelInputs,
    iterations: int,
    n_patients: int,
    seed: int,
    calibration_mode: str = "frozen",
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    tasks = [
        (iteration, sample_psa_overrides(inputs, rng))
        for iteration in range(1, iterations + 1)
    ]
    worker_count = max(1, int(workers))
    if worker_count == 1 or iterations <= 1:
        rows = [
            _psa_iteration_row(inputs, iteration, overrides, n_patients, seed, calibration_mode)
            for iteration, overrides in tasks
        ]
    else:
        chunks = _chunk_psa_tasks(tasks, min(worker_count, iterations))
        chunk_args = [
            (inputs, chunk, n_patients, seed, calibration_mode)
            for chunk in chunks
        ]
        rows = []
        with ProcessPoolExecutor(max_workers=min(worker_count, len(chunks))) as executor:
            for chunk_rows in executor.map(_run_psa_chunk, chunk_args):
                rows.extend(chunk_rows)
        rows.sort(key=lambda row: int(row["iteration"]))

    psa_results = pd.DataFrame(rows)
    wtp_values = sorted(
        set(np.linspace(0, 300000, 61).round(0).astype(int).tolist() + [99665, 298995])
    )
    ceac_rows = []
    for wtp in wtp_values:
        inc_nmb = wtp * psa_results["incremental_qaly"] - psa_results["incremental_cost"]
        erenumab_ce = inc_nmb > 0
        ceac_rows.append(
            {
                "wtp_cny_per_qaly": float(wtp),
                "probability_erenumab_plus_bsc_cost_effective": float(np.mean(erenumab_ce)),
                "probability_bsc_cost_effective": float(np.mean(~erenumab_ce)),
                "probability_cost_effective": float(np.mean(erenumab_ce)),
                "mean_incremental_nmb": float(np.mean(inc_nmb)),
            }
        )
    ceac = pd.DataFrame(ceac_rows)
    return psa_results, ceac
