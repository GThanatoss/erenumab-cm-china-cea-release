from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import truncnorm

from .data import ModelInputs
from .utilities import MMD_MAX, MMD_MIN, _truncated_normal_params_for_moments, state_indices_from_mmd


INTERVENTION_LABEL_ZH = "Erenumab preventive treatment"
COMPARATOR_LABEL_ZH = "No preventive treatment"
INTERVENTION_LABEL_EN = "Erenumab preventive treatment"
COMPARATOR_LABEL_EN = "No preventive treatment"


ZH_SHEET_NAMES = {
    "Inputs": "Inputs",
    "Basecase_Results": "Basecase_Results",
    "Cycle_Trace": "Cycle_Trace",
    "Calibration_Diagnostics": "Calibration_Diagnostics",
    "OWSA": "OWSA",
    "PSA_Results": "PSA_Results",
    "CEAC": "Cost-effectiveness acceptability curve",
    "Scenario_Results": "Scenario_Results",
    "Threshold_Prices": "Threshold_Prices",
    "Figures_Index": "Figures_Index",
}


ZH_COLUMNS = {
    "section": "section",
    "key": "parameter",
    "value": "value",
    "unit": "unit",
    "status": "status",
    "notes": "notes",
    "arm": "arm",
    "total_cost": "total cost",
    "total_qaly": "total QALY",
    "erenumab_drug_cost": "Erenumab drug cost",
    "acute_rescue_medication_cost": "Acute medication cost",
    "non_drug_medical_service_cost": "Non-drug medical service cost",
    "erenumab_cycles_or_doses_per_patient": "Erenumab cycles per patient",
    "erenumab_total_doses": "Total erenumab boxes",
    "week12_mean_mmd": "Week 12 mean MMD",
    "week12_responder50": "Week 12 >=50% MMD responder rate",
    "week12_mmd_lt15": "Week 12 MMD<15 proportion",
    "nmb_1x_gdp_cny": "1x GDP NMB",
    "nmb_3x_gdp_cny": "3x GDP NMB",
    "inmb_1x_gdp_cny": "1x GDP INMB",
    "inmb_3x_gdp_cny": "3x GDP INMB",
    "icer_cny_per_qaly": "ICER( CNY/QALY)",
    "cycle": "cycle",
    "week_start": "week start",
    "week_end": "week end",
    "mean_mmd": "mean MMD",
    "mean_utility": "mean utility",
    "qaly": "QALY",
    "acute_medication_days": "Acute medication days",
    "erenumab_on_treatment_rate": "erenumabon-treatment rate",
    "share_S1": "S1share",
    "share_S2": "S2share",
    "share_S3": "S3share",
    "share_S4": "S4share",
    "share_S5": "S5share",
    "share_S6": "S6share",
    "method": "method",
    "target_mean_mmd_week12": "Target Week 12 mean MMD",
    "simulated_mean_mmd_week12": "Simulated Week 12 mean MMD",
    "residual_mean_mmd_week12": "Week 12 mean MMD residual",
    "target_responder50_week12": "Target Week 12 >=50% MMD responder rate",
    "simulated_responder50_week12": "Simulated Week 12 >=50% MMD responder rate",
    "residual_responder50_week12": "Week 12 >=50% MMD responder rate residual",
    "target_mmd_lt15_week12": "Target Week 12 MMD<15 proportion",
    "simulated_mmd_lt15_week12": "Simulated Week 12 MMD<15 proportion",
    "residual_mmd_lt15_week12": "Week 12 MMD<15 proportion residual",
    "objective_value": "objective value",
    "rank_correlation_weight": "rank-correlation weight",
    "uses_frozen_calibration_parameters": "uses frozen calibration parameters",
    "calibration_seed": "calibration seed",
    "state_value_seed": "state value seed",
    "uses_continuous_week12_distribution": "uses continuous Week 12 distribution",
    "mix_weight_low": "low-MMD component weight",
    "component_mean_low": "low-MMD component mean",
    "component_sd_low": "low-MMD component SD",
    "component_mean_high": "high-MMD component mean",
    "component_sd_high": "high-MMD component SD",
    "continuous_draw_seed": "continuous draw seed",
    "smoothness_penalty": "smoothness penalty",
    "local_continuity_penalty": "local continuity penalty",
    "frozen_share_S1": "diagnostic S1 share",
    "frozen_share_S2": "diagnostic S2 share",
    "frozen_share_S3": "diagnostic S3 share",
    "frozen_share_S4": "diagnostic S4 share",
    "frozen_share_S5": "diagnostic S5 share",
    "frozen_share_S6": "diagnostic S6 share",
    "parameter_id": "parameter ID",
    "name_cn": "parameter name",
    "base_value": "base value",
    "lower_value": "lower value",
    "upper_value": "upper value",
    "wtp_cny_per_qaly": "Willingness-to-pay threshold (CNY/QALY)",
    "base_inmb_cny": "base INMB",
    "lower_inmb_cny": "lower INMB",
    "upper_inmb_cny": "upper INMB",
    "min_inmb_cny": "minimum INMB",
    "max_inmb_cny": "maximum INMB",
    "inmb_range_width": "INMB range width",
    "base_icer_cny_per_qaly": "Base ICER (CNY/QALY)",
    "lower_icer_cny_per_qaly": "LowerICER( CNY/QALY)",
    "upper_icer_cny_per_qaly": "UpperICER( CNY/QALY)",
    "min_icer_cny_per_qaly": "Minimum ICER (CNY/QALY)",
    "max_icer_cny_per_qaly": "Maximum ICER (CNY/QALY)",
    "icer_range_width": "ICER range",
    "range_width": "ranking range width",
    "iteration": "iteration",
    "n_patients": "Patients",
    "scenario_id": "scenario ID",
    "scenario_name": "scenario name",
    "time_horizon_weeks": "time horizon (weeks)",
    "total_cycles": "Total cycles",
    "threshold_price_per_70mg_box_cny": "threshold price (CNY per 70 mg box)",
    "current_price_per_70mg_box_cny": "current price (CNY per 70 mg box)",
    "price_ratio_to_current": "threshold/current price ratio",
    "allowable_erenumab_drug_cost_cny": "allowable erenumab drug cost",
    "current_erenumab_drug_cost_cny": "current erenumab drug cost",
    "non_drug_incremental_cost_cny": "non-drug incremental cost",
    "incremental_cost_at_current_price_cny": "incremental cost at current price",
    "incremental_cost": "incremental cost",
    "incremental_qaly": "Incremental QALY",
    "intervention_total_cost": "intervention total cost",
    "intervention_total_qaly": "intervention total QALY",
    "comparator_total_cost": "comparator total cost",
    "comparator_total_qaly": "comparator total QALY",
    "sampled_acute_medication_cost_per_day": "sampled acute medication cost",
    "sampled_disc_allcause_12w": "sampled 0-12 week double-blind all-cause discontinuation rate",
    "sampled_disc_allcause_64w": "sampled long-term all-cause discontinuation rate",
    "sampled_mmd_change_w12_bsc": "sampled Week 12 MMD change(No preventive treatment)",
    "sampled_mmd_change_w12_erenumab_plus_bsc": "sampled Week 12 MMD change(Erenumab preventive treatment)",
    "sampled_mmd_lt15_week12_bsc": "sampled Week 12 MMD<15 proportion(No preventive treatment)",
    "sampled_mmd_lt15_week12_erenumab_plus_bsc": "sampled Week 12 MMD<15 proportion(Erenumab preventive treatment)",
    "sampled_non_drug_medical_service_cost_per_4w_2025": "sampled Non-drug medical service cost",
    "sampled_responder50_week12_bsc": "sampled Week 12 >=50% MMD responder rate(No preventive treatment)",
    "sampled_responder50_week12_erenumab_plus_bsc": "sampled Week 12 >=50% MMD responder rate(Erenumab preventive treatment)",
    "sampled_utility_0_3": "sampled 0-3 MMD utility",
    "sampled_utility_4_9": "sampled 4-9 MMD utility",
    "sampled_utility_10_14": "sampled 10-14 MMD utility",
    "sampled_utility_15_19": "sampled 15-19 MMD utility",
    "sampled_utility_20_23": "sampled 20-23 MMD utility",
    "sampled_utility_24_28": "sampled 24-28 MMD utility",
    "sampled_utility_24_30": "sampled 24-28 MMD utility",
    "probability_erenumab_plus_bsc_cost_effective": "Erenumab preventive treatment cost-effective probability",
    "probability_bsc_cost_effective": "No preventive treatment cost-effective probability",
    "probability_cost_effective": "intervention strategy cost-effective probability",
    "mean_incremental_nmb": "mean incremental net monetary benefit",
    "figure": "figure",
    "path": "path",
}


EN_COLUMNS = {
    "arm": "arm",
    "name_cn": "parameter_name",
    "sampled_mmd_change_w12_bsc": "sampled_week12_mmd_change_no_preventive_treatment",
    "sampled_mmd_change_w12_erenumab_plus_bsc": "sampled_week12_mmd_change_erenumab_preventive_treatment",
    "sampled_mmd_lt15_week12_bsc": "sampled_week12_mmd_lt15_no_preventive_treatment",
    "sampled_mmd_lt15_week12_erenumab_plus_bsc": "sampled_week12_mmd_lt15_erenumab_preventive_treatment",
    "sampled_responder50_week12_bsc": "sampled_week12_responder50_no_preventive_treatment",
    "sampled_responder50_week12_erenumab_plus_bsc": "sampled_week12_responder50_erenumab_preventive_treatment",
    "probability_erenumab_plus_bsc_cost_effective": "probability_erenumab_preventive_treatment_cost_effective",
    "probability_bsc_cost_effective": "probability_no_preventive_treatment_cost_effective",
    "intervention_total_cost": "erenumab_preventive_treatment_total_cost",
    "intervention_total_qaly": "erenumab_preventive_treatment_total_qaly",
    "comparator_total_cost": "no_preventive_treatment_total_cost",
    "comparator_total_qaly": "no_preventive_treatment_total_qaly",
}


ZH_VALUES = {
    "run_settings": "Run settings",
    "model": "Model settings",
    "baseline_mmd_distribution": "Baseline MMD distribution",
    "cost_parameters": "Cost parameters",
    "analysis": "Analysis type",
    "basecase_n_patients": "Base-case patients",
    "owsa_n_patients": "OWSA patients",
    "psa_iterations": "PSA iterations",
    "psa_n_patients": "PSA patients per iteration",
    "seed": "Random seed",
    "calibration_mode": "Calibration mode",
    "language": "Output language",
    "language_request": "Output language request",
    "perspective": "Perspective",
    "currency": "Currency",
    "cycle_length_weeks": "Cycle length (weeks)",
    "total_cycles": "Total cycles",
    "decision_tree_cycles": "Short-term cycles",
    "maintenance_cycles": "Maintenance cycles",
    "time_horizon_weeks": "Time horizon (weeks)",
    "discount_rate_annual": "Annual discount rate",
    "wtp_1x_gdp_cny_per_qaly": "1x GDP threshold",
    "wtp_3x_gdp_cny_per_qaly": "3x GDP threshold",
    "main_outcome": "Primary outcome",
    "distribution": "Distribution",
    "mean": "Mean",
    "sd": "Standard deviation",
    "lower": "Lower bound",
    "upper": "Upper bound",
    "n_simulated_patients": "Patients",
    "random_seed": "Random seed",
    "source_id": "Source ID",
    "erenumab_price_per_4w": "Erenumab 70 mg box price",
    "chira_diagnosis_nondrug_cost_usd": "CHIRA diagnosis and non-drug medical service cost",
    "chira_exchange_rate": "CHIRA exchange rate",
    "medical_service_cpi_adjustment_2017_to_2025": "Medical service CPI adjustment factor 2017 to 2025",
    "non_drug_medical_service_cost_annual_2025": "2025 non-drug medical service cost",
    "non_drug_medical_service_cost_per_4w_2025": "2025 non-drug medical service cost per 4 weeks",
    "acute_medication_cost_per_day": "Weighted daily acute medication cost",
    "erenumab_plus_bsc": INTERVENTION_LABEL_ZH,
    "bsc": COMPARATOR_LABEL_ZH,
    "incremental": "Incremental",
    "all": "All analyses",
    "frozen": "Frozen calibration",
    "recalibrate": "Recalibrate",
    "zh": "Chinese",
    "en": "English",
    "both": "English and Chinese",
    "sourced": "Sourced",
    "derived": "Derived",
    "base_case_assumption": "Base-case assumption",
    "frozen_state_distribution_rank_calibration": "Frozen state-distribution rank calibration",
    "frozen_smooth_continuous_mmd_calibration": "Frozen smooth continuous MMD calibration",
    "smooth_continuous_mmd_calibration": "Smooth continuous MMD calibration",
    "owsa_tornado": "One-way sensitivity tornado diagram",
    "psa_ceac": "PSA cost-effectiveness acceptability curve",
    "psa_scatter": "PSA incremental cost-effectiveness scatter plot",
    "threshold_price_inmb": "Threshold price plot",
    "baseline_mmd_distribution": "Baseline MMD distribution plot",
    "week12_mmd_continuous_distribution": "Calibrated Week 12 continuous MMD distribution plot",
    "week12_mmd_distribution": "Calibrated Week 12 MMD state-share plot",
    "Legacy state-distribution calibration": "Legacy state-distribution calibration scenario",
    "12-week time horizon": "12-week time horizon scenario",
    "Threshold price at 1x gdp WTP": "Threshold price at 1x GDP WTP",
    "Threshold price at 3x gdp WTP": "Threshold price at 3x GDP WTP",
}


EN_VALUES = {
    "erenumab_plus_bsc": INTERVENTION_LABEL_EN,
    "bsc": COMPARATOR_LABEL_EN,
    "incremental": "Incremental",
}


ZH_PARAMETER_NAMES = {
    "erenumab_price_per_4w": "Erenumab cost",
    "mmd_change_w12_erenumab": "Week 12 MMD change(Erenumab preventive treatment)",
    "mmd_change_w12_bsc": "Week 12 MMD change(No preventive treatment)",
    "utility_4_9": "4-9 MMD utility",
    "disc_allcause_12w": "Short-term all-cause discontinuation rate",
    "disc_allcause_64w": "Long-term all-cause discontinuation rate",
    "utility_20_23": "20-23 MMD utility",
    "utility_15_19": "15-19 MMD utility",
    "utility_0_3": "0-3 MMD utility",
    "utility_24_28": "24-28 MMD utility",
    "utility_24_30": "24-28 MMD utility",
    "utility_10_14": "10-14 MMD utility",
    "baseline_mmd_mean": "Baseline MMD mean",
    "baseline_mmd_sd": "Baseline MMD standard deviation",
    "acute_medication_cost_per_day": "Daily acute medication cost",
    "non_drug_medical_service_cost_per_4w_2025": "Non-drug medical service cost",
}


def _zh_font_family() -> str:
    for font_path in [
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            return font_manager.FontProperties(fname=str(font_path)).get_name()
    return "DejaVu Sans"


def _inputs_sheet(inputs: ModelInputs, settings: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, value in settings.items():
        rows.append({"section": "run_settings", "key": key, "value": value})
    for key, value in inputs.config["model"].items():
        rows.append({"section": "model", "key": key, "value": value})
    for key, value in inputs.config["baseline_mmd_distribution"].items():
        rows.append({"section": "baseline_mmd_distribution", "key": key, "value": value})
    for _, row in inputs.cost_parameters.iterrows():
        rows.append(
            {
                "section": "cost_parameters",
                "key": row["parameter_id"],
                "value": row["value"],
                "unit": row.get("unit"),
                "status": row.get("status"),
                "notes": row.get("notes"),
            }
        )
    return pd.DataFrame(rows)


def _safe_figure_name(path: Path) -> str:
    return str(path.resolve())


def _translate_scalar(value: Any, language: str) -> Any:
    if not isinstance(value, str):
        return value
    if language == "zh":
        return ZH_VALUES.get(value, value)
    if language == "en":
        return EN_VALUES.get(value, value)
    return value


def _translate_frame(frame: pd.DataFrame, language: str) -> pd.DataFrame:
    translated = frame.copy()
    if {"parameter_id", "name_cn"}.issubset(translated.columns):
        if language == "zh":
            translated["name_cn"] = translated.apply(
                lambda row: ZH_PARAMETER_NAMES.get(str(row["parameter_id"]), row["name_cn"]),
                axis=1,
            )
        elif language == "en":
            translated["name_cn"] = translated.apply(
                lambda row: _publishable_parameter_name(str(row["parameter_id"]), language),
                axis=1,
            )
    column_map = ZH_COLUMNS if language == "zh" else EN_COLUMNS if language == "en" else {}
    translated = translated.rename(columns={key: value for key, value in column_map.items() if key in translated.columns})
    for column in translated.columns:
        if translated[column].dtype == object:
            translated[column] = translated[column].map(lambda value: _translate_scalar(value, language))
    return translated


def _sheet_name(name: str, language: str) -> str:
    return ZH_SHEET_NAMES.get(name, name) if language == "zh" else name


def _font_context(language: str) -> dict[str, Any]:
    if language == "zh":
        family = _zh_font_family()
        return {
            "font.family": "sans-serif",
            "font.sans-serif": [family],
            "axes.unicode_minus": False,
        }
    return {"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"]}


def _publishable_parameter_name(parameter_id: str, language: str = "en") -> str:
    if language == "zh":
        return ZH_PARAMETER_NAMES.get(parameter_id, parameter_id.replace("_", " "))
    names = {
        "erenumab_price_per_4w": "Cost of erenumab",
        "mmd_change_w12_erenumab": "Week 12 MMD change (erenumab preventive treatment)",
        "mmd_change_w12_bsc": "Week 12 MMD change (no preventive treatment)",
        "utility_4_9": "Utility of 4-9 MMD",
        "disc_allcause_12w": "Short-term all-cause discontinuation rate",
        "disc_allcause_64w": "Long-term all-cause discontinuation rate",
        "utility_20_23": "Utility of 20-23 MMD",
        "utility_15_19": "Utility of 15-19 MMD",
        "utility_0_3": "Utility of 0-3 MMD",
        "utility_24_28": "Utility of 24-28 MMD",
        "utility_24_30": "Utility of 24-28 MMD",
        "utility_10_14": "Utility of 10-14 MMD",
        "baseline_mmd_mean": "Baseline MMD mean",
        "baseline_mmd_sd": "Baseline MMD standard deviation",
        "acute_medication_cost_per_day": "Daily cost of acute medication",
        "non_drug_medical_service_cost_per_4w_2025": "Non-drug medical service cost",
    }
    return names.get(parameter_id, parameter_id.replace("_", " "))


def _strategy_label(arm: str, language: str) -> str:
    labels = {
        "zh": {
            "intervention": INTERVENTION_LABEL_ZH,
            "comparator": COMPARATOR_LABEL_ZH,
        },
        "en": {
            "intervention": INTERVENTION_LABEL_EN,
            "comparator": COMPARATOR_LABEL_EN,
        },
    }
    return labels.get(language, labels["en"])[arm]


def _draw_tornado_segment(ax, y_value: float, base_value: float, endpoint: float, color: str, label: str | None) -> None:
    left = min(base_value, endpoint)
    width = abs(endpoint - base_value)
    ax.barh(
        y_value,
        width,
        left=left,
        height=0.64,
        color=color,
        edgecolor="black",
        linewidth=1.0,
        label=label,
        zorder=3,
    )


def _draw_baseline_mmd_distribution(
    output_dir: Path,
    inputs: ModelInputs,
    base_output,
    language: str,
) -> Path:
    baseline = np.asarray(base_output.baseline_mmd, dtype=float)
    cfg = inputs.config["baseline_mmd_distribution"]
    mean = float(cfg["mean"])
    sd = float(cfg["sd"])
    lower = float(cfg["lower"])
    upper = float(cfg["upper"])
    loc, scale = _truncated_normal_params_for_moments(mean, sd, lower, upper)
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    x = np.linspace(lower, upper, 600)
    density = truncnorm.pdf(x, a, b, loc=loc, scale=scale)

    with plt.rc_context(_font_context(language)):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        bins = np.arange(lower, upper + 0.5, 0.5)
        ax.hist(
            baseline,
            bins=bins,
            density=True,
            color="#6BA7B5",
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            label="Simulated patients" if language == "zh" else "Simulated patients",
        )
        ax.plot(
            x,
            density,
            color="#8C2D5E",
            linewidth=2.8,
            label="Theoretical truncated-normal density" if language == "zh" else "Theoretical truncated-normal density",
        )
        ax.axvline(mean, color="black", linewidth=2.0, label=("Target mean 19.2" if language == "zh" else "Target mean 19.2"))
        ax.axvline(lower, color="#555555", linewidth=1.4, linestyle="--")
        ax.axvline(upper, color="#555555", linewidth=1.4, linestyle="--")
        ax.text(
            lower + 0.15,
            ax.get_ylim()[1] * 0.92,
            "Lower 8" if language == "zh" else "Lower 8",
            ha="left",
            va="top",
            fontsize=11,
            color="#444444",
        )
        ax.text(
            upper - 0.15,
            ax.get_ylim()[1] * 0.92,
            "Upper 28" if language == "zh" else "Upper 28",
            ha="right",
            va="top",
            fontsize=11,
            color="#444444",
        )
        if language == "zh":
            summary = (
                f"n n={len(baseline):,}\n"
                f"sample mean={np.mean(baseline):.3f}\n"
                f"sample SD={np.std(baseline, ddof=1):.3f}"
            )
        else:
            summary = (
                f"n={len(baseline):,}\n"
                f"sample mean={np.mean(baseline):.3f}\n"
                f"sample SD={np.std(baseline, ddof=1):.3f}"
            )
        fig.text(
            0.79,
            0.78,
            summary,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.95},
        )
        ax.set_title("Baseline MMD distribution" if language == "zh" else "Baseline MMD Distribution", fontsize=18, pad=15)
        ax.set_xlabel("Baseline MMD (days per 4 weeks)" if language == "zh" else "Baseline MMD (days per 4 weeks)", fontsize=13)
        ax.set_ylabel("Density" if language == "zh" else "Density", fontsize=13)
        ax.set_xlim(lower - 0.5, upper + 0.5)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.17),
            ncol=3,
            fontsize=11,
            handlelength=2.6,
            columnspacing=1.8,
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.subplots_adjust(left=0.10, right=0.76, top=0.88, bottom=0.24)
    path = output_dir / ("baseline_mmd_distribution_Chinese.png" if language == "zh" else "baseline_mmd_distribution.png")
    with plt.rc_context(_font_context(language)):
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _draw_week12_mmd_distribution(output_dir: Path, inputs: ModelInputs, base_output, language: str) -> Path:
    intervention = np.asarray(base_output.calibration_results[inputs.intervention_arm].week12_mmd, dtype=float)
    comparator = np.asarray(base_output.calibration_results[inputs.comparator_arm].week12_mmd, dtype=float)
    intervention_label = _strategy_label("intervention", language)
    comparator_label = _strategy_label("comparator", language)
    states = inputs.mmd_states.sort_values("lower_bound").reset_index(drop=True)
    state_labels = [
        str(row["state_name"]).replace(" MMD", "\nMMD")
        for _, row in states.iterrows()
    ]
    intervention_indices = state_indices_from_mmd(intervention, states)
    comparator_indices = state_indices_from_mmd(comparator, states)
    intervention_shares = np.array(
        [np.mean(intervention_indices == idx) for idx in range(len(states))],
        dtype=float,
    )
    comparator_shares = np.array(
        [np.mean(comparator_indices == idx) for idx in range(len(states))],
        dtype=float,
    )
    with plt.rc_context(_font_context(language)):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        x = np.arange(len(states))
        width = 0.36
        comparator_bars = ax.bar(
            x - width / 2,
            comparator_shares,
            width,
            color="#7A7A7A",
            edgecolor="black",
            linewidth=0.7,
            label=comparator_label,
            zorder=3,
        )
        intervention_bars = ax.bar(
            x + width / 2,
            intervention_shares,
            width,
            color="#2C7FB8",
            edgecolor="black",
            linewidth=0.7,
            label=intervention_label,
            zorder=3,
        )
        for bars in [comparator_bars, intervention_bars]:
            for bar in bars:
                height = float(bar.get_height())
                if height >= 0.03:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.008,
                        f"{height:.1%}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
        if language == "zh":
            summary = (
                f"{intervention_label}:\n"
                f"Mean={np.mean(intervention):.2f},MMD<15={np.mean(intervention < 15):.1%}\n"
                f"{comparator_label}:\n"
                f"Mean={np.mean(comparator):.2f},MMD<15={np.mean(comparator < 15):.1%}"
            )
        else:
            summary = (
                f"{intervention_label}: mean={np.mean(intervention):.2f}, MMD<15={np.mean(intervention < 15):.1%}\n"
                f"{comparator_label}: mean={np.mean(comparator):.2f}, MMD<15={np.mean(comparator < 15):.1%}"
            )
        ax.text(
            0.98,
            0.97,
            summary,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.95},
        )
        ax.set_title("Calibrated Week 12 MMD State Shares" if language == "zh" else "Calibrated Week 12 MMD State Shares", fontsize=18, pad=15)
        ax.set_xlabel("MMD state" if language == "zh" else "MMD state", fontsize=13)
        ax.set_ylabel("Patient share" if language == "zh" else "Patient share", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(state_labels, fontsize=11)
        y_max = max(0.55, float(max(np.max(intervention_shares), np.max(comparator_shares))) + 0.10)
        ax.set_ylim(0.0, y_max)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
        ax.legend(frameon=False, loc="upper left", fontsize=11)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
    path = output_dir / ("week12_mmd_distribution_Chinese.png" if language == "zh" else "week12_mmd_distribution.png")
    with plt.rc_context(_font_context(language)):
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _draw_week12_mmd_continuous_distribution(
    output_dir: Path,
    inputs: ModelInputs,
    base_output,
    language: str,
) -> Path:
    intervention = np.asarray(base_output.calibration_results[inputs.intervention_arm].week12_mmd, dtype=float)
    comparator = np.asarray(base_output.calibration_results[inputs.comparator_arm].week12_mmd, dtype=float)
    intervention_label = _strategy_label("intervention", language)
    comparator_label = _strategy_label("comparator", language)
    bins = np.arange(MMD_MIN, MMD_MAX + 0.5, 0.5)

    with plt.rc_context(_font_context(language)):
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        ax.hist(
            comparator,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color="#7A7A7A",
            alpha=0.26,
            edgecolor="#4C4C4C",
            linewidth=1.4,
            label=comparator_label,
            zorder=2,
        )
        ax.hist(
            intervention,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color="#2C7FB8",
            alpha=0.30,
            edgecolor="#1F5E8A",
            linewidth=1.4,
            label=intervention_label,
            zorder=3,
        )
        ax.axvline(15.0, color="black", linestyle="--", linewidth=1.6, zorder=4)
        if language == "zh":
            summary = (
                f"{intervention_label}:\n"
                f"Mean={np.mean(intervention):.2f},>=50% MMD responder rate={base_output.calibration_results[inputs.intervention_arm].responder50_week12:.1%},MMD<15={np.mean(intervention < 15):.1%}\n"
                f"{comparator_label}:\n"
                f"Mean={np.mean(comparator):.2f},>=50% MMD responder rate={base_output.calibration_results[inputs.comparator_arm].responder50_week12:.1%},MMD<15={np.mean(comparator < 15):.1%}"
            )
        else:
            summary = (
                f"{intervention_label}: mean={np.mean(intervention):.2f}, responder50={base_output.calibration_results[inputs.intervention_arm].responder50_week12:.1%}, MMD<15={np.mean(intervention < 15):.1%}\n"
                f"{comparator_label}: mean={np.mean(comparator):.2f}, responder50={base_output.calibration_results[inputs.comparator_arm].responder50_week12:.1%}, MMD<15={np.mean(comparator < 15):.1%}"
            )
        ax.text(
            0.98,
            0.97,
            summary,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.40", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.95},
        )
        ax.set_title(
            "Calibrated Week 12 continuous MMD distribution" if language == "zh" else "Calibrated Week 12 Continuous MMD Distribution",
            fontsize=18,
            pad=15,
        )
        ax.set_xlabel("Week 12 MMD (days per 4 weeks)" if language == "zh" else "Week 12 MMD (days per 4 weeks)", fontsize=13)
        ax.set_ylabel("Density" if language == "zh" else "Density", fontsize=13)
        ax.set_xlim(MMD_MIN, MMD_MAX)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.3)
        ax.legend(frameon=False, loc="upper left", fontsize=11)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.14)
    path = output_dir / (
        "week12_mmd_continuous_distribution_Chinese.png"
        if language == "zh"
        else "week12_mmd_continuous_distribution.png"
    )
    with plt.rc_context(_font_context(language)):
        fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def _draw_threshold_price_inmb(
    output_dir: Path,
    threshold_prices: pd.DataFrame,
    language: str,
) -> Path | None:
    required_columns = {
        "scenario_id",
        "wtp_cny_per_qaly",
        "threshold_price_per_70mg_box_cny",
        "current_price_per_70mg_box_cny",
        "non_drug_incremental_cost_cny",
        "incremental_qaly",
        "erenumab_cycles_or_doses_per_patient",
    }
    if threshold_prices.empty or not required_columns.issubset(threshold_prices.columns):
        return None

    plot_data = threshold_prices.copy()
    current_price = float(plot_data["current_price_per_70mg_box_cny"].iloc[0])
    max_threshold_price = float(np.nanmax(plot_data["threshold_price_per_70mg_box_cny"].to_numpy(dtype=float)))
    x_max = max(current_price * 1.08, max_threshold_price * 1.25, 1.0)
    prices = np.linspace(0.0, x_max, 400)
    colors = {
        "threshold_price_1x_gdp": "#2F6F73",
        "threshold_price_3x_gdp": "#9A4D76",
    }

    with plt.rc_context(_font_context(language)):
        fig, ax = plt.subplots(figsize=(8.8, 5.4))
        y_values: list[np.ndarray] = []
        for _, row in plot_data.sort_values("wtp_cny_per_qaly").iterrows():
            scenario_id = str(row["scenario_id"])
            wtp = float(row["wtp_cny_per_qaly"])
            incremental_qaly = float(row["incremental_qaly"])
            non_drug_incremental_cost = float(row["non_drug_incremental_cost_cny"])
            doses_per_patient = float(row["erenumab_cycles_or_doses_per_patient"])
            threshold_price = float(row["threshold_price_per_70mg_box_cny"])
            inmb = wtp * incremental_qaly - (non_drug_incremental_cost + prices * doses_per_patient)
            y_values.append(inmb)

            if scenario_id.endswith("1x_gdp"):
                label = "1x GDP" if language == "zh" else "1x GDP"
            elif scenario_id.endswith("3x_gdp"):
                label = "3x GDP" if language == "zh" else "3x GDP"
            else:
                label = scenario_id.replace("_", " ")
            color = colors.get(scenario_id, "#386FA4")
            ax.plot(prices, inmb, color=color, linewidth=2.8, label=label, zorder=4)
            ax.scatter([threshold_price], [0.0], s=46, color=color, edgecolor="white", linewidth=0.8, zorder=6)
            ax.axvline(threshold_price, color=color, linestyle="--", linewidth=1.1, alpha=0.75, zorder=2)
            ax.annotate(
                f"{label}\n{threshold_price:,.2f}",
                xy=(threshold_price, 0.0),
                xytext=(8, 12 if scenario_id.endswith("3x_gdp") else -34),
                textcoords="offset points",
                ha="left",
                va="bottom" if scenario_id.endswith("3x_gdp") else "top",
                fontsize=10,
                color=color,
            )

        ax.axhline(0.0, color="black", linestyle="-", linewidth=1.4, zorder=3)
        ax.text(
            x_max * 0.58,
            0.0,
            "INMB=0",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )
        ax.axvline(current_price, color="#555555", linestyle=":", linewidth=1.6, zorder=2)
        current_price_label = (
            f"Current price {current_price:,.2f} CNY"
            if language == "zh"
            else f"Current price {current_price:,.2f} CNY"
        )
        ax.annotate(
            current_price_label,
            xy=(current_price, 0.0),
            xytext=(-8, 10),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=10,
            color="#444444",
        )
        all_y = np.concatenate(y_values + [np.array([0.0])])
        y_min = float(np.nanmin(all_y))
        y_max = float(np.nanmax(all_y))
        y_pad = max((y_max - y_min) * 0.12, 1000.0)
        ax.set_xlim(0.0, x_max)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.set_xlabel(
            "Erenumab price (CNY per 70 mg box)" if language == "zh" else "Erenumab price (CNY per 70 mg box)",
            fontsize=13,
        )
        ax.set_ylabel("INMB (CNY)" if language == "zh" else "INMB (CNY)", fontsize=13)
        ax.set_title(
            "Incremental net monetary benefit by erenumab price" if language == "zh" else "Incremental net monetary benefit by erenumab price",
            fontsize=17,
            pad=14,
        )
        ax.grid(axis="y", color="#D3D3D3", linestyle="--", linewidth=0.9, alpha=0.75, zorder=0)
        ax.tick_params(axis="both", labelsize=11, width=1.2, length=5)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(1.6)
            ax.spines[spine].set_color("black")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.legend(
            loc="lower left",
            bbox_to_anchor=(0.18, 0.08),
            ncol=2,
            frameon=False,
            fontsize=12,
            handlelength=2.6,
        )
        fig.subplots_adjust(left=0.15, right=0.96, top=0.88, bottom=0.16)

    path = output_dir / ("threshold_price_inmb_Chinese.png" if language == "zh" else "threshold_price_inmb.png")
    with plt.rc_context(_font_context(language)):
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_figures(
    output_dir: Path,
    inputs: ModelInputs,
    base_output,
    owsa: pd.DataFrame | None,
    psa_results: pd.DataFrame | None,
    ceac: pd.DataFrame | None,
    threshold_prices: pd.DataFrame | None = None,
    language: str = "en",
) -> dict[str, Path]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, Path] = {}

    figures["baseline_mmd_distribution"] = _draw_baseline_mmd_distribution(
        figure_dir, inputs, base_output, language
    )
    figures["week12_mmd_continuous_distribution"] = _draw_week12_mmd_continuous_distribution(
        figure_dir, inputs, base_output, language
    )
    figures["week12_mmd_distribution"] = _draw_week12_mmd_distribution(
        figure_dir, inputs, base_output, language
    )

    if owsa is not None and not owsa.empty:
        plot_data = (
            owsa.sort_values("range_width", ascending=False)
            .head(10)
            .sort_values("range_width", ascending=True)
            .reset_index(drop=True)
        )
        base_inmb = float(plot_data["base_inmb_cny"].iloc[0])
        y = np.arange(len(plot_data))
        lower_color = "#2F6F73"
        upper_color = "#9A4D76"
        with plt.rc_context(_font_context(language)):
            fig, ax = plt.subplots(figsize=(12.0, 6.1))
            for i, row in plot_data.iterrows():
                _draw_tornado_segment(
                    ax,
                    y[i],
                    base_inmb,
                    float(row["lower_inmb_cny"]),
                    lower_color,
                    ("Lower" if language == "zh" else "Lower") if i == 0 else None,
                )
                _draw_tornado_segment(
                    ax,
                    y[i],
                    base_inmb,
                    float(row["upper_inmb_cny"]),
                    upper_color,
                    ("Upper" if language == "zh" else "Upper") if i == 0 else None,
                )
            ax.axvline(base_inmb, color="black", linewidth=3.0, zorder=4)
            ax.grid(axis="x", linestyle="--", linewidth=1.0, alpha=0.25, zorder=0)
            ax.set_yticks(y)
            ax.set_yticklabels(
                [_publishable_parameter_name(str(v), language) for v in plot_data["parameter_id"]],
                fontsize=12,
            )
            ax.tick_params(axis="x", labelsize=12, width=1.4, length=6)
            ax.tick_params(axis="y", width=1.4, length=6, pad=6)
            all_values = np.concatenate(
                [
                    plot_data["lower_inmb_cny"].to_numpy(dtype=float),
                    plot_data["upper_inmb_cny"].to_numpy(dtype=float),
                    np.array([base_inmb]),
                ]
            )
            x_min = float(np.nanmin(all_values))
            x_max = float(np.nanmax(all_values))
            x_range = max(x_max - x_min, 1.0)
            left_pad = max(x_range * 0.08, 500.0)
            right_pad = max(x_range * 0.06, 500.0)
            ax.set_xlim(x_min - left_pad, x_max + right_pad)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.set_xlabel(
                "INMB at 3x GDP per capita WTP, CNY"
                if language == "zh"
                else "INMB at 3x GDP per capita WTP, CNY",
                fontsize=15,
            )
            ax.set_title(
                (
                    f"{INTERVENTION_LABEL_ZH} vs. {COMPARATOR_LABEL_ZH}"
                    if language == "zh"
                    else f"{INTERVENTION_LABEL_EN} vs. {COMPARATOR_LABEL_EN}"
                ),
                fontsize=20,
                pad=18,
            )
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_linewidth(2.4)
                ax.spines[spine].set_color("black")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.20),
                ncol=2,
                frameon=False,
                fontsize=14,
                handlelength=2.2,
            )
            fig.subplots_adjust(left=0.43, right=0.98, top=0.86, bottom=0.23)
        path = figure_dir / ("owsa_tornado_Chinese.png" if language == "zh" else "owsa_tornado.png")
        with plt.rc_context(_font_context(language)):
            fig.savefig(path, dpi=300)
        plt.close(fig)
        figures["owsa_tornado"] = path

    if ceac is not None and not ceac.empty:
        erenumab_col = "probability_erenumab_plus_bsc_cost_effective"
        bsc_col = "probability_bsc_cost_effective"
        with plt.rc_context(_font_context(language)):
            fig, ax = plt.subplots(figsize=(8.4, 5.2))
            ax.plot(
                ceac["wtp_cny_per_qaly"],
                ceac[bsc_col],
                color="#2D2D2D",
                linewidth=2.8,
                label=_strategy_label("comparator", language),
                zorder=5,
            )
            ax.plot(
                ceac["wtp_cny_per_qaly"],
                ceac[erenumab_col],
                color="#1F77B4",
                linewidth=2.8,
                label=_strategy_label("intervention", language),
                zorder=5,
            )
            threshold_labels = [("1x GDP" if language == "zh" else "1x GDP"), ("3x GDP" if language == "zh" else "3x GDP")]
            for threshold, threshold_label in zip([99665, 298995], threshold_labels):
                ax.axvline(threshold, color="#A7A7A7", linestyle="--", linewidth=1.1, zorder=1)
                ax.text(
                    threshold + 3500,
                    0.08,
                    threshold_label,
                    rotation=90,
                    ha="left",
                    va="bottom",
                    fontsize=10,
                    color="#555555",
                )
            ax.set_xlim(0, 305000)
            ax.set_ylim(-0.025, 1.025)
            ax.set_xticks(np.arange(0, 300001, 50000))
            ax.set_yticks(np.linspace(0, 1, 6))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}"))
            ax.set_xlabel("Willingness-to-pay threshold (CNY/QALY)" if language == "zh" else "Willingness-to-pay threshold (CNY/QALY)", fontsize=13)
            ax.set_ylabel("Probability cost-effective" if language == "zh" else "Probability cost-effective", fontsize=13)
            ax.set_title("Cost-effectiveness acceptability curve" if language == "zh" else "Cost-effectiveness Acceptability Curve", fontsize=17, pad=14)
            ax.grid(axis="y", color="#D3D3D3", linestyle="--", linewidth=0.9, alpha=0.75, zorder=0)
            ax.tick_params(axis="both", labelsize=11, width=1.2, length=5)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_linewidth(1.6)
                ax.spines[spine].set_color("black")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, -0.19),
                ncol=2,
                frameon=False,
                fontsize=12,
                handlelength=2.6,
            )
            fig.subplots_adjust(left=0.13, right=0.96, top=0.88, bottom=0.24)
        path = figure_dir / ("psa_ceac_Chinese.png" if language == "zh" else "psa_ceac.png")
        with plt.rc_context(_font_context(language)):
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        figures["psa_ceac"] = path

    if threshold_prices is not None and not threshold_prices.empty:
        path = _draw_threshold_price_inmb(figure_dir, threshold_prices, language)
        if path is not None:
            figures["threshold_price_inmb"] = path

    if psa_results is not None and not psa_results.empty:
        with plt.rc_context(_font_context(language)):
            fig, ax = plt.subplots(figsize=(8, 5))
            x = psa_results["incremental_qaly"].to_numpy()
            y = psa_results["incremental_cost"].to_numpy()
            ax.scatter(
                x,
                y,
                s=20,
                alpha=0.65,
                color="#386FA4",
                edgecolors="none",
            )
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            y_min = float(np.nanmin(y))
            y_max = float(np.nanmax(y))
            x_pad = max((x_max - x_min) * 0.08, 0.001)
            y_pad = max((y_max - y_min) * 0.12, 250.0)
            x_left = min(0.0, x_min - x_pad)
            x_right = x_max + x_pad
            y_bottom = y_min - y_pad
            y_top = y_max + y_pad
            ax.set_xlim(x_left, x_right)
            ax.set_ylim(y_bottom, y_top)
            if y_bottom <= 0 <= y_top:
                ax.axhline(0, color="#777777", linewidth=1)
            if x_left <= 0 <= x_right:
                ax.axvline(0, color="#777777", linewidth=1)
            ax.set_xlabel("Incremental QALY" if language == "zh" else "Incremental QALY")
            ax.set_ylabel("Incremental cost (CNY)" if language == "zh" else "Incremental cost (CNY)")
            ax.set_title("PSA incremental cost-effectiveness scatter plot" if language == "zh" else "PSA incremental cost-effectiveness scatter")
            ax.grid(alpha=0.25)
            fig.tight_layout()
        path = figure_dir / ("psa_scatter_Chinese.png" if language == "zh" else "psa_scatter.png")
        with plt.rc_context(_font_context(language)):
            fig.savefig(path, dpi=200)
        plt.close(fig)
        figures["psa_scatter"] = path

    return figures


def write_results_workbook(
    output_path: Path,
    inputs: ModelInputs,
    base_output,
    run_settings: dict[str, Any],
    owsa: pd.DataFrame | None = None,
    psa_results: pd.DataFrame | None = None,
    ceac: pd.DataFrame | None = None,
    scenario_results: pd.DataFrame | None = None,
    threshold_prices: pd.DataFrame | None = None,
    figures: dict[str, Path] | None = None,
    language: str = "en",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figures = figures or {}
    figures_index = pd.DataFrame(
        [{"figure": key, "path": _safe_figure_name(path)} for key, path in figures.items()]
    )

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        _translate_frame(_inputs_sheet(inputs, run_settings), language).to_excel(
            writer, sheet_name=_sheet_name("Inputs", language), index=False
        )
        _translate_frame(base_output.results, language).to_excel(
            writer, sheet_name=_sheet_name("Basecase_Results", language), index=False
        )
        _translate_frame(base_output.cycle_trace, language).to_excel(
            writer, sheet_name=_sheet_name("Cycle_Trace", language), index=False
        )
        _translate_frame(base_output.calibration_diagnostics, language).to_excel(
            writer, sheet_name=_sheet_name("Calibration_Diagnostics", language), index=False
        )
        _translate_frame(owsa if owsa is not None else pd.DataFrame(), language).to_excel(
            writer, sheet_name=_sheet_name("OWSA", language), index=False
        )
        _translate_frame(psa_results if psa_results is not None else pd.DataFrame(), language).to_excel(
            writer, sheet_name=_sheet_name("PSA_Results", language), index=False
        )
        _translate_frame(ceac if ceac is not None else pd.DataFrame(), language).to_excel(
            writer, sheet_name=_sheet_name("CEAC", language), index=False
        )
        _translate_frame(scenario_results if scenario_results is not None else pd.DataFrame(), language).to_excel(
            writer, sheet_name=_sheet_name("Scenario_Results", language), index=False
        )
        _translate_frame(threshold_prices if threshold_prices is not None else pd.DataFrame(), language).to_excel(
            writer, sheet_name=_sheet_name("Threshold_Prices", language), index=False
        )
        _translate_frame(figures_index, language).to_excel(
            writer, sheet_name=_sheet_name("Figures_Index", language), index=False
        )

        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            for column_cells in sheet.columns:
                max_length = 0
                column_letter = column_cells[0].column_letter
                for cell in column_cells[:200]:
                    value = "" if cell.value is None else str(cell.value)
                    max_length = max(max_length, len(value))
                sheet.column_dimensions[column_letter].width = min(max(max_length + 2, 10), 42)
