from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from data_source import ModelData, load_model_data


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelInputs:
    raw: ModelData
    config: dict[str, Any]
    mmd_states: pd.DataFrame
    clinical_trajectory: pd.DataFrame
    acute_medication_days: pd.DataFrame
    acute_rescue_drug_costs: pd.DataFrame
    discontinuation_parameters: pd.DataFrame
    cost_parameters: pd.DataFrame
    scenario_settings: pd.DataFrame
    psa_parameters: pd.DataFrame
    calibration_parameters: pd.DataFrame
    sources: pd.DataFrame

    @property
    def intervention_arm(self) -> str:
        return str(self.config["arms"]["intervention"])

    @property
    def comparator_arm(self) -> str:
        return str(self.config["arms"]["comparator"])

    @property
    def cycle_length_weeks(self) -> int:
        return int(self.config["model"]["cycle_length_weeks"])

    @property
    def total_cycles(self) -> int:
        return int(self.config["model"]["total_cycles"])

    @property
    def decision_tree_cycles(self) -> int:
        return int(self.config["model"]["decision_tree_cycles"])

    @property
    def time_horizon_weeks(self) -> int:
        return int(self.config["model"]["time_horizon_weeks"])


def _table_frame(data: ModelData, table_name: str) -> pd.DataFrame:
    return pd.DataFrame(data.table(table_name))


def load_inputs(data_dir: str | Path | None = None) -> ModelInputs:
    raw = load_model_data(data_dir)
    errors, warnings = raw.validate()
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Input data validation failed:\n{joined}")
    for warning in warnings:
        print(f"Data warning: {warning}")

    states = _table_frame(raw, "mmd_states").sort_values("lower_bound").reset_index(drop=True)
    return ModelInputs(
        raw=raw,
        config=raw.config,
        mmd_states=states,
        clinical_trajectory=_table_frame(raw, "clinical_trajectory"),
        acute_medication_days=_table_frame(raw, "acute_medication_days"),
        acute_rescue_drug_costs=_table_frame(raw, "acute_rescue_drug_costs"),
        discontinuation_parameters=_table_frame(raw, "discontinuation_parameters"),
        cost_parameters=_table_frame(raw, "cost_parameters"),
        scenario_settings=_table_frame(raw, "scenario_settings"),
        psa_parameters=_table_frame(raw, "psa_parameters"),
        calibration_parameters=_table_frame(raw, "calibration_parameters")
        if "calibration_parameters" in raw.tables
        else pd.DataFrame(),
        sources=_table_frame(raw, "sources"),
    )


def records_by_id(frame: pd.DataFrame, id_column: str) -> dict[str, dict[str, Any]]:
    return {str(row[id_column]): row.to_dict() for _, row in frame.iterrows()}


def cost_value(inputs: ModelInputs, parameter_id: str, overrides: dict[str, Any] | None = None) -> float:
    if overrides and parameter_id in overrides:
        return float(overrides[parameter_id])
    rows = inputs.cost_parameters.loc[inputs.cost_parameters["parameter_id"] == parameter_id]
    if rows.empty:
        raise KeyError(f"Missing cost parameter: {parameter_id}")
    value = rows.iloc[0]["value"]
    if pd.isna(value):
        raise ValueError(f"Cost parameter has no numeric value: {parameter_id}")
    return float(value)


def cost_range(inputs: ModelInputs, parameter_id: str) -> tuple[float | None, float | None]:
    rows = inputs.cost_parameters.loc[inputs.cost_parameters["parameter_id"] == parameter_id]
    if rows.empty:
        return None, None
    row = rows.iloc[0]
    lower = None if pd.isna(row.get("lower")) else float(row["lower"])
    upper = None if pd.isna(row.get("upper")) else float(row["upper"])
    return lower, upper


def discontinuation_row(inputs: ModelInputs, parameter_id: str) -> dict[str, Any]:
    rows = inputs.discontinuation_parameters.loc[
        inputs.discontinuation_parameters["parameter_id"] == parameter_id
    ]
    if rows.empty:
        raise KeyError(f"Missing discontinuation parameter: {parameter_id}")
    return rows.iloc[0].to_dict()


def target_value(
    inputs: ModelInputs,
    arm: str,
    target_name: str,
    overrides: dict[str, Any] | None = None,
) -> float:
    key = f"{target_name}_{arm}"
    if overrides and key in overrides:
        return float(overrides[key])
    return float(inputs.config["calibration_targets"][arm][target_name])


def trajectory_value(
    inputs: ModelInputs,
    arm: str,
    timepoint: str,
    value_column: str,
    overrides: dict[str, Any] | None = None,
) -> float:
    if timepoint == "week12" and value_column == "mmd_change":
        override_key = f"mmd_change_w12_{arm}"
        if overrides and override_key in overrides:
            return float(overrides[override_key])
    rows = inputs.clinical_trajectory.loc[
        (inputs.clinical_trajectory["record_type"] == "observed")
        & (inputs.clinical_trajectory["arm"] == arm)
        & (inputs.clinical_trajectory["timepoint"] == timepoint)
    ]
    if rows.empty:
        raise KeyError(f"Missing trajectory row for {arm} {timepoint}")
    return float(rows.iloc[0][value_column])


def acute_medication_day_value(inputs: ModelInputs, arm: str, timepoint: str) -> float:
    rows = inputs.acute_medication_days.loc[
        (inputs.acute_medication_days["arm"] == arm)
        & (inputs.acute_medication_days["timepoint"] == timepoint)
    ]
    if rows.empty:
        raise KeyError(f"Missing acute medication row for {arm} {timepoint}")
    return float(rows.iloc[0]["acute_med_days"])
