from __future__ import annotations

from typing import Any

from .data import ModelInputs, cost_value, discontinuation_row
from .utilities import probability_to_cycle_probability


def erenumab_price_per_cycle(inputs: ModelInputs, overrides: dict[str, Any] | None = None) -> float:
    return cost_value(inputs, "erenumab_price_per_4w", overrides)


def acute_medication_cost_per_day(inputs: ModelInputs, overrides: dict[str, Any] | None = None) -> float:
    return cost_value(inputs, "acute_medication_cost_per_day", overrides)


def non_drug_medical_service_cost_per_cycle(
    inputs: ModelInputs,
    overrides: dict[str, Any] | None = None,
) -> float:
    return cost_value(inputs, "non_drug_medical_service_cost_per_4w_2025", overrides)


def discontinuation_cycle_probability(
    inputs: ModelInputs,
    parameter_id: str,
    overrides: dict[str, Any] | None = None,
) -> float:
    row = discontinuation_row(inputs, parameter_id)
    if overrides and parameter_id in overrides:
        return probability_to_cycle_probability(float(overrides[parameter_id]), float(row["time_window_cycles"]))
    return float(row["converted_4w_probability"])

