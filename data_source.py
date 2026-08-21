from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"
PARAMETERS_FILE = "parameters.yaml"


def _load_parameters(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    try:
        import yaml
    except ModuleNotFoundError:
        text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
        return json.loads(text)
    parsed = yaml.safe_load(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return parsed


def _coerce(value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if any(ch in value for ch in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows_without_comments = (line for line in handle if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows_without_comments)
        return [{key: _coerce(value or "") for key, value in row.items()} for row in reader]


@dataclass(frozen=True)
class ModelData:
    data_dir: Path
    config: dict[str, Any]
    tables: dict[str, list[dict[str, Any]]]

    def table(self, name: str) -> list[dict[str, Any]]:
        try:
            return self.tables[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.tables))
            raise KeyError(f"Unknown table '{name}'. Available tables: {available}") from exc

    def by_id(self, table_name: str, id_column: str | None = None) -> dict[str, dict[str, Any]]:
        rows = self.table(table_name)
        if not rows:
            return {}
        if id_column is None:
            for candidate in ("parameter_id", "scenario_id", "state_id", "source_id"):
                if candidate in rows[0]:
                    id_column = candidate
                    break
        if id_column is None:
            raise ValueError(f"Cannot infer id column for table '{table_name}'")
        output: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = row.get(id_column)
            if not key:
                raise ValueError(f"Missing {id_column} in table '{table_name}': {row}")
            if str(key) in output:
                raise ValueError(f"Duplicate {id_column}='{key}' in table '{table_name}'")
            output[str(key)] = row
        return output

    def cost_value(self, parameter_id: str) -> float:
        row = self.by_id("cost_parameters")[parameter_id]
        value = row.get("value")
        if not isinstance(value, (int, float)):
            raise TypeError(f"Cost parameter '{parameter_id}' is not numeric: {value!r}")
        return float(value)

    def weighted_acute_rescue_daily_cost(
        self,
        cost_column: str = "daily_cost_cny",
        require_complete: bool = True,
    ) -> float | None:
        rows = self.table("acute_rescue_drug_costs")
        total = 0.0
        missing: list[str] = []
        for row in rows:
            proportion = row.get("proportion")
            daily_cost = row.get(cost_column)
            parameter_id = str(row.get("parameter_id"))
            if not isinstance(proportion, (int, float)):
                raise TypeError(f"Invalid acute rescue proportion for {parameter_id}: {proportion!r}")
            if not isinstance(daily_cost, (int, float)):
                missing.append(parameter_id)
                continue
            total += float(proportion) * float(daily_cost)
        if missing and require_complete:
            raise ValueError(
                "Missing daily_cost_cny for acute rescue rows: " + ", ".join(missing)
            )
        if missing:
            return None
        return total

    def medical_service_cpi_adjustment(self) -> float:
        rows = self.table("medical_service_cpi")
        factor = 1.0
        for row in rows:
            if row.get("included_in_2017_to_2025_adjustment") is True:
                index_value = row.get("index_value")
                if not isinstance(index_value, (int, float)):
                    raise TypeError(f"Invalid medical service CPI index row: {row}")
                factor *= float(index_value) / 100.0
        return factor

    def scenario(self, scenario_id: str = "basecase") -> dict[str, Any]:
        return self.by_id("scenario_settings")[scenario_id]

    def validate(self) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        data_files = self.config.get("data_files", {})
        for table_name, file_name in data_files.items():
            if table_name not in self.tables:
                errors.append(f"Configured table '{table_name}' was not loaded")
            if not (self.data_dir / str(file_name)).exists():
                errors.append(f"Configured file missing: {file_name}")

        required_tables = {
            "mmd_states",
            "clinical_trajectory",
            "acute_medication_days",
            "acute_rescue_drug_costs",
            "medical_service_cpi",
            "discontinuation_parameters",
            "cost_parameters",
            "scenario_settings",
            "psa_parameters",
            "sources",
        }
        missing_tables = required_tables - set(self.tables)
        if missing_tables:
            errors.append(f"Missing required tables: {sorted(missing_tables)}")

        source_rows = self.tables.get("sources", [])
        if source_rows:
            source_columns = set(source_rows[0])
            expected_source_columns = {"source_id", "reference"}
            if source_columns != expected_source_columns:
                errors.append(
                    "sources.csv must contain exactly the columns source_id and reference"
                )
        source_ids = {str(row["source_id"]) for row in source_rows if row.get("source_id")}
        if len(source_ids) != len(source_rows):
            errors.append("sources.csv source_id values must be non-empty and unique")
        for table_name, rows in self.tables.items():
            if table_name == "sources" or not rows or "source_id" not in rows[0]:
                continue
            for row in rows:
                raw_source = row.get("source_id")
                if not raw_source:
                    errors.append(f"{table_name} row is missing source_id: {row}")
                    continue
                for source_id in str(raw_source).split(";"):
                    source_id = source_id.strip()
                    if source_id and source_id not in source_ids:
                        errors.append(f"{table_name} references unknown source_id '{source_id}'")

        for row in self.tables.get("mmd_states", []):
            utility = row.get("utility")
            if not isinstance(utility, (int, float)) or not 0 <= float(utility) <= 1:
                errors.append(f"Invalid utility in mmd_states: {row}")

        robustness = self.config.get("calibration_robustness", {})
        weights = robustness.get("mmd_lt15_weights")
        if not isinstance(weights, list) or not weights:
            errors.append("calibration_robustness.mmd_lt15_weights must be a non-empty list")
        elif any(not isinstance(value, (int, float)) or float(value) < 0.0 for value in weights):
            errors.append("calibration_robustness.mmd_lt15_weights must be nonnegative numbers")
        for key in ("primary_mean_mmd_tolerance", "primary_responder50_tolerance"):
            value = robustness.get(key)
            if not isinstance(value, (int, float)) or float(value) <= 0.0:
                errors.append(f"calibration_robustness.{key} must be positive")

        cost_ids = self.by_id("cost_parameters") if "cost_parameters" in self.tables else {}
        for required_id in ("erenumab_price_per_4w", "non_drug_medical_service_cost_per_4w_2025"):
            if required_id not in cost_ids:
                errors.append(f"Missing required cost parameter '{required_id}'")

        cpi_row = cost_ids.get("medical_service_cpi_adjustment_2017_to_2025")
        if cpi_row:
            derived_cpi = self.medical_service_cpi_adjustment()
            recorded_cpi = cpi_row.get("value")
            if isinstance(recorded_cpi, (int, float)) and abs(float(recorded_cpi) - derived_cpi) > 1e-8:
                errors.append(
                    "medical_service_cpi_adjustment_2017_to_2025 does not match "
                    f"medical_service_cpi.csv derived value {derived_cpi:.10f}"
                )

        non_drug_annual = cost_ids.get("non_drug_medical_service_cost_annual_2025")
        non_drug_cycle = cost_ids.get("non_drug_medical_service_cost_per_4w_2025")
        if non_drug_annual and non_drug_cycle:
            annual_value = non_drug_annual.get("value")
            cycle_value = non_drug_cycle.get("value")
            if isinstance(annual_value, (int, float)) and isinstance(cycle_value, (int, float)):
                derived_cycle = float(annual_value) / float(self.config["model"]["total_cycles"])
                if abs(float(cycle_value) - derived_cycle) > 1e-5:
                    errors.append(
                        "non_drug_medical_service_cost_per_4w_2025 does not match annual cost / total cycles "
                        f"({derived_cycle:.6f})"
                    )

        for optional_id in ("acute_medication_cost_per_day", "constipation_management_cost"):
            row = cost_ids.get(optional_id)
            if row and row.get("status") in {"incomplete", "missing_value", "pending_daily_unit_costs"}:
                warnings.append(f"{optional_id} requires review before final analysis")

        acute_rescue_rows = self.tables.get("acute_rescue_drug_costs", [])
        if acute_rescue_rows:
            total_proportion = sum(float(row.get("proportion") or 0) for row in acute_rescue_rows)
            if abs(total_proportion - 1.0) > 1e-6:
                errors.append(f"acute_rescue_drug_costs proportion sums to {total_proportion:.6f}, expected 1.0")
            missing_daily_cost = [
                str(row.get("parameter_id"))
                for row in acute_rescue_rows
                if not isinstance(row.get("daily_cost_cny"), (int, float))
            ]
            if missing_daily_cost:
                warnings.append(
                    "acute_rescue_drug_costs daily_cost_cny is pending for: "
                    + ", ".join(missing_daily_cost)
                )
            else:
                derived_mean = self.weighted_acute_rescue_daily_cost("daily_cost_cny")
                derived_lower = self.weighted_acute_rescue_daily_cost("lower")
                derived_upper = self.weighted_acute_rescue_daily_cost("upper")
                acute_cost_row = cost_ids.get("acute_medication_cost_per_day")
                if acute_cost_row:
                    for column, derived_value in (
                        ("value", derived_mean),
                        ("lower", derived_lower),
                        ("upper", derived_upper),
                    ):
                        recorded = acute_cost_row.get(column)
                        if isinstance(recorded, (int, float)) and isinstance(derived_value, (int, float)):
                            if abs(float(recorded) - float(derived_value)) > 1e-5:
                                errors.append(
                                    "acute_medication_cost_per_day "
                                    f"{column}={recorded} does not match acute_rescue_drug_costs derived "
                                    f"{derived_value:.6f}"
                                )

        discontinuation_ids = (
            self.by_id("discontinuation_parameters") if "discontinuation_parameters" in self.tables else {}
        )
        discontinuation_cfg = self.config.get("discontinuation", {})
        for config_key in ("short_term_rule", "maintenance_rule", "lack_efficacy_share_rule"):
            parameter_id = str(discontinuation_cfg.get(config_key) or "")
            if not parameter_id or parameter_id not in discontinuation_ids:
                errors.append(f"Missing discontinuation parameter configured by '{config_key}': {parameter_id}")
                continue
            value = discontinuation_ids[parameter_id].get("original_value")
            if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                errors.append(f"Invalid probability for discontinuation parameter '{parameter_id}': {value}")

        retention_profile = discontinuation_cfg.get("retained_effect_fractions_by_cycle")
        if not isinstance(retention_profile, list) or not retention_profile:
            errors.append("discontinuation.retained_effect_fractions_by_cycle must be a non-empty list")
        else:
            try:
                retention_values = [float(value) for value in retention_profile]
            except (TypeError, ValueError):
                errors.append("Post-discontinuation retained-effect fractions must be numeric")
            else:
                if any(value < 0 or value > 1 for value in retention_values):
                    errors.append("Post-discontinuation retained-effect fractions must be between 0 and 1")
                if any(left < right for left, right in zip(retention_values, retention_values[1:])):
                    errors.append("Post-discontinuation retained-effect fractions must be non-increasing")
        terminal_retention = discontinuation_cfg.get("retained_effect_fraction_after_profile")
        if not isinstance(terminal_retention, (int, float)) or float(terminal_retention) != 0.0:
            errors.append("retained_effect_fraction_after_profile must be 0 for complete effect loss")

        scenarios = self.by_id("scenario_settings") if "scenario_settings" in self.tables else {}
        if "basecase" not in scenarios:
            errors.append("Missing basecase scenario")

        trajectory = self.tables.get("clinical_trajectory", [])
        week12_arms = {row.get("arm") for row in trajectory if row.get("timepoint") == "week12"}
        if {"erenumab_plus_bsc", "bsc"} - week12_arms:
            errors.append("clinical_trajectory must include week12 rows for both arms")

        return errors, warnings


def load_model_data(data_dir: str | Path | None = None) -> ModelData:
    resolved_data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    resolved_data_dir = resolved_data_dir.resolve()
    config = _load_parameters(resolved_data_dir / PARAMETERS_FILE)
    tables: dict[str, list[dict[str, Any]]] = {}
    for table_name, file_name in config.get("data_files", {}).items():
        tables[table_name] = _read_csv(resolved_data_dir / str(file_name))
    return ModelData(data_dir=resolved_data_dir, config=config, tables=tables)


def main() -> int:
    data = load_model_data()
    errors, warnings = data.validate()
    print(f"Data directory: {data.data_dir}")
    print(f"Loaded tables: {', '.join(sorted(data.tables))}")
    print(f"MMD states: {len(data.table('mmd_states'))}")
    print(f"Scenarios: {len(data.table('scenario_settings'))}")
    print(f"Base-case erenumab price per 4w: {data.cost_value('erenumab_price_per_4w'):.2f} CNY")
    print(
        "Base-case non-drug medical service cost per 4w: "
        f"{data.cost_value('non_drug_medical_service_cost_per_4w_2025'):.4f} CNY"
    )
    print(f"Medical service CPI adjustment factor: {data.medical_service_cpi_adjustment():.10f}")
    acute_cost = data.weighted_acute_rescue_daily_cost(require_complete=False)
    if acute_cost is None:
        print("Weighted acute rescue daily cost: pending daily_cost_cny inputs")
    else:
        print(f"Weighted acute rescue daily cost: {acute_cost:.4f} CNY/day")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
