from __future__ import annotations

import csv
import unittest
from pathlib import Path

from data_source import load_model_data
from src.data import load_inputs
from src.sensitivity import run_time_horizon_scenarios


ROOT = Path(__file__).resolve().parents[1]


class PublicDataTests(unittest.TestCase):
    def test_sources_schema_and_cross_references(self) -> None:
        source_path = ROOT / "data" / "sources.csv"
        with source_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        self.assertEqual(reader.fieldnames, ["source_id", "reference"])
        self.assertEqual(len(rows), 13)

        model_data = load_model_data(ROOT / "data")
        errors, _ = model_data.validate()
        self.assertEqual(errors, [])
        declared_sources = {row["source_id"] for row in rows}
        referenced_sources: set[str] = set()
        for table_name, table_rows in model_data.tables.items():
            if table_name == "sources":
                continue
            for row in table_rows:
                for source_id in str(row.get("source_id") or "").split(";"):
                    if source_id.strip():
                        referenced_sources.add(source_id.strip())

        def collect_config_sources(value: object) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    if key == "source_id":
                        for source_id in str(nested).split(";"):
                            if source_id.strip():
                                referenced_sources.add(source_id.strip())
                    else:
                        collect_config_sources(nested)
            elif isinstance(value, list):
                for nested in value:
                    collect_config_sources(nested)

        collect_config_sources(model_data.config)
        self.assertEqual(declared_sources, referenced_sources)

    def test_state_inputs_and_scenarios_contain_only_public_model_paths(self) -> None:
        inputs = load_inputs(ROOT / "data")
        self.assertEqual(
            list(inputs.mmd_states.columns),
            [
                "state_id",
                "state_name",
                "lower_bound",
                "upper_bound",
                "representative_mmd",
                "utility",
                "source_id",
                "notes",
            ],
        )

        scenarios = run_time_horizon_scenarios(
            inputs,
            n_patients=200,
            seed=314159,
        )
        self.assertEqual(
            set(scenarios["scenario_id"]),
            {"12_week_time_horizon", "immediate_post_discontinuation_loss"},
        )


if __name__ == "__main__":
    unittest.main()
