from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_inputs
from src.sensitivity import run_calibration_robustness_analysis
from src.simulation import SimulationSettings, run_basecase_simulation


ROOT = Path(__file__).resolve().parents[1]


class CalibrationRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs(ROOT / "data")
        cls.settings = SimulationSettings(n_patients=300, seed=314159)
        cls.base_output = run_basecase_simulation(cls.inputs, cls.settings)

    def run_small_analysis(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return run_calibration_robustness_analysis(
            self.inputs,
            self.base_output,
            screening_n_patients=180,
            seed=314159,
            weights=[0.0, 0.6],
            de_maxiter=1,
            de_popsize=3,
            local_maxiter=6,
        )

    def test_analysis_is_separate_reproducible_and_reports_compatible_ranges(self) -> None:
        base_before = self.base_output.results.copy(deep=True)
        first_results, first_summary = self.run_small_analysis()
        second_results, second_summary = self.run_small_analysis()

        pd.testing.assert_frame_equal(first_results, second_results)
        pd.testing.assert_frame_equal(first_summary, second_summary)
        pd.testing.assert_frame_equal(self.base_output.results, base_before)

        self.assertEqual(len(first_results), 3)
        self.assertEqual(
            int(np.sum(first_results["analysis_type"] == "prespecified_target_weight")),
            2,
        )
        base_reference = first_results.loc[
            first_results["analysis_type"] == "basecase_reference"
        ].iloc[0]
        self.assertTrue(np.isnan(float(base_reference["screening_n_patients"])))
        self.assertIn("incremental_qaly", first_results.columns)
        self.assertIn("icer_cny_per_qaly", first_results.columns)

        compatible_candidates = first_results.loc[
            (first_results["analysis_type"] == "prespecified_target_weight")
            & first_results["primary_targets_compatible"]
        ]
        summary = first_summary.iloc[0]
        self.assertEqual(
            int(summary["n_primary_compatible_target_weights"]),
            len(compatible_candidates),
        )
        self.assertEqual(bool(summary["economic_range_available"]), not compatible_candidates.empty)
        if compatible_candidates.empty:
            self.assertTrue(np.isnan(float(summary["incremental_qaly_min"])))
        else:
            self.assertAlmostEqual(
                float(summary["incremental_qaly_min"]),
                float(compatible_candidates["incremental_qaly"].min()),
            )
        self.assertIn("not a confidence", str(summary["range_interpretation"]))

    def test_default_basecase_is_unchanged_after_robustness_analysis(self) -> None:
        self.run_small_analysis()
        rerun = run_basecase_simulation(self.inputs, self.settings)
        pd.testing.assert_frame_equal(rerun.results, self.base_output.results)
        pd.testing.assert_frame_equal(rerun.cycle_trace, self.base_output.cycle_trace)


if __name__ == "__main__":
    unittest.main()
