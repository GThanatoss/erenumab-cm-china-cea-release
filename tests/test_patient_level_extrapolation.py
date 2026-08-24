from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.costs import discontinuation_cycle_probability
from src.data import load_inputs
from src.simulation import (
    DISC_REASON_LACK_EFFICACY,
    DISC_REASON_OTHER,
    SimulationSettings,
    _patient_level_maintenance_mmd,
    _retained_effect_fraction,
    run_basecase_simulation,
)


ROOT = Path(__file__).resolve().parents[1]


class PatientLevelExtrapolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs(ROOT / "data")

    def test_maintenance_probability_conversion(self) -> None:
        actual = discontinuation_cycle_probability(self.inputs, "disc_treatment_52w")
        expected = 1.0 - (1.0 - 0.228) ** (1.0 / 13.0)
        self.assertAlmostEqual(actual, expected, places=10)

    def test_retained_effect_profile(self) -> None:
        cycles_since = np.array([1, 2, 3, 4, 8])
        actual = _retained_effect_fraction(cycles_since, (0.91, 0.64, 0.45))
        np.testing.assert_allclose(actual, np.array([0.91, 0.64, 0.45, 0.0, 0.0]))

    def test_reason_specific_mmd_paths(self) -> None:
        erenumab = np.array([10.0, 10.0, 15.0, 9.0])
        comparator = np.array([14.0, 12.0, 12.0, 13.0])
        reasons = np.array(
            [DISC_REASON_OTHER, DISC_REASON_OTHER, DISC_REASON_OTHER, DISC_REASON_LACK_EFFICACY],
            dtype=np.int8,
        )
        stop_cycle = np.array([3, 3, 3, 3], dtype=np.int16)

        cycle_4 = _patient_level_maintenance_mmd(
            erenumab, comparator, 4, reasons, stop_cycle, (0.91, 0.64, 0.45)
        )
        np.testing.assert_allclose(cycle_4, np.array([10.36, 10.18, 15.0, 13.0]))

        cycle_7 = _patient_level_maintenance_mmd(
            erenumab, comparator, 7, reasons, stop_cycle, (0.91, 0.64, 0.45)
        )
        np.testing.assert_allclose(cycle_7, np.array([14.0, 12.0, 15.0, 13.0]))

    def test_first_twelve_weeks_do_not_depend_on_maintenance_horizon(self) -> None:
        short = run_basecase_simulation(
            self.inputs,
            SimulationSettings(n_patients=3000, seed=314159, total_cycles=3),
        )
        full = run_basecase_simulation(
            self.inputs,
            SimulationSettings(n_patients=3000, seed=314159),
        )
        full_early = full.cycle_trace.loc[full.cycle_trace["cycle"] <= 3].reset_index(drop=True)
        pd.testing.assert_frame_equal(short.cycle_trace.reset_index(drop=True), full_early)

    def test_simulation_is_reproducible_and_matches_targeted_discontinuation(self) -> None:
        settings = SimulationSettings(n_patients=20000, seed=314159)
        first = run_basecase_simulation(self.inputs, settings)
        second = run_basecase_simulation(self.inputs, settings)
        pd.testing.assert_frame_equal(first.results, second.results)
        pd.testing.assert_frame_equal(first.cycle_trace, second.cycle_trace)

        intervention = first.results.loc[first.results["arm"] == self.inputs.intervention_arm].iloc[0]
        intervention_trace = first.cycle_trace.loc[
            first.cycle_trace["arm"] == self.inputs.intervention_arm
        ].reset_index(drop=True)
        self.assertTrue(
            np.all(np.diff(intervention_trace["erenumab_on_treatment_rate"].to_numpy()) <= 0.0)
        )
        np.testing.assert_allclose(
            intervention_trace["erenumab_drug_cost"].to_numpy(),
            intervention_trace["erenumab_on_treatment_rate"].to_numpy() * 3498.0,
        )
        expected_on_treatment = (1.0 - 0.021505) * (1.0 - 0.228) ** (10.0 / 13.0)
        self.assertAlmostEqual(
            float(intervention["end_horizon_on_treatment_rate"]), expected_on_treatment, delta=0.015
        )
        lack = float(intervention["end_horizon_discontinued_lack_efficacy_rate"])
        other = float(intervention["end_horizon_discontinued_other_reason_rate"])
        self.assertAlmostEqual(
            float(intervention["end_horizon_on_treatment_rate"]) + lack + other,
            1.0,
            places=12,
        )
        self.assertAlmostEqual(lack / (lack + other), 39.0 / 139.0, delta=0.03)

    def test_generic_empty_retention_profile_override_does_not_increase_qalys(self) -> None:
        settings = SimulationSettings(n_patients=5000, seed=314159)
        basecase = run_basecase_simulation(self.inputs, settings)
        immediate = run_basecase_simulation(
            self.inputs,
            settings,
            overrides={"_retention_profile": []},
        )
        base_qaly = float(
            basecase.results.loc[basecase.results["arm"] == self.inputs.intervention_arm, "total_qaly"].iloc[0]
        )
        immediate_qaly = float(
            immediate.results.loc[immediate.results["arm"] == self.inputs.intervention_arm, "total_qaly"].iloc[0]
        )
        self.assertLessEqual(immediate_qaly, base_qaly)


if __name__ == "__main__":
    unittest.main()
