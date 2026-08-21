from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_inputs
from src.sensitivity import run_probabilistic_sensitivity, sample_psa_overrides


ROOT = Path(__file__).resolve().parents[1]


class PsaPriceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_inputs(ROOT / "data")

    def test_price_draws_are_positive_reproducible_and_centered_on_basecase(self) -> None:
        def draw_prices(seed: int) -> np.ndarray:
            rng = np.random.default_rng(seed)
            return np.array(
                [sample_psa_overrides(self.inputs, rng)["erenumab_price_per_4w"] for _ in range(5000)]
            )

        first = draw_prices(314159)
        second = draw_prices(314159)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first > 0.0))
        self.assertAlmostEqual(float(np.mean(first)), 3498.0, delta=20.0)

    def test_psa_output_contains_sampled_price_and_ceac_uses_psa_results(self) -> None:
        first_psa, first_ceac = run_probabilistic_sensitivity(
            self.inputs,
            iterations=8,
            n_patients=300,
            seed=314159,
            workers=1,
        )
        second_psa, second_ceac = run_probabilistic_sensitivity(
            self.inputs,
            iterations=8,
            n_patients=300,
            seed=314159,
            workers=1,
        )
        pd.testing.assert_frame_equal(first_psa, second_psa)
        pd.testing.assert_frame_equal(first_ceac, second_ceac)

        price_column = "sampled_erenumab_price_per_4w"
        self.assertIn(price_column, first_psa.columns)
        self.assertTrue(np.all(first_psa[price_column].to_numpy() > 0.0))

        ceac_row = first_ceac.loc[first_ceac["wtp_cny_per_qaly"] == 298995.0].iloc[0]
        incremental_nmb = (
            298995.0 * first_psa["incremental_qaly"] - first_psa["incremental_cost"]
        )
        self.assertAlmostEqual(
            float(ceac_row["probability_cost_effective"]),
            float(np.mean(incremental_nmb > 0.0)),
        )
        self.assertAlmostEqual(
            float(ceac_row["mean_incremental_nmb"]),
            float(np.mean(incremental_nmb)),
        )


if __name__ == "__main__":
    unittest.main()
