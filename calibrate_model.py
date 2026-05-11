from __future__ import annotations

import argparse
from pathlib import Path

from src.calibration import frozen_calibration_rows
from src.data import load_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and freeze Week 12 calibration parameters into data/calibration_parameters.csv"
    )
    parser.add_argument("--data-dir", default="data", help="Input data directory.")
    parser.add_argument(
        "--output",
        default=None,
        help="Calibration CSV output path. Defaults to <data-dir>/calibration_parameters.csv.",
    )
    parser.add_argument("--n-patients", type=int, default=None, help="Calibration patient count.")
    parser.add_argument("--seed", type=int, default=None, help="Calibration random seed.")
    parser.add_argument(
        "--method",
        choices=["continuous", "state"],
        default="continuous",
        help="Calibration method to freeze. 'state' is retained only for legacy validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = load_inputs(args.data_dir)
    baseline_cfg = inputs.config["baseline_mmd_distribution"]
    n_patients = int(args.n_patients or baseline_cfg["n_simulated_patients"])
    seed = int(args.seed or baseline_cfg["random_seed"])
    output_path = Path(args.output) if args.output else Path(args.data_dir) / "calibration_parameters.csv"

    rows = frozen_calibration_rows(
        inputs,
        n_patients=n_patients,
        seed=seed,
        calibration_method=args.method,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Frozen calibration parameters written: {output_path.resolve()}")
    print(rows[["arm", "method", "simulated_mean_mmd_week12", "simulated_responder50_week12", "simulated_mmd_lt15_week12"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
