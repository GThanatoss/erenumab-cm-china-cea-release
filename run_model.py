from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_inputs
from src.outputs import write_figures, write_results_workbook
from src.sensitivity import (
    run_calibration_robustness_analysis,
    run_one_way_sensitivity,
    run_probabilistic_sensitivity,
    run_time_horizon_scenarios,
    threshold_price_analysis,
)
from src.simulation import SimulationSettings, run_basecase_simulation


DEFAULT_PSA_WORKERS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="China erenumab chronic migraine pharmacoeconomic model"
    )
    parser.add_argument(
        "--analysis",
        choices=["basecase", "owsa", "psa", "calibration", "all"],
        default="all",
        help="Analysis set to run.",
    )
    parser.add_argument("--data-dir", default="data", help="Input data directory.")
    parser.add_argument("--output-dir", default="results", help="Output directory.")
    parser.add_argument("--n-patients", type=int, default=None, help="Base-case patient count.")
    parser.add_argument("--owsa-patients", type=int, default=10000, help="OWSA patient count per run.")
    parser.add_argument(
        "--calibration-patients",
        type=int,
        default=10000,
        help="Patient count used to fit each calibration target-weight candidate.",
    )
    parser.add_argument("--psa-iterations", type=int, default=100, help="Number of PSA iterations.")
    parser.add_argument("--psa-patients", type=int, default=10000, help="PSA patient count per iteration.")
    parser.add_argument(
        "--psa-workers",
        type=int,
        default=DEFAULT_PSA_WORKERS,
        help="Parallel worker processes for PSA. Default 16 matches Intel Core i7-13700 physical cores; use 1 for serial execution.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--calibration-mode",
        choices=["frozen", "recalibrate"],
        default="frozen",
        help="Use frozen calibration parameters from data source or recalibrate during the run.",
    )
    parser.add_argument(
        "--language",
        choices=["en"],
        default="en",
        help="Output language for workbook and figures.",
    )
    return parser.parse_args()


def _outputs_for_language(base_output_dir: Path, language: str) -> tuple[Path, Path]:
    return base_output_dir, base_output_dir / "model_results.xlsx"


def _language_jobs(output_dir: Path, language: str) -> list[tuple[str, Path, Path]]:
    lang_dir, workbook = _outputs_for_language(output_dir, language)
    return [(language, lang_dir, workbook)]


def main() -> int:
    args = parse_args()
    inputs = load_inputs(args.data_dir)
    baseline_cfg = inputs.config["baseline_mmd_distribution"]
    n_patients = int(args.n_patients or baseline_cfg["n_simulated_patients"])
    seed = int(args.seed or baseline_cfg["random_seed"])
    output_dir = Path(args.output_dir)

    print(f"Running base-case simulation with n={n_patients}, seed={seed}")
    base_settings = SimulationSettings(
        n_patients=n_patients,
        seed=seed,
        scenario_id="basecase",
        calibration_mode=args.calibration_mode,
    )
    base_output = run_basecase_simulation(
        inputs,
        base_settings,
        overrides=None,
    )

    owsa = None
    psa_results = None
    ceac = None
    calibration_robustness = None
    calibration_summary = None
    if args.analysis in {"owsa", "all"}:
        print(f"Running OWSA with n={args.owsa_patients}")
        owsa = run_one_way_sensitivity(
            inputs,
            base_output,
            args.owsa_patients,
            seed,
            calibration_mode=args.calibration_mode,
        )
    if args.analysis in {"psa", "all"}:
        print(f"Running PSA with iterations={args.psa_iterations}, n={args.psa_patients}")
        psa_results, ceac = run_probabilistic_sensitivity(
            inputs,
            iterations=args.psa_iterations,
            n_patients=args.psa_patients,
            seed=seed + 50000,
            calibration_mode=args.calibration_mode,
            workers=args.psa_workers,
        )
    if args.analysis in {"calibration", "all"}:
        print(
            "Running calibration robustness analysis across prespecified target weights "
            f"with screening n={args.calibration_patients}"
        )
        calibration_robustness, calibration_summary = run_calibration_robustness_analysis(
            inputs,
            base_output,
            screening_n_patients=args.calibration_patients,
            seed=seed,
        )

    print("Running scenario analyses")
    scenario_results = run_time_horizon_scenarios(
        inputs,
        n_patients=n_patients,
        seed=seed,
        calibration_mode=args.calibration_mode,
    )
    threshold_prices = threshold_price_analysis(inputs, base_output)

    run_settings_base = {
        "analysis": args.analysis,
        "basecase_n_patients": n_patients,
        "owsa_n_patients": args.owsa_patients if args.analysis in {"owsa", "all"} else "",
        "calibration_screening_n_patients": (
            args.calibration_patients if args.analysis in {"calibration", "all"} else ""
        ),
        "psa_iterations": args.psa_iterations if args.analysis in {"psa", "all"} else "",
        "psa_n_patients": args.psa_patients if args.analysis in {"psa", "all"} else "",
        "psa_workers": args.psa_workers if args.analysis in {"psa", "all"} else "",
        "seed": seed,
        "calibration_mode": args.calibration_mode,
        "language": args.language,
    }

    written_workbooks: list[Path] = []
    written_figures: list[tuple[str, Path]] = []
    for language, language_output_dir, workbook_path in _language_jobs(output_dir, args.language):
        figures = write_figures(
            language_output_dir,
            inputs,
            base_output,
            owsa,
            psa_results,
            ceac,
            calibration_robustness=calibration_robustness,
            threshold_prices=threshold_prices,
            language=language,
        )
        run_settings = dict(run_settings_base)
        run_settings["language"] = language
        write_results_workbook(
            workbook_path,
            inputs,
            base_output,
            run_settings,
            owsa=owsa,
            psa_results=psa_results,
            ceac=ceac,
            calibration_robustness=calibration_robustness,
            calibration_summary=calibration_summary,
            scenario_results=scenario_results,
            threshold_prices=threshold_prices,
            figures=figures,
            language=language,
        )
        written_workbooks.append(workbook_path)
        written_figures.extend(figures.items())

    incremental = base_output.results.loc[base_output.results["arm"] == "incremental"].iloc[0]
    for workbook_path in written_workbooks:
        print(f"Workbook: {workbook_path.resolve()}")
    for name, path in written_figures:
        print(f"Figure {name}: {path.resolve()}")
    print(
        "Base-case ICER: "
        f"{float(incremental['icer_cny_per_qaly']):,.2f} CNY/QALY "
        f"(incremental cost {float(incremental['total_cost']):,.2f}, "
        f"incremental QALY {float(incremental['total_qaly']):.6f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
