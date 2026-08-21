# China erenumab chronic migraine cost-effectiveness model

This repository contains a Python implementation of a China health-system cost-utility model comparing erenumab preventive treatment with no preventive treatment for adults with chronic migraine.

## Files

- `run_model.py`: main analysis entry point.
- `calibrate_model.py`: regenerates frozen Week 12 calibration parameters when calibration inputs change.
- `validate_data_source.py`: checks that all input files can be read and that derived acute rescue medication cost is internally consistent.
- `data_source.py`: lightweight data reader and validator.
- `src/`: model, calibration, cost, sensitivity, and output modules.
- `data/`: model input tables.
- `docs/model_description.md`: model structure and analysis methods.
- `docs/parameter_table.csv`: model parameter table.
- `results/`: output directory.

## Installation

Use Python 3.10 or later.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Validate the input data

```bash
python validate_data_source.py
```

Expected validation output includes the weighted acute rescue daily cost:

```text
Validation passed.
Weighted acute rescue daily cost: 31.7499 CNY/day
```

## Run the formal analysis

The command below runs the base case, calibration robustness analysis across prespecified target weights, one-way sensitivity analysis, probabilistic sensitivity analysis with 1000 iterations, scenario analyses, and threshold price analysis.

```bash
python run_model.py --analysis all --psa-iterations 1000 --psa-workers 16 --calibration-mode frozen --language en --output-dir results
```

Outputs are written to:

- `results/model_results.xlsx`
- `results/figures/owsa_tornado.png`
- `results/figures/psa_ceac.png`
- `results/figures/psa_scatter.png`
- `results/figures/threshold_price_inmb.png`
- `results/figures/baseline_mmd_distribution.png`
- `results/figures/week12_mmd_continuous_distribution.png`
- `results/figures/week12_mmd_distribution.png`
- `results/figures/calibration_robustness.png`

The workbook includes `Calibration_Robustness` and `Calibration_Summary` sheets. Their ranges describe the observed results across prespecified target weights and are not confidence or credible intervals.

To run only the base case plus calibration robustness analysis:

```bash
python run_model.py --analysis calibration --calibration-patients 10000 --seed 314159 --output-dir results_calibration
```

## Fast smoke test

Use this command to confirm the full workflow without running a full PSA:

```bash
python run_model.py --analysis all --n-patients 2000 --calibration-patients 500 --owsa-patients 1000 --psa-iterations 4 --psa-patients 1000 --psa-workers 2 --seed 314159 --calibration-mode frozen --language en --output-dir results_smoke
```

## Recalibration

Routine runs use frozen Week 12 calibration parameters stored in `data/calibration_parameters.csv`. Recalibration is only needed after changing baseline MMD distribution settings or Week 12 calibration targets.

```bash
python calibrate_model.py --data-dir data --output data/calibration_parameters.csv
```

## Post-Week-12 extrapolation

The maintenance-period model simulates treatment discontinuation at patient level. Patients remaining on treatment retain their Week-12 effect. Treatment discontinuers are assigned to lack of efficacy or another reason using the observed counts from the 52-week open-label extension. Lack-of-efficacy discontinuers return immediately to their paired comparator level; other-reason discontinuers retain 91%, 64%, and 45% of positive incremental benefit in the first three 4-week cycles after discontinuation, with no retained benefit thereafter. Drug cost stops after discontinuation, while clinical benefit can persist temporarily.

Run the model tests with:

```bash
python -m unittest discover -s tests -v
```

## License

The code is released under the MIT License. The input tables cite published sources and documented study inputs.
