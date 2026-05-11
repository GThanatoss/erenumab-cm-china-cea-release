# Model description

## Objective and PICO

The model evaluates the cost utility of erenumab preventive treatment versus no preventive treatment for adults with chronic migraine in China.

- Population: adults with chronic migraine represented by the DRAGON trial baseline monthly migraine day profile.
- Intervention: erenumab preventive treatment, 70 mg every 4 weeks.
- Comparator: no preventive treatment.
- Perspective: China health-system direct medical cost perspective.
- Time horizon: 52 weeks in the base case.
- Cycle length: 4 weeks.
- Outcome: costs in CNY, QALYs, ICER, NMB, and INMB.
- Willingness-to-pay thresholds: 1x GDP per capita and 3x GDP per capita.

## Model structure

The model uses 13 four-week cycles. The first 3 cycles represent the double-blind 0-12 week trial period. The remaining 10 cycles represent the post-Week 12 maintenance period.

For each arm, simulated patients receive a baseline MMD value and calibrated Week 12 MMD value. MMD values are mapped to six utility states:

- 0-3 MMD
- 4-9 MMD
- 10-14 MMD
- 15-19 MMD
- 20-23 MMD
- 24-28 MMD

The final state keeps the utility previously assigned to the 24-30 MMD state, but the upper bound is set to 28 days to match a 4-week cycle.

## Baseline MMD

Baseline MMD is sampled from a truncated normal distribution bounded at 8 and 28 days per 4 weeks. The distribution parameters are solved so that the post-truncation theoretical mean is 19.2 and the post-truncation theoretical standard deviation is 5.4.

## Week 12 calibration

Routine analysis uses frozen continuous Week 12 calibration parameters from `data/calibration_parameters.csv`.

The main calibration method generates Week 12 MMD as a continuous outcome rather than directly assigning patients to MMD utility states. The calibrated Week 12 distribution is represented by a two-component truncated-normal marginal distribution and a rank-ordering parameter linking Week 12 MMD to baseline MMD.

The calibration targets are:

- Week 12 mean MMD.
- Week 12 >=50% MMD responder rate.
- Week 12 MMD<15 proportion.

MMD utility states are used after calibration for utility mapping and diagnostic reporting. They are not used as direct calibration targets in the main analysis.

## Discontinuation

Short-term discontinuation uses the 0-12 week double-blind all-cause discontinuation rate from DRAGON. Long-term maintenance discontinuation uses the all-cause discontinuation reported in Tepper et al.'s 52-week open-label extension study; the reported 25.9% value corresponds to the 64-week total study period, comprising the 12-week double-blind phase plus the 52-week open-label extension.

During Weeks 0-12, observed trial effects are retained and discontinuation does not further reduce efficacy or acute medication use. After Week 12, erenumab drug cost follows the open-label all-cause discontinuation rate. Erenumab MMD and acute medication days gradually move toward the no-preventive-treatment arm according to the additional cumulative discontinuation beyond the short-term discontinuation rate.

For maintenance cycle `k`, the model uses:

```text
D_short(k) = 1 - (1 - p_short)^k
D_long(k) = 1 - (1 - p_long)^k
q(k) = max(0, D_long(k) - D_short(k))
```

Erenumab MMD and acute medication use are then shifted toward the comparator arm by `q(k)`.

## Costing

Costs are direct medical costs in CNY. The base case includes:

- Erenumab drug acquisition cost.
- Acute rescue medication cost.
- Non-drug medical service cost.

Erenumab cost assumes one 70 mg box per 4-week cycle while on treatment. The base-case price is 3498 CNY per 70 mg box.

Acute rescue medication cost is calculated as acute medication days multiplied by the weighted daily acute rescue medication cost. The weighted daily cost is derived from category proportions and representative daily costs in `data/acute_rescue_drug_costs.csv`.

Non-drug medical service cost is derived from the CHIRA estimate and inflated to 2025 using medical service CPI.

Adverse event costs are not included in the base case.

## Sensitivity analyses

One-way sensitivity analysis varies model parameters over predefined ranges and reports INMB at the 3x GDP per capita willingness-to-pay threshold. Week 12 MMD change is the only Week 12 efficacy input varied in OWSA; changing it triggers recalibration.

PSA samples clinical probabilities and utilities from beta or beta-PERT distributions, costs from gamma distributions where applicable, and MMD change from normal distributions. Utility sampling is sorted within each PSA iteration to preserve monotonicity across MMD states.

Scenario analyses include:

- 12-week time horizon.
- Legacy state-distribution Week 12 calibration.
- Threshold erenumab price at 1x and 3x GDP per capita willingness-to-pay thresholds.

## Outputs

The model writes an Excel workbook and publication-style figures:

- Base-case results.
- Cycle trace.
- Calibration diagnostics.
- OWSA table and tornado diagram.
- PSA iterations, CEAC, and incremental cost-effectiveness scatter plot.
- Threshold price table and INMB-price figure.
- Baseline MMD distribution figure.
- Week 12 continuous MMD distribution figure.
- Week 12 MMD state-share figure.
