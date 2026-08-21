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

The 24-28 MMD state uses a utility of 0.38 and an upper bound of 28 days to match a 4-week cycle.

## Baseline MMD

Baseline MMD is sampled from a truncated normal distribution bounded at 8 and 28 days per 4 weeks. The distribution parameters are solved so that the post-truncation theoretical mean is 19.2 and the post-truncation theoretical standard deviation is 5.4.

## Week 12 calibration

Routine analysis uses frozen continuous Week 12 calibration parameters from `data/calibration_parameters.csv`.

The continuous calibration method generates Week 12 MMD as a continuous outcome rather than directly assigning patients to MMD utility states. The calibrated Week 12 distribution is represented by a two-component truncated-normal marginal distribution and a rank-ordering parameter linking Week 12 MMD to baseline MMD.

The calibration targets are:

- Week 12 mean MMD.
- Week 12 >=50% MMD responder rate.
- Week 12 MMD<15 proportion.

MMD utility states are used after calibration for utility mapping and diagnostic reporting. They are not used as direct calibration targets.

Week 12 mean MMD and the >=50% MMD responder rate are treated as primary calibration targets. MMD<15 is a secondary soft target because its source endpoint was defined using headache days rather than MMD. The base-case objective weight for this soft target is 0.6; this numerical fitting weight is not interpreted as a clinical standard error or probability.

## Discontinuation

Short-term discontinuation uses the 0-12 week double-blind all-cause discontinuation rate from DRAGON. Maintenance discontinuation uses the investigational-product discontinuation reported in Tepper et al.'s 52-week open-label treatment phase: 139/609 patients (22.8%) discontinued open-label treatment. This is distinct from the 158/609 patients (25.9%) who discontinued the overall study. The 22.8% treatment-discontinuation probability is converted to a constant 4-week probability of 1.9709% and is applied after Week 12 only to patients who remain on treatment.

During Weeks 0-12, observed trial effects are retained and discontinuation does not further reduce efficacy or acute medication use. Drug cost is charged only while the patient is on treatment. After Week 12, clinical effects are extrapolated at patient level. Patients who remain on treatment retain their Week-12 MMD and acute-medication levels.

Among the 139 treatment discontinuations in the open-label study, 39 (28.1%) were due to lack of efficacy. Each simulated discontinuation is therefore assigned to lack of efficacy with probability 39/139 and to another reason otherwise. A patient who discontinues for lack of efficacy returns immediately to the paired comparator Week-12 level. For other-reason discontinuers, define the patient's positive Week-12 incremental MMD benefit as:

```text
delta_i = max(0, MMD_comparator_i_week12 - MMD_erenumab_i_week12)
MMD_i(s) = MMD_erenumab_i_week12 + (1 - R(s)) * delta_i
```

The retained-effect fractions are `R(1)=0.91`, `R(2)=0.64`, `R(3)=0.45`, and `R(s>=4)=0`. Acute medication days use the same patient-level discontinuation state, reason, and retained-effect profile. Consequently, drug acquisition cost stops after discontinuation while clinical benefit can persist temporarily.

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

PSA samples clinical probabilities and utilities from beta or beta-PERT distributions, costs from gamma distributions where applicable, and MMD change from normal distributions. The erenumab price is sampled from a gamma distribution with mean CNY 3,498 per 70 mg box. Its dispersion is derived from an assumed uncertainty range of CNY 2,798.4 to CNY 4,197.6 (plus or minus 20% of the base-case price); this assumption is not a statistical confidence interval. Utility sampling is sorted within each PSA iteration to preserve monotonicity across MMD states.

The 52-week treatment-discontinuation probability and the conditional lack-of-efficacy share are included in OWSA and PSA using the observed counts from the open-label study. The retained-effect profile is treated as structural uncertainty rather than assigned an unsupported probability distribution.

A separate calibration robustness analysis across prespecified target weights evaluates whether the economic conclusion depends on how strongly the secondary MMD<15 target influences calibration. It uses the base-case two-component continuous distribution, parameter bounds, primary targets, patient cohort, extrapolation, costing, and utility mapping. The MMD<15 objective weight is varied over the prespecified values 0, 0.15, 0.3, 0.6, 1.2, 2.4, and 4.8. Each candidate calibration is fitted using common random numbers and evaluated on the common base-case cohort. Candidate pairs are classified as compatible with the primary evidence when both arm-specific mean-MMD residuals are within 0.5 days and both responder-rate residuals are within 0.03. MMD<15 is reported but is not used as an acceptance criterion because of the endpoint-definition mismatch.

The calibration robustness analysis reports the observed minimum and maximum incremental QALY, ICER, and INMB across primary-compatible prespecified target weights. These ranges are deterministic structural-robustness results, not confidence intervals, credible intervals, or posterior calibration distributions. The frozen base-case calibration is used for the base case and PSA; the weight analysis is reported separately.

Scenario analyses include:

- 12-week time horizon.
- Immediate post-discontinuation loss of positive incremental benefit.
- Threshold erenumab price at 1x and 3x GDP per capita willingness-to-pay thresholds.

## Outputs

The model writes an Excel workbook and publication-style figures:

- Base-case results.
- Cycle trace.
- Calibration diagnostics.
- Calibration robustness results across prespecified target weights, a primary-compatible outcome range summary, and a calibration-versus-economic-outcome figure.
- OWSA table and tornado diagram.
- PSA iterations, CEAC, and incremental cost-effectiveness scatter plot.
- Threshold price table and INMB-price figure.
- Baseline MMD distribution figure.
- Week 12 continuous MMD distribution figure.
- Week 12 MMD state-share figure.
