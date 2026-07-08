# Prediction of Urgent Mechanical Circulatory Support During Chronic Total Occlusion Percutaneous Coronary Intervention

Pipeline for urgent MCS prediction after CTO-PCI, reproducing the analyses in the companion notebook. The manuscript's Supplementary Table S1 is the source of truth for TRIPOD+AI (2024) reporting — this repository is the underlying analysis code.

**Deployable model**: Firth penalized logistic regression on 8 pre-specified predictors, calibrated for deployment with FLIC (intercept correction) and a uniform-shrinkage factor derived from bootstrap optimism correction.

**Benchmark**: The strongest model, by out-of-fold AUC, from a 12-algorithm bake-off on the same pre-specified predictor set (ExtraTrees by default — configurable via `benchmark_model`). Kept only as a discrimination ceiling; the full TRIPOD battery (calibration, DCA, optimism, heterogeneity, fairness, specification) is run for the deployable model only.

**External comparison**: Reconstructed PROGRESS-CTO nomogram (Karacsonyi et al., AJC 2023) plus pairwise DeLong tests.

---

## Setup

### Clone the repository

```bash
git clone https://github.com/foivos-all/urgent-mcs-cto-pci-prediction.git
cd urgent-mcs-cto-pci-prediction
```

### Prerequisites

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/) — fast Python package manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
uv sync
```

### Prepare data

Place `for_score.csv` in `data/`. The CSV must contain the target `lv_assist2_aae___2` and use Latin-1 encoding.

If `data_path` doesn't exist and `use_synth_if_missing` is `true` (default), the pipeline generates a synthetic cohort matching the expected schema instead of failing — useful for a dry run of the full pipeline (including every plot) without the real registry export.

---

## Configuration

Edit `config.yaml` in the project root. Top-level keys control the full pipeline.

| Key | Default | Description |
|---|---|---|
| `data_path` | `data/for_score.csv` | Path to input CSV |
| `target` | `lv_assist2_aae___2` | Target column (urgent MCS) |
| `random_state` | `42` | Random seed |
| `test_size` | `0.20` | Held-out test fraction |
| `cv_splits` | `5` | Stratified CV folds |
| `n_boot_ci` | `2000` | Bootstrap iterations for AUC CI |
| `n_repeated_cv` | `20` | Repeats for the repeated-CV discrimination check (Section 6 only — the 12-model bake-off itself is tuned with a single, non-repeated `cv_splits`-fold split) |
| `cat_max_levels` | `20` | Max levels for categorical detection |
| `fast_mode` | `false` | Skip SVM/MLP, reduce bootstraps |

### `tripod:` section

| Key | Default | Description |
|---|---|---|
| `pre_specified_predictors` | 8 predictors | The pre-specified feature set for the deployable model |
| `deployable_variant` | `flic` | Post-hoc calibration applied on top of the Firth fit for deployment: `firth` (no correction) \| `flic` (default — re-estimates only the intercept via ML, keeping the Firth slopes) \| `flac` (augmented-data refit). Firth is always fit first; the odds-ratio table always uses the Firth slopes regardless of this setting |
| `benchmark_model` | `ExtraTrees` | Which bake-off model to carry forward as the ML benchmark (falls back to the top non-Firth model by OOF AUC if absent) |
| `use_synth_if_missing` | `true` | Generate a synthetic cohort when `data_path` doesn't exist (dry run only) |
| `plausible_bounds` | `{lvef: [5,80], length: [1,200], age: [18,110]}` | Physiologic ranges — values outside become NaN |
| `score_increments` | `{lvef: -10, length: 10, age: 10}` | Clinical increments for the point score |
| `points_max` | `10` | Maximum points in the scoring system |
| `n_boot_optimism` | `500` | Bootstrap iterations for optimism-corrected AUC and calibration slope, which also feeds the deployment shrinkage factor |
| `exclude_planned_mcs` | `true` | Exclude planned/prophylactic MCS from derivation |
| `planned_mcs_col` | `lv_assist2_aae___1` | Column flagging planned MCS |
| `site_col` | `center` | Site column (for heterogeneity analysis) |
| `year_col` | `year_of_procedure` | Year column (for temporal split) |
| `pub_vars` | `{retro, lvef, length}` | Variable mapping for published PROGRESS-CTO score |
| `pub_pts` | `{retro_yes: 45, ...}` | Nomogram weights for published score |
| `published_betas` | `null` | Override with explicit logistic coefficients |

---

## Usage

> If the script hangs, xgboost may be probing CUDA. Set:
> ```bash
> export CUDA_VISIBLE_DEVICES=-1
> ```

### Run the full pipeline

```bash
uv run python -m bakeoff.main
```

#### Options

```bash
uv run python -m bakeoff.main --data-path /path/to/for_score.csv --output-dir /tmp/results
uv run python -m bakeoff.main --n-boot-optimism 2000
uv run python -m bakeoff.main --fast-mode   # skip SVM/MLP for quick iteration
uv run python -m bakeoff.main --config /path/to/config.yaml
```

---

## Pipeline sections

| Section | Description | Output |
|---|---|---|
| 0b | Sample-size adequacy — pmsampsize a-priori grid (data-driven verdict repeated after Section 6) | Console |
| 1 | Data loading, cohort derivation, physiologic cleaning | Console |
| 2 | Variable typing (binary / categorical / continuous) | `variable_typing.csv` |
| 2b | Table 1 — participant characteristics by outcome | `table1.csv` |
| 3 | Missing data table | `missingness.csv` |
| 4 | 80/20 split, EPV | Console |
| 5a | Train Firth LR (deployable) + single-pass 12-model bake-off; benchmark selected by out-of-fold AUC | `bakeoff_results.csv`, `precision_recall.csv` |
| 5b | Why Firth — discrimination vs calibration for the top-3 bake-off models + Firth | `why_firth_table.csv` |
| 5c | Parsimony sweep — Firth over the pre-specified clinical-priority order | `firth_k_sweep.csv` |
| 5d | Marginal contribution — leave-one-out Firth LR | `firth_leave_one_out.csv`, `plots/firth_loo.png` |
| 6 | Discrimination — OOF AUC, repeated CV, test AUC, data-driven pmsampsize verdict | `discrimination.csv` |
| 7 | Calibration — intercept, slope, Brier, plot | `plots/calibration_curve.png` |
| 8 | Decision-curve analysis | `plots/dca_curve.png` |
| 9 | Bootstrap optimism correction (AUC + calibration slope) | Console |
| 10 | Site-clustered (GroupKFold, capped at 10 folds) + temporal split | Console |
| 11 | Fairness — subgroup AUC across 20+ strata | `subgroup_performance.csv` |
| 11b | MICE sensitivity — IterativeImputer | Console |
| 12 | External comparison — PROGRESS-CTO + DeLong | `delong_comparison.csv` |
| 14 | Odds ratios (Firth-only), point score, risk equation (deployed FLIC + shrinkage coefficients) | 3 CSV files + equation |
| 14b | Sensitivity — reduced model dropping age & occlusion length, DeLong vs the full 8-predictor model | `reduced_model_specification.csv` |
| 15 | Open science — environment info | `environment.json` |
| 17 | Save all results | `results.json` |
| 18 | Observed vs predicted incidence by deployable-model point strata (whole/test/training cohorts, 3 bin schemes) | `deployable_patient_counts_by_exact_point.csv`, `deployable_observed_predicted_incidence_by_point_strata.csv` + 2 plots (png + pdf each) |

### Model calibration

The deployable pipeline fits plain Firth logistic regression first (for de-biased odds ratios), then applies `deployable_variant` (default `flic`, an ML-corrected intercept) and, at Section 14, a uniform-shrinkage factor derived from the Section 9 bootstrap-optimism-corrected calibration slope (falling back to a van Houwelingen heuristic, clipped to `[0.5, 1.0]`). The odds-ratio table and point score always use the de-biased Firth coefficients; the saved model and risk equation use the deployed (FLIC + shrinkage) coefficients — this is what `predict_proba` actually returns.

### Plots (`tripod_outputs/plots/`)

| File | Description |
|---|---|
| `bakeoff_boxplot.png` | CV-AUC boxplot across all 12 bake-off models |
| `precision_recall_oof.png` | OOF precision-recall curves, all models |
| `why_firth_panel.png` | Calibration / ROC / decision-curve panel — top-3 bake-off models + Firth |
| `firth_k_sweep.png` | OOF AUC vs number of pre-specified predictors (parsimony sweep) |
| `calibration_curve.png` | Firth LR calibration (10-bin quantile) |
| `roc_curve.png` | Firth LR OOF ROC curve |
| `dca_curve.png` | Decision-curve analysis |
| `deployable_observed_predicted_incidence_by_point_strata.png` / `.pdf` | Observed vs predicted incidence — 3 point-bin schemes × 3 cohorts (3×3 grid) |
| `deployable_observed_predicted_incidence_7point_whole_cohort.png` / `.pdf` | Same, standalone 7-point scheme, whole cohort only |

### Deployable model (`tripod_outputs/`)

| File | Description |
|---|---|
| `final_logreg_firth.pkl` | Serialized pipeline (Firth + FLIC + shrinkage) + metadata, incl. `variant`/`shrinkage` (see Programmatic usage below) |
| `logreg_firth_specification.csv` | Odds ratios (Firth de-biased), 95% CIs, p-values |
| `logreg_firth_point_score.csv` | Integer point score per predictor |
| `logreg_firth_risk_equation.txt` | Plain-text logit + probability equation, using the deployed (FLIC + shrinkage) coefficients |
| `reduced_model_specification.csv` | Odds ratios for the 6-predictor sensitivity model (Section 14b) |

---

## Programmatic usage

```python
from bakeoff.predict import load_model, predict_from_dict

pipeline, metadata = load_model("tripod_outputs/final_logreg_firth.pkl")
result = predict_from_dict(
    {"age_manual_input": 72.0, "retro": 1.0, "left_ventr_ejection_fract": 45.0, ...},
    pipeline, metadata,
)
print(result["probability_positive"])   # 0.0 to 1.0
```
