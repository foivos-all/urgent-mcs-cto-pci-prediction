# Prediction of Urgent Mechanical Circulatory Support During Chronic Total Occlusion Percutaneous Coronary Intervention

Full TRIPOD+AI (2024)-compliant pipeline for urgent MCS prediction after CTO-PCI. Covers all 27 code-addressable TRIPOD+AI items; reproduces the analyses in the companion notebook.

**Deployable model**: Firth penalized logistic regression on 8 pre-specified predictors (item 22).

**Benchmark**: Tuned Bernoulli Naive Bayes on the full candidate set with SelectKBest (discrimination only).

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
| `n_repeated_cv` | `20` | Repeated CV repetitions |
| `cat_max_levels` | `20` | Max levels for categorical detection |
| `fast_mode` | `false` | Skip SVM/MLP, reduce bootstraps |
| `k_grid` | `[10, 15, 25, "all"]` | Feature-count candidates for SelectKBest |
| `yesno_na_vars` | _(10 vars)_ | Columns with 1=yes/2=no/3=NA coding |
| `redundant_groups` | _(5 groups)_ | Collinear groups for redundancy reduction |

### `tripod:` section

| Key | Default | Description |
|---|---|---|
| `pre_specified_predictors` | 8 predictors | The pre-specified feature set for the deployable model |
| `plausible_bounds` | `{lvef: [5,80], length: [1,200], age: [18,110]}` | Physiologic ranges — values outside become NaN |
| `score_increments` | `{lvef: -10, length: 10, age: 10}` | Clinical increments for the point score |
| `points_max` | `10` | Maximum points in the scoring system |
| `n_boot_optimism` | `500` | Bootstrap iterations for optimism-corrected AUC |
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

This runs all 21 sections matching the companion notebook.

#### Options

```bash
uv run python -m bakeoff.main --data-path /path/to/for_score.csv --output-dir /tmp/results
uv run python -m bakeoff.main --n-boot-optimism 2000
uv run python -m bakeoff.main --fast-mode   # skip SVM/MLP for quick iteration
```

---

## Pipeline sections

| Section | TRIPOD item | Description | Output |
|---|---|---|---|
| 1 | 5, 6 | Data loading, cohort derivation, physiologic cleaning | Console |
| 2 | 7 | Variable typing (binary / categorical / continuous) | `variable_typing.csv` |
| 3 | 9 | Missing data table | `missingness.csv` |
| 4 | 8 | 80/20 split, redundancy reduction, EPV | Console |
| 5a | 12, 13, 15 | Train Firth LR (deployable) + NB_Bernoulli (benchmark) | Console |
| 5b | — | Marginal contribution — leave-one-out Firth LR | `marginal_contribution.csv` |
| 5c | 12, 23a | Multi-model bake-off (11 models + Firth) | `bakeoff_results.csv` |
| 6 | 12e, 23a | Discrimination — OOF AUC, repeated CV, test AUC | `discrimination.csv` |
| 7 | 12e, 12f | Calibration — intercept, slope, Brier, plot | `plots/calibration_curve.png` |
| 8 | 12e | Decision-curve analysis | `plots/dca_curve.png` |
| 9 | 12 | Bootstrap optimism correction (AUC + slope) | Console |
| 10 | 23b | Site-clustered (GroupKFold) + temporal split | Console |
| 11 | 14, 23a | Fairness — subgroup AUC across 20+ strata | `subgroup_performance.csv` |
| 11b | 9, 12 | MICE sensitivity — IterativeImputer | Console |
| 12 | 12, 23a | External comparison — PROGRESS-CTO + DeLong | `delong_comparison.csv` |
| 14 | 22, 12g | Odds ratios, point score, risk equation, shrinkage | 3 CSV files + equation |
| 15 | 18 | Open science — environment info | `environment.json` |
| 16 | — | TRIPOD+AI 27-item checklist | `tripod_ai_checklist.csv` |
| 17 | — | Save all results | `results.json` |

### Plots (`tripod_outputs/plots/`)

| File | Description |
|---|---|
| `comparison_auc.png` | Bar chart — Firth LR vs tuned models (CV AUC) |
| `comparison_roc.png` | OOF ROC — Firth LR vs NB_Bernoulli |
| `comparison_calibration.png` | Calibration curves — Firth LR vs NB_Bernoulli |
| `calibration_curve.png` | Firth LR calibration (10-bin quantile) |
| `roc_curve.png` | Firth LR OOF ROC curve |
| `dca_curve.png` | Decision-curve analysis |

### Deployable model (`tripod_outputs/`)

| File | Description |
|---|---|
| `final_logreg_firth.pkl` | Serialized pipeline + metadata (loaded by dashboard) |
| `logreg_firth_specification.csv` | Odds ratios, 95% CIs, p-values |
| `logreg_firth_point_score.csv` | Integer point score per predictor |
| `logreg_firth_risk_equation.txt` | Plain-text logit + probability equation |

---

## Dashboard (Streamlit)

```bash
# After running main.py
uv run streamlit run src/bakeoff/dashboard.py
```

- 8 pre-specified predictors, auto-imputation, risk score
- AI explanation via OpenAI (optional, set `OPENAI_API_KEY`)

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
