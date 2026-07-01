# Prediction of Urgent Mechanical Circulatory Support During Chronic Total Occlusion Percutaneous Coronary Intervention

Two complementary pipelines for the CTO-PCI adverse outcome prediction project:

1. **TRIPOD+AI pipeline** (primary) — Firth logistic regression on 8 pre-specified predictors, with optimism correction, calibration, DCA, point score, and risk equation. This is the **deployable model** used by the dashboard.
2. **Model bake-off** (secondary) — 11-model discrimination benchmark on 64 features with CV-safe hyperparameter tuning, for internal reporting only.

---

## Setup

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

This creates a virtual environment (`.venv`) and installs all dependencies.

### Prepare data

Place `for_score.csv` in `data/`. The CSV must contain the target `lv_assist2_aae___2` and use Latin-1 encoding.

---

## Configuration

Edit `config.yaml` in the project root. Key sections:

### TRIPOD section (`tripod:`)

| Key | Default | Description |
|---|---|---|
| `pre_specified_predictors` | 8 predictors | The pre-specified feature set for the deployable model |
| `plausible_bounds` | `{lvef: [5,80], length: [1,200], age: [18,110]}` | Physiologic ranges → values outside become NaN |
| `score_increments` | `{lvef: -10, length: 10, age: 10}` | Clinical increments for the point score |
| `points_max` | `10` | Maximum points in the scoring system |
| `n_boot_optimism` | `500` | Bootstrap iterations for optimism-corrected AUC |
| `tripod_output_dir` | `tripod_outputs` | Output directory for TRIPOD results |
| `exclude_planned_mcs` | `true` | Exclude planned/prophylactic MCS from derivation |
| `planned_mcs_col` | `lv_assist2_aae___1` | Column flagging planned MCS |

### Bake-off section

| Key | Default | Description |
|---|---|---|
| `data_path` | `data/for_score.csv` | Path to input CSV |
| `target` | `lv_assist2_aae___2` | Target column |
| `output_dir` | `output` | Directory for bake-off results |
| `k_grid` | `[15, 25, 50, "all"]` | Feature counts to evaluate |
| `fast_mode` | `false` | Skip slow models (SVM, MLP) |
| `top_features` | _(list of 15)_ | Featured inputs in the old bake-off dashboard (not used by TRIPOD dashboard) |

---

## Usage

> **If the script hangs on startup**, xgboost may be probing CUDA devices. Set:
> ```bash
> export CUDA_VISIBLE_DEVICES=-1
> ```

### 1. TRIPOD+AI pipeline (deployable model)

Train and validate the Firth logistic regression on the 8 pre-specified predictors:

```bash
uv run python -m bakeoff.tripod_main
```

#### Options

```bash
# Custom config
uv run python -m bakeoff.tripod_main --config my_config.yaml

# Override paths
uv run python -m bakeoff.tripod_main --data-path /path/to/for_score.csv --output-dir /tmp/tripod

# More bootstrap iterations
uv run python -m bakeoff.tripod_main --n-boot-optimism 2000
```

#### Output (`tripod_outputs/`)

| File | Description |
|---|---|
| `final_logreg_firth.pkl` | Serialized pipeline + metadata |
| `logreg_firth_specification.csv` | Odds ratios, CIs, p-values |
| `logreg_firth_point_score.csv` | Integer point score per predictor |
| `logreg_firth_risk_equation.txt` | Plain-text logit equation |
| `calibration_curve.png` | Calibration plot (OOF) |
| `roc_curve.png` | ROC curve (OOF) |
| `dca_curve.png` | Decision-curve analysis |

#### Key results

- OOF AUC with bootstrap CI
- Optimism-corrected AUC (bootstrap)
- Calibration slope & Brier score
- DCA net benefit curve

### 2. Dashboard (Streamlit)

A clinical web app using the TRIPOD deployable model:

```bash
# After running tripod_main (so tripod_outputs/final_logreg_firth.pkl exists)
uv run streamlit run src/bakeoff/dashboard.py
```

- **8 pre-specified predictors** shown as the input form
- Appropriate input types: number inputs (with plausible bounds) for continuous, Yes/No selects for binary
- Unfilled fields imputed automatically
- **AI explanation** — click "Explain with AI" for a 2-3 sentence clinical summary (requires OpenAI API key)

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"   # optional
uv run streamlit run src/bakeoff/dashboard.py
```

Or paste the API key into the dashboard's password field (session-only).

### 3. Model bake-off (discrimination benchmark)

For internal comparison only — does **not** feed into the dashboard:

```bash
uv run python -m bakeoff.main

# Fast mode (skip SVM/MLP)
uv run python -m bakeoff.main --fast-mode

# Custom paths
uv run python -m bakeoff.main --data-path /path/to/for_score.csv --output-dir /tmp/bakeoff
```

#### Models evaluated

LR, NB_Gaussian, NB_Bernoulli, KNN, XGBoost, RandomForest, ExtraTrees, AdaBoost, HistGBM, SVM, MLP

#### Output (`output/`)

- `variable_categorization.csv` — variable type classification
- `per_model_tuning_results.csv` — best settings per model, sorted by CV AUC
- `fixed_k_train_test.csv` — CV and test AUC at K = 15, 25, 50
- `best_model.pkl` — full tuned pipeline + metadata (used by the old bake-off dashboard only)
- 7 plot PNGs

---

## Pipeline details

### TRIPOD pipeline

1. **Load & clean** — drop index columns, physiologic cleaning (implausible → NaN), exclude planned MCS
2. **Preprocess** — median imputation + standard scaling for continuous, most-frequent imputation for binary
3. **Train Firth LR** — penalized logistic regression on all 8 pre-specified predictors (no feature selection)
4. **OOF evaluation** — 5-fold stratified cross-validation with bootstrap CI for AUC
5. **Bootstrap optimism correction** — apparent AUC − average optimism from 500+ bootstrap samples
6. **Calibration** — calibration-in-the-large, calibration slope (statsmodels), Brier score, calibration curve plot
7. **DCA** — decision-curve analysis across clinically relevant thresholds
8. **Model specification** — odds ratios, 95% CIs, p-values from the Firth estimator
9. **Point score** — integer scoring system based on log-odds per clinical increment
10. **Risk equation** — plain-text logit + probability formula
11. **Serialize** — save pipeline + metadata for the dashboard

### Bake-off pipeline

1. Load data, drop index/missing/constant/duplicate columns
2. Classify variables (binary / categorical / continuous)
3. Redundancy reduction for predefined collinear groups
4. Stratified 80/20 train/test split
5. Per-model `GridSearchCV` tuning of K + hyperparameters, scored by CV ROC-AUC
6. Fixed-K evaluation at K = 15, 25, 50
7. Summary table + plots

---

## Programmatic usage

### TRIPOD model

```python
from bakeoff.predict import load_model, predict_from_dict, list_features

pipeline, metadata = load_model("tripod_outputs/final_logreg_firth.pkl")

# List features
feat = list_features(metadata)
# feat["all"] -> 8 predictors
# feat["binary"] -> binary ones
# feat["continuous"] -> continuous ones

# Predict from partial input
result = predict_from_dict(
    {
        "age_manual_input": 72.0,
        "retro": 1.0,
        "calcification_med_sev": 1.0,
        "peripheral_arterial_diseas": 0.0,
        "proximal_cap_ambiguity": 1.0,
        "acs": 0.0,
        "occlusion_length_mm": 35.0,
        "left_ventr_ejection_fract": 45.0,
    },
    pipeline,
    metadata,
)
# result["probability_positive"] -> 0.0 to 1.0
```

### AI explanation

```python
from bakeoff.explain import explain_prediction

result = explain_prediction(
    pipeline,
    {"age_manual_input": 72.0, "retro": 1.0},
    metadata,
    api_key="sk-...",
)
print(result["explanation"])
print(result["top_contributors"])
```

---

## Note on reporting

The **TRIPOD model** (Firth LR, 8 pre-specified predictors) is the deployable model with optimism-corrected AUC, calibration, and decision-curve analysis. The **bake-off** is a separate discrimination benchmark for internal reporting — its best-performing model (typically NB_Bernoulli) achieves a higher AUC but uses 64 features and is poorly calibrated.
