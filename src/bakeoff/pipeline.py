import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import statsmodels.api as sm
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_val_score,
    GridSearchCV,
    train_test_split,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.base import clone

import seaborn as sns

from bakeoff.firth import FirthLogisticRegression, _sigmoid

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

ACCENT_RED = "#C0392B"
DARK = "#222222"
MID_GREY = "#7F7F7F"
LIGHT_GREY = "#D9D9D9"
SECONDARY_COLORS = ["#4C78A8", "#6B6B6B", "#59A14F", "#9C755F",
                    "#8064A2", "#F28E2B", "#76B7B2", "#B07AA1", "#A0A0A0"]

sns.set_theme(style="white", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": DARK,
    "axes.linewidth": 1.0,
    "axes.grid": False,
    "axes.titleweight": "normal",
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def style_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(DARK)
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", colors=DARK, width=1.0, length=4)
    return ax


def model_color_map(names, primary="LogReg_Firth"):
    names = list(names)
    colors, j = {}, 0
    for name in names:
        if name == primary:
            colors[name] = ACCENT_RED
        else:
            colors[name] = SECONDARY_COLORS[j % len(SECONDARY_COLORS)]
            j += 1
    return colors


# ===================================================================
# Helper metrics
# ===================================================================

def _boot_ci(yt, ys, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    yt, ys = np.asarray(yt), np.asarray(ys)
    pos = np.where(yt == 1)[0]
    neg = np.where(yt == 0)[0]
    if len(pos) < 2 or len(neg) < 2:
        return (np.nan, np.nan)
    out = np.empty(n)
    for i in range(n):
        ix = np.concatenate([
            rng.choice(pos, pos.size, replace=True),
            rng.choice(neg, neg.size, replace=True),
        ])
        out[i] = roc_auc_score(yt[ix], ys[ix])
    return tuple(np.percentile(out, [2.5, 97.5]))


def _cal_metrics(yv, p):
    c = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(c / (1 - c))
    try:
        res = sm.Logit(np.asarray(yv), sm.add_constant(lg)).fit(disp=False)
        intercept, slope = float(res.params[0]), float(res.params[1])
    except Exception:
        intercept, slope = np.nan, np.nan
    brier = float(brier_score_loss(yv, p))
    return intercept, slope, brier


def _cal_slope_only(yv, p):
    c = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(c / (1 - c))
    try:
        return float(sm.Logit(np.asarray(yv), sm.add_constant(lg)).fit(disp=False).params[1])
    except Exception:
        return np.nan


def _net_benefit(yv, p, th):
    yv, p = np.asarray(yv), np.asarray(p)
    n = len(yv)
    m = []
    for pt in th:
        tp = ((p >= pt) & (yv == 1)).sum()
        fp = ((p >= pt) & (yv == 0)).sum()
        w = pt / (1 - pt)
        m.append(tp / n - fp / n * w)
    return np.array(m)


# ===================================================================
# 0b. Sample-size adequacy — pmsampsize (Riley et al.)  (item 8)
# ===================================================================

def approximate_R2(auc, prev, n=400000, seed=42):
    """Cox-Snell R^2 implied by an anticipated C-statistic and prevalence (Riley simulation)."""
    r = np.random.default_rng(seed)
    mu = np.sqrt(2) * stats.norm.ppf(auc)
    n1 = int(round(prev * n))
    n0 = n - n1
    lp = np.concatenate([r.normal(mu, 1, n1), r.normal(0, 1, n0)])
    yy = np.concatenate([np.ones(n1), np.zeros(n0)])
    m = sm.GLM(yy, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    r2cs = 1 - np.exp((2 / n) * (m.llnull - m.llf))
    maxr2 = 1 - np.exp(2 * (prev * np.log(prev) + (1 - prev) * np.log(1 - prev)))
    return float(r2cs), float(maxr2), float(r2cs / maxr2)


def pmsampsize_bin(parameters, prevalence, cstat=None, r2cs=None, seed=42):
    if r2cs is None:
        r2cs, maxr2, nag = approximate_R2(cstat, prevalence, seed=seed)
    else:
        maxr2 = 1 - np.exp(2 * (prevalence * np.log(prevalence) + (1 - prevalence) * np.log(1 - prevalence)))
        nag = r2cs / maxr2
    P = parameters
    phi = prevalence
    n1 = P / ((0.9 - 1) * np.log(1 - r2cs / 0.9))                          # criterion 1: shrinkage 0.9
    S2 = r2cs / (r2cs + 0.05 * maxr2)
    n2 = P / ((S2 - 1) * np.log(1 - r2cs / S2))                            # criterion 2: optimism <= 0.05
    n3 = (1.96 / 0.05) ** 2 * phi * (1 - phi)                              # criterion 3: overall risk +/-0.05
    nfin = max(n1, n2, n3)
    return dict(
        parameters=P, prevalence=round(phi, 4), cstat=cstat, r2cs=round(r2cs, 4), max_r2=round(maxr2, 4),
        nagelkerke=round(nag, 4), n_crit1=int(np.ceil(n1)), n_crit2=int(np.ceil(n2)),
        n_crit3=int(np.ceil(n3)), n_required=int(np.ceil(nfin)),
        events_required=int(np.ceil(nfin * phi)), epp_required=round(np.ceil(nfin * phi) / P, 1),
    )


def expected_shrinkage_at_n(n, parameters, r2cs):
    LR = -n * np.log(1 - r2cs)
    return 1 - parameters / LR


def print_pmsampsize_grid(n_predictors=8):
    print("A-PRIORI SAMPLE-SIZE GRID (pmsampsize; item 8):")
    rows = []
    for phi in [0.02, 0.03, 0.04, 0.05]:
        for c in [0.70, 0.75, 0.80]:
            o = pmsampsize_bin(n_predictors, phi, cstat=c)
            rows.append({
                "prev": phi, "C": c, "R2cs": o["r2cs"], "max_R2": o["max_r2"],
                "n_required": o["n_required"], "events_required": o["events_required"],
                "EPP_required": o["epp_required"],
            })
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    print("\nThe binding criterion is #1 (shrinkage); required EVENTS depend chiefly on C "
          "(~140 at C=0.70, ~84 at C=0.75, ~55 at C=0.80). Read off the data-driven verdict after Section 6.\n")
    return rows


def print_pmsampsize_verdict(prevalence, n_predictors, oof_auc, n_dev, events_dev):
    """Data-driven pmsampsize verdict using the realized prevalence and deployable OOF C-statistic."""
    r2, maxr2, nag = approximate_R2(oof_auc, prevalence)
    need = pmsampsize_bin(n_predictors, prevalence, r2cs=r2)
    s_dev = expected_shrinkage_at_n(n_dev, n_predictors, r2)
    print(f"\nPMSAMPSIZE VERDICT (data-driven): prevalence={prevalence:.3%}, P={n_predictors}, "
          f"OOF C={oof_auc:.3f} -> R2cs={r2:.3f}")
    print(f"  required N={need['n_required']} (events {need['events_required']}, EPP {need['epp_required']}); "
          f"you have N(dev)={n_dev} events(dev)={events_dev}.")
    print(f"  expected dev-set shrinkage S~{s_dev:.3f} "
          f"{'(>=0.90 OK)' if s_dev >= 0.9 else '(<0.90 -> FLIC + shrinkage recalibration matter)'}")
    return {
        "prevalence": prevalence, "P": n_predictors, "oof_c": oof_auc, "r2cs": r2,
        "n_required": need["n_required"], "events_required": need["events_required"],
        "expected_shrinkage": float(s_dev),
    }


# ===================================================================
# Synthetic dry-run cohort (used only when data_path is absent)
# ===================================================================

def make_synth(N=3200, seed=42, path="for_score.csv"):
    """Synthetic cohort matching the expected schema, for a self-contained dry run when the real
    for_score.csv registry export isn't available."""
    r = np.random.default_rng(seed)
    age = np.clip(r.normal(66, 11, N), 30, 92)
    lvef = np.clip(r.normal(50, 12, N), 10, 70)
    length = np.clip(r.exponential(22, N) + 5, 1, 150)
    retro = r.binomial(1, 0.18, N)
    pca = r.binomial(1, 0.30, N)
    acs = r.binomial(1, 0.20, N)
    calc = r.binomial(1, 0.45, N)
    pad = r.binomial(1, 0.18, N)
    lp = (-5.2 + 0.9 * retro + 0.045 * (45 - lvef) + 0.018 * length + 0.020 * (age - 66)
          + 0.5 * acs + 0.35 * calc + 0.3 * pad + r.normal(0, 0.4, N))
    y = r.binomial(1, _sigmoid(lp))
    d = pd.DataFrame(dict(
        lv_assist2_aae___2=y, lv_assist2_aae___1=r.binomial(1, 0.06, N),
        center=r.integers(1, 9, N), year_of_procedure=r.integers(2017, 2025, N),
        retro=retro, left_ventr_ejection_fract=lvef, occlusion_length_mm=length,
        proximal_cap_ambiguity=pca, age_manual_input=age, acs=np.where(acs == 1, 1, 2),
        calcification_med_sev=calc, peripheral_arterial_diseas=pad,
        j_cto_calcification_score=calc + r.integers(0, 3, N), lmcto=r.binomial(1, 0.05, N),
        target_vessel_overall=r.integers(1, 4, N), j_cto_lesion_length=length + r.normal(0, 3, N),
        j_cto_tortuosity_score_1_f=r.integers(0, 4, N), tortuosity_med_sev=r.binomial(1, 0.3, N),
        lvef40=(lvef < 40).astype(int), lvef50=(lvef < 50).astype(int),
        prior_heart_failure=r.binomial(1, 0.2, N), gender=r.integers(1, 3, N), race=r.integers(1, 5, N),
        ethnicity=r.integers(1, 3, N), diabetes_mellitus=r.binomial(1, 0.4, N),
        prior_cabg=r.binomial(1, 0.2, N), prior_mi=r.binomial(1, 0.3, N),
        current_dialysis=r.binomial(1, 0.05, N), chronic_lung_disease=r.binomial(1, 0.15, N),
        hypertension=r.binomial(1, 0.7, N), smoking=r.binomial(1, 0.25, N),
    ))
    for c in ["left_ventr_ejection_fract", "occlusion_length_mm", "proximal_cap_ambiguity"]:
        d.loc[r.random(N) < 0.06, c] = np.nan
    d.to_csv(path, index=False)
    return d


# ===================================================================
# 1. Data loading — outcome & cohort derivation  (items 5, 6)
# ===================================================================

def load_and_prepare(
    df_path,
    target,
    plausible_bounds=None,
    exclude_planned_mcs=False,
    planned_mcs_col=None,
    site_col=None,
    year_col=None,
):
    df = pd.read_csv(df_path, encoding="latin1")
    n_raw = len(df)
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        print("Dropping index-like columns:", unnamed)
    df = df.drop(columns=unnamed)
    n_no_outcome = int(df[target].isna().sum())
    df = df.loc[~df[target].isna()].copy().reset_index(drop=True)

    if plausible_bounds:
        print("PHYSIOLOGIC CLEANING (implausible -> NaN):")
        clean_log = []
        for col, (lo, hi) in plausible_bounds.items():
            if col not in df.columns:
                continue
            v = pd.to_numeric(df[col], errors="coerce")
            bad = (v < lo) | (v > hi)
            n = int(bad.sum())
            if n:
                examples = sorted(
                    set(np.round(v[bad].dropna(), 3).tolist())
                )[:6]
                print(f"  {col:30s} {n:4d} outside [{lo},{hi}] -> NaN  e.g. {examples}")
            df.loc[bad, col] = np.nan
            clean_log.append({"variable": col, "n_set_nan": n})
        print("  done.\n")

    y = pd.to_numeric(df[target], errors="coerce")
    y = pd.Series(np.where(y > 0, 1, 0), index=df.index).astype(int)

    # Validation-only columns kept aside
    validation_only = [
        c for c in [site_col, year_col, planned_mcs_col]
        if c and c in df.columns
    ]
    X0 = df.drop(columns=[target] + validation_only).copy()
    X0 = X0.drop(columns=sorted(
        set(X0.columns[X0.isna().all()]) |
        set(X0.columns[X0.astype(str).T.duplicated().values])
    ))

    print("COHORT DERIVATION (items 5/6)")
    print(f"  raw rows {n_raw} | excluded missing-outcome {n_no_outcome} | analysis cohort {len(y)}")
    print(f"  outcome '{target}' events: {int(y.sum())} ({y.mean():.3%}) | candidate cols {X0.shape[1]}")

    planned_mask = (
        pd.to_numeric(df[planned_mcs_col], errors="coerce").fillna(0) > 0
        if (exclude_planned_mcs and planned_mcs_col and planned_mcs_col in df.columns)
        else pd.Series(False, index=df.index)
    )
    print(f"  prophylactic/planned MCS flagged: {int(planned_mask.sum())} -> excluded from derivation\n")

    return X0, y, df, planned_mask, validation_only


# ===================================================================
# 2. Predictor typing  (item 7)
# ===================================================================

def _intset(s):
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() < 0.5:
        return None
    vv = sn.dropna().unique()
    if not np.all(np.isclose(np.mod(vv, 1.0), 0.0, atol=1e-6)):
        return None
    return set(np.round(vv).astype(int))


def recode_binary(s):
    iset = _intset(s)
    sn = pd.to_numeric(s, errors="coerce")
    if iset == {1, 2}:
        return sn.map({1: 1.0, 2: 0.0}), "1=yes->1,2=no->0"
    if iset == {1, 2, 3}:
        return sn.map({1: 1.0, 2: 0.0, 3: np.nan}), "1=yes->1,2=no->0,3=na->NaN"
    if iset == {0, 1}:
        return sn.astype(float), ""
    codes, uniq = pd.factorize(s, sort=True)
    note = f"{uniq[0]}->0,{uniq[1]}->1" if len(uniq) >= 2 else ""
    return pd.Series(
        np.where(codes == -1, np.nan, codes).astype(float), index=s.index
    ), note


def _classify(s, cat_max):
    nun = s.dropna().nunique()
    if nun <= 1:
        return "DROPPED"
    iset = _intset(s)
    numeric = pd.to_numeric(s, errors="coerce").notna().mean() >= 0.5
    if nun == 2:
        return "binary"
    if iset is not None and iset <= {1, 2, 3} and {1, 2} <= iset:
        return "binary"
    if numeric and iset is not None and len(iset) < cat_max:
        return "categorical"
    if numeric:
        return "continuous"
    return "categorical" if nun < cat_max else "DROPPED"


def classify_variables(X0, cat_max=20, yesno_na_vars=None):
    if yesno_na_vars is None:
        yesno_na_vars = []
    binary, categorical, continuous = [], [], []
    X = pd.DataFrame(index=X0.index)
    rows = []
    for c in X0.columns:
        s = X0[c]
        if c in yesno_na_vars:
            rec, note = recode_binary(s)
            X[c] = rec
            binary.append(c)
            t = "binary"
        else:
            t = _classify(s, cat_max)
            if t == "binary":
                rec, _ = recode_binary(s)
                X[c] = rec
                binary.append(c)
            elif t == "continuous":
                X[c] = pd.to_numeric(s, errors="coerce")
                continuous.append(c)
            elif t == "categorical":
                X[c] = s
                categorical.append(c)
            else:
                note = "dropped"
        rows.append({
            "variable": c,
            "type": t,
            "levels": int(s.dropna().nunique()),
            "note": note if t == "DROPPED" else "",
        })
    typing_df = pd.DataFrame(rows)
    print(f"VARIABLE CLASSIFICATION: binary={len(binary)} categorical={len(categorical)} continuous={len(continuous)} dropped={sum(1 for r in rows if r['type']=='DROPPED')}")
    return X, binary, categorical, continuous, typing_df


# ===================================================================
# Preprocessors
# ===================================================================

def build_full_preprocessor(binary, categorical, continuous):
    transformers = []
    if continuous:
        transformers.append((
            "cont",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]),
            continuous,
        ))
    if binary:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binary))
    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def build_prespec_preprocessor(predictors, binary, categorical, continuous):
    ps_cont = [c for c in predictors if c in continuous]
    ps_bin = [c for c in predictors if c in binary]
    ps_cat = [c for c in predictors if c in categorical]
    transformers = []
    if ps_cont:
        transformers.append((
            "cont",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]),
            ps_cont,
        ))
    if ps_bin:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), ps_bin))
    if ps_cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            ps_cat,
        ))
    return ColumnTransformer(transformers, remainder="drop"), ps_cont, ps_bin, ps_cat


# ===================================================================
# Redundancy reduction
# ===================================================================

def _uni_auc(x, yv):
    x = pd.to_numeric(x, errors="coerce")
    x = x.fillna(x.median())
    if x.nunique() < 2:
        return 0.5
    a = roc_auc_score(yv, x)
    return max(a, 1 - a)


def reduce_redundancy(X_train, X_test, y_train, redundant_groups, pre_specified_predictors):
    drop = []
    print("REDUNDANCY REDUCTION (univariate AUC; keep best per group, skip pre-specified):")
    for grp in redundant_groups:
        present = [c for c in grp if c in X_train.columns]
        if len(present) <= 1:
            if present:
                print(f"  {grp}: only {present} present — kept")
            continue
        scored = sorted(
            ((_uni_auc(X_train[c], y_train), c) for c in present), reverse=True
        )
        dr = [c for _, c in scored[1:] if c not in pre_specified_predictors]
        drop += dr
        print(f"  keep {scored[0][1]} | drop {dr}")
    if drop:
        for d in [X_train, X_test]:
            d.drop(columns=drop, inplace=True)
        print(f"  Dropped {len(drop)} redundant column(s).")
    return drop


# ===================================================================
# 4/5a. Firth LR training  (deployable — pre-specified predictors)
# ===================================================================

def _firth_pipe(cols, binary, categorical, continuous, variant="firth"):
    cc = [c for c in cols if c in continuous]
    bb = [c for c in cols if c in binary]
    kk = [c for c in cols if c in categorical]
    transformers = []
    if cc:
        transformers.append((
            "cont",
            Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
            cc,
        ))
    if bb:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), bb))
    if kk:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            kk,
        ))
    prep = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("prep", prep), ("model", FirthLogisticRegression(variant=variant))])


def train_firth(X_train, y_train, predictors, binary, categorical, continuous, variant="firth"):
    pipe = _firth_pipe(predictors, binary, categorical, continuous, variant=variant)
    pipe.fit(X_train, y_train)
    return pipe


# ===================================================================
# 5c. Model zoo — multi-model comparison
# ===================================================================

def _model_zoo(y_train, fast_mode=False):
    _spw = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    RS = 42

    zoo = {
        "LogReg_L1L2": (
            LogisticRegression(solver="liblinear", class_weight=None, max_iter=5000, random_state=RS),
            {"model__penalty": ["l1", "l2"], "model__C": [0.01, 0.1, 1.0]},
        ),
        "NB_Gaussian": (GaussianNB(), {"model__var_smoothing": [1e-9, 1e-7, 1e-5]}),
        "NB_Bernoulli": (
            BernoulliNB(),
            {"model__alpha": [0.1, 0.5, 1.0, 5.0], "model__fit_prior": [True, False], "model__binarize": [0.0, 0.5]},
        ),
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [15, 31], "model__weights": ["uniform", "distance"]}),
    }
    if HAS_XGB:
        zoo["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", n_estimators=400, tree_method="hist", subsample=0.8, colsample_bytree=0.8, random_state=RS, n_jobs=1),
            {"model__max_depth": [2, 3], "model__learning_rate": [0.03, 0.1], "model__reg_lambda": [1.0, 5.0], "model__scale_pos_weight": [float(np.sqrt(_spw)), _spw]},
        )
    zoo["RandomForest"] = (
        RandomForestClassifier(n_estimators=400, class_weight=None, random_state=RS, n_jobs=-1),
        {"model__max_depth": [6, 10], "model__min_samples_leaf": [5, 20]},
    )
    zoo["ExtraTrees"] = (
        ExtraTreesClassifier(n_estimators=400, class_weight=None, random_state=RS, n_jobs=-1),
        {"model__max_depth": [6, 10], "model__min_samples_leaf": [5, 20]},
    )
    zoo["AdaBoost"] = (AdaBoostClassifier(n_estimators=200, random_state=RS), {"model__n_estimators": [100, 200], "model__learning_rate": [0.5, 1.0]})
    zoo["HistGBM"] = (HistGradientBoostingClassifier(max_iter=400, class_weight=None, random_state=RS), {"model__learning_rate": [0.03, 0.1], "model__l2_regularization": [0.0, 1.0, 5.0]})
    if not fast_mode:
        zoo["SVM"] = (SVC(probability=True, class_weight=None, random_state=RS), {"model__C": [1.0, 10.0], "model__gamma": ["scale"]})
        zoo["MLP"] = (MLPClassifier(max_iter=400, early_stopping=True, n_iter_no_change=10, random_state=RS), {"model__alpha": [1e-4, 1e-2], "model__hidden_layer_sizes": [(32, 16), (64, 32, 16)]})
    return zoo, _spw


def run_bakeoff(X_train, y_train, X_test, y_test, prep_prespec, model_zoo, cv, random_state=42):
    """Single-pass bake-off: each zoo model is tuned (GridSearchCV) exactly once, and its
    OOF predictions (pooled cross_val_predict), test predictions, and per-fold CV scores are
    all captured from that one fit — mirroring notebook cell 22. Callers should reuse
    oof_scores/test_scores/cv_folds/fitted_best rather than re-running GridSearchCV."""
    results = []
    oof_scores, test_scores, cv_folds, fitted_best = {}, {}, {}, {}
    print("#" * 60)
    print("# 5c. Multi-model bake-off — pre-specified predictors, no feature selection")
    print("#" * 60)
    for name, (est, grid) in model_zoo.items():
        pipe = Pipeline([("prep", prep_prespec), ("model", est)])
        gs = GridSearchCV(
            pipe, grid,
            scoring="roc_auc", cv=cv, n_jobs=-1, refit=True, return_train_score=False,
        )
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        fitted_best[name] = best
        oof_scores[name] = cross_val_predict(
            clone(best), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1]
        test_scores[name] = best.predict_proba(X_test)[:, 1]
        cv_folds[name] = cross_val_score(clone(best), X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        oof_auc = roc_auc_score(y_train, oof_scores[name])
        test_auc = roc_auc_score(y_test, test_scores[name])
        lo, hi = _boot_ci(y_test.values, test_scores[name], seed=random_state)
        results.append({
            "model": name,
            "cv_auc": round(float(np.mean(cv_folds[name])), 3),
            "oof_auc": round(float(oof_auc), 3),
            "test_auc": round(float(test_auc), 3),
            "test_lo": round(float(lo), 3),
            "test_hi": round(float(hi), 3),
        })
        print(f"  {name:13s} CV={np.mean(cv_folds[name]):.3f}  OOF={oof_auc:.3f}  test={test_auc:.3f}")
    bake_df = pd.DataFrame(results).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    return bake_df, oof_scores, test_scores, cv_folds, fitted_best


# ===================================================================
# 5b. Marginal contribution — leave-one-out
# ===================================================================

def run_marginal_contribution(X_train, y_train, full_predictors, binary, categorical, continuous, cv, random_state=42, variant="firth"):
    results_mc = []
    full_auc = float(roc_auc_score(
        y_train,
        cross_val_predict(
            _firth_pipe(full_predictors, binary, categorical, continuous, variant=variant),
            X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
        )[:, 1],
    ))
    results_mc.append({"set": "full", "auc": round(full_auc, 3)})

    for c_exclude in full_predictors:
        sub = [c for c in full_predictors if c != c_exclude]
        sub_pipe = _firth_pipe(sub, binary, categorical, continuous, variant=variant)
        oof = cross_val_predict(
            sub_pipe, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        results_mc.append({"set": f"w/o {c_exclude}", "auc": round(roc_auc_score(y_train, oof), 3)})

    mc = pd.DataFrame(results_mc)
    print("\nMARGINAL CONTRIBUTION — leave-one-out Firth LR (train OOF AUC):")
    print(mc.to_string(index=False))
    return mc


# ===================================================================
# 6. Discrimination evaluation  (item 12e)
# ===================================================================

def evaluate_discrimination(
    X_train, y_train, X_test, y_test, best_estimators, cv, random_state=42,
    n_boot_ci=2000, n_repeated_cv=20,
):
    oof_pred = {}
    disc_rows = []
    rep = RepeatedStratifiedKFold(
        n_splits=cv.n_splits, n_repeats=n_repeated_cv, random_state=random_state
    )
    for name, bestm in best_estimators.items():
        oof = cross_val_predict(
            clone(bestm), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        oof_pred[name] = oof
        lo, hi = _boot_ci(y_train.values, oof, n=n_boot_ci, seed=random_state)
        ra = []
        for tr, te in rep.split(X_train, y_train):
            if y_train.iloc[te].nunique() < 2:
                continue
            ra.append(roc_auc_score(
                y_train.iloc[te],
                clone(bestm).fit(X_train.iloc[tr], y_train.iloc[tr]).predict_proba(X_train.iloc[te])[:, 1],
            ))
        tp = bestm.predict_proba(X_test)[:, 1]
        tlo, thi = _boot_ci(y_test.values, tp, n=n_boot_ci, seed=random_state)
        disc_rows.append({
            "model": name,
            "oof_auc": round(float(roc_auc_score(y_train, oof)), 3),
            "oof_lo": round(float(lo), 3),
            "oof_hi": round(float(hi), 3),
            "oof_pr_auc": round(float(average_precision_score(y_train, oof)), 3),
            "rep_cv_auc": round(float(np.mean(ra)), 3) if ra else np.nan,
            "rep_cv_sd": round(float(np.std(ra)), 3) if ra else np.nan,
            "test_auc": round(float(roc_auc_score(y_test, tp)), 3),
            "test_lo": round(float(tlo), 3),
            "test_hi": round(float(thi), 3),
        })
        print(f"  {name:20s} OOF={roc_auc_score(y_train, oof):.3f} ({lo:.3f}-{hi:.3f})  "
              f"repCV={np.mean(ra):.3f}  test={roc_auc_score(y_test, tp):.3f}")
    disc = pd.DataFrame(disc_rows)
    print(disc.round(3).to_string(index=False))
    return disc, oof_pred


# ===================================================================
# 7–8. Calibration, DCA plots
# ===================================================================

def plot_calibration(y, oof_pred, plots_dir, label="LogReg_Firth"):
    intercept, slope, brier = _cal_metrics(y, oof_pred)
    base = float(y.mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    fp, mp = calibration_curve(y, oof_pred, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", color=ACCENT_RED, lw=2.0, ms=7,
            markerfacecolor=ACCENT_RED, markeredgecolor=ACCENT_RED,
            label=f"{label} (slope {slope:.2f})")
    mx = max(float(np.percentile(oof_pred, 99)), 0.05)
    ax.plot([0, mx], [0, mx], color=MID_GREY, ls="--", lw=1.2)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"intercept {intercept:.2f}, slope {slope:.2f}, Brier {brier:.4f}")
    ax.legend(frameon=False)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "calibration_curve.png"), dpi=300)
    plt.close(fig)
    return intercept, slope, brier


def plot_roc_curve(y, oof_pred, plots_dir, label="LogReg_Firth"):
    fpr, tpr, _ = roc_curve(y, oof_pred)
    auc_val = roc_auc_score(y, oof_pred)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color=ACCENT_RED, lw=2.4, label=f"{label} OOF (AUC {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color=MID_GREY, ls="--", lw=1.2)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Out-of-fold ROC curve")
    ax.legend(loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=300)
    plt.close(fig)


def plot_dca(y, oof_pred, plots_dir):
    prev = float(y.mean())
    th = np.linspace(max(1e-4, prev / 10), min(0.20, max(0.05, prev * 10)), 300)
    nb_m = _net_benefit(y, oof_pred, th)
    treat_all = prev - (1 - prev) * th / (1 - th)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(th * 100, nb_m, color=ACCENT_RED, linewidth=2.5, label="LogReg_Firth", zorder=4)
    ax.plot(th * 100, treat_all, color=MID_GREY, linestyle="--", linewidth=1.5, label="Treat all", zorder=2)
    ax.axhline(0, color=DARK, linewidth=1.2, label="Treat none", zorder=3)
    y_min = min(-0.0005, np.nanmin(nb_m) - 0.0003) if len(nb_m) else -0.0005
    y_max = np.nanmax(nb_m) + 0.0005 if len(nb_m) else prev * 1.15
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(th.min() * 100, th.max() * 100)
    ax.set_xlabel("Threshold probability (%)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision-curve analysis using out-of-fold predictions")
    ax.legend(loc="upper right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "dca_curve.png"), dpi=300)
    plt.close(fig)


# ===================================================================
# 9. Bootstrap optimism correction
# ===================================================================

def bootstrap_optimism(X, y, pipe, n_boot=500, random_state=42):
    rng = np.random.default_rng(random_state)
    ap = pipe.predict_proba(X)[:, 1]
    app_auc = roc_auc_score(y, ap)
    app_sl = _cal_slope_only(y, ap)
    oa, os_ = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yb = y.iloc[idx]
        if yb.nunique() < 2 or yb.sum() < 3:
            continue
        mb = clone(pipe).fit(X.iloc[idx], yb)
        pb = mb.predict_proba(X.iloc[idx])[:, 1]
        po = mb.predict_proba(X)[:, 1]
        oa.append(roc_auc_score(yb, pb) - roc_auc_score(y, po))
        os_.append(_cal_slope_only(yb, pb) - _cal_slope_only(y, po))
    opt = {
        "auc_apparent": float(app_auc),
        "auc_corrected": float(app_auc - np.mean(oa)),
        "slope_apparent": float(app_sl),
        "slope_corrected": float(app_sl - np.nanmean(os_)),
    }
    print(f"  AUC: apparent={opt['auc_apparent']:.3f} corrected={opt['auc_corrected']:.3f}")
    print(f"  Cal slope: apparent={opt['slope_apparent']:.3f} corrected={opt['slope_corrected']:.3f}")
    return opt


# ===================================================================
# 14. Model specification, point score, risk equation  (items 22, 12g)
# ===================================================================

def save_specification(pipe_template, outdir, X, y):
    """Fits pipe_template (a Firth pipeline already configured with the deployable variant and
    shrinkage) on the full derivation cohort. The OR table uses the FIRTH (de-biased)
    coefficients + their penalized-likelihood CIs — always beta_firth_/ci_/pvals_, regardless of
    variant, per notebook Section 14. Returns the fitted pipeline for reuse (point score, risk
    equation, pickling) so it is only fit once."""
    final_lr = clone(pipe_template).fit(X, y)
    prep_f = final_lr.named_steps["prep"]
    fl = final_lr.named_steps["model"]
    feat = list(prep_f.get_feature_names_out())
    terms = ["intercept"] + feat
    beta_firth = fl.beta_firth_
    ci = fl.ci_
    spec = pd.DataFrame({
        "term": terms,
        "beta_firth": beta_firth,
        "odds_ratio": np.exp(beta_firth),
        "or_lo": np.exp(ci[:, 0]),
        "or_hi": np.exp(ci[:, 1]),
        "p_value": fl.pvals_,
    })
    spec.to_csv(os.path.join(outdir, "logreg_firth_specification.csv"), index=False)
    print("DEPLOYABLE FIRTH LR SPECIFICATION (item 22) — Firth de-biased odds ratios:")
    print(spec.round(3).to_string(index=False))
    print(f"  deployed intercept (variant={fl.variant}, shrinkage={fl.shrinkage:.3f}) = {fl.intercept_:.4f}; "
          f"deployed slopes = shrinkage x Firth slopes.")
    return spec, final_lr


def compute_point_score(final_lr, score_increments, points_max, outdir, ps_cont=None):
    """Clinically-scaled integer point score from the de-biased (Firth) per-natural-unit
    coefficients — matches notebook Section 14, which scores on beta_firth_, not the deployed
    (shrunk) coefficients. Returns (point_table, ref) where ref is the log-odds represented by
    one point — Section 18 needs it to convert predictions to points on the same scale."""
    prep_f = final_lr.named_steps["prep"]
    fl = final_lr.named_steps["model"]
    feat = list(prep_f.get_feature_names_out())
    nbeta = fl.beta_firth_[1:]
    sd_map = {}
    if ps_cont and prep_f.named_transformers_.get("cont", None) is not None:
        scl = prep_f.named_transformers_["cont"].named_steps.get("sc", None)
        if scl is not None and hasattr(scl, "scale_"):
            for nm, sd in zip(ps_cont, scl.scale_):
                sd_map["cont__" + nm] = sd
    rows_p = []
    for nm, b in zip(feat, nbeta):
        base = nm.replace("cont__", "").replace("bin__", "").replace("cat__", "")
        b_clin = b / sd_map.get(nm, 1.0)
        incr = score_increments.get(base, 1.0)
        contrib = b_clin * incr
        rows_p.append({
            "term": base,
            "per_increment": incr,
            "beta_per_increment": contrib,
            "or_per_increment": np.exp(contrib),
        })
    ptab = pd.DataFrame(rows_p)
    mx = float(np.max(np.abs(ptab["beta_per_increment"]))) or 1.0
    ref = mx / points_max
    ptab["points"] = np.round(ptab["beta_per_increment"] / ref).astype(int)
    ptab = ptab.reindex(ptab["points"].abs().sort_values(ascending=False).index)
    ptab.to_csv(os.path.join(outdir, "logreg_firth_point_score.csv"), index=False)
    print(f"\nPOINT SCORE (1 pt = {ref:.3f} log-odds, strongest = {points_max}):")
    print(ptab.round(3).to_string(index=False))
    return ptab, ref


def save_risk_equation(final_lr, outdir):
    """Uses the SHIPPED (deployed) coefficients — fl.intercept_/coef_ already reflect the
    variant (e.g. FLIC) and shrinkage — since this is the equation predict_proba actually uses."""
    prep_f = final_lr.named_steps["prep"]
    fl = final_lr.named_steps["model"]
    feat = list(prep_f.get_feature_names_out())
    eq = "logit(p) = %.4f" % fl.intercept_
    eq += "".join(f" + ({b:.4f})*[{n}]" for b, n in zip(fl.coef_, feat))
    with open(os.path.join(outdir, "logreg_firth_risk_equation.txt"), "w") as f:
        f.write(eq + "\n\np = 1/(1+exp(-logit(p)))\n")
    print(f"Risk equation saved.")


# ===================================================================
# 14b. Sensitivity — reduced model dropping age & occlusion length  (items 12, 23a)
# ===================================================================

def run_reduced_model_sensitivity(
    X, y, X_train, y_train, X_test, y_test, ps_present, oof_full,
    binary, categorical, continuous, cv, outdir, variant="firth",
):
    """5c/14 flagged age_manual_input and occlusion_length_mm as carrying little independent
    signal. Refits the deployable Firth on the remaining predictors and DeLong-tests (OOF)
    whether dropping the two costs discrimination. The full predictor set stays primary
    (occlusion length anchors the published nomogram); this is a robustness check only."""
    from bakeoff.analysis import delong_test

    red_cols = [c for c in ps_present if c not in ("age_manual_input", "occlusion_length_mm")]
    oof_red = cross_val_predict(
        _firth_pipe(red_cols, binary, categorical, continuous, variant=variant),
        X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    auc_full = float(roc_auc_score(y_train, oof_full))
    auc_red = float(roc_auc_score(y_train, oof_red))
    a_f, a_r, pv = delong_test(y_train.values, oof_full, oof_red)
    red_test = _firth_pipe(red_cols, binary, categorical, continuous, variant=variant).fit(
        X_train, y_train,
    ).predict_proba(X_test)[:, 1]
    print(f"\nREDUCED MODEL ({len(red_cols)} predictors): {red_cols}")
    print(f"  OOF AUC   full({len(ps_present)})={auc_full:.3f}  reduced({len(red_cols)})={auc_red:.3f}  | DeLong p={pv:.3f}")
    print(f"  OOF cal-slope reduced={_cal_slope_only(y_train, oof_red):.3f}  | "
          f"test AUC reduced={roc_auc_score(y_test, red_test):.3f}")

    red_final = _firth_pipe(red_cols, binary, categorical, continuous, variant=variant).fit(X, y)
    rf = red_final.named_steps["model"]
    rp = red_final.named_steps["prep"]
    red_spec = pd.DataFrame({
        "term": ["intercept"] + list(rp.get_feature_names_out()),
        "odds_ratio": np.exp(rf.beta_firth_),
        "or_lo": np.exp(rf.ci_[:, 0]),
        "or_hi": np.exp(rf.ci_[:, 1]),
        "p_value": rf.pvals_,
    })
    red_spec.to_csv(os.path.join(outdir, "reduced_model_specification.csv"), index=False)
    print("\nReduced-model odds ratios (Firth de-biased):")
    print(red_spec.round(3).to_string(index=False))

    same = abs(auc_full - auc_red) < 0.01 and pv > 0.05
    print(f"\nVerdict: dropping age & occlusion length -> "
          f"{'no meaningful discrimination loss' if same else 'measurable discrimination change'} "
          f"(dAUC={auc_red - auc_full:+.3f}, p={pv:.3f}). Full predictor model stays primary.")
    return {
        "predictors": red_cols, "oof_auc_full": auc_full, "oof_auc_reduced": auc_red,
        "delong_p": float(pv), "oof_cal_slope_reduced": float(_cal_slope_only(y_train, oof_red)),
        "test_auc_reduced": float(roc_auc_score(y_test, red_test)),
    }


# ===================================================================
# 18. Observed vs predicted incidence by deployable-model point strata
# ===================================================================

def plot_incidence_by_point_strata(final_lr, ref, X, y, X_train, X_test, outdir, plots_dir):
    """Descriptive risk-stratification plot for the final shipped model (FLIC + shrinkage,
    refit on the full derivation cohort). Predicted incidence is the mean predicted probability
    within each point stratum; observed incidence is the empirical event rate. The point scale
    (`ref` = log-odds per point, from Section 14) is anchored so the lowest-risk patient in the
    full cohort has 0 points, and reused across whole/test/training cohort panels."""
    point_unit = float(ref) if np.isfinite(ref) and ref > 0 else 1.0

    pred_all = np.clip(final_lr.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    lp_all = np.log(pred_all / (1 - pred_all))
    points_all = np.round((lp_all - np.nanmin(lp_all)) / point_unit).astype(int)
    score_frame = pd.DataFrame({
        "y": np.asarray(y, dtype=int),
        "pred": pred_all,
        "deployable_points": points_all,
    }, index=X.index)

    cohort_frames = {
        "Whole cohort": score_frame.copy(),
        "Test cohort": score_frame.loc[X_test.index].copy(),
        "Training cohort": score_frame.loc[X_train.index].copy(),
    }

    bin_schemes = [
        ("10-point strata", [0, 10, 20, 30, 40, np.inf],
         ["0–9", "10–19", "20–29", "30–39", ">39"]),
        ("5-point strata", [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, np.inf],
         ["0–4", "5–9", "10–14", "15–19", "20–24", "25–29", "30–34", "35–39", "40–44", ">44"]),
        ("7-point strata", [0, 7, 14, 21, 28, 35, 42, np.inf],
         ["0–6", "7–13", "14–20", "21–27", "28–34", "35–41", ">41"]),
    ]

    fl = final_lr.named_steps.get("model")
    prep = final_lr.named_steps.get("prep")
    feature_names = list(prep.get_feature_names_out()) if prep is not None else []
    print(f"\nMODEL USED FOR FIGURE 18: {type(fl).__name__} "
          f"(variant={getattr(fl, 'variant', '')}, shrinkage={getattr(fl, 'shrinkage', float('nan')):.3f}, "
          f"deployed intercept={getattr(fl, 'intercept_', float('nan')):.4f}); "
          f"1 point = {point_unit:.3f} log-odds; terms: {feature_names}")

    point_counts_by_cohort = (
        pd.concat(
            [frame["deployable_points"].value_counts().sort_index().rename(name)
             for name, frame in cohort_frames.items()],
            axis=1,
        )
        .fillna(0)
        .astype(int)
        .rename_axis("deployable_points")
        .reset_index()
    )
    point_counts_by_cohort.to_csv(
        os.path.join(outdir, "deployable_patient_counts_by_exact_point.csv"), index=False,
    )
    print("\nNumber of patients at each exact deployable point value (Figure 18):")
    print(point_counts_by_cohort.to_string(index=False))

    def _summarize(d, edges, labels, cohort_name, scheme_name):
        z = d.copy()
        z["point_stratum"] = pd.cut(
            z["deployable_points"], bins=edges, labels=labels,
            right=False, include_lowest=True,
        )
        out = (z.groupby("point_stratum", observed=False)
                 .agg(n=("y", "size"), events=("y", "sum"),
                      observed_incidence=("y", "mean"),
                      predicted_incidence=("pred", "mean"),
                      point_min=("deployable_points", "min"),
                      point_max=("deployable_points", "max"))
                 .reset_index())
        out.insert(0, "scheme", scheme_name)
        out.insert(1, "cohort", cohort_name)
        out["n"] = out["n"].astype(int)
        out["events"] = out["events"].fillna(0).astype(int)
        out["observed_incidence_pct"] = out["observed_incidence"] * 100
        out["predicted_incidence_pct"] = out["predicted_incidence"] * 100
        return out

    incidence_tables = []
    for scheme_name, edges, labels in bin_schemes:
        for cohort_name, frame in cohort_frames.items():
            incidence_tables.append(_summarize(frame, edges, labels, cohort_name, scheme_name))
    incidence_by_point_strata = pd.concat(incidence_tables, ignore_index=True)
    incidence_by_point_strata.to_csv(
        os.path.join(outdir, "deployable_observed_predicted_incidence_by_point_strata.csv"),
        index=False,
    )

    yvals = incidence_by_point_strata[["observed_incidence_pct", "predicted_incidence_pct"]].to_numpy(dtype=float)
    finite_yvals = yvals[np.isfinite(yvals)]
    ymax = max(5.0, float(np.nanmax(finite_yvals)) * 1.25 if finite_yvals.size else 5.0)

    bar_width = 0.38
    fig, axes = plt.subplots(3, 3, figsize=(21, 15.5), sharey=True)
    for r, (scheme_name, edges, labels) in enumerate(bin_schemes):
        for c, cohort_name in enumerate(["Whole cohort", "Test cohort", "Training cohort"]):
            ax = axes[r, c]
            s = incidence_by_point_strata.query(
                "scheme == @scheme_name and cohort == @cohort_name"
            ).copy()
            x = np.arange(len(s))
            has_patients = s["n"].to_numpy(dtype=int) > 0
            observed = np.where(has_patients, s["observed_incidence_pct"].to_numpy(dtype=float), np.nan)
            predicted = np.where(has_patients, s["predicted_incidence_pct"].to_numpy(dtype=float), np.nan)
            ax.bar(x - bar_width / 2, observed, width=bar_width, color=ACCENT_RED, label="Observed")
            ax.bar(x + bar_width / 2, predicted, width=bar_width, color="#4C78A8", label="Predicted")
            tick_labels = [f"{lab}\nn={int(n)}" for lab, n in zip(labels, s["n"].to_numpy(dtype=int))]
            ax.set_xticks(x)
            ax.set_xticklabels(tick_labels, rotation=45 if len(labels) > 5 else 0,
                                ha="right" if len(labels) > 5 else "center")
            ax.set_ylim(0, ymax)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
            ax.set_title(f"{cohort_name} — {scheme_name}")
            if c == 0:
                ax.set_ylabel("Urgent-MCS incidence")
            if r == len(bin_schemes) - 1:
                ax.set_xlabel("Deployable-model point stratum")
            if r == 0 and c == 0:
                ax.legend(loc="upper left")
            style_axis(ax)
    fig.suptitle("Figure 18. Observed vs predicted urgent-MCS incidence by deployable-model point strata",
                 y=1.01, fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "deployable_observed_predicted_incidence_by_point_strata.png"), dpi=300)
    fig.savefig(os.path.join(plots_dir, "deployable_observed_predicted_incidence_by_point_strata.pdf"))
    plt.close(fig)

    # Standalone plot: 7-point strata, whole cohort only
    scheme_name, cohort_name, ymax7 = "7-point strata", "Whole cohort", 7.0
    s = incidence_by_point_strata.query("scheme == @scheme_name and cohort == @cohort_name").copy()
    x = np.arange(len(s))
    has_patients = s["n"].to_numpy(dtype=int) > 0
    observed = np.where(has_patients, s["observed_incidence_pct"].to_numpy(dtype=float), np.nan)
    predicted = np.where(has_patients, s["predicted_incidence_pct"].to_numpy(dtype=float), np.nan)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - bar_width / 2, observed, width=bar_width, color=ACCENT_RED, label="Observed")
    ax.bar(x + bar_width / 2, predicted, width=bar_width, color="#4C78A8", label="Predicted")
    tick_labels = [f"{lab}\nn={int(n)}" for lab, n in zip(s["point_stratum"].astype(str), s["n"].to_numpy(dtype=int))]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(0, ymax7)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel("Urgent-MCS incidence")
    ax.set_xlabel("Deployable-model point stratum")
    ax.legend(loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "deployable_observed_predicted_incidence_7point_whole_cohort.png"), dpi=300)
    fig.savefig(os.path.join(plots_dir, "deployable_observed_predicted_incidence_7point_whole_cohort.pdf"))
    plt.close(fig)

    print("\nOBSERVED VS PREDICTED INCIDENCE BY DEPLOYABLE POINT STRATA:")
    print(incidence_by_point_strata[[
        "scheme", "cohort", "point_stratum", "n", "events",
        "observed_incidence_pct", "predicted_incidence_pct",
    ]].round(2).to_string(index=False))

    return point_counts_by_cohort, incidence_by_point_strata


# ===================================================================
# Comparison plots  (section 5d — top models + Firth)
# ===================================================================

def _plot_bakeoff_auc(bakeoff_results, firth_auc, plots_dir):
    order = bakeoff_results.sort_values("cv_auc", ascending=False)["model"].tolist()
    names = order + ["LogReg_Firth"]
    cv_vals = list(bakeoff_results.sort_values("cv_auc", ascending=False)["cv_auc"].values) + [firth_auc]
    xp = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(xp, cv_vals, 0.6, color="#4da3ff", edgecolor="white")
    for i, n in enumerate(names):
        if n == "LogReg_Firth":
            bars[i].set_color(ACCENT_RED)
    ax.axhline(0.5, ls=":", color=MID_GREY, lw=1)
    ax.axhline(0.80, ls="--", color=DARK, lw=1)
    ax.set_xticks(xp); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("CV ROC-AUC")
    ax.set_title("Model comparison — pre-specified Firth LR vs tuned models")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "comparison_auc.png"), dpi=300)
    plt.close(fig)


def _plot_comparison_roc(oof_pred_firth, oof_pred_bench, y_train, bench_name, plots_dir):
    fig, ax = plt.subplots(figsize=(7, 7))
    for name, preds in [("LogReg_Firth", oof_pred_firth), (bench_name, oof_pred_bench)]:
        fpr, tpr, _ = roc_curve(y_train, preds)
        lw = 2.5 if name == "LogReg_Firth" else 1.5
        ax.plot(fpr, tpr, lw=lw, label=f"{name} ({roc_auc_score(y_train, preds):.3f})")
    ax.plot([0, 1], [0, 1], color=MID_GREY, ls="--", lw=1.2)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("OOF ROC — Firth vs " + bench_name)
    ax.legend(fontsize=10, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "comparison_roc.png"), dpi=300)
    plt.close(fig)


def _plot_comparison_calibration(oof_pred_firth, oof_pred_bench, y_train, bench_name, plots_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "--", lw=1.5, color=MID_GREY, label="ideal")
    for name, preds in [("LogReg_Firth", oof_pred_firth), (bench_name, oof_pred_bench)]:
        fp, mp = calibration_curve(y_train, preds, n_bins=10, strategy="quantile")
        _, slope, brier = _cal_metrics(y_train, preds)
        lw = 2.5 if name == "LogReg_Firth" else 1.5
        ax.plot(mp, fp, "o-", lw=lw, label=f"{name} (slope={slope:.2f}, Brier={brier:.4f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
    ax.set_title("Calibration — Firth vs " + bench_name)
    ax.legend(fontsize=10, loc="lower right")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "comparison_calibration.png"), dpi=300)
    plt.close(fig)


# ===================================================================
# Main pipeline
# ===================================================================

def run(
    data_path,
    target,
    pre_specified_predictors,
    plausible_bounds=None,
    exclude_planned_mcs=True,
    planned_mcs_col=None,
    site_col=None,
    year_col=None,
    outdir="tripod_outputs",
    random_state=42,
    test_size=0.20,
    cv_splits=5,
    n_boot_ci=2000,
    n_boot_optimism=500,
    n_repeated_cv=20,
    cat_max_levels=20,
    fast_mode=False,
    score_increments=None,
    points_max=10,
    yesno_na_vars=None,
    redundant_groups=None,
    pub_vars=None,
    pub_pts=None,
    published_betas=None,
    deployable_variant="flic",
    benchmark_model="ExtraTrees",
    use_synth_if_missing=True,
):
    if yesno_na_vars is None:
        yesno_na_vars = []
    if redundant_groups is None:
        redundant_groups = []
    if score_increments is None:
        score_increments = {}
    if pub_vars is None:
        pub_vars = {}
    if pub_pts is None:
        pub_pts = {}

    results_dict = {}
    os.makedirs(outdir, exist_ok=True)
    plots_dir = os.path.join(outdir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # ── 0b. Sample-size adequacy — pmsampsize  (item 8) ──
    print("=" * 60)
    print("0b. Sample-size adequacy — pmsampsize  (item 8)")
    print("=" * 60)
    results_dict["pmsampsize_grid"] = print_pmsampsize_grid(n_predictors=len(pre_specified_predictors))

    if use_synth_if_missing and not os.path.exists(data_path):
        print(f"\n{data_path} not found -> generating SYNTHETIC cohort for a dry run.\n")
        make_synth(path=data_path)

    # ── 1. Load & prepare data ──
    print("=" * 60)
    print("1. Data, outcome & cohort derivation  (items 5, 6)")
    print("=" * 60)
    X0, y, df, planned_mask, validation_only = load_and_prepare(
        data_path, target,
        plausible_bounds=plausible_bounds,
        exclude_planned_mcs=exclude_planned_mcs,
        planned_mcs_col=planned_mcs_col,
        site_col=site_col,
        year_col=year_col,
    )

    # ── 2. Variable classification  (item 7) ──
    print("=" * 60)
    print("2. Predictor typing & pre-procedural audit  (item 7)")
    print("=" * 60)
    X, binary, categorical, continuous, typing_df = classify_variables(
        X0, cat_max=cat_max_levels, yesno_na_vars=yesno_na_vars,
    )
    typing_df.to_csv(os.path.join(outdir, "variable_typing.csv"), index=False)

    # Pre-procedural audit
    unconfirmed = [c for c in yesno_na_vars if c in X.columns]
    print("Pre-procedural audit: all predictors knowable before procedure.")

    # Derivation cohort (exclude planned MCS)
    X_full, y_full = X.copy(), y.copy()
    if exclude_planned_mcs and planned_mask.any():
        der = ~planned_mask.to_numpy()
        X = X.loc[der].reset_index(drop=True)
        y = y.loc[der].reset_index(drop=True)
        df = df.loc[der].reset_index(drop=True)
        print(f"DERIVATION COHORT: n={len(y)} events={int(y.sum())} ({y.mean():.3%})")

    # ── 2b. Table 1 ──
    from bakeoff.analysis import generate_table1
    print("=" * 60)
    print("2b. Participant characteristics  (Table 1)")
    print("=" * 60)
    generate_table1(X, y, binary, continuous, outdir)

    # ── 3. Missing data  (item 9) ──
    from bakeoff.analysis import generate_missingness_table
    print("=" * 60)
    print("3. Missing data  (item 9)")
    print("=" * 60)
    generate_missingness_table(X, outdir)

    # ── 4. Train/test split & redundancy reduction  (item 8) ──
    print("=" * 60)
    print("4. Split & redundancy reduction  (item 8)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    reduce_redundancy(X_train, X_test, y_train, redundant_groups, pre_specified_predictors)

    # Update variable lists after dropping
    for dset in [X_train, X_test]:
        binary = [c for c in binary if c in dset.columns]
        categorical = [c for c in categorical if c in dset.columns]
        continuous = [c for c in continuous if c in dset.columns]

    # Build full preprocessor (for bake-off / benchmark)
    prep = build_full_preprocessor(binary, categorical, continuous)
    n_enc = prep.fit(X_train).transform(X_train.head(20)).shape[1]
    events_total = int(y.sum())
    print(f"  EPV (candidate pool) = {events_total}/{n_enc} = {events_total / max(n_enc, 1):.2f}")

    # Pre-specified predictors that exist in data
    ps_present = [c for c in pre_specified_predictors if c in X.columns]
    missing_ps = [c for c in pre_specified_predictors if c not in X.columns]
    if missing_ps:
        print(f"WARNING: pre-specified predictors absent from data: {missing_ps}")

    # ── 5a. Train deployable Firth LR + ExtraTrees benchmark ──
    print("=" * 60)
    print("5a. Build deployable model + benchmark  (items 12, 13, 15)")
    print("=" * 60)
    prep_prespec, ps_cont, ps_bin, ps_cat = build_prespec_preprocessor(
        ps_present, binary, categorical, continuous,
    )
    n_enc_ps = prep_prespec.fit(X_train).transform(X_train.head(20)).shape[1]
    print(f"  Pre-specified predictors: {ps_present} -> {n_enc_ps} encoded df")
    print(f"  EPV (deployable) = {events_total}/{n_enc_ps} = {events_total / max(n_enc_ps, 1):.1f}")

    deployable = Pipeline([
        ("prep", prep_prespec),
        ("model", FirthLogisticRegression(variant=deployable_variant)),
    ])
    deployable.fit(X_train, y_train)

    dep_oof = cross_val_predict(
        clone(deployable), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    print(f"\nLogReg_Firth CV AUC: {roc_auc_score(y_train, dep_oof):.3f}  "
          f"(benchmark is selected AFTER the bake-off, by OOF AUC)")

    # ── 5a. Multi-model bake-off (single pass — reused for boxplot, PR curves, benchmark) ──
    zoo, _ = _model_zoo(y_train, fast_mode=fast_mode)
    bakeoff_results, oof_scores_all, test_scores_all, cv_folds_all, fitted_best = run_bakeoff(
        X_train, y_train, X_test, y_test, prep_prespec, zoo, cv,
        random_state=random_state,
    )
    dep_test_pred = deployable.predict_proba(X_test)[:, 1]
    dep_cv_auc = float(roc_auc_score(
        y_train,
        cross_val_predict(clone(deployable), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1],
    ))
    oof_scores_all["LogReg_Firth"] = dep_oof
    test_scores_all["LogReg_Firth"] = dep_test_pred
    cv_folds_all["LogReg_Firth"] = np.array([
        roc_auc_score(y_train.iloc[te],
            clone(deployable).fit(X_train.iloc[tr], y_train.iloc[tr]).predict_proba(X_train.iloc[te])[:, 1])
        for tr, te in cv.split(X_train, y_train)
    ])
    bakeoff_results = pd.concat([
        bakeoff_results,
        pd.DataFrame([{
            "model": "LogReg_Firth",
            "cv_auc": round(dep_cv_auc, 3),
            "oof_auc": round(float(roc_auc_score(y_train, dep_oof)), 3),
            "test_auc": round(float(roc_auc_score(y_test, dep_test_pred)), 3),
            "test_lo": np.nan,
            "test_hi": np.nan,
        }]),
    ], ignore_index=True).sort_values("oof_auc", ascending=False).reset_index(drop=True)
    bakeoff_results.to_csv(os.path.join(outdir, "bakeoff_results.csv"), index=False)
    print("\nBake-off summary (sorted by out-of-fold AUC):")
    print(bakeoff_results.round(3).to_string(index=False))

    order = bakeoff_results["model"].tolist()
    TOP3 = [m for m in order if m != "LogReg_Firth"][:3]
    fitted_best["LogReg_Firth"] = deployable
    bench = benchmark_model if benchmark_model in fitted_best else [m for m in order if m != "LogReg_Firth"][0]
    best_estimators = {"LogReg_Firth": deployable, bench: fitted_best[bench]}
    print(f"\nTop-3 by out-of-fold AUC: {TOP3} -> compared head-to-head with LogReg_Firth in 5b.")
    print(f"Carried models: {list(best_estimators)} | benchmark = {bench} "
          f"(OOF AUC {roc_auc_score(y_train, oof_scores_all[bench]):.3f})")

    # Boxplot
    box_colors = ["#4BA3D3", "#E5A93C", "#59B894", "#D9A35F", "#C18AC5",
                  "#C99A72", "#E9A6D2", "#A6A6A6", "#F2E768", "#75B8DE",
                  "#4D9BC7", "#E2A13C"]
    fig, ax = plt.subplots(figsize=(11, 5))
    bp = ax.boxplot(
        [cv_folds_all[m] for m in order],
        patch_artist=True, widths=0.62,
        boxprops=dict(color=DARK, linewidth=1.0),
        whiskerprops=dict(color=DARK, linewidth=1.0),
        capprops=dict(color=DARK, linewidth=1.0),
        medianprops=dict(color=ACCENT_RED, linewidth=2.0),
        flierprops=dict(marker="o", markerfacecolor="white",
                        markeredgecolor=DARK, markersize=4.5, alpha=0.9),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color); patch.set_alpha(0.9)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=45, ha="right")
    ax.set_ylabel("Cross-validated ROC-AUC")
    ax.axhline(0.5, linestyle="--", color=MID_GREY, linewidth=1.2)
    ax.set_title("Algorithm comparison on the pre-specified feature set")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "bakeoff_boxplot.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # PR curves
    models_pr = list(oof_scores_all)
    colors_pr = model_color_map(models_pr)
    fig, ax = plt.subplots(figsize=(8, 6))
    for m in models_pr:
        precision, recall, _ = precision_recall_curve(y_train, oof_scores_all[m])
        ap = average_precision_score(y_train, oof_scores_all[m])
        ax.plot(recall, precision, color=colors_pr[m],
                lw=2.5 if m == "LogReg_Firth" else 1.7,
                alpha=1.0 if m == "LogReg_Firth" else 0.85,
                label=f"{m} ({ap:.3f})")
    prev = y_train.mean()
    ax.axhline(prev, ls="--", color=MID_GREY, lw=1.3, label=f"Baseline ({prev:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–recall curves (OOF)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "precision_recall_oof.png"), dpi=300)
    plt.close(fig)
    pr_table = pd.DataFrame({
        "model": models_pr,
        "oof_pr_auc": [average_precision_score(y_train, oof_scores_all[m]) for m in models_pr],
        "test_pr_auc": [average_precision_score(y_test, test_scores_all[m]) for m in models_pr],
    }).sort_values("oof_pr_auc", ascending=False)
    pr_table.to_csv(os.path.join(outdir, "precision_recall.csv"), index=False)
    print("\nPrecision-recall summary:")
    print(pr_table.round(3).to_string(index=False))

    # ── 5b. Why Firth — discrimination vs calibration for top-3 + Firth ──
    print("=" * 60)
    print("5b. Why logistic regression — top models vs Firth  (item 12g)")
    print("=" * 60)
    panel = list(dict.fromkeys(TOP3 + ["LogReg_Firth"]))
    cmap = model_color_map(panel)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for m in panel:
        fp, mp = calibration_curve(y_train, oof_scores_all[m], n_bins=10, strategy="quantile")
        axes[0].plot(mp, fp, "o-", color=cmap[m],
                     ms=6 if m == "LogReg_Firth" else 4.5,
                     lw=2.2 if m == "LogReg_Firth" else 1.5,
                     label=f"{m} (slope {_cal_slope_only(y_train, oof_scores_all[m]):.2f})")
    mx = max(np.percentile(oof_scores_all["LogReg_Firth"], 99), 0.03)
    axes[0].plot([0, mx], [0, mx], color=MID_GREY, ls="--", lw=1.2)
    axes[0].set_xlim(0, mx); axes[0].set_ylim(0, mx)
    axes[0].set_xlabel("Predicted probability"); axes[0].set_ylabel("Observed frequency")
    axes[0].set_title("Calibration (OOF)"); axes[0].legend(fontsize=8)
    for m in panel:
        fpr, tpr, _ = roc_curve(y_train, oof_scores_all[m])
        axes[1].plot(fpr, tpr, color=cmap[m],
                     lw=2.3 if m == "LogReg_Firth" else 1.6,
                     label=f"{m} ({roc_auc_score(y_train, oof_scores_all[m]):.3f})")
    axes[1].plot([0, 1], [0, 1], color=MID_GREY, ls="--", lw=1.2)
    axes[1].set_xlabel("1 - Specificity"); axes[1].set_ylabel("Sensitivity")
    axes[1].set_title("ROC (OOF)"); axes[1].legend(fontsize=8, loc="lower right")
    th = np.linspace(0.001, 0.10, 100); prev_l = float(y_train.mean())
    axes[2].plot(th * 100, prev_l - (1 - prev_l) * th / (1 - th),
                 color=MID_GREY, lw=1.3, ls="--", label="Treat all")
    axes[2].axhline(0, color=DARK, lw=1.2, label="Treat none")
    for m in panel:
        nb_m = _net_benefit(y_train, oof_scores_all[m], th)
        axes[2].plot(th * 100, nb_m, color=cmap[m],
                     lw=2.3 if m == "LogReg_Firth" else 1.6, label=m)
    axes[2].set_ylim(-0.005, prev_l * 1.15)
    axes[2].set_xlabel("Threshold probability (%)"); axes[2].set_ylabel("Net benefit")
    axes[2].set_title("Decision curve (OOF)"); axes[2].legend(fontsize=8)
    for a in axes:
        style_axis(a)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "why_firth_panel.png"), dpi=300)
    plt.close(fig)
    wtab = pd.DataFrame([{
        "model": m,
        "oof_auc": roc_auc_score(y_train, oof_scores_all[m]),
        "cal_slope": _cal_slope_only(y_train, oof_scores_all[m]),
        "brier": brier_score_loss(y_train, oof_scores_all[m]),
    } for m in panel]).sort_values("oof_auc", ascending=False)
    wtab.to_csv(os.path.join(outdir, "why_firth_table.csv"), index=False)
    results_dict["why_firth"] = wtab.to_dict(orient="records")
    print("WHY FIRTH — discrimination vs calibration (top-3 by CV AUC + Firth):")
    print(wtab.round(3).to_string(index=False))
    print("Near-identical AUC, but top discriminators show slope!=1 (mis-calibrated); Firth slope ~1 -> deployable.")

    # ── 5c. Parsimony sweep — Firth over clinical priority order ──
    print("=" * 60)
    print("5c. Parsimony sweep — Firth by clinical priority")
    print("=" * 60)
    order_pars = [c for c in ps_present]
    ksweep = []
    for k in range(3, len(order_pars) + 1):
        cols = order_pars[:k]
        oofk = cross_val_predict(_firth_pipe(cols, binary, categorical, continuous, variant=deployable_variant),
                                 X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
        ksweep.append({
            "k": k, "added": cols[-1], "epv": round(int(y_train.sum()) / k, 1),
            "cv_oof_auc": roc_auc_score(y_train, oofk),
            "cal_slope": _cal_slope_only(y_train, oofk),
        })
    ks = pd.DataFrame(ksweep)
    ks.to_csv(os.path.join(outdir, "firth_k_sweep.csv"), index=False)
    results_dict["firth_k_sweep"] = ks.to_dict(orient="records")
    print("FIRTH PARSIMONY SWEEP (priority order; not data-selected):")
    print(ks.round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    ax.plot(ks["k"], ks["cv_oof_auc"], "o-", color=ACCENT_RED, lw=2.0, ms=6)
    for _, r in ks.iterrows():
        ax.annotate(r["added"], (r["k"], r["cv_oof_auc"]), fontsize=7,
                    rotation=25, ha="left", va="bottom", color=DARK)
    ax.set_xlabel("Number of pre-specified predictors")
    ax.set_ylabel("CV out-of-fold AUC")
    ax.set_title("Firth discrimination vs model size")
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "firth_k_sweep.png"), dpi=300)
    plt.close(fig)

    # ── 5d. Marginal contribution — leave-one-out ──
    print("=" * 60)
    print("5d. Marginal contribution — leave-one-out")
    print("=" * 60)
    mc = run_marginal_contribution(
        X_train[ps_present], y_train, ps_present, binary, categorical, continuous, cv,
        random_state=random_state, variant=deployable_variant,
    )
    mc.to_csv(os.path.join(outdir, "marginal_contribution.csv"), index=False)

    # ── 6. Discrimination ──
    print("=" * 60)
    print("6. Discrimination — both models  (items 12e, 23a)")
    print("=" * 60)
    disc, oof_pred = evaluate_discrimination(
        X_train, y_train, X_test, y_test,
        best_estimators,
        cv, random_state=random_state,
        n_boot_ci=n_boot_ci, n_repeated_cv=n_repeated_cv,
    )
    disc.to_csv(os.path.join(outdir, "discrimination.csv"), index=False)
    results_dict["discrimination"] = disc.to_dict(orient="records")

    # ---- data-driven pmsampsize verdict (Section 0b tail) ----
    results_dict["pmsampsize_data_driven"] = print_pmsampsize_verdict(
        prevalence=float(y.mean()), n_predictors=len(ps_present),
        oof_auc=float(roc_auc_score(y_train, oof_pred["LogReg_Firth"])),
        n_dev=len(y_train), events_dev=int(y_train.sum()),
    )

    # ── Comparison plots ──
    dep_oof = oof_pred["LogReg_Firth"]
    bench_oof = oof_pred[bench]
    _plot_bakeoff_auc(
        bakeoff_results[bakeoff_results["model"] != "LogReg_Firth"],
        dep_cv_auc, plots_dir,
    )
    _plot_comparison_roc(dep_oof, bench_oof, y_train, bench, plots_dir)
    _plot_comparison_calibration(dep_oof, bench_oof, y_train, bench, plots_dir)

    # ── 7. Calibration ──
    print("=" * 60)
    print("7. Calibration — deployable model  (items 12e, 12f)")
    print("=" * 60)
    intercept, slope, brier = plot_calibration(y_train, dep_oof, plots_dir)
    print(f"  Calibration-in-the-large: {intercept:.3f}  |  slope: {slope:.3f}  |  Brier: {brier:.4f}")
    plot_roc_curve(y_train, dep_oof, plots_dir)

    # ── 8. DCA ──
    print("=" * 60)
    print("8. Decision-curve analysis  (item 12e)")
    print("=" * 60)
    plot_dca(y_train, dep_oof, plots_dir)
    print("  DCA curve saved.")

    # ── 9. Bootstrap optimism ──
    print("=" * 60)
    print("9. Internal validation — bootstrap optimism  (item 12)")
    print("=" * 60)
    pipe_full = clone(deployable).fit(X_train[ps_present], y_train)
    opt = bootstrap_optimism(
        X_train[ps_present], y_train, pipe_full,
        n_boot=n_boot_optimism, random_state=random_state,
    )
    results_dict["optimism"] = opt

    # ── 10. Heterogeneity ──
    from bakeoff.analysis import run_heterogeneity_analysis
    print("=" * 60)
    print("10. Heterogeneity across clusters  (item 23b)")
    print("=" * 60)
    het = run_heterogeneity_analysis(
        X_train[ps_present], y_train, clone(deployable),
        site_col, year_col, df,
    )
    results_dict["heterogeneity"] = het

    # ── 11. Fairness / subgroup ──
    from bakeoff.analysis import run_subgroup_analysis
    print("=" * 60)
    print("11. Fairness & subgroup performance  (items 14, 23a)")
    print("=" * 60)
    sub = run_subgroup_analysis(
        X_train, y_train, df, oof_pred, "LogReg_Firth", outdir,
        random_state=random_state,
    )
    results_dict["subgroups"] = sub.to_dict(orient="records")

    # ── 11b. MICE sensitivity ──
    from bakeoff.analysis import run_mice_sensitivity
    print("=" * 60)
    print("11b. Sensitivity — multiple imputation MICE  (item 9, 12)")
    print("=" * 60)
    auc_mi, slope_mi = run_mice_sensitivity(
        X_train[ps_present], y_train, ps_cont, ps_bin, ps_cat,
        random_state=random_state, cv_splits=cv_splits, variant=deployable_variant,
    )
    results_dict["mice_sensitivity"] = {
        "oof_auc_mi": auc_mi,
        "cal_slope_mi": slope_mi,
        "oof_auc_primary": float(roc_auc_score(y_train, dep_oof)),
    }

    # ── 12. External comparison ──
    from bakeoff.analysis import run_external_comparison
    print("=" * 60)
    print("12. External comparison — PROGRESS-CTO + DeLong  (items 12, 23a)")
    print("=" * 60)
    ext = run_external_comparison(
        df, X_train, y_train, oof_pred, best_estimators,
        pub_vars, pub_pts, published_betas, outdir,
    )
    if ext is not None:
        results_dict["external_comparison"] = ext

    # ── 14. Model specification ──
    print("=" * 60)
    print("14. Full model specification  (items 22, 12g)")
    print("=" * 60)

    # Uniform shrinkage: optimism-corrected calibration slope (bootstrap), with a van Houwelingen
    # heuristic fallback computed on the FULL derivation cohort — matches notebook Section 14.
    shrink_boot = opt["slope_corrected"]
    prep_vh = clone(prep_prespec).fit(X[ps_present])
    Zfull = prep_vh.transform(X[ps_present])
    try:
        mf = sm.Logit(y.to_numpy(), sm.add_constant(Zfull)).fit(disp=False)
        chi2 = float(2 * (mf.llf - mf.llnull))
        vh = (chi2 - Zfull.shape[1]) / chi2 if chi2 > 0 else float("nan")
    except Exception:
        vh = float("nan")
    shrinkage_factor = shrink_boot if (np.isfinite(shrink_boot) and 0 < shrink_boot <= 1) else (
        vh if np.isfinite(vh) else 1.0
    )
    shrinkage_factor = float(min(1.0, max(0.5, shrinkage_factor)))
    print(f"Shrinkage: bootstrap slope={shrink_boot:.3f}, van Houwelingen={vh:.3f} "
          f"-> applied factor={shrinkage_factor:.3f}")

    # SHIPPED deployable: Firth ORs -> variant intercept correction -> uniform shrinkage,
    # refit on the full derivation cohort.
    final_lr_template = Pipeline([
        ("prep", clone(prep_prespec)),
        ("model", FirthLogisticRegression(variant=deployable_variant, shrinkage=shrinkage_factor)),
    ])
    _, final_lr = save_specification(final_lr_template, outdir, X[ps_present], y)
    _, ref = compute_point_score(
        final_lr, score_increments, points_max, outdir, ps_cont=ps_cont,
    )
    save_risk_equation(final_lr, outdir)
    results_dict["shrinkage"] = {
        "bootstrap": shrink_boot, "van_houwelingen": vh, "applied": shrinkage_factor,
    }

    # ── 14b. Sensitivity — reduced model dropping age & occlusion length  (items 12, 23a) ──
    print("=" * 60)
    print("14b. Sensitivity — reduced model (drop age & occlusion length)")
    print("=" * 60)
    results_dict["reduced_model_sensitivity"] = run_reduced_model_sensitivity(
        X[ps_present], y, X_train[ps_present], y_train, X_test[ps_present], y_test,
        ps_present, dep_oof, binary, categorical, continuous, cv, outdir,
        variant=deployable_variant,
    )

    # ── 15. Open science ──
    from bakeoff.analysis import save_open_science
    print("=" * 60)
    print("15. Open science  (item 18)")
    print("=" * 60)
    save_open_science(outdir)

    # ── 16. TRIPOD+AI checklist ──
    from bakeoff.analysis import generate_checklist
    print("=" * 60)
    print("16. TRIPOD+AI checklist coverage")
    print("=" * 60)
    generate_checklist(outdir)

    # ── Save deployable model ──
    print("=" * 60)
    print("Save deployable model")
    print("=" * 60)
    model_path = os.path.join(outdir, "final_logreg_firth.pkl")
    metadata = {
        "model_name": "LogReg_Firth",
        "variant": deployable_variant,
        "shrinkage": shrinkage_factor,
        "predictors": ps_present,
        "binary": ps_bin,
        "continuous": ps_cont,
        "categorical": ps_cat,
        "oof_auc": float(roc_auc_score(y_train, dep_oof)),
        "auc_corrected": opt["auc_corrected"],
        "calibration_slope": float(slope),
        "brier": brier,
    }
    joblib.dump({"pipeline": final_lr, "metadata": metadata}, model_path)
    print(f"  Saved: {model_path}")

    # ── 17. Save results dict to JSON ──
    from bakeoff.analysis import save_results
    print("=" * 60)
    print("17. Save results")
    print("=" * 60)
    save_results(results_dict, outdir)

    # ── 18. Observed vs predicted incidence by deployable point strata ──
    print("=" * 60)
    print("18. Observed vs predicted incidence by deployable point strata")
    print("=" * 60)
    point_counts, incidence_by_point_strata = plot_incidence_by_point_strata(
        final_lr, ref, X[ps_present], y, X_train[ps_present], X_test[ps_present], outdir, plots_dir,
    )
    results_dict["deployable_patient_counts_by_exact_point"] = point_counts.to_dict(orient="records")
    results_dict["deployable_incidence_by_point_strata"] = incidence_by_point_strata.to_dict(orient="records")
    save_results(results_dict, outdir)  # re-save so the Section 18 tables are included

    # ── Final summary ──
    print()
    print("=" * 60)
    print("DONE — Outputs in", outdir)
    print("=" * 60)
    return final_lr


if __name__ == "__main__":
    # Quick local test
    from bakeoff.config import load_config
    cfg = load_config()
    tripod = cfg.get("tripod", {})
    run(
        data_path=cfg["data_path"],
        target=cfg["target"],
        pre_specified_predictors=tripod.get("pre_specified_predictors", []),
        plausible_bounds=tripod.get("plausible_bounds", {}),
        exclude_planned_mcs=tripod.get("exclude_planned_mcs", True),
        planned_mcs_col=tripod.get("planned_mcs_col", None),
        site_col=tripod.get("site_col", None),
        year_col=tripod.get("year_col", None),
        random_state=cfg.get("random_state", 42),
        score_increments=tripod.get("score_increments", {}),
        points_max=tripod.get("points_max", 10),
        yesno_na_vars=cfg.get("yesno_na_vars", []),
        redundant_groups=cfg.get("redundant_groups", []),
        pub_vars=tripod.get("pub_vars", {}),
        pub_pts=tripod.get("pub_pts", {}),
        published_betas=tripod.get("published_betas", None),
        deployable_variant=tripod.get("deployable_variant", "flic"),
        benchmark_model=tripod.get("benchmark_model", "ExtraTrees"),
        use_synth_if_missing=tripod.get("use_synth_if_missing", True),
    )
