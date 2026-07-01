import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.base import clone
import seaborn as sns

from bakeoff.firth import FirthLogisticRegression

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


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
    prev = yv.mean()
    m = []
    for pt in th:
        tp = ((p >= pt) & (yv == 1)).sum()
        fp = ((p >= pt) & (yv == 0)).sum()
        w = pt / (1 - pt)
        m.append(tp / n - fp / n * w)
    return np.array(m)


def load_and_prepare(df_path, target, predictors, plausible_bounds=None, exclude_planned_mcs=False, planned_mcs_col=None):
    df = pd.read_csv(df_path, encoding="latin1")
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed:
        print("Dropping index-like columns:", unnamed)
    df = df.drop(columns=unnamed)
    df = df.loc[~df[target].isna()].copy().reset_index(drop=True)
    if plausible_bounds:
        print("PHYSIOLOGIC CLEANING (implausible -> NaN):")
        for col, (lo, hi) in plausible_bounds.items():
            if col not in df.columns:
                continue
            v = pd.to_numeric(df[col], errors="coerce")
            bad = (v < lo) | (v > hi)
            n = int(bad.sum())
            if n:
                examples = sorted(set(np.round(v[bad].dropna(), 3).tolist()))[:6]
                print(f"  {col:30s} {n:4d} outside [{lo},{hi}] -> NaN  e.g. {examples}")
            df.loc[bad, col] = np.nan
        print("  done.\n")
    y = pd.to_numeric(df[target], errors="coerce")
    y = pd.Series(np.where(y > 0, 1, 0), index=df.index).astype(int)
    if exclude_planned_mcs and planned_mcs_col and planned_mcs_col in df.columns:
        planned = pd.to_numeric(df[planned_mcs_col], errors="coerce").fillna(0) > 0
        print(f"Prophylactic/planned MCS flagged: {int(planned.sum())} -> excluded from derivation\n")
    else:
        planned = pd.Series(False, index=df.index)
    if planned.any():
        idx = ~planned
        X = df.loc[idx, predictors].copy()
        y = y[idx].reset_index(drop=True)
        X = X.reset_index(drop=True)
    else:
        X = df[predictors].copy()
    missing_predictors = [c for c in predictors if c not in X.columns]
    if missing_predictors:
        print(f"WARNING: missing predictors: {missing_predictors}")
    X = X[[c for c in predictors if c in X.columns]]
    print(f"Cohort: n={len(y)}, events={int(y.sum())} ({y.mean():.3%}), predictors={X.shape[1]}")
    return X, y


def build_preprocessor(predictors, continuous_cols):
    binary = [c for c in predictors if c not in continuous_cols]
    continuous = [c for c in predictors if c in continuous_cols]
    return ColumnTransformer([
        ("cont", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), continuous),
        ("bin", SimpleImputer(strategy="most_frequent"), binary),
    ], remainder="drop"), binary, continuous


def train_firth(X_train, y_train, predictors, continuous_cols):
    prep, binary, continuous = build_preprocessor(predictors, continuous_cols)
    pipe = Pipeline([
        ("prep", prep),
        ("model", FirthLogisticRegression(max_iter=200, tol=1e-8)),
    ])
    pipe.fit(X_train, y_train)
    return pipe, prep, binary, continuous


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
    optimism = {
        "auc_apparent": float(app_auc),
        "auc_corrected": float(app_auc - np.mean(oa)),
        "slope_apparent": float(app_sl),
        "slope_corrected": float(app_sl - np.nanmean(os_)),
    }
    return optimism


def save_specification(pipe, outdir, X, y):
    final_lr = clone(pipe).fit(X, y)
    prep_f = final_lr.named_steps["prep"]
    fl = final_lr.named_steps["model"]
    feat = list(prep_f.get_feature_names_out())
    terms = ["intercept"] + feat
    beta = fl.beta_
    ci = fl.ci_
    spec = pd.DataFrame({
        "term": terms,
        "beta": beta,
        "odds_ratio": np.exp(beta),
        "or_lo": np.exp(ci[:, 0]),
        "or_hi": np.exp(ci[:, 1]),
        "p_value": fl.pvals_,
    })
    spec.to_csv(os.path.join(outdir, "logreg_firth_specification.csv"), index=False)
    print("DEPLOYABLE FIRTH LR SPECIFICATION (item 22):")
    print(spec.round(3).to_string(index=False))
    return spec, prep_f, fl


def compute_point_score(pipe, prep_f, fl, score_increments, points_max, outdir):
    feat = list(prep_f.get_feature_names_out())
    nbeta = fl.beta_[1:]
    sd_map = {}
    if hasattr(prep_f.named_transformers_.get("cont", None), "named_steps"):
        scl = prep_f.named_transformers_["cont"].named_steps.get("sc", None)
        if scl is not None and hasattr(scl, "scale_"):
            for nm, sd in zip(continuous_cols_global, scl.scale_):
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
    print(f"\nPOINT SCORE (1 point = {ref:.3f} log-odds, strongest = {points_max}):")
    print(ptab.round(3).to_string(index=False))
    return ptab


def save_risk_equation(pipe, prep_f, fl, outdir):
    feat = list(prep_f.get_feature_names_out())
    nbeta = fl.beta_[1:]
    eq = "logit(p) = %.4f" % fl.beta_[0]
    eq += "".join(f" + ({b:.4f})*[{n}]" for b, n in zip(nbeta, feat))
    with open(os.path.join(outdir, "logreg_firth_risk_equation.txt"), "w") as f:
        f.write(eq + "\n\np = 1/(1+exp(-logit(p)))\n")
    print(f"\nRisk equation saved to {outdir}/logreg_firth_risk_equation.txt")


def plot_calibration(y, oof_pred, outdir, cal_metrics_tuple):
    intercept, slope, brier = cal_metrics_tuple
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", lw=1, label="ideal")
    fp, mp = calibration_curve(y, oof_pred, n_bins=10, strategy="quantile")
    ax.plot(mp, fp, "o-", label=f"LogReg_Firth (slope {slope:.2f})")
    mx = max(float(np.percentile(oof_pred, 99)), 0.05)
    ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("predicted"); ax.set_ylabel("observed")
    ax.set_title("OOF calibration — Firth LR"); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "calibration_curve.png"), dpi=200)
    plt.close(fig)


def plot_roc_curve(y, oof_pred, outdir):
    fpr, tpr, _ = roc_curve(y, oof_pred)
    auc_val = roc_auc_score(y, oof_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"LogReg_Firth OOF ({auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
    ax.set_title("OOF ROC — Firth LR"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "roc_curve.png"), dpi=200)
    plt.close(fig)


def plot_dca(y, oof_pred, outdir):
    prev = float(y.mean())
    th = np.linspace(max(1e-4, prev / 10), min(0.20, max(0.05, prev * 10)), 150)
    nb_m = _net_benefit(y, oof_pred, th)
    treat_all = prev - (1 - prev) * th / (1 - th)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(th * 100, nb_m, label="LogReg_Firth")
    ax.plot(th * 100, treat_all, ls="--", c="grey", label="treat all")
    ax.axhline(0, lw=1, c="k", label="treat none")
    ax.set_xlabel("threshold probability (%)"); ax.set_ylabel("net benefit")
    ax.set_title("Decision-curve analysis (OOF) — Firth LR"); ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "dca_curve.png"), dpi=200)
    plt.close(fig)


# Global for point score helper
continuous_cols_global = []


def run(
    data_path,
    target,
    predictors,
    plausible_bounds=None,
    exclude_planned_mcs=False,
    planned_mcs_col=None,
    outdir="tripod_outputs",
    random_state=42,
    n_boot_optimism=500,
    score_increments=None,
    points_max=10,
):
    global continuous_cols_global

    os.makedirs(outdir, exist_ok=True)

    print("#" * 60)
    print("# Load & prepare data")
    print("#" * 60)
    X, y = load_and_prepare(
        data_path, target, predictors,
        plausible_bounds=plausible_bounds,
        exclude_planned_mcs=exclude_planned_mcs,
        planned_mcs_col=planned_mcs_col,
    )

    continuous_cols = list(plausible_bounds.keys()) if plausible_bounds else []
    continuous_cols_global = [c for c in continuous_cols if c in predictors]
    binary_cols = [c for c in predictors if c not in continuous_cols_global]

    print("\n" + "#" * 60)
    print("# Train Firth logistic regression")
    print("#" * 60)
    pipe, prep, bin_list, cont_list = train_firth(X, y, predictors, continuous_cols_global)
    oof = cross_val_predict(
        clone(pipe), X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
        method="predict_proba", n_jobs=-1,
    )[:, 1]
    oof_auc = roc_auc_score(y, oof)
    lo, hi = _boot_ci(y.values, oof, n=2000, seed=random_state)
    intercept, slope, brier = _cal_metrics(y, oof)
    print(f"OOF AUC: {oof_auc:.3f} ({lo:.3f} - {hi:.3f})")
    print(f"Calibration slope: {slope:.3f}  |  Brier: {brier:.4f}\n")

    print("#" * 60)
    print("# Bootstrap optimism correction")
    print("#" * 60)
    optimism = bootstrap_optimism(X, y, pipe, n_boot=n_boot_optimism, random_state=random_state)
    print(f"AUC: apparent={optimism['auc_apparent']:.3f}, corrected={optimism['auc_corrected']:.3f}")
    print(f"Calibration slope: apparent={optimism['slope_apparent']:.3f}, corrected={optimism['slope_corrected']:.3f}\n")

    print("#" * 60)
    print("# Plots")
    print("#" * 60)
    plot_calibration(y, oof, outdir, (intercept, slope, brier))
    plot_roc_curve(y, oof, outdir)
    plot_dca(y, oof, outdir)
    print(f"Plots saved to {outdir}/")

    print("\n" + "#" * 60)
    print("# Model specification & point score")
    print("#" * 60)
    save_specification(pipe, outdir, X, y)
    compute_point_score(
        pipe, prep, pipe.named_steps["model"],
        score_increments or {}, points_max, outdir,
    )
    save_risk_equation(pipe, prep, pipe.named_steps["model"], outdir)

    print("\n" + "#" * 60)
    print("# Save model")
    print("#" * 60)
    final_lr = clone(pipe).fit(X, y)
    model_path = os.path.join(outdir, "final_logreg_firth.pkl")
    metadata = {
        "model_name": "LogReg_Firth",
        "predictors": predictors,
        "binary": binary_cols,
        "continuous": continuous_cols_global,
        "oof_auc": float(oof_auc),
        "auc_corrected": optimism["auc_corrected"],
        "calibration_slope": float(slope),
        "brier": brier,
    }
    joblib.dump({"pipeline": final_lr, "metadata": metadata}, model_path)
    print(f"Saved: {model_path}\n")

    print("=" * 60)
    print(f"DEPLOYABLE MODEL: LogReg_Firth")
    print(f"  OOF AUC:          {oof_auc:.3f} ({lo:.3f} - {hi:.3f})")
    print(f"  AUC (optimism):   {optimism['auc_corrected']:.3f}")
    print(f"  Calibration slope: {slope:.3f} (corrected: {optimism['slope_corrected']:.3f})")
    print(f"  Brier:            {brier:.4f}")
    print(f"  Predictors:       {len(predictors)} ({len(binary_cols)} binary, {len(continuous_cols_global)} continuous)")
    print("=" * 60)
    return pipe
