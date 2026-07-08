import os
import json
import platform
import numpy as np
import pandas as pd
from scipy import stats
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    GroupKFold,
)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import clone

import statsmodels.api as sm

from bakeoff.firth import FirthLogisticRegression


# ----------------
# Helper metrics
# ----------------

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


def _cal_slope_only(yv, p):
    c = np.clip(p, 1e-6, 1 - 1e-6)
    lg = np.log(c / (1 - c))
    try:
        return float(sm.Logit(np.asarray(yv), sm.add_constant(lg)).fit(disp=False).params[1])
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# 1b. Table 1 — Participant characteristics by outcome  (item 20)
# ---------------------------------------------------------------------------

def generate_table1(X, y, binary, continuous, output_dir):
    g0, g1 = (y == 0), (y == 1)
    out = []
    for c in continuous:
        s = pd.to_numeric(X[c], errors="coerce")
        _f = lambda mask: (
            f"{s[mask].median():.1f} [{s[mask].quantile(.25):.1f}"
            f"-{s[mask].quantile(.75):.1f}]"
        )
        out.append({
            "variable": c,
            "type": "median [IQR]",
            "no_event": _f(g0.values),
            "event": _f(g1.values),
            "missing_%": round(s.isna().mean() * 100, 1),
        })
    for c in binary:
        s = pd.to_numeric(X[c], errors="coerce")
        _f = lambda mask: (
            f"{int(s[mask].sum())} ({s[mask].mean() * 100:.1f}%)"
        )
        out.append({
            "variable": c,
            "type": "n (%)",
            "no_event": _f(g0.values),
            "event": _f(g1.values),
            "missing_%": round(s.isna().mean() * 100, 1),
        })
    t1 = pd.DataFrame(out)
    t1.to_csv(os.path.join(output_dir, "table1.csv"), index=False)
    print("\nTABLE 1 — Participant characteristics by outcome (item 20):")
    print(t1.to_string(index=False))
    return t1


# ---------------------------------------------------------------------------
# 3. Missing data table
# ---------------------------------------------------------------------------

def generate_missingness_table(X, output_dir):
    miss = (X.isna().mean() * 100).round(2).sort_values(ascending=False)
    miss = miss[miss > 0]
    miss.to_csv(os.path.join(output_dir, "missingness.csv"), header=["pct_missing"])
    print("MISSINGNESS (%):")
    print(miss.to_string() if len(miss) else "  none")
    print("Imputation (fit in-fold): continuous=median, binary/categorical=most_frequent.\n")
    return miss


# ---------------------------------------------------------------------------
# 10. Heterogeneity across clusters — site & temporal  (item 23b)
# ---------------------------------------------------------------------------

def run_heterogeneity_analysis(
    X, y, best_model, site_col, year_col, df_original
):
    site_rows, temp_rows = [], []

    # Site-clustered cross-validation
    g = (
        df_original.loc[X.index, site_col].fillna("NA").astype(str).to_numpy()
        if site_col and site_col in df_original.columns
        else None
    )
    yr = (
        pd.to_numeric(df_original.loc[X.index, year_col], errors="coerce")
        if year_col and year_col in df_original.columns
        else None
    )

    if g is not None:
        ns = min(10, int(pd.Series(g).nunique()))
        a = []
        if ns >= 2:
            for tr, te in GroupKFold(n_splits=ns).split(X, y, g):
                if y.iloc[te].nunique() < 2 or y.iloc[tr].nunique() < 2:
                    continue
                m = clone(best_model).fit(X.iloc[tr], y.iloc[tr])
                p = m.predict_proba(X.iloc[te])[:, 1]
                a.append(roc_auc_score(y.iloc[te], p))
        site_rows.append({
            "model": "LogReg_Firth",
            "auc_mean": float(np.mean(a)) if a else np.nan,
            "auc_sd": float(np.std(a)) if a else np.nan,
            "n_folds": len(a),
            "n_sites": int(pd.Series(g).nunique()),
        })

    # Temporal split (70% quantile cutoff)
    if yr is not None:
        v = yr.notna()
        cut = float(yr[v].quantile(0.70))
        trm = v & (yr <= cut)
        tem = v & (yr > cut)
        if y.loc[trm].nunique() == 2 and y.loc[tem].nunique() == 2:
            m = clone(best_model).fit(X.loc[trm], y.loc[trm])
            p = m.predict_proba(X.loc[tem])[:, 1]
            temp_rows.append({
                "model": "LogReg_Firth",
                "cutoff": cut,
                "n_test": int(tem.sum()),
                "events_test": int(y.loc[tem].sum()),
                "auc": float(roc_auc_score(y.loc[tem], p)),
                "cal_slope": float(_cal_slope_only(y.loc[tem], p)),
            })

    site = pd.DataFrame(site_rows)
    temp = pd.DataFrame(temp_rows)
    results = {
        "site_clustered": site.to_dict(orient="records"),
        "temporal": temp.to_dict(orient="records"),
    }

    print("\nSITE-CLUSTERED (leave-centres-out, item 23b):")
    print(site.round(3).to_string(index=False) if len(site) else "  (no site data)")
    print("\nTEMPORAL (train <= cutoff, test > cutoff):")
    print(temp.round(3).to_string(index=False) if len(temp) else "  (no year data)")
    print("Report site & temporal AUCs as generalization estimates.\n")
    return results


# ---------------------------------------------------------------------------
# 11. Fairness & subgroup performance  (items 14, 23a)
# ---------------------------------------------------------------------------

SUBGROUP_VARS = [
    "gender", "race", "ethnicity", "diabetes_mellitus", "acs",
    "prior_cabg", "prior_mi", "current_dialysis", "chronic_lung_disease",
    "hypertension", "peripheral_arterial_diseas", "smoking",
]

MIN_SUB_EVENTS = 10


def _bin01(s):
    sn = pd.to_numeric(s, errors="coerce")
    u = set(np.round(sn.dropna().unique()).astype(int))
    if u and u <= {1, 2, 3}:
        return sn.map({1: 1.0, 2: 0.0, 3: np.nan})
    return sn


def run_subgroup_analysis(
    X, y, df_original, oof_pred, recommended_model, output_dir, random_state=42
):
    dfx = df_original.loc[X.index]
    defs = []
    if "age_manual_input" in dfx.columns:
        a = pd.to_numeric(dfx["age_manual_input"], errors="coerce")
        defs.append(("age", "<70", (a < 70).values))
        defs.append(("age", ">=70", (a >= 70).values))
    if "left_ventr_ejection_fract" in dfx.columns:
        e = pd.to_numeric(dfx["left_ventr_ejection_fract"], errors="coerce")
        defs.append(("LVEF", "<40", (e < 40).values))
        defs.append(("LVEF", ">=40", (e >= 40).values))
    for col in SUBGROUP_VARS:
        if col in dfx.columns:
            b = _bin01(dfx[col])
            for lv in sorted(b.dropna().unique()):
                label = str(int(lv)) if float(lv).is_integer() else str(lv)
                defs.append((col, label, (b == lv).values))

    rows = []
    for sg, lv, mask in defs:
        m = np.asarray(mask)
        yy = y[m]
        pp = oof_pred[recommended_model][m]
        if yy.nunique() < 2 or yy.sum() < MIN_SUB_EVENTS or (1 - yy).sum() < MIN_SUB_EVENTS:
            rows.append({
                "model": recommended_model,
                "subgroup": sg,
                "level": lv,
                "n": int(m.sum()),
                "events": int(yy.sum()),
                "auc": np.nan,
                "lo": np.nan,
                "hi": np.nan,
            })
        else:
            lo, hi = _boot_ci(yy.values, pp, seed=random_state)
            rows.append({
                "model": recommended_model,
                "subgroup": sg,
                "level": lv,
                "n": int(m.sum()),
                "events": int(yy.sum()),
                "auc": round(float(roc_auc_score(yy, pp)), 3),
                "lo": round(float(lo), 3),
                "hi": round(float(hi), 3),
            })
    sub = pd.DataFrame(rows)
    sub.to_csv(os.path.join(output_dir, "subgroup_performance.csv"), index=False)
    ov = float(roc_auc_score(y, oof_pred[recommended_model]))
    print(f"\nFAIRNESS / SUBGROUP — deployable model audited (overall OOF AUC {ov:.3f}):")
    print(sub.round(3).to_string(index=False))
    print(
        f"\n  AUC shown where stratum has >={MIN_SUB_EVENTS} events. "
        "Flag any stratum whose CI excludes the overall AUC."
    )
    return sub


# ---------------------------------------------------------------------------
# 11b. MICE sensitivity  (item 9, 12)
# ---------------------------------------------------------------------------

def run_mice_sensitivity(
    X, y, ps_cont, ps_bin, ps_cat, random_state=42, cv_splits=5, variant="firth"
):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    transformers = []
    if ps_cont:
        transformers.append((
            "cont",
            Pipeline([
                ("imp", IterativeImputer(
                    random_state=random_state, max_iter=10, sample_posterior=True
                )),
                ("sc", StandardScaler()),
            ]),
            ps_cont,
        ))
    if ps_bin:
        transformers.append((
            "bin", SimpleImputer(strategy="most_frequent"), ps_bin
        ))
    if ps_cat:
        transformers.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            ps_cat,
        ))
    prep_mi = ColumnTransformer(transformers, remainder="drop")
    oof = cross_val_predict(
        Pipeline([("prep", prep_mi), ("model", FirthLogisticRegression(variant=variant))]),
        X, y, cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]
    auc_mi = float(roc_auc_score(y, oof))
    slope_mi = float(_cal_slope_only(y, oof))
    print(f"\nMULTIPLE-IMPUTATION (MICE) SENSITIVITY:")
    print(f"  MICE OOF AUC {auc_mi:.3f}  |  cal slope {slope_mi:.3f}")
    return auc_mi, slope_mi


# ---------------------------------------------------------------------------
# 12. External comparison — PROGRESS-CTO score + DeLong  (items 12, 23a)
# ---------------------------------------------------------------------------

def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds, m):
    n = preds.shape[1] - m
    pos = preds[:, :m]
    neg = preds[:, m:]
    k = preds.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    cov = np.cov((tz[:, :m] - tx) / n) / m + np.cov(1.0 - (tz[:, m:] - ty) / m) / n
    return aucs, cov


def delong_test(y_true, p1, p2):
    """Pairwise DeLong test. Returns (auc1, auc2, p_value)."""
    y_true = np.asarray(y_true, float)
    m = int(y_true.sum())
    if m < 2 or (len(y_true) - m) < 2:
        return np.nan, np.nan, np.nan
    order = (-y_true).argsort(kind="mergesort")
    preds = np.vstack((np.asarray(p1, float), np.asarray(p2, float)))[:, order]
    aucs, cov = _fast_delong(preds, m)
    l = np.array([[1.0, -1.0]])
    var = float((l @ cov @ l.T).item())
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(2 * (1 - stats.norm.cdf(abs(z))))


def run_external_comparison(
    df, X, y, oof_pred, best_estimators,
    pub_vars, pub_pts, published_betas, output_dir,
):
    req = [pub_vars["retro"], pub_vars["lvef"], pub_vars["length"]]
    missing = [c for c in req if c not in df.columns]
    if missing:
        print("Published-score comparison skipped; missing columns:", missing)
        return None

    retro = pd.to_numeric(df.loc[X.index, pub_vars["retro"]], errors="coerce").fillna(0)
    lvef = pd.to_numeric(df.loc[X.index, pub_vars["lvef"]], errors="coerce")
    lvef = lvef.fillna(lvef.median())
    length = pd.to_numeric(df.loc[X.index, pub_vars["length"]], errors="coerce")
    length = length.fillna(length.median())

    if published_betas is not None:
        lp = (
            published_betas.get("intercept", 0)
            + published_betas.get("retro", 0) * retro
            + published_betas.get("lvef_per10_low", 0) * ((50 - lvef) / 10)
            + published_betas.get("len_per10", 0) * (length / 10)
        )
        pub_score = lp.to_numpy()
        note = "explicit PUBLISHED_BETAS"
    else:
        pts = (
            pub_pts["retro_yes"] * (retro > 0).astype(float)
            + np.clip((pub_pts["lvef_ref"] - lvef) * pub_pts["lvef_per_pct_below_ref"], 0, None)
            + np.clip(length, 0, None) * pub_pts["length_per_mm"]
        )
        pub_score = pts.to_numpy()
        note = "nomogram-reconstructed point total"

    pub_auc = float(roc_auc_score(y, pub_score))
    print(f"\nEXTERNAL COMPARISON — published PROGRESS-CTO score ({note}):")
    print(f"  Published score discrimination in this cohort: AUC = {pub_auc:.3f}")

    comp_rows = []
    for name in best_estimators:
        a_m, a_p, p_val = delong_test(y, oof_pred[name], pub_score)
        comp_rows.append({
            "comparison": f"{name} vs published",
            "auc_model": round(a_m, 3),
            "auc_published": round(a_p, 3),
            "delong_p": round(p_val, 4),
        })
    names = list(best_estimators.keys())
    if len(names) == 2:
        a1, a2, p_val = delong_test(y, oof_pred[names[0]], oof_pred[names[1]])
        comp_rows.append({
            "comparison": f"{names[0]} vs {names[1]}",
            "auc_model": round(a1, 3),
            "auc_published": round(a2, 3),
            "delong_p": round(p_val, 4),
        })

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(os.path.join(output_dir, "delong_comparison.csv"), index=False)
    print(comp.to_string(index=False))
    print("\n  Published score applied as-is (not refit); DeLong compares ranking.\n")
    return {
        "published_auc": pub_auc,
        "note": note,
        "pairwise": comp.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# 15. Open science  (item 18, partial)
# ---------------------------------------------------------------------------

def save_open_science(output_dir):
    env = {
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with open(os.path.join(output_dir, "environment.json"), "w") as f:
        json.dump(env, f, indent=2)
    print("\nOPEN SCIENCE — environment saved to environment.json")


# ---------------------------------------------------------------------------
# 16. TRIPOD+AI checklist  (all 27 items)
# ---------------------------------------------------------------------------

def generate_checklist(output_dir):
    checklist = [
        ("1 Title", "MANUSCRIPT", "development+validation prediction-model study"),
        ("2 Abstract", "MANUSCRIPT", "TRIPOD+AI for Abstracts"),
        ("3 Background", "MANUSCRIPT", "prior urgent-MCS score; BCIS-3 context"),
        ("4 Objectives", "MANUSCRIPT", "development & validation objective"),
        ("5 Data source/eligibility", "CODE", "Section 1: derivation cohort (prophylactic-MCS excluded); flow"),
        ("6 Outcome", "CODE", "Section 1"),
        ("7 Predictors", "CODE", "Section 2 typing + pre-procedural audit"),
        ("8 Sample size", "CODE", "Section 4 EPV"),
        ("9 Missing data", "CODE", "Section 3"),
        ("10 Data preparation", "CODE", "Sections 2/4"),
        ("11 Model type / candidates", "CODE", "Sec 0/5: pre-specified Firth/FLIC (deployable) + ExtraTrees benchmark; EPV rationale"),
        ("12 Analytical methods", "CODE", "Sections 5-9,12 incl. external comparison"),
        ("13 Class imbalance", "CODE", "Firth Jeffreys penalty + FLIC/FLAC calibration; no resampling"),
        ("14 Fairness", "CODE", "Section 11"),
        ("15 Model output", "CODE", "Section 5: probability"),
        ("16 Risk groups", "PARTIAL", "Section 14 point score -> define cut-points in manuscript"),
        ("17 Evaluation/validation design", "CODE", "Sections 6,9,10"),
        ("18 Open science", "PARTIAL", "Section 15"),
        ("19 Patient & public involvement", "MANUSCRIPT", "describe PPI or state none"),
        ("20 Participants/Table 1", "CODE", "Section 1b"),
        ("21 Participants & events per analysis", "CODE", "Sections 1,4,6"),
        ("22 Model specification", "CODE", "Sec 14: deployable Firth/FLIC — equation+OR+points+shrinkage"),
        ("23a Performance + subgroups + comparison", "CODE", "Sec 6-12 on deployable; ExtraTrees/published as benchmarks (DeLong, OOF + test)"),
        ("23b Heterogeneity across clusters", "CODE", "Section 10 site & temporal"),
        ("24 Model updating", "PARTIAL", "Sec 7 recalibration + Sec 14 shrinkage factor"),
        ("25 Usability/implementation", "MANUSCRIPT", "how clinicians apply the score"),
        ("26 Interpretation/limitations", "MANUSCRIPT", "EPV, prophylactic bias, temporal drift"),
        ("27 Supplementary/future work", "MANUSCRIPT", "LVEF-alone vs multivariable question"),
    ]
    ck = pd.DataFrame(checklist, columns=["item", "status", "where"])
    ck.to_csv(os.path.join(output_dir, "tripod_ai_checklist.csv"), index=False)
    print("\nTRIPOD+AI CHECKLIST (all 27 items):")
    print(ck.to_string(index=False))
    print("\nCounts:", ck["status"].value_counts().to_dict())
    return ck


# ---------------------------------------------------------------------------
# 17. Save results to JSON
# ---------------------------------------------------------------------------

def _js(v):
    if isinstance(v, dict):
        return {str(k): _js(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_js(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def save_results(results_dict, output_dir):
    json.dump(
        _js(results_dict),
        open(os.path.join(output_dir, "results.json"), "w"),
        indent=2,
    )
    print(f"\nResults saved to {output_dir}/results.json")
