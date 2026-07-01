import numpy as np
import pandas as pd
from functools import partial
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, average_precision_score


FIXED_KS = [15, 25, 50]


def bootstrap_ci(yt, ys, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    yt = np.asarray(yt)
    ys = np.asarray(ys)
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


def tune_all_models(
    X_train,
    y_train,
    X_test,
    y_test,
    prep,
    cv,
    k_grid,
    zoo,
    n_boot_ci=2000,
    random_state=42,
):
    mi = partial(mutual_info_classif, random_state=random_state)
    results = []
    oof_scores = {}
    test_scores = {}
    k_curves = {}
    cv_results_by_model = {}
    best_estimators = {}

    for name, (est, grid) in zoo.items():
        pipe = Pipeline([
            ("prep", prep),
            ("selector", SelectKBest(score_func=mi)),
            ("model", est),
        ])
        gs = GridSearchCV(
            pipe,
            {"selector__k": k_grid, **grid},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        gs.fit(X_train, y_train)

        cr = pd.DataFrame(gs.cv_results_)
        cv_results_by_model[name] = cr
        k_curves[name] = (
            cr.assign(k=cr["param_selector__k"].astype(str))
            .groupby("k")["mean_test_score"]
            .max()
        )
        best_est = gs.best_estimator_
        best_estimators[name] = best_est
        oof = cross_val_predict(
            best_est, X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        s_te = best_est.predict_proba(X_test)[:, 1]
        oof_scores[name] = oof
        test_scores[name] = s_te
        lo, hi = bootstrap_ci(y_test.values, s_te, n=n_boot_ci, seed=random_state)
        best_params = {
            k.replace("model__", ""): v
            for k, v in gs.best_params_.items()
            if k != "selector__k"
        }
        entry = {
            "model": name,
            "best_k": gs.best_params_["selector__k"],
            "cv_auc": gs.best_score_,
            "cv_sd": float(cr.loc[gs.best_index_, "std_test_score"]),
            "oof_train_auc": roc_auc_score(y_train, oof),
            "test_auc": roc_auc_score(y_test, s_te),
            "test_lo": lo,
            "test_hi": hi,
            "test_pr_auc": average_precision_score(y_test, s_te),
            "best_params": best_params,
        }
        results.append(entry)
        print(
            f"{name:13s} bestK={str(gs.best_params_['selector__k']):>4} "
            f"CV={gs.best_score_:.3f}  test={roc_auc_score(y_test, s_te):.3f}  {best_params}"
        )
    res = pd.DataFrame(results).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    return res, oof_scores, test_scores, k_curves, cv_results_by_model, best_estimators


def evaluate_fixed_ks(
    X_train,
    y_train,
    X_test,
    y_test,
    prep,
    cv,
    zoo,
    cv_results_by_model,
    k_grid,
    fixed_ks=None,
    n_boot_ci=2000,
    random_state=42,
):
    if fixed_ks is None:
        fixed_ks = FIXED_KS
    mi = partial(mutual_info_classif, random_state=random_state)
    fixed_rows = []
    fixed_test = {}
    for name, cr in cv_results_by_model.items():
        base = zoo[name][0]
        for K in fixed_ks:
            sub = cr[cr["param_selector__k"].apply(lambda v: v == K)]
            if sub.empty:
                continue
            best = sub.loc[sub["mean_test_score"].idxmax()]
            pipe = Pipeline([
                ("prep", prep),
                ("selector", SelectKBest(score_func=mi)),
                ("model", clone(base)),
            ])
            pipe.set_params(**best["params"]).fit(X_train, y_train)
            s = pipe.predict_proba(X_test)[:, 1]
            fixed_test[(name, K)] = s
            lo, hi = bootstrap_ci(y_test.values, s, n=n_boot_ci, seed=random_state)
            fixed_rows.append({
                "model": name,
                "K": K,
                "cv_auc": float(best["mean_test_score"]),
                "test_auc": roc_auc_score(y_test, s),
                "test_lo": lo,
                "test_hi": hi,
                "test_pr_auc": average_precision_score(y_test, s),
            })
    fixed_df = pd.DataFrame(fixed_rows)
    order = (
        fixed_df.groupby("model")["cv_auc"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    wide = (
        fixed_df.pivot(index="model", columns="K", values=["cv_auc", "test_auc"])
        .reindex(order)
    )
    wide.columns = [f"{m}_K{k}" for m, k in wide.columns]
    print("Fixed-K models \u2014 CV (train) and test ROC-AUC at K = 15 / 25 / 50:\n")
    print(wide.round(3).to_string())
    return fixed_df, fixed_test, order
