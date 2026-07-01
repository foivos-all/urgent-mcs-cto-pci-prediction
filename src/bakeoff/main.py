import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import joblib

import numpy as np

from bakeoff.config import load_config, YESNO_NA_VARS, REDUNDANT_GROUPS, FORCE_TYPES
from bakeoff.data import (
    load_data,
    classify_variables,
    reduce_redundancy,
    train_val_split,
)
from bakeoff.preprocessing import build_preprocessor, build_cv, get_k_grid
from bakeoff.models import model_zoo
from bakeoff.training import tune_all_models, evaluate_fixed_ks
from bakeoff.plotting import generate_all_plots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Per-model tuned bake-off — CV-safe hyperparameter & K selection"
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: config.yaml in project root)",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override path to for_score.csv",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for results and plots",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=None,
        help="Skip slow models (SVM, MLP)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    else:
        cfg.setdefault("output_dir", "output")
    if args.fast_mode is not None:
        cfg["fast_mode"] = args.fast_mode

    data_path = cfg["data_path"]
    target = cfg["target"]
    output_dir = cfg["output_dir"]
    random_state = cfg.get("random_state", 42)
    test_size = cfg.get("test_size", 0.20)
    cv_splits = cfg.get("cv_splits", 5)
    n_boot_ci = cfg.get("n_boot_ci", 2000)
    cat_max_levels = cfg.get("cat_max_levels", 20)
    k_grid_config = cfg.get("k_grid", [15, 25, 50, "all"])
    fast_mode = cfg.get("fast_mode", False)
    recode_123 = cfg.get("recode_123_as_yesno_na", True)

    os.makedirs(output_dir, exist_ok=True)

    print("#" * 60)
    print("# 1. Load data")
    print("#" * 60)
    X0, y = load_data(data_path, target)

    print("\n" + "#" * 60)
    print("# 2. Variable classification")
    print("#" * 60)
    X, binary, categorical, continuous, dropped, typing_table = classify_variables(
        X0,
        cat_max=cat_max_levels,
        recode123=recode_123,
        yesno_na_vars=YESNO_NA_VARS,
        force_types=FORCE_TYPES,
    )
    typing_table.to_csv(os.path.join(output_dir, "variable_categorization.csv"), index=False)
    print(f"\nSaved {output_dir}/variable_categorization.csv")

    print("\n" + "#" * 60)
    print("# 3. Train/test split + redundancy reduction")
    print("#" * 60)
    X_train, X_test, y_train, y_test = train_val_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_test, binary, categorical, continuous = reduce_redundancy(
        X_train,
        X_test,
        y_train,
        REDUNDANT_GROUPS,
        binary,
        categorical,
        continuous,
        random_state=random_state,
    )

    print("\n" + "#" * 60)
    print("# 4. Build preprocessor")
    print("#" * 60)
    prep = build_preprocessor(binary, categorical, continuous)
    cv = build_cv(cv_splits, random_state)
    k_grid, n_enc = get_k_grid(prep, X_train, k_grid_config)
    print(f"Train events: {int(y_train.sum())} | Test events: {int(y_test.sum())}")

    print("\n" + "#" * 60)
    print("# 5. Per-model tuning of K + hyperparameters (CV-safe)")
    print("#" * 60)
    zoo, spw = model_zoo(y_train, fast_mode=fast_mode)
    res, oof_scores, test_scores, k_curves, cv_results_by_model, best_estimators = tune_all_models(
        X_train,
        y_train,
        X_test,
        y_test,
        prep,
        cv,
        k_grid,
        zoo,
        n_boot_ci=n_boot_ci,
        random_state=random_state,
    )

    print("\n" + "#" * 60)
    print("# 6. Summary table (best settings per model)")
    print("#" * 60)
    show = res[
        ["model", "best_k", "cv_auc", "cv_sd", "oof_train_auc",
         "test_auc", "test_lo", "test_hi", "test_pr_auc"]
    ].copy()
    print("Best tuned settings per model (sorted by cross-validated ROC-AUC):\n")
    print(show.round(3).to_string(index=False))
    print("\nBest hyperparameters / penalty per model:")
    for _, r in res.iterrows():
        print(f"  {r['model']:13s} k={str(r['best_k']):<4} {r['best_params']}")
    res_out = res.assign(best_params=res["best_params"].astype(str))
    res_out.to_csv(os.path.join(output_dir, "per_model_tuning_results.csv"), index=False)
    print(f"\nSaved {output_dir}/per_model_tuning_results.csv")

    print("\n" + "#" * 60)
    print("# 7. Fixed-K models (15 / 25 / 50)")
    print("#" * 60)
    fixed_df, fixed_test, order = evaluate_fixed_ks(
        X_train,
        y_train,
        X_test,
        y_test,
        prep,
        cv,
        zoo,
        cv_results_by_model,
        k_grid,
        fixed_ks=[15, 25, 50],
        n_boot_ci=n_boot_ci,
        random_state=random_state,
    )
    fixed_df.to_csv(os.path.join(output_dir, "fixed_k_train_test.csv"), index=False)
    print(f"\nSaved {output_dir}/fixed_k_train_test.csv")

    print("\n" + "#" * 60)
    print("# 8. Generate plots")
    print("#" * 60)
    generate_all_plots(
        output_dir,
        binary,
        categorical,
        continuous,
        dropped,
        fixed_df,
        fixed_test,
        order,
        k_curves,
        k_grid,
        res,
        oof_scores,
        test_scores,
        y_train,
        y_test,
    )
    print(f"Plots saved to {output_dir}/")

    print("\n" + "#" * 60)
    print("# 9. Save best model + feature metadata")
    print("#" * 60)
    best_model = res.iloc[0]
    best_name = best_model["model"]
    best_pipeline = best_estimators[best_name]
    feature_names = binary + categorical + continuous
    metadata = {
        "model_name": best_name,
        "feature_names": feature_names,
        "binary": binary,
        "categorical": categorical,
        "continuous": continuous,
        "cv_auc": best_model["cv_auc"],
        "test_auc": best_model["test_auc"],
        "best_k": best_model["best_k"],
        "best_params": best_model["best_params"],
    }
    model_path = os.path.join(output_dir, "best_model.pkl")
    joblib.dump({"pipeline": best_pipeline, "metadata": metadata}, model_path)
    print(f"Saved best model ({best_name}) to {model_path}")
    print(f"  Features: {len(feature_names)} ({len(binary)} binary, {len(categorical)} categorical, {len(continuous)} continuous)")

    print("\n" + "=" * 60)
    print(f"BEST MODEL: {best_name}")
    print(f"  Cross-validated AUC: {best_model['cv_auc']:.3f} ± {best_model['cv_sd']:.3f}")
    print(f"  Test AUC:            {best_model['test_auc']:.3f}")
    print(f"  Best K:              {best_model['best_k']}")
    print(f"  Best params:         {best_model['best_params']}")
    print(f"  Saved pipeline:      {model_path}")
    print("=" * 60)
    print("\nLead with the CV (train) AUC; test bars/curves are one ~22-positive split (wide CIs).")
    print("At ~0.5% prevalence PR-AUC stays low; this is discrimination, not a usable yes/no classifier.")
    print("\nDone.")


if __name__ == "__main__":
    main()
