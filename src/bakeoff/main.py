import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from bakeoff.config import load_config
from bakeoff.pipeline import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="TRIPOD+AI — CTO-PCI adverse outcome prediction pipeline"
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
        help="Override output directory",
    )
    parser.add_argument(
        "--n-boot-optimism",
        type=int,
        default=None,
        help="Override number of bootstrap iterations for optimism correction",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=None,
        help="Skip slow models (SVM, MLP) and reduce bootstraps",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tripod = cfg.get("tripod", {})

    run(
        data_path=args.data_path or cfg["data_path"],
        target=cfg["target"],
        pre_specified_predictors=tripod.get("pre_specified_predictors", []),
        plausible_bounds=tripod.get("plausible_bounds", {}),
        exclude_planned_mcs=tripod.get("exclude_planned_mcs", True),
        planned_mcs_col=tripod.get("planned_mcs_col", None),
        site_col=tripod.get("site_col", None),
        year_col=tripod.get("year_col", None),
        outdir=args.output_dir or tripod.get("tripod_output_dir", "tripod_outputs"),
        random_state=cfg.get("random_state", 42),
        test_size=cfg.get("test_size", 0.20),
        cv_splits=cfg.get("cv_splits", 5),
        n_boot_ci=cfg.get("n_boot_ci", 2000),
        n_boot_optimism=args.n_boot_optimism or tripod.get("n_boot_optimism", 500),
        n_repeated_cv=cfg.get("n_repeated_cv", 20),
        cat_max_levels=cfg.get("cat_max_levels", 20),
        k_grid=cfg.get("k_grid", [10, 15, 25, "all"]),
        fast_mode=args.fast_mode if args.fast_mode is not None else cfg.get("fast_mode", False),
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


if __name__ == "__main__":
    main()
