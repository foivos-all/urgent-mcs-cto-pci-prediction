import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import yaml

from bakeoff.config import load_config
from bakeoff.tripod import run


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="TRIPOD+AI-compliant Firth logistic regression pipeline"
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
        help="Override output directory for TRIPOD results",
    )
    parser.add_argument(
        "--n-boot-optimism",
        type=int,
        default=None,
        help="Override number of bootstrap iterations for optimism correction",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tripod = cfg.get("tripod", {})

    data_path = args.data_path or cfg["data_path"]
    outdir = args.output_dir or tripod.get("tripod_output_dir", "tripod_outputs")
    target = cfg["target"]
    predictors = tripod.get("pre_specified_predictors", [])
    plausible_bounds = tripod.get("plausible_bounds", {})
    exclude_planned_mcs = tripod.get("exclude_planned_mcs", True)
    planned_mcs_col = tripod.get("planned_mcs_col", None)
    n_boot_optimism = args.n_boot_optimism or tripod.get("n_boot_optimism", 500)
    score_increments = tripod.get("score_increments", {})
    points_max = tripod.get("points_max", 10)
    random_state = cfg.get("random_state", 42)

    if not predictors:
        print("ERROR: No pre_specified_predictors defined in config[tripod].", file=sys.stderr)
        sys.exit(1)

    print("TRIPOD+AI Clinical Prediction Model — Firth Logistic Regression")
    print(f"  Data:    {data_path}")
    print(f"  Target:  {target}")
    print(f"  Predictors ({len(predictors)}): {', '.join(predictors)}")
    print(f"  Output:  {outdir}")
    print(f"  Optimism bootstrap iterations: {n_boot_optimism}")
    print()

    run(
        data_path=data_path,
        target=target,
        predictors=predictors,
        plausible_bounds=plausible_bounds,
        exclude_planned_mcs=exclude_planned_mcs,
        planned_mcs_col=planned_mcs_col,
        outdir=outdir,
        random_state=random_state,
        n_boot_optimism=n_boot_optimism,
        score_increments=score_increments,
        points_max=points_max,
    )


if __name__ == "__main__":
    main()
