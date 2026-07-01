import os
import yaml

_CONFIG = None

YESNO_NA_VARS = [
    "use_of_acei", "use_of_arb", "use_of_aspirin", "use_of_ezetimibe", "use_of_pcsk9i",
    "use_of_statins", "ranolazine_yesno", "beta_blockers_yesno", "long_acting_nitr_yesno",
]

REDUNDANT_GROUPS = [
    ["calcification_med_sev", "j_cto_calcification_score"],
    ["lmcto", "target_vessel_overall"],
    ["j_cto_lesion_length", "occlusion_length_mm"],
    ["j_cto_tortuosity_score_1_f", "tortuosity_med_sev"],
    ["left_ventr_ejection_fract", "lvef40", "lvef50", "prior_heart_failure"],
]

FORCE_TYPES = {}


def load_config(config_path=None):
    global _CONFIG
    if config_path is None:
        config_path = os.environ.get(
            "BAKEOFF_CONFIG",
            os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"),
        )
    config_path = os.path.abspath(config_path)
    with open(config_path) as f:
        _CONFIG = yaml.safe_load(f)
    _CONFIG["_config_path"] = config_path
    return _CONFIG


def get_config():
    if _CONFIG is None:
        raise RuntimeError("config not loaded – call load_config() first")
    return _CONFIG
