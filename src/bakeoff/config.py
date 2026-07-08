import os
import yaml

_CONFIG = None


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
