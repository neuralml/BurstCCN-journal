import yaml
import os


def deep_merge_dicts(default, override):
    """
    Recursively merges two dictionaries.
    Values from `override` take precedence,
    but missing keys are filled from `default`.
    """
    for key, value in default.items():
        if key not in override:
            override[key] = value
        elif isinstance(value, dict) and isinstance(override[key], dict):
            deep_merge_dicts(value, override[key])  # Recursive merge for nested dicts
    return override


def load_config(config_path=None):
    """
    Loads the default config (`configs/default.yaml`) first,
    then overrides it with the specified config (if provided).
    """
    default_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs/default.yaml"))

    with open(default_config_path, "r") as f:
        default_config = yaml.safe_load(f)

    if config_path:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)
        # Merge user config with default config
        return deep_merge_dicts(default_config, user_config)

    return default_config  # Return defaults if no user config is specified
