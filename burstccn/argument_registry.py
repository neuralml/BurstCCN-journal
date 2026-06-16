import argparse


class ArgumentRegistry:
    _defaults = {
        "run_name": {"type": str, "help": "Name of the run"},
        "run_id": {"type": str, "help": "Existing wandb run ID to copy config from"},

        "sweep_id": {"type": str, "help": "Sweep ID for wandb agent"},
        "count": {"type": int, "help": "Number of sweep runs to execute"},

        "config": {"type": str, "help": "Config file path"},

        "dataset": {"type": str, "help": "Dataset to use"},
        "n_epochs": {"type": int, "help": "Number of epochs to train with"},
        "use_validation": {"type": lambda x: (str(x).lower() == 'true'), "help": "Whether to the validation set"},
        "train_batch_size": {"type": int, "help": "Training dataset batch size"},
        "test_batch_size": {"type": int, "help": "Testing dataset batch size"},

        "model_type": {"type": str, "help": "Model type"},
        "use_backprop": {"type": lambda x: (str(x).lower() == 'true'), "help": "Use backprop implementation instead of the model"},

        "seed": {"type": int, "help": "The seed number", "default": 1},

        "optimiser_type": {"type": str, "help": "The optimiser to train feedforward weights with"},
        "lr": {"type": float, "help": "Learning rate"},
        "momentum": {"type": float, "help": "Momentum for optimiser"},
        "weight_decay": {"type": float, "help": "Weight decay for optimiser"},

        "Y_learning": {"type": lambda x: (str(x).lower() == 'true'), "help": "Scale of the Y feedback weights"},
        "Y_lr": {"type": float, "help": "Learning rate for Y feedback weights"},

        "Q_learning": {"type": lambda x: (str(x).lower() == 'true'), "help": "Scale of the Q feedback weights"},
        "Q_lr": {"type": float, "help": "Learning rate for Y feedback weights"},

        "hidden_activation_function": {"type": str, "help": "Hidden layer activation function"},
        "output_activation_function": {"type": str, "help": "Output layer activation function"},
        "loss_type": {"type": str, "help": "Loss function to use"},

        "p_baseline": {"type": float, "help": "Baseline burst probability"},

        "W_scale": {"type": float, "help": "Scale of the feedforward weights"},

        "Y_mode": {"type": str, "help": "Feedback mode for Y feedback weights",
                   "choices": ["tied", "symmetric_init", "random_init"]},
        "Y_scale": {"type": float, "help": "Scale of the Y feedback weights"},

        "Q_mode": {"type": str, "help": "Feedback mode for Y feedback weights.",
                   "choices": ["tied", "symmetric_init", "random_init"]},
        "Q_scale": {"type": float, "help": "Scale of the Q feedback weights"},

        "log_interval": {"type": int, "help": "Interval in batches between each log call"},
        "max_stagnant_epochs": {"type": int, "help": "Maximum number of epochs without improvement before terminating run"},

        "num_workers": {"type": int, "help": "Number of dataset workers"},
        "prefetch_factor": {"type": int, "help": "Data loading prefetch factor"}
    }
    # Predefined defaults for known parameters.
    # Registry storage for registered parameters.
    _registry = {}

    # Single parser instance.
    _parser = None
    # Dictionary to store the values.
    _parsed_params = {}

    # Base config
    _base_config = {}

    @classmethod
    def register(cls, name: str, **overrides):
        """
        Register a parameter by its name.
        If the parameter is known (in _defaults), its default metadata is used.
        Overrides can update or replace those defaults.
        Ensures that 'type' is always specified if no default value exists.
        """
        meta = cls._defaults.get(name, {}).copy()
        meta.update(overrides)

        # Ensure 'type' is present if there's no 'default' value
        if "default" not in meta and "type" not in meta:
            raise ValueError(f"Missing 'type' for parameter '{name}' with no default value.")

        cls._registry[name] = meta

    @classmethod
    def register_list(cls, items, **overrides):
        for item in items:
            if isinstance(item, str):
                cls.register(item, **overrides)
            elif isinstance(item, tuple) and len(item) == 2:
                name, specific_overrides = item
                # Merge common overrides with specific ones (specific wins).
                merged_overrides = overrides.copy()
                merged_overrides.update(specific_overrides)
                cls.register(name, **merged_overrides)
            else:
                raise ValueError("Each registration must be a string or a tuple of (name, overrides_dict)")

    # @classmethod
    # def update_parameters_from_config(cls, config: dict):
    #     """
    #     For each key in the external config (e.g. wandb.config), add it to
    #     the registry if it hasn't been registered already.
    #     The value from the config is used as the default.
    #     """
    #     for key, value in config.items():
    #         # assert key in cls._registry, f"{key} is not registered in the registry: {list(cls._registry.keys())}"
    #         if key not in cls._parameters:
    #             cls._parameters[key] = value
    #     return cls._parameters

    @classmethod
    def _init_parser(cls):
        if cls._parser is None:
            cls._parser = argparse.ArgumentParser(description="Training script arguments")
        return cls._parser

    @classmethod
    def update_parser(cls):
        parser = cls._init_parser()
        for name, meta in cls._registry.items():
            if not any(action.dest == name for action in parser._actions):
                arg_kwargs = {k: meta[k] for k in ("type", "help", "choices") if k in meta}
                parser.add_argument(f"--{name}", default=argparse.SUPPRESS, **arg_kwargs)
        return parser

    @classmethod
    def parse(cls):
        parser = cls.update_parser()
        cli_args, _ = parser.parse_known_args()
        # Update _parameters only with provided (non-None) values.
        for key, value in vars(cli_args).items():
            # if value is not None:
            cls._parsed_params[key] = value
        return cls._parsed_params

    @classmethod
    def set_base_config(cls, config: dict):
        cls._base_config = config

    @classmethod
    def validate_required(cls):
        config = cls.get_config()
        missing = []
        type_errors = []

        for name, meta in cls._registry.items():
            param_required = meta.get("required", False)
            param_value = config.get(name, None)

            # Check for missing required parameters
            if param_required and param_value is None:
                missing.append(f"--{name}")

            # Check for type mismatches
            expected_type = meta.get("type")
            if param_value is not None and expected_type:
                try:
                    # Attempt type conversion
                    expected_type(param_value)
                except (ValueError, TypeError):
                    type_errors.append(
                        f"--{name} should be of type {expected_type.__name__}, but got {type(param_value).__name__}")

        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"The following required arguments are missing: {missing_str}")

        if type_errors:
            type_errors_str = "\n".join(type_errors)
            raise ValueError(f"Type validation errors:\n{type_errors_str}")

    @classmethod
    def get_config(cls):
        cls.parse()

        return_dict = dict()
        for key in cls._registry:
            if key in cls._parsed_params:
                return_dict[key] = cls._parsed_params[key]
            elif key in cls._base_config:
                return_dict[key] = cls._base_config[key]
            elif "default" in cls._registry[key]:
                return_dict[key] = cls._registry[key]["default"]
            else:
                if cls._registry[key].get("required", False):
                    raise ValueError(f"the following missing arguments are required: --{key}")
                # assert not cls._registry[key].get("required", False)

        return return_dict

    @classmethod
    def get(cls, key):
        return cls.get_config()[key]

    @classmethod
    def contains(cls, key):
        return key in cls.get_config()
