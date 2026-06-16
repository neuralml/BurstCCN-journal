from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline

from burstccn.models.networks.ann_networks import FullyConnectedANN
from burstccn.models.networks.burstccn_networks import FullyConnectedBurstCCN
from plotting.analysis.rl_model_utils import DQN, invphi, move_start_location, phi, prep_state
from plotting.analysis.results_store_base import (
    ResultsStore,
    WandbResultsStore,
    file_cache_decorator,
    smooth_arrays,
    summarise_metric,
    arrays_from_summary,
)


class RLWandbResultsStore(WandbResultsStore):
    def __init__(self):
        super().__init__(
            project_name="BurstCCN_RL",
            cache_path="rl_wandb_results_cache.pkl",
        )
        self.wandb_interface.rename_episode_keys_backwards_compatible = False

        self.EPISODE_KEY = "episode"
        self.TEST_SCORE_KEY = "episode/avg_test_score"

    def get_group_params(self, group):
        GROUP_PARAM_SETS = {
            "performance": dict(
                model_types=["ann", "burstccn"],
                modes=["fa", "kp", "tied"],
            )
        }
        return GROUP_PARAM_SETS[group]

    def get_wandb_run_name(self, group, **kwargs):
        if group == "performance":
            model_type = kwargs["model_type"]
            mode = kwargs["mode"]
            return model_type if mode == "tied" else f"{model_type}_{mode}"

        raise ValueError(f"Invalid group: {group}")

    def get_wandb_group_name(self, group, **kwargs):
        group_dict = {
            "performance": "may_runs2",
        }
        return group_dict[group]

    def get_wandb_run_filter(self, group, **kwargs):
        return {
            "run_name": self.get_wandb_run_name(group, **kwargs),
            "group": self.get_wandb_group_name(group, **kwargs),
        }

    def get_mean_log_episode_data(self, value_key, group="performance", sigma=10, **kwargs):
        run_filter = self.get_wandb_run_filter(group, **kwargs)
        df = self.fetch(
            **run_filter,
            keys=[self.EPISODE_KEY, value_key],
        )

        summary = summarise_metric(
            df,
            step_col=self.EPISODE_KEY,
            value_col=value_key,
            sample_col="run_id",
            err="sem",
        )
        episodes, mean, stderr = arrays_from_summary(summary, step_col=self.EPISODE_KEY)

        if sigma is not None:
            return smooth_arrays(episodes, mean, stderr, sigma=sigma)

        return episodes, mean, stderr

    @file_cache_decorator()
    def get_mean_test_score_by_episode(self, group="performance", **kwargs):
        return self.get_mean_log_episode_data(self.TEST_SCORE_KEY, group=group, **kwargs)


class RLOfflineResultsStore(ResultsStore):
    MODEL_RUN_NAMES = {
        "burstccn_fa": "burstccn_fa_feb1_d90uessn",
        "burstccn": "burstccn_oi6zapol",
        "tmp": "tmp",

    }
    LEGACY_MODEL_RUN_NAMES = {
        "burstccn_fa_legacy": "BurstCCN_best_fa_from_sweep",
    }

    def __init__(self, model_factory=None):
        super().__init__(cache_path="rl_offline_results_cache.pkl")
        self.model_factory = model_factory
        self.print_data_distributions = True
        self._printed_distribution_keys = set()

    def get_rl_data_dir(self):
        return Path(__file__).parent / "rl_data"

    def get_legacy_rl_data_dir(self):
        return Path(__file__).parents[2] / "plotting_old" / "analysis" / "rl_data"

    def get_checkpoint_run_name(self, model_type):
        if model_type in self.LEGACY_MODEL_RUN_NAMES:
            return self.LEGACY_MODEL_RUN_NAMES[model_type]

        try:
            return self.MODEL_RUN_NAMES[model_type]
        except KeyError as e:
            raise ValueError(
                f"No RL checkpoint run name configured for model_type={model_type!r}. "
                f"Known model types: {list(self.MODEL_RUN_NAMES) + list(self.LEGACY_MODEL_RUN_NAMES)}"
            ) from e

    def load_generated_states_data(self):
        path = self.get_rl_data_dir() / "states_and_descriptors.pkl"
        with path.open("rb") as f:
            states, descriptors = pickle.load(f)
        return states, descriptors

    def load_run_config(self, model_type):
        run_name = self.get_checkpoint_run_name(model_type)
        config_path = self.get_rl_data_dir() / run_name / "config.yaml"
        return OmegaConf.load(config_path)

    def build_model_from_config(self, model_type, device):
        cfg = self.load_run_config(model_type)
        model_cfg = cfg.model

        if model_cfg.model_type == "burstccn":
            model = FullyConnectedBurstCCN(model_cfg)
        elif model_cfg.model_type == "ann":
            model = FullyConnectedANN(model_cfg)
        else:
            raise ValueError(f"Unsupported RL model_type in config: {model_cfg.model_type!r}")

        return model.to(device)

    def get_model_layers(self, model):
        if hasattr(model, "classification_layers"):
            return model.classification_layers
        return model.get_layers()

    def get_layer_somatic_potential(self, layer):
        if hasattr(layer, "v"):
            return layer.v
        return layer.soma

    def get_action_size(self, model_type, model):
        if model_type in self.LEGACY_MODEL_RUN_NAMES:
            return int(getattr(self.get_model_layers(model)[-1], "out_features", 4))

        cfg = self.load_run_config(model_type)
        return int(getattr(cfg.task, "n_actions", getattr(model, "n_outputs", 4)))

    def get_generated_states_and_maps(self):
        states, descriptors = self.load_generated_states_data()
        hole_maps = [self.get_hole_map(state) for state in states]
        return states, hole_maps, descriptors

    def get_batched_states_tensor(self):
        states, _ = self.load_generated_states_data()
        return torch.from_numpy(np.stack(states)).float()

    def get_agent_location(self, state, map_size=3, one_hot=False):
        position_map = state[:map_size * map_size]
        if one_hot:
            return position_map

        agent_index = np.argmax(position_map)
        return divmod(agent_index, map_size)

    def get_hole_map(self, full_state, map_size=3):
        hole_map_state = full_state[map_size ** 2:]
        return hole_map_state.reshape(map_size, map_size)

    def create_state(self, agent_row, agent_col, hole_map, map_size=3):
        position = np.zeros((map_size, map_size))
        position[agent_row, agent_col] = 1.0
        state = np.concatenate([position.ravel(), hole_map.ravel()])
        return torch.as_tensor(state, dtype=torch.float32).reshape(1, -1)

    def create_environment(self, map_desc):
        import gymnasium as gym

        env = gym.make("FrozenLake-v1", desc=map_desc, is_slippery=False, render_mode="rgb_array_list")
        env.reset()
        return env

    def get_model(self, model_type, model_id):
        if model_type in self.LEGACY_MODEL_RUN_NAMES:
            return self.get_legacy_model(model_type, model_id)

        device = "cpu"
        run_name = self.get_checkpoint_run_name(model_type)
        model_path = self.get_rl_data_dir() / run_name / f"model_{model_id}.pth"

        if self.model_factory is None:
            model = self.build_model_from_config(model_type, device=device)
        else:
            model = self.model_factory(model_type=model_type, model_id=model_id, device=device)

        loaded_state_dict = torch.load(model_path, map_location=device)
        if isinstance(loaded_state_dict, dict) and "model_state_dict" in loaded_state_dict:
            loaded_state_dict = loaded_state_dict["model_state_dict"]
        loaded_state_dict = {
            key.replace("classification_layers", "layers"): value
            for key, value in loaded_state_dict.items()
        }
        model.load_state_dict(loaded_state_dict)
        model.eval()
        return model

    def get_legacy_model(self, model_type, model_id):
        from plotting_old.analysis.rl_data.tmp_functions import BurstCCN

        if model_type != "burstccn_fa_legacy":
            raise ValueError(f"Unsupported legacy RL model_type: {model_type!r}")

        device = "cpu"
        model = BurstCCN(
            n_inputs=18,
            n_outputs=4,
            p_baseline=0.5,
            n_hidden_layers=3,
            n_hidden_units=64,
            W_scale=1.894118339471331,
            Y_mode="random_init",
            Q_mode="tied",
            Y_scale=2.067546502102279,
            Q_scale=1,
            Y_learning=True,
            Q_learning=False,
            device=device,
        )

        run_name = self.get_checkpoint_run_name(model_type)
        model_path = self.get_legacy_rl_data_dir() / run_name / f"model_{model_id}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model

    def scale_final_hidden_feedback_weights(self, model, scale):
        layers = self.get_model_layers(model)
        if len(layers) < 2:
            return

        final_hidden_layer = layers[-2]
        with torch.no_grad():
            for weight_name in ["Y_weight", "Q_weight"]:
                weight = getattr(final_hidden_layer, weight_name, None)
                if weight is not None:
                    weight.mul_(scale)

    def get_model_layer_names(self, model_type, model_id=601):
        model = self.get_model(model_type, model_id)
        return [f"fc{i + 1}" for i in range(len(self.get_model_layers(model)))]

    @staticmethod
    def decode_model_output_to_q_values(output):
        return invphi(output)

    @file_cache_decorator()
    def get_model_layer_activities(self, model_type, model_id, state):
        model = self.get_model(model_type, model_id)
        model.forward(state)

        extract = lambda s: s.detach().cpu().numpy()
        layer_names = self.get_model_layer_names(model_type, model_id)
        layers = self.get_model_layers(model)

        return {
            "somatic_potentials": {
                layer_name: extract(self.get_layer_somatic_potential(layer))
                for layer, layer_name in zip(layers, layer_names)
            },
            "event_rates": {
                layer_name: extract(layer.e) for layer, layer_name in zip(layers, layer_names)
            },
            "output_Q_values": self.decode_model_output_to_q_values(extract(layers[-1].e)),
        }

    def get_all_model_Q_values(self, model_type, model_id, hole_map, map_size=3):
        model = self.get_model(model_type, model_id)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        rows, cols = map_size, map_size
        goal_location = (rows - 1, cols - 1)
        action_size = self.get_action_size(model_type, model)
        q_values = np.zeros((rows, cols, action_size))

        with torch.no_grad():
            for row in range(rows):
                for col in range(cols):
                    if (row, col) == goal_location or hole_map[row, col] == 1:
                        continue

                    state = self.create_state(row, col, hole_map, map_size=map_size).to(device)
                    output = model(state)
                    q_values[row, col] = self.decode_model_output_to_q_values(output).detach().cpu().numpy()[0]

        return q_values

    def is_ice_hole_in_direction(self, state, direction, map_size=3):
        total_cells = map_size * map_size
        agent_row, agent_col = self.get_agent_location(state, map_size=map_size, one_hot=False)

        direction_offsets = {
            "left": (0, -1),
            "right": (0, 1),
            "up": (-1, 0),
            "down": (1, 0),
        }

        if direction not in direction_offsets:
            raise ValueError(f"Invalid direction {direction!r}")

        dr, dc = direction_offsets[direction]
        neighbor_row, neighbor_col = agent_row + dr, agent_col + dc

        if not (0 <= neighbor_row < map_size and 0 <= neighbor_col < map_size):
            return False

        neighbor_index = neighbor_row * map_size + neighbor_col
        return state[total_cells + neighbor_index] == 1

    def calculate_relative_hole_locations(self, state):
        directions = ["up", "down", "left", "right"]
        return [self.is_ice_hole_in_direction(state, direction) for direction in directions]

    def get_distance_to_goal(self, state, map_size=3):
        agent_row, agent_col = self.get_agent_location(state, map_size=map_size, one_hot=False)
        goal_row, goal_col = map_size - 1, map_size - 1
        return abs(goal_row - agent_row) + abs(goal_col - agent_col)

    def get_predictor_data_from_actor(self, predictor_data_type, **kwargs):
        predictor_types = {
            "somatic_potentials",
            "event_rates",
            "burst_rates",
            "burst_probabilities",
            "apical_potentials",
            "delta_b",
            "delta_p",
            "output_Q_values",
            "Q_value_prediction_errors",
            "Q_value_prediction_errors_vec",
            "action",
        }

        if predictor_data_type not in predictor_types:
            raise ValueError(f"Unsupported predictor data type: {predictor_data_type}")

        activities = self.get_model_layer_prediction_errors(kwargs["model_type"], kwargs.get("model_id", 601))
        self.print_decoder_predictor_distribution_summary(
            activities,
            model_type=kwargs["model_type"],
            model_id=kwargs.get("model_id", 601),
            layer_names=kwargs.get("layer_names", [kwargs.get("layer_name", "fc2")]),
        )
        predictor_data = activities[predictor_data_type]

        if "layer_name" in kwargs:
            predictor_data = predictor_data[kwargs["layer_name"]]

        if "layer_names" in kwargs:
            layer_names = kwargs["layer_names"]
            arrays = [predictor_data[name] for name in layer_names]
            predictor_data = np.concatenate(arrays, axis=1)

        return predictor_data

    def get_decoder_target_data(self, decoder_data_type, **kwargs):
        states, _, _ = self.get_generated_states_and_maps()

        if decoder_data_type == "agent_location":
            agent_locations = np.array([self.get_agent_location(state, one_hot=True) for state in states])
            return np.argmax(agent_locations, axis=1)

        if decoder_data_type == "relative_hole_locations":
            return np.array([self.calculate_relative_hole_locations(state) for state in states])

        if decoder_data_type == "Q_value_prediction_errors":
            return self.get_predictor_data_from_actor(
                "Q_value_prediction_errors",
                model_type=kwargs["model_type"],
                model_id=kwargs["model_id"],
            )

        if decoder_data_type == "Q_value_prediction_errors_vec":
            return self.get_predictor_data_from_actor(
                "Q_value_prediction_errors_vec",
                model_type=kwargs["model_type"],
                model_id=kwargs["model_id"],
            )

        if decoder_data_type == "action":
            action = self.get_predictor_data_from_actor(
                "action",
                model_type=kwargs["model_type"],
                model_id=kwargs["model_id"],
            )
            return np.argmax(action, axis=1)

        if decoder_data_type == "is_safe_action":
            activities = self.get_model_layer_prediction_errors(kwargs["model_type"], kwargs.get("model_id", 601))
            return activities["is_safe_action"].ravel().astype(int)

        if decoder_data_type == "distance_to_goal":
            distances = np.array([self.get_distance_to_goal(state) for state in states])
            return preprocessing.StandardScaler().fit_transform(distances.reshape(-1, 1))

        if decoder_data_type == "action_Q_value":
            actions = self.get_decoder_target_data("action", **kwargs)
            q_values = self.get_decoder_target_data("trained_Q_values", **kwargs)
            action_q_values = q_values[np.arange(len(q_values)), actions]
            return preprocessing.StandardScaler().fit_transform(action_q_values.reshape(-1, 1))

        if decoder_data_type == "trained_Q_values":
            batched_state = self.get_batched_states_tensor()
            activities = self.get_model_layer_activities(kwargs["model_type"], kwargs.get("model_id", 601), batched_state)
            return activities["output_Q_values"]

        raise ValueError(f"Unsupported decoder data type: {decoder_data_type}")

    def calculate_decoding_error(self, predictor_data_type, decoder_target_type, decoder_type=None, n_splits=5, **kwargs):
        predictor_data = self.get_predictor_data_from_actor(predictor_data_type, **kwargs)
        decoder_target_data = self.get_decoder_target_data(decoder_target_type, **kwargs)

        if decoder_type is None:
            if decoder_target_type in {
                "trained_Q_values",
                "Q_value_prediction_errors",
                "Q_value_prediction_errors_vec",
                "distance_to_goal",
                "action_Q_value",
            }:
                decoder_type = "regression"
            elif decoder_target_type in {"agent_location", "action", "is_safe_action"}:
                decoder_type = "classification"
            elif decoder_target_type == "relative_hole_locations":
                decoder_type = "multi-classification"
            else:
                raise ValueError(f"No decoder type specified for {decoder_target_type}")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        if decoder_type == "regression":
            decoder_target_data = preprocessing.StandardScaler().fit_transform(decoder_target_data)
            if decoder_target_data.ndim == 2 and decoder_target_data.shape[1] == 1:
                decoder_target_data = decoder_target_data.ravel()

            decoder = make_pipeline(
                preprocessing.StandardScaler(),
                RidgeCV(alphas=np.logspace(-4, 4, 20)),
            )
            scores = cross_val_score(decoder, predictor_data, decoder_target_data, cv=kf)
            scores = np.clip(scores, 0.0, 1.0)
        elif decoder_type == "classification":
            decoder = make_pipeline(
                preprocessing.StandardScaler(),
                LogisticRegression(max_iter=1000),
            )
            scores = cross_val_score(decoder, predictor_data, decoder_target_data, cv=kf)
        elif decoder_type == "multi-classification":
            decoder = make_pipeline(
                preprocessing.StandardScaler(),
                MultiOutputClassifier(LogisticRegression(max_iter=1000)),
            )
            scores = cross_val_score(decoder, predictor_data, decoder_target_data, cv=kf, scoring="accuracy")
        else:
            raise ValueError(f"Unknown decoder type {decoder_type}")

        score_mean = scores.mean()
        score_stderr = scores.std() / np.sqrt(n_splits)
        return score_mean, score_stderr

    def get_model_layer_prediction_errors(self, model_type, model_id, map_size=3, gamma=0.95):
        return self._get_model_layer_prediction_errors(model_type, model_id, map_size=map_size, gamma=gamma, cache_version=8)

    def print_decoder_predictor_distribution_summary(self, data, model_type, model_id=601, layer_names=None):
        if not self.print_data_distributions:
            return

        layer_names = list(layer_names or ["fc2"])
        print_key = (model_type, model_id, tuple(layer_names))
        if print_key in self._printed_distribution_keys:
            return

        self._printed_distribution_keys.add(print_key)

        predictor_data_types = [
            "somatic_potentials",
            "event_rates",
            "burst_rates",
            "apical_potentials",
            "burst_probabilities",
            "delta_b",
        ]

        def summarise_array(values):
            values = np.asarray(values)
            flat_values = values.reshape(-1).astype(float)
            return {
                "shape": str(values.shape),
                "mean": np.mean(flat_values),
                "std": np.std(flat_values),
                "min": np.min(flat_values),
                "median": np.median(flat_values),
                "max": np.max(flat_values),
            }

        print(
            f"\nRL decoder predictor distributions "
            f"(model_type={model_type}, model_id={model_id}, layers={','.join(layer_names)})"
        )
        print(f"{'data type':<22} {'shape':<14} {'mean':>10} {'std':>10} {'min':>10} {'median':>10} {'max':>10}")
        print("-" * 90)

        for data_type in predictor_data_types:
            if data_type not in data:
                continue

            layer_arrays = [
                data[data_type][layer_name]
                for layer_name in layer_names
                if layer_name in data[data_type]
            ]
            if not layer_arrays:
                continue

            values = np.concatenate(layer_arrays, axis=1)
            summary = summarise_array(values)
            print(
                f"{data_type:<22} {summary['shape']:<14} "
                f"{summary['mean']:>10.4g} {summary['std']:>10.4g} "
                f"{summary['min']:>10.4g} {summary['median']:>10.4g} {summary['max']:>10.4g}"
            )

    @file_cache_decorator()
    def _get_model_layer_prediction_errors(self, model_type, model_id, map_size=3, gamma=0.95, cache_version=7):
        model = self.get_model(model_type, model_id)
        states, _, descriptors = self.get_generated_states_and_maps()

        state_size = map_size * map_size * 2
        use_conv = False
        rgb = False
        device = "cpu"
        action_size = self.get_action_size(model_type, model)
        dqn = DQN(model, state_size, action_size=action_size, device=device, args={"use_conv": use_conv, "rgb": rgb}, mem_size=2000)

        extract = lambda x: x.detach().cpu().numpy()
        layer_names = self.get_model_layer_names(model_type, model_id)
        layers = self.get_model_layers(model)

        data_dict = {
            "somatic_potentials": {name: [] for name in layer_names},
            "event_rates": {name: [] for name in layer_names},
            "burst_rates": {name: [] for name in layer_names},
            "burst_probabilities": {name: [] for name in layer_names},
            "apical_potentials": {name: [] for name in layer_names[:-1]},
            "delta_b": {name: [] for name in layer_names},
            "delta_p": {name: [] for name in layer_names},
            "output_Q_values": [],
            "Q_value_prediction_errors": [],
            "Q_value_prediction_errors_vec": [],
            "agent_location": [],
            "relative_hole_locations": [],
            "action": [],
            "is_safe_action": [],
            "state_value": [],
        }

        for state, map_desc in zip(states, descriptors):
            new_map_desc = move_start_location(map_desc, state)
            env = self.create_environment(new_map_desc)
            env.reset()

            action = dqn.act(state, epsilon=0.0)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = prep_state(next_state, env, rgb, curr_map=new_map_desc)

            tensor_state = torch.from_numpy(state).float().to(device).reshape(1, -1)
            next_state = torch.from_numpy(next_state).float().to(device).reshape(1, -1)

            next_state_output = dqn.forward(next_state)
            target = phi(reward + (1 - done) * gamma * torch.max(invphi(next_state_output), dim=1)[0].detach())

            state_output = dqn.forward(tensor_state)
            t = state_output.clone()
            t[0, action] = target
            model.backward(t)

            prediction_error = state_output[0, action] - target
            t_detached = t.detach()
            prediction_error_vec = (state_output.detach() - t_detached).squeeze(0).cpu().numpy()

            action_vec = np.zeros((action_size,))
            action_vec[action] = 1.0
            state_value = torch.max(state_output.detach(), dim=1)[0].item()

            data_dict["Q_value_prediction_errors"].append(prediction_error.item())
            data_dict["Q_value_prediction_errors_vec"].append(prediction_error_vec)
            data_dict["agent_location"].append(self.get_agent_location(state, one_hot=True))
            data_dict["relative_hole_locations"].append(self.calculate_relative_hole_locations(state))
            data_dict["output_Q_values"].append(self.decode_model_output_to_q_values(extract(layers[-1].e)))
            data_dict["action"].append(action_vec)
            data_dict["is_safe_action"].append(float(not (done and reward <= 0)))
            data_dict["state_value"].append(state_value)

            for layer, name in zip(layers, layer_names):
                data_dict["somatic_potentials"][name].append(extract(self.get_layer_somatic_potential(layer)))
                data_dict["event_rates"][name].append(extract(layer.e))
                data_dict["burst_rates"][name].append(extract(layer.b_t))
                data_dict["burst_probabilities"][name].append(extract(layer.p_t))
                data_dict["delta_b"][name].append(extract(layer.b_t - layer.b))
                data_dict["delta_p"][name].append(extract(layer.p_t - layer.p_baseline))

                if name != layer_names[-1]:
                    data_dict["apical_potentials"][name].append(extract(layer.apic))

        for key in [
            "somatic_potentials",
            "event_rates",
            "burst_rates",
            "burst_probabilities",
            "apical_potentials",
            "delta_b",
            "delta_p",
        ]:
            data_dict[key] = {name: np.vstack(values) for name, values in data_dict[key].items()}

        for key in [
            "Q_value_prediction_errors",
            "Q_value_prediction_errors_vec",
            "output_Q_values",
            "relative_hole_locations",
            "agent_location",
            "action",
            "is_safe_action",
            "state_value",
        ]:
            data_dict[key] = np.vstack(data_dict[key])

        return data_dict
