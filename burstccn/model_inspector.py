import torch

from burstccn.models.networks.base import AutogradNetwork
from burstccn.utils import similarity_angle

STATE_REGISTRY = {
    "angle_bp": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global",        "func": "get_global_grad_angle_bp", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_grad_angle_bp", "layers": "hidden"},
            {"name": "global_average", "func": "get_global_grad_angle_average_bp", "layers": "all"},
            {"name": "global_hidden_average", "func": "get_global_grad_angle_average_bp", "layers": "hidden"},
            {"name": "global_weighted", "func": "get_global_grad_angle_weighted_bp", "layers": "all"},
            {"name": "global_hidden_weighted", "func": "get_global_grad_angle_weighted_bp", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_angle_bp",
            "layers": "all",
        },
    },
    "angle_fa": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global",        "func": "get_global_grad_angle_fa", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_grad_angle_fa", "layers": "hidden"},
            {"name": "global_average", "func": "get_global_grad_angle_average_fa", "layers": "all"},
            {"name": "global_hidden_average", "func": "get_global_grad_angle_average_fa", "layers": "hidden"},
            {"name": "global_weighted", "func": "get_global_grad_angle_weighted_fa", "layers": "all"},
            {"name": "global_hidden_weighted", "func": "get_global_grad_angle_weighted_fa", "layers": "hidden"},

        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_angle_fa",
            "layers": "all",
        },
    },
    "grad_norm": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global",        "func": "get_global_grad_norm", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_grad_norm", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_norm",
            "layers": "all",
        },
    },
    "grad_norm_bp": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global",        "func": "get_global_grad_norm_bp", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_grad_norm_bp", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_norm_bp",
            "layers": "all",
        },
    },
    "grad_norm_fa": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global",        "func": "get_global_grad_norm_fa", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_grad_norm_fa", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_norm_fa",
            "layers": "all",
        },
    },
    "grad_norm_Y": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global", "func": "get_global_grad_norm_Y", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_norm_Y",
            "layers": "hidden",
        },
    },
    "grad_norm_ratio_bp": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global",        "func": "get_global_grad_norm_ratio_bp", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_grad_norm_ratio_bp", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_norm_ratio_bp",
            "layers": "all",
        },
    },
    "grad_norm_ratio_fa": {
        "require_teacher": True,
        "require_noiseless": True,
        "global_states": [
            {"name": "global",        "func": "get_global_grad_norm_ratio_fa", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_grad_norm_ratio_fa", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_grad_norm_ratio_fa",
            "layers": "all",
        },
    },
    "angle_WY": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_angle_WY", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_angle_WY",
            "layers": "hidden",
        },
    },
    "angle_QY": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_angle_QY", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_angle_QY",
            "layers": "hidden",
        },
    },
    "angle_W_pyr_intn": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_angle_W_pyr_intn", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_angle_W_pyr_intn",
            "layers": "hidden",
        },
    },
    "angle_Y_intn_pyr": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_angle_Y_intn_pyr", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_angle_Y_intn_pyr",
            "layers": "hidden",
        },
    },
    "apical_magnitude": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_apical_magnitude", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_apical_magnitude",
            "layers": "hidden",
        },
    },
    "apical_variance": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_apical_variance", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_apical_variance",
            "layers": "hidden",
        },
    },
    "apical_max": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_apical_max", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_apical_max",
            "layers": "hidden",
        },
    },
    "burst_prob_change_magnitude": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_burst_prob_change_magnitude", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_burst_prob_change_magnitude",
            "layers": "all",
        },
    },
    "burst_prob_change_magnitude_top95": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_burst_prob_change_magnitude_top95",
            "layers": "all",
        },
    },
    "burst_prob_change_variance": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_burst_prob_change_variance", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_burst_prob_change_variance",
            "layers": "all",
        },
    },
    "burst_prob_change_max": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_burst_prob_change_max", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_burst_prob_change_max",
            "layers": "all",
        },
    },
    "burst_rate_change_magnitude": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_burst_rate_change_magnitude", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_burst_rate_change_magnitude",
            "layers": "all",
        },
    },
    "burst_rate_change_variance": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_burst_rate_change_variance", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_burst_rate_change_variance",
            "layers": "all",
        },
    },
    "burst_rate_change_max": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_burst_rate_change_max", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_burst_rate_change_max",
            "layers": "all",
        },
    },
    "event_rate_variance": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_event_rate_variance", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_event_rate_variance",
            "layers": "all",
        },
    },
    "event_rate_derivative_mean": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_event_rate_derivative_mean", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_event_rate_derivative_mean",
            "layers": "all",
        },
    },
    "event_rate_saturation_factor": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_event_rate_saturation_factor", "layers": "all"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_event_rate_saturation_factor",
            "layers": "all",
        },
    },
    "W_norm": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_W_norm", "layers": "all"},
            {"name": "global_hidden", "func": "get_global_W_norm", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_W_norm",
            "layers": "all",
        },
    },
    "Y_norm": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_Y_norm", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_Y_norm",
            "layers": "hidden",
        },
    },
    "Q_norm": {
        "require_teacher": False,
        "require_noiseless": False,
        "global_states": [
            {"name": "global", "func": "get_global_Q_norm", "layers": "hidden"},
        ],
        "layer_states": {
            "name_template": "layer_{layer_index}",
            "func": "get_layer_Q_norm",
            "layers": "hidden",
        },
    },
}


# Flattens 4-dim convolutional state to 2-dims
def _flatten_state(state):
    if state.ndim > 2:
        state = state.flatten(start_dim=1)
    return state


class ModelInspector:
    def __init__(self, model):
        self.model = model
        self.state_funcs = {}

        num_layers = len(model.get_layers())
        self.layer_sets = {
            "all": list(range(num_layers)),
            "hidden": list(range(num_layers - 1)),  # exclude output
            "output": [num_layers - 1],
        }

        self.register_states(model.loggable_state_types)
        self.add_internal_state_hooks()

        self._cache = {}

    def _expand_layers(self, spec):
        if isinstance(spec, str):
            try:
                return self.layer_sets[spec]
            except KeyError:
                raise KeyError(f"Unknown layer set '{spec}'. Known: {list(self.layer_sets.keys())}")
        return list(spec)

    def _register_state(self, state_name, func, require_teacher=False, require_noiseless=False, func_kwargs=None):
        entry = {
            "func": func,
            "require_teacher": require_teacher,
            "require_noiseless": require_noiseless,
        }
        if func_kwargs is not None:
            entry["func_kwargs"] = func_kwargs

        self.state_funcs[state_name] = entry

    def register_states(self, internal_state_types):
        for internal_state_type in internal_state_types:
            self.register_state(internal_state_type)

    def register_state(self, internal_state_type):
        if internal_state_type not in STATE_REGISTRY:
            raise NotImplementedError(f"Internal state type '{internal_state_type}' is not supported.")

        prefix = f"{internal_state_type}/"
        meta = STATE_REGISTRY[internal_state_type]
        require_teacher = meta.get("require_teacher", False)
        require_noiseless = meta.get("require_noiseless", False)

        # --- Global states ---
        for g in meta.get("global_states", []):
            func = getattr(self, g["func"])  # resolve "func" string to bound method
            layers = self._expand_layers(g["layers"])  # expand "all"/"hidden"/"output" or list
            self._register_state(
                state_name=prefix + g["name"],
                func=func,
                require_teacher=require_teacher,
                require_noiseless=require_noiseless,
                func_kwargs={"layer_indices": layers},
            )

        # --- Layer states ---
        if "layer_states" in meta:
            ls = meta["layer_states"]
            func = getattr(self, ls["func"])  # resolve "func" string
            layers = self._expand_layers(ls["layers"])
            name_tmpl = ls["name_template"]
            for layer_index in layers:
                self._register_state(
                    state_name=prefix + name_tmpl.format(layer_index=layer_index),
                    func=func,
                    require_teacher=require_teacher,
                    require_noiseless=require_noiseless,
                    func_kwargs={"layer_index": layer_index},
                )

    def add_internal_state_hooks(self):
        if isinstance(self.model, AutogradNetwork):
            print("Adding state hooks...")
            if "angle_bp" in self.model.loggable_state_types:
                layers = self.model.get_layers()
                for layer in layers:
                    def bp_grad_store_hook(grad, layer=layer):
                        if getattr(layer, 'store_grad_bp', False):
                            layer.W_weight.grad_bp = grad.detach().clone()
                        return grad

                    layer.W_weight.register_hook(bp_grad_store_hook)

    def get_internal_model_state(self, with_teacher=True, with_forward_noise=False):
        self._cache.clear()

        state_dict = {}

        for name, info in self.state_funcs.items():
            if info["require_teacher"] and not with_teacher:
                continue
            if info["require_noiseless"] and with_forward_noise:
                continue
            func_kwargs = info.get("func_kwargs", {})
            # state_name = name.format(**state_kwargs)
            state_dict[name] = info["func"](**func_kwargs)

        return state_dict

    def get_layer_weight_grads(self, layer_index, grad_type='default'):
        layer_weight = self.model.get_layer(layer_index).W_weight
        if grad_type == 'default':
            grads = layer_weight.grad
        elif grad_type == 'bp':
            grads = layer_weight.grad_bp
        elif grad_type == 'fa':
            grads = layer_weight.grad_fa
        else:
            raise NotImplementedError(f"Unsupported gradient type: {grad_type}")

        return grads.detach().flatten() if grads is not None else torch.zeros_like(layer_weight).detach().flatten()

    def get_weight_grads(self, layer_indices, grad_type='default'):
        all_grads = torch.cat([self.get_layer_weight_grads(layer_index, grad_type=grad_type)
                               for layer_index in layer_indices]).flatten()
        return all_grads

    def get_layer_W_weights(self, layer_index):
        # return self.model.W_weight_parameters()[layer_index].flatten()
        layers = self.model.get_layers()
        return layers[layer_index].W_weight.flatten()

    def get_W_weights(self, layer_indices):
        all_W_weights = torch.cat([self.get_layer_W_weights(layer_index) for layer_index in layer_indices]).flatten()
        return all_W_weights

    def get_layer_Q_weights(self, layer_index):
        # return self.model.Q_weights()[layer_index].flatten()
        return self.model.get_parameters("Q_weight")[layer_index].flatten()

    def get_Q_weights(self, layer_indices):
        all_Q_weights = torch.cat([self.get_layer_Q_weights(layer_index) for layer_index in layer_indices]).flatten()
        return all_Q_weights

    def get_layer_Y_weights(self, layer_index):
        # return self.model.Y_weights()[layer_index].flatten()
        return self.model.get_parameters("Y_weight")[layer_index].flatten()

    def get_Y_weights(self, layer_indices):
        all_Y_weights = torch.cat([self.get_layer_Y_weights(layer_index) for layer_index in layer_indices]).flatten()
        return all_Y_weights

    def get_layer_Y_weight_grads(self, layer_index):
        layer = self.model.get_layer(layer_index)
        grads = layer.Y_weight.grad
        return grads.detach().flatten() if grads is not None else torch.zeros_like(layer.Y_weight).detach().flatten()

    def get_Y_weight_grads(self, layer_indices):
        all_grads = torch.cat([self.get_layer_Y_weight_grads(layer_index) for layer_index in layer_indices]).flatten()
        return all_grads

    def get_layer_pyr_intn_weights(self, layer_index):
        # return self.model.Y_weights()[layer_index].flatten()
        return self.model.get_parameters("pyr_intn_weight")[layer_index].flatten()

    def get_pyr_intn_weights(self, layer_indices):
        all_pyr_intn_weights = torch.cat([self.get_layer_pyr_intn_weights(layer_index) for layer_index in layer_indices]).flatten()
        return all_pyr_intn_weights

    def get_layer_intn_pyr_weights(self, layer_index):
        # return self.model.Y_weights()[layer_index].flatten()
        return self.model.get_parameters("intn_pyr_weight")[layer_index].flatten()

    def get_intn_pyr_weights(self, layer_indices):
        all_intn_pyr_weights = torch.cat([self.get_layer_intn_pyr_weights(layer_index) for layer_index in layer_indices]).flatten()
        return all_intn_pyr_weights

    def get_layer_grad_angle(self, layer_index, grad_type1, grad_type2):
        key = ("layer_angle", layer_index, grad_type1, grad_type2)
        if key in self._cache:
            return self._cache[key]

        grads1 = self.get_layer_weight_grads(layer_index, grad_type=grad_type1)
        grads2 = self.get_layer_weight_grads(layer_index, grad_type=grad_type2)

        a = similarity_angle(grads1, grads2)
        self._cache[key] = a
        return a

    def get_global_grad_angle(self, layer_indices, grad_type1, grad_type2):
        grads1 = self.get_weight_grads(layer_indices, grad_type=grad_type1)
        grads2 = self.get_weight_grads(layer_indices, grad_type=grad_type2)
        return similarity_angle(grads1, grads2)

    def get_layer_angle_WY(self, layer_index):
        W_weights = self.get_layer_W_weights(layer_index+1)
        Y_weights = self.get_layer_Y_weights(layer_index)
        return similarity_angle(W_weights, Y_weights)

    def get_global_angle_WY(self, layer_indices):
        W_weights = self.get_W_weights([layer_index+1 for layer_index in layer_indices])
        Y_weights = self.get_Y_weights(layer_indices)
        return similarity_angle(W_weights, Y_weights)

    def get_layer_angle_QY(self, layer_index):
        Q_weights = self.get_layer_Q_weights(layer_index)
        Y_weights = self.get_layer_Y_weights(layer_index)
        return similarity_angle(-Q_weights, Y_weights)

    def get_global_angle_QY(self, layer_indices):
        Q_weights = self.get_Q_weights(layer_indices)
        Y_weights = self.get_Y_weights(layer_indices)
        return similarity_angle(-Q_weights, Y_weights)

    def get_layer_angle_W_pyr_intn(self, layer_index):
        W_weights = self.get_layer_W_weights(layer_index+1)
        pyr_intn_weights = self.get_layer_pyr_intn_weights(layer_index)
        return similarity_angle(W_weights, pyr_intn_weights)

    def get_global_angle_W_pyr_intn(self, layer_indices):
        W_weights = self.get_W_weights([layer_index+1 for layer_index in layer_indices])
        pyr_intn_weights = self.get_pyr_intn_weights(layer_indices)
        return similarity_angle(W_weights, pyr_intn_weights)

    def get_layer_angle_Y_intn_pyr(self, layer_index):
        Y_weights = self.get_layer_Y_weights(layer_index)
        intn_pyr_weights = self.get_layer_intn_pyr_weights(layer_index)
        return similarity_angle(Y_weights, intn_pyr_weights)

    def get_global_angle_Y_intn_pyr(self, layer_indices):
        Y_weights = self.get_Y_weights(layer_indices)
        intn_pyr_weights = self.get_intn_pyr_weights(layer_indices)
        return similarity_angle(Y_weights, intn_pyr_weights)

    def get_layer_grad_norm(self, layer_index, grad_type='default'):
        key = ("layer_norm", layer_index, grad_type)
        if key in self._cache:
            return self._cache[key]

        grads = self.get_layer_weight_grads(layer_index, grad_type=grad_type)
        n = grads.norm()

        self._cache[key] = n
        return n

    def get_global_grad_norm(self, layer_indices, grad_type='default'):
        grads = self.get_weight_grads(layer_indices, grad_type=grad_type)
        return grads.norm()

    def get_global_grad_norm_ratio(self, layer_indices, grad_type1, grad_type2):
        norm1 = self.get_global_grad_norm(layer_indices, grad_type=grad_type1)
        norm2 = self.get_global_grad_norm(layer_indices, grad_type=grad_type2)
        return norm1 / norm2

    def get_layer_grad_norm_ratio(self, layer_index, grad_type1, grad_type2):
        norm1 = self.get_layer_grad_norm(layer_index, grad_type=grad_type1)
        norm2 = self.get_layer_grad_norm(layer_index, grad_type=grad_type2)
        return norm1 / norm2

    def get_layer_grad_angle_bp(self, layer_index):
        return self.get_layer_grad_angle(layer_index, grad_type1="default", grad_type2="bp")

    def get_global_grad_angle_bp(self, layer_indices):
        return self.get_global_grad_angle(layer_indices, grad_type1="default", grad_type2="bp")

    def get_layer_grad_angle_fa(self, layer_index):
        return self.get_layer_grad_angle(layer_index, grad_type1="default", grad_type2="fa")

    def get_global_grad_angle_fa(self, layer_indices):
        return self.get_global_grad_angle(layer_indices, grad_type1="default", grad_type2="fa")

    def get_global_grad_norm_bp(self, layer_indices):
        return self.get_global_grad_norm(layer_indices, grad_type='bp')

    def get_global_grad_norm_fa(self, layer_indices):
        return self.get_global_grad_norm(layer_indices, grad_type='fa')

    def get_layer_grad_norm_bp(self, layer_index):
        return self.get_layer_grad_norm(layer_index, grad_type='bp')

    def get_layer_grad_norm_fa(self, layer_index):
        return self.get_layer_grad_norm(layer_index, grad_type='fa')

    def get_layer_grad_norm_Y(self, layer_index):
        grads = self.get_layer_Y_weight_grads(layer_index)
        return grads.norm()

    def get_global_grad_norm_Y(self, layer_indices):
        grads = self.get_Y_weight_grads(layer_indices)
        return grads.norm()

    def get_global_grad_norm_ratio_bp(self, layer_indices):
        return self.get_global_grad_norm_ratio(layer_indices, grad_type1='default', grad_type2='bp')

    def get_global_grad_norm_ratio_fa(self, layer_indices):
        return self.get_global_grad_norm_ratio(layer_indices, grad_type1='default', grad_type2='fa')

    def get_layer_grad_norm_ratio_bp(self, layer_index):
        return self.get_layer_grad_norm_ratio(layer_index, grad_type1='default', grad_type2='bp')

    def get_layer_grad_norm_ratio_fa(self, layer_index):
        return self.get_layer_grad_norm_ratio(layer_index, grad_type1='default', grad_type2='fa')

    def get_layer_apical(self, layer_index):
        layer = self.model.get_layer(layer_index)
        layer_apical = layer.get_state('apic')

        return _flatten_state(layer_apical)

    def get_layer_burst_prob_change(self, layer_index):
        layer = self.model.get_layer(layer_index)

        layer_burst_prob = layer.get_state('p')
        layer_burst_prob_teacher = layer.get_state('p_t')
        layer_burst_prob_change = layer_burst_prob_teacher - layer_burst_prob
        return _flatten_state(layer_burst_prob_change)

    def get_layer_burst_rate_change(self, layer_index):
        layer = self.model.get_layer(layer_index)

        layer_burst_rate = layer.get_state('b')
        layer_burst_rate_teacher = layer.get_state('b_t')
        layer_burst_rate_change = layer_burst_rate_teacher - layer_burst_rate
        return _flatten_state(layer_burst_rate_change)

    def get_layer_event_rate(self, layer_index):
        layer = self.model.get_layer(layer_index)
        layer_event_rate = layer.get_state('e')
        return _flatten_state(layer_event_rate)

    def get_layer_event_rate_deriv(self, layer_index):
        layer = self.model.get_layer(layer_index)
        # layer_event_rate = layer.get_state('e')
        layer_soma = layer.get_state('soma')
        layer_event_rate_deriv = layer.f_deriv(layer_soma)
        return _flatten_state(layer_event_rate_deriv)

    def get_global_apical_magnitude(self, layer_indices):
        layer_apicals = torch.cat([self.get_layer_apical(layer_index) for layer_index in layer_indices], dim=1)
        return layer_apicals.abs().mean()

    def get_layer_apical_magnitude(self, layer_index):
        layer_apical = self.get_layer_apical(layer_index)
        return layer_apical.abs().mean()

    def get_global_apical_variance(self, layer_indices):
        layer_apicals = torch.cat([self.get_layer_apical(layer_index) for layer_index in layer_indices], dim=1)
        return layer_apicals.var()

    def get_layer_apical_variance(self, layer_index):
        layer_apical = self.get_layer_apical(layer_index)
        return layer_apical.var()

    def get_global_apical_max(self, layer_indices):
        layer_apicals = torch.cat([self.get_layer_apical(layer_index) for layer_index in layer_indices], dim=1)
        return layer_apicals.abs().max()

    def get_layer_apical_max(self, layer_index):
        layer_apical = self.get_layer_apical(layer_index)
        return layer_apical.abs().max()

    def get_global_burst_prob_change_magnitude(self, layer_indices):
        layer_burst_prob_changes = torch.cat([self.get_layer_burst_prob_change(layer_index) for layer_index in layer_indices], dim=1)
        return layer_burst_prob_changes.abs().mean()

    def get_layer_burst_prob_change_magnitude(self, layer_index):
        layer_burst_prob_change = self.get_layer_burst_prob_change(layer_index)
        return layer_burst_prob_change.abs().mean()

    def top95_abs(self, input):
        abs = input.abs()
        top95 = abs.quantile(0.95)
        top95_abs = abs[abs > top95]
        return top95_abs

    def get_layer_burst_prob_change_magnitude_top95(self, layer_index):
        layer_burst_prob_change = self.get_layer_burst_prob_change(layer_index)
        return self.top95_abs(layer_burst_prob_change).mean()

    def get_global_burst_prob_change_variance(self, layer_indices):
        layer_burst_prob_changes = torch.cat([self.get_layer_burst_prob_change(layer_index) for layer_index in layer_indices], dim=1)
        return layer_burst_prob_changes.var()

    def get_layer_burst_prob_change_variance(self, layer_index):
        layer_burst_prob_change = self.get_layer_burst_prob_change(layer_index)
        return layer_burst_prob_change.var()

    def get_global_burst_prob_change_max(self, layer_indices):
        layer_burst_prob_changes = torch.cat([self.get_layer_burst_prob_change(layer_index) for layer_index in layer_indices], dim=1)
        return layer_burst_prob_changes.abs().max()

    def get_layer_burst_prob_change_max(self, layer_index):
        layer_burst_prob_change = self.get_layer_burst_prob_change(layer_index)
        return layer_burst_prob_change.abs().max()

    def get_global_burst_rate_change_magnitude(self, layer_indices):
        layer_burst_rate_changes = torch.cat([self.get_layer_burst_rate_change(layer_index) for layer_index in layer_indices], dim=1)
        return layer_burst_rate_changes.abs().mean()

    def get_layer_burst_rate_change_magnitude(self, layer_index):
        layer_burst_rate_change = self.get_layer_burst_rate_change(layer_index)
        return layer_burst_rate_change.abs().mean()

    def get_global_burst_rate_change_variance(self, layer_indices):
        layer_burst_rate_changes = torch.cat([self.get_layer_burst_rate_change(layer_index) for layer_index in layer_indices], dim=1)
        return layer_burst_rate_changes.var()

    def get_layer_burst_rate_change_variance(self, layer_index):
        layer_burst_rate_change = self.get_layer_burst_rate_change(layer_index)
        return layer_burst_rate_change.var()

    def get_global_burst_rate_change_max(self, layer_indices):
        layer_burst_rate_changes = torch.cat([self.get_layer_burst_rate_change(layer_index) for layer_index in layer_indices], dim=1)
        return layer_burst_rate_changes.abs().max()

    def get_layer_burst_rate_change_max(self, layer_index):
        layer_burst_rate_change = self.get_layer_burst_rate_change(layer_index)
        return layer_burst_rate_change.abs().max()

    def get_global_event_rate_variance(self, layer_indices):
        layer_event_rates = torch.cat([self.get_layer_event_rate(layer_index) for layer_index in layer_indices], dim=1)
        return layer_event_rates.var()

    def get_layer_event_rate_variance(self, layer_index):
        layer_event_rate = self.get_layer_event_rate(layer_index)
        return layer_event_rate.var()

    def get_global_event_rate_derivative_mean(self, layer_indices):
        layer_event_rates_derivatives = torch.cat([self.get_layer_event_rate_deriv(layer_index) for layer_index in layer_indices], dim=1)
        # layer_event_rates_derivatives = layer_event_rates * (1 - layer_event_rates)
        return layer_event_rates_derivatives.mean()

    def get_layer_event_rate_derivative_mean(self, layer_index):
        layer_event_rate_derivatives = self.get_layer_event_rate_deriv(layer_index)
        # layer_event_rate_derivatives = layer_event_rate * (1 - layer_event_rate)
        return layer_event_rate_derivatives.mean()

    def get_global_event_rate_saturation_factor(self, layer_indices):
        # layer_event_rates = torch.cat([self.get_layer_event_rate(layer_index) for layer_index in layer_indices], dim=1)
        # saturated = (layer_event_rates <= 0.01) | (layer_event_rates >= 0.99)

        layer_event_rates_derivatives = torch.cat([self.get_layer_event_rate_deriv(layer_index) for layer_index in layer_indices], dim=1)
        deriv_threshold = 1e-2
        saturated = layer_event_rates_derivatives.abs() < deriv_threshold
        return saturated.float().mean()

    def get_layer_event_rate_saturation_factor(self, layer_index):
        # layer_event_rate = self.get_layer_event_rate(layer_index)
        # saturated = (layer_event_rate <= 0.01) | (layer_event_rate >= 0.99)

        layer_event_rate_derivatives = self.get_layer_event_rate_deriv(layer_index)
        deriv_threshold = 1e-2
        saturated = layer_event_rate_derivatives.abs() < deriv_threshold
        return saturated.float().mean()

    # ----------------------------
    # Weight norms (layer + global)
    # ----------------------------
    def get_layer_W_norm(self, layer_index, p=2):
        W = self.get_layer_W_weights(layer_index)
        return W.norm(p=p)

    def get_global_W_norm(self, layer_indices, p=2):
        W = self.get_W_weights(layer_indices)
        return W.norm(p=p)

    def get_layer_Y_norm(self, layer_index, p=2):
        Y = self.get_layer_Y_weights(layer_index)
        return Y.norm(p=p)

    def get_global_Y_norm(self, layer_indices, p=2):
        Y = self.get_Y_weights(layer_indices)
        return Y.norm(p=p)

    def get_layer_Q_norm(self, layer_index, p=2):
        Q = self.get_layer_Q_weights(layer_index)
        return Q.norm(p=p)

    def get_global_Q_norm(self, layer_indices, p=2):
        Q = self.get_Q_weights(layer_indices)
        return Q.norm(p=p)

    def get_global_grad_angle_average(self, layer_indices, grad_type1, grad_type2, weight_by="norm", eps=1e-12, weighted=False):
        angles = []
        weights = []

        for li in layer_indices:
            a = self.get_layer_grad_angle(li, grad_type1=grad_type1, grad_type2=grad_type2)
            w = self.get_layer_grad_norm(li, grad_type=grad_type1)

            if weight_by == "norm_sq":
                w = w * w
            elif weight_by != "norm":
                raise ValueError(f"Unknown weight_by: {weight_by}")

            angles.append(a)
            weights.append(w)

        angles = torch.tensor(angles)
        weights = torch.tensor(weights)

        if weighted:
            denom = weights.sum().clamp_min(eps)
            return ((angles * weights).sum() / denom).item()
        else:
            return angles.mean().item()

    def get_global_grad_angle_average_bp(self, layer_indices, weight_by="norm"):
        return self.get_global_grad_angle_average(layer_indices, grad_type1="default", grad_type2="bp",
                                                   weight_by=weight_by)

    def get_global_grad_angle_average_fa(self, layer_indices, weight_by="norm"):
        return self.get_global_grad_angle_average(layer_indices, grad_type1="default", grad_type2="fa",
                                                   weight_by=weight_by)

    def get_global_grad_angle_weighted_bp(self, layer_indices, weight_by="norm"):
        return self.get_global_grad_angle_average(layer_indices, grad_type1="default", grad_type2="bp",
                                                   weight_by=weight_by, weighted=True)

    def get_global_grad_angle_weighted_fa(self, layer_indices, weight_by="norm"):
        return self.get_global_grad_angle_average(layer_indices, grad_type1="default", grad_type2="fa",
                                                   weight_by=weight_by, weighted=True)
