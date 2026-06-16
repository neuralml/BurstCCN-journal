import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Callable

from torch.nn.modules.utils import _pair

import torch
from omegaconf import DictConfig

from torch import nn
from burstccn.optimisers import CombinedOptimiser


class NetworkBase(nn.Module, ABC):
    optimiser_info: Dict[str, Optional[DictConfig]]

    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList()

    def get_layers(self):
        return self.layers

    def get_layer(self, layer_index):
        layers = self.get_layers()
        return layers[layer_index]

    def get_dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def forward(self, x, forward_noise=None, **kwargs):
        layers = self.get_layers()
        dtype = self.get_dtype()
        return self.forward_policy(x, layers=layers, forward_noise=forward_noise, dtype=dtype, **kwargs)

    def backward(self, target, **kwargs):
        layers = self.get_layers()
        return self.backward_policy(target, layers=layers, **kwargs)

    @abstractmethod
    def _init_parameters(self):
        pass

    @abstractmethod
    def apply_weight_constraints(self):
        pass

    def _load_parameters(self, param_prefix, model_filename):
        model_path = Path("saved_models") / model_filename
        loaded_state_dict = torch.load(model_path, map_location='cpu')
        loaded_state_dict = {k.replace("classification_layers", "layers"): v for k, v in loaded_state_dict.items()}

        weight_params = self.get_parameters(f"{param_prefix}_weight")
        bias_params = self.get_parameters(f"{param_prefix}_bias")

        param_ids = set(id(p) for p in weight_params + bias_params)

        param_name_map = dict(self.named_parameters())
        param_names = [n for n, p in param_name_map.items() if id(p) in param_ids]

        filtered_state_dict = {
            n: loaded_state_dict[n]
            for n in param_names
            if n in loaded_state_dict
        }

        # Error check for missing parameters
        missing = [n for n in param_names if n not in loaded_state_dict]
        if missing:
            raise KeyError(
                f"Missing parameters in {model_path}: {missing} "
                f"(found: {list(filtered_state_dict.keys())})"
            )

        print(f"Loading parameters: {list(filtered_state_dict.keys())} from {model_path}")
        self.load_state_dict(filtered_state_dict, strict=False)

    def get_parameters(self, parameter_name, exclude_layers=None):
        exclude_layers = exclude_layers or []
        parameters = []
        layers = self.get_layers()
        for i, layer in enumerate(layers):
            if i in exclude_layers:
                continue
            param = getattr(layer, parameter_name, None)
            if param is not None:
                parameters.append(param)

        return parameters

    # def create_optimiser(self):
    #     return CombinedOptimiser([
    #         self._create_sub_optimiser(param_prefix, cfg)
    #         for param_prefix, cfg in self.optimiser_info.items()
    #         if cfg is not None
    #     ])

    def create_optimiser(self, lr_scheduler_context=None):
        opt_list = []
        lr_scheduler_list = []
        for param_prefix, cfg in self.optimiser_info.items():
            if cfg is not None:
                opt = self._create_sub_optimiser(param_prefix, cfg)
                if hasattr(cfg, "lr_scheduler"):
                    steps_per_epoch = lr_scheduler_context.get('steps_per_epoch', None)
                    total_epochs = lr_scheduler_context.get('total_epochs', None)
                    assert cfg.lr_scheduler.type == 'cosine_warmup'
                    warmup_epochs = cfg.lr_scheduler.warmup_epochs
                    warmup_steps = warmup_epochs * steps_per_epoch
                    total_steps = total_epochs * steps_per_epoch

                    def lr_lambda(step, warmup_steps=warmup_steps, total_steps=total_steps):
                        if step < warmup_steps:
                            return (step + 1) / warmup_steps  # scales initial lr
                        t = (step - warmup_steps) / (total_steps - warmup_steps)
                        return 0.5 * (1 + math.cos(math.pi * t))  # cosine 1 → 0

                    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
                else:
                    lr_scheduler = None

                opt_list.append(opt)
                lr_scheduler_list.append(lr_scheduler)

        return CombinedOptimiser(optimiser_list=opt_list, lr_scheduler_list=lr_scheduler_list)

    def _create_sub_optimiser(self, param_prefix, cfg):
        fixed_layers = getattr(cfg, "fixed_layers", None)
        weight_params = self.get_parameters(f"{param_prefix}_weight", exclude_layers=fixed_layers)

        bias_params = self.get_parameters(f"{param_prefix}_bias", exclude_layers=fixed_layers)
        # bias_params += self.get_parameters(f"{param_prefix}_gamma", exclude_layers=fixed_layers)

        if not weight_params and not bias_params:
            return None

        if "type" not in cfg:
            raise ValueError("optimiser_config must include 'type' ('sgd' or 'adam').")

        param_groups = []
        if weight_params:
            param_groups.append({
                "params": weight_params,
                "weight_decay": cfg.weight_decay
            })
        if bias_params:
            param_groups.append({
                "params": bias_params,
                "weight_decay": 0.0
            })

        if cfg.type == "sgd":
            momentum = getattr(cfg, "momentum", 0.0)
            return torch.optim.SGD(param_groups, lr=cfg.lr, momentum=momentum)
        elif cfg.type == "adam":
            return torch.optim.Adam(param_groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
        else:
            raise ValueError(f"Unsupported optimiser type: {cfg.type}")

    def _init_W_parameters(self):
        if self.W_mode == 'random_init':
            self._init_W_parameters_random()
        elif self.W_mode == 'load_init':
            self._load_parameters('W', self.W_load_file)
        else:
            raise ValueError('Unknown weight mode: ' + self.W_mode)

    def _init_W_parameters_random(self):
        layers = self.get_layers()
        for layer_index, layer in enumerate(layers):
            if layer.activation_function == 'sigmoid':
                nn.init.xavier_normal_(layer.W_weight, gain=self.W_scale)
                nn.init.constant_(layer.W_bias, 0)
            elif layer.activation_function in ['relu', 'ln_relu', 'relu_ln', 'ms_relu_n', 'relu_rms']:
                # nn.init.kaiming_normal_(layer.W_weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(layer.W_weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(layer.W_bias, 0)
                if hasattr(self, "W_scale"):
                    layer.W_weight.data *= self.W_scale
            elif layer.activation_function in ['ms_softplus_n']:
                nn.init.kaiming_normal_(layer.W_weight, mode='fan_in', nonlinearity='relu')

                # gain = 2.0  # approx compensation for slope ~0.5
                # layer.W_weight.data *= gain / math.sqrt(2)  # adjust from sqrt(2) to 2

                nn.init.constant_(layer.W_bias, 0)
                if hasattr(self, "W_scale"):
                    layer.W_weight.data *= self.W_scale
            elif layer.activation_function == 'softmax':
                nn.init.xavier_normal_(layer.W_weight, gain=self.W_scale)
                # nn.init.normal_(layer.W_weight, mean=0, std=0.001)
                # nn.init.normal_(layer.W_weight, 0, 0.01)

                nn.init.constant_(layer.W_bias, 0)
            else:
                raise NotImplementedError(layer.activation_function)

    def backward_bp(self, target):
        layers = self.get_layers()
        delta_bp = layers[-1].backward_bp(target)
        for layer in layers[-2::-1]:
            delta_bp = layer.backward_bp(delta_bp)

    def backward_fa(self, target):
        layers = self.get_layers()
        next_delta_fa = layers[-1].backward_fa(target)
        for layer in layers[-2::-1]:
            next_delta_fa = layer.backward_fa(next_delta_fa)

    def zero_grad_bp(self):
        layers = self.get_layers()
        for layer in layers:
            if hasattr(layer, 'W_weight'): layer.W_weight.grad_bp = None
            if hasattr(layer, 'W_bias'): layer.W_bias.grad_bp = None

    def zero_grad_fa(self):
        layers = self.get_layers()
        for layer in layers:
            if hasattr(layer, 'W_weight'): layer.W_weight.grad_fa = None
            if hasattr(layer, 'W_bias'): layer.W_bias.grad_fa = None


class AutogradNetwork(NetworkBase, ABC):
    def set_store_grad_bp(self, store_grad_bp):
        layers = self.get_layers()
        for layer in layers:
            layer.store_grad_bp = store_grad_bp


class ManualGradNetwork(NetworkBase, ABC):
    pass


class NetworkFactory(ABC):
    @abstractmethod
    def create_layers(self, layer_builder, layer_kwargs):
        pass

    @abstractmethod
    def build_layer_kwargs(self, **kwargs):
        pass


class FullyConnectedNetworkFactory(NetworkFactory):
    def __init__(self, cfg: DictConfig):
        self.n_inputs = cfg.n_inputs
        self.n_outputs = cfg.n_outputs
        self.n_hidden_layers = cfg.n_hidden_layers
        self.n_hidden_units = cfg.n_hidden_units

    def create_layers(self, layer_builder, layer_kwargs):
        layers = nn.ModuleList()
        if isinstance(self.n_hidden_units, int):
            hidden_sizes = [self.n_hidden_units] * self.n_hidden_layers
        else:
            hidden_sizes = self.n_hidden_units
            assert len(hidden_sizes) == self.n_hidden_layers, \
                f"Expected {self.n_hidden_layers} hidden sizes, got {len(hidden_sizes)}"

        layer_sizes = [self.n_inputs] + hidden_sizes + [self.n_outputs]

        n_layers = len(layer_sizes)
        # Build hidden layers only (exclude last layer which is for output)
        for layer_index in range(n_layers - 2):
            in_size = layer_sizes[layer_index]
            out_size = layer_sizes[layer_index + 1]
            next_size = layer_sizes[layer_index + 2]
            layers.append(layer_builder.create_hidden_layer(in_size, out_size, next_size,
                                                            **layer_kwargs[layer_index]))

        output_index = n_layers - 2
        layers.append(layer_builder.create_output_layer(layer_sizes[-2], self.n_outputs,
                                                        **layer_kwargs[output_index]))
        return layers

    def build_layer_kwargs(self, **kwargs):
        hidden_kwargs = kwargs.get("hidden_kwargs", {})
        output_kwargs = kwargs.get("output_kwargs", {})

        hidden_kwargs_list = kwargs.get("hidden_kwargs_list", None)
        if not hidden_kwargs_list:
            hidden_kwargs_list = [{} for _ in range(self.n_hidden_layers)]

        layer_kwargs = [{**hidden_kwargs, **hidden_kwargs_list[i]} for i in range(self.n_hidden_layers)] + [output_kwargs]

        return layer_kwargs


class ConvNetworkFactory(NetworkFactory):
    def __init__(self, cfg: DictConfig):
        self.in_shape = tuple(cfg.in_shape)
        self.n_outputs = cfg.n_outputs

        def _as_pair_list(vals):
            return [_pair(val) for val in vals]

        self.in_channels = list(cfg.in_channels)
        self.out_channels = list(cfg.out_channels)
        self.next_channels = self.out_channels[1:]
        self.kernel_sizes = _as_pair_list(cfg.kernel_sizes)
        self.next_kernel_sizes = self.kernel_sizes[1:]
        self.strides = _as_pair_list(cfg.strides)
        self.next_strides = self.strides[1:]

        self.paddings = _as_pair_list(cfg.paddings)
        self.dilations = _as_pair_list(cfg.dilations)
        self.groups = list(cfg.groups)

        self.n_conv_layers = len(self.in_channels)
        self.n_hidden_layers = cfg.n_hidden_layers
        self.n_hidden_units = cfg.n_hidden_units

    def create_layers(self, layer_builder, layer_kwargs):
        layers = nn.ModuleList()
        in_shape = self.in_shape

        n_conv_layers = len(self.in_channels)
        for layer_index in range(n_conv_layers - 1):
            layer = layer_builder.create_hidden_conv_layer(in_channels=self.in_channels[layer_index],
                                                           out_channels=self.out_channels[layer_index],
                                                           in_shape=in_shape,
                                                           kernel_size=self.kernel_sizes[layer_index],
                                                           stride=self.strides[layer_index],
                                                           padding=self.paddings[layer_index],
                                                           dilation=self.dilations[layer_index],
                                                           groups=self.groups[layer_index],
                                                           padding_mode='zeros',
                                                           next_channels=self.next_channels[layer_index],
                                                           next_kernel_size=self.next_kernel_sizes[layer_index],
                                                           next_stride=self.next_strides[layer_index],
                                                           **layer_kwargs[layer_index])
            layers.append(layer)
            in_shape = layer.out_shape

        fc_hidden_sizes = [self.n_hidden_units] * self.n_hidden_layers + [self.n_outputs]

        final_conv_layer_index = n_conv_layers - 1
        final_conv_layer = layer_builder.create_final_conv_layer(in_channels=self.in_channels[-1],
                                                                 out_channels=self.out_channels[-1],
                                                                 in_shape=in_shape,
                                                                 kernel_size=self.kernel_sizes[-1],
                                                                 stride=self.strides[-1],
                                                                 padding=self.paddings[-1],
                                                                 dilation=self.dilations[-1],
                                                                 groups=self.groups[-1],
                                                                 padding_mode='zeros',
                                                                 next_features=fc_hidden_sizes[0],
                                                                 **layer_kwargs[final_conv_layer_index])
        layers.append(final_conv_layer)

        fc_hidden_sizes.insert(0, final_conv_layer.out_features)

        for fc_layer_index in range(len(fc_hidden_sizes) - 2):
            in_size = fc_hidden_sizes[fc_layer_index]
            out_size = fc_hidden_sizes[fc_layer_index + 1]
            next_size = fc_hidden_sizes[fc_layer_index + 2]
            hidden_layer_index = n_conv_layers + fc_layer_index
            layers.append(layer_builder.create_hidden_layer(in_size, out_size, next_size,
                                                            **layer_kwargs[hidden_layer_index]))

        output_layer_index = n_conv_layers + self.n_hidden_layers
        layers.append(layer_builder.create_output_layer(fc_hidden_sizes[-2], self.n_outputs,
                                                        **layer_kwargs[output_layer_index]))
        return layers

    def build_layer_kwargs(self, **kwargs):
        hidden_kwargs = kwargs.get("hidden_kwargs", {})
        output_kwargs = kwargs.get("output_kwargs", {})

        hidden_kwargs_list = kwargs.get("hidden_kwargs_list", None)
        total_hidden_layers = self.n_conv_layers + self.n_hidden_layers

        if hidden_kwargs_list is None:
            hidden_kwargs_list = [{} for _ in range(total_hidden_layers)]

        layer_kwargs = [{**hidden_kwargs, **hidden_kwargs_list[i]} for i in range(total_hidden_layers)] + [output_kwargs]

        return layer_kwargs


class ForwardPolicy(Callable, ABC):
    @abstractmethod
    def __call__(self, x, layers, forward_noise=None, dtype=torch.float32, **kwargs):
        pass


class FullyConnectedForwardPolicy(ForwardPolicy):
    def __call__(self, x, layers, forward_noise=None, dtype=torch.float32, **kwargs):
        x = x.to(dtype)
        x = x.view(x.size()[0], -1)

        for layer in layers:
            x = layer.forward(x, forward_noise=forward_noise)

        return x


class ConvForwardPolicy(ForwardPolicy):
    def __call__(self, x, layers, forward_noise=None, dtype=torch.float32, **kwargs):
        x = x.to(dtype)

        for layer in layers:
            x = layer.forward(x, forward_noise=forward_noise)

        return x


class FullyConnectedNPForwardPolicy(ForwardPolicy):
    def __call__(self, x, layers, forward_noise=None, dtype=torch.float32, **kwargs):
        x = x.to(dtype)
        x = x.view(x.size()[0], -1)

        perturb = kwargs['perturb']

        for layer in layers:
            x = layer.forward(x, perturb=perturb)

        return x


class BackwardPolicy(Callable):
    @abstractmethod
    def __call__(self, target, layers, **kwargs):
        pass


class SingleBackwardPolicy(BackwardPolicy):
    def __call__(self, target, layers, **kwargs):
        feedback_state = layers[-1].backward(target)
        for layer in layers[-2::-1]:
            feedback_state = layer.backward(feedback_state)


class MultiBackwardPolicy(BackwardPolicy):
    def __call__(self, target, layers, **kwargs):
        feedback_state = layers[-1].backward(target)
        for layer in layers[-2::-1]:
            feedback_state = layer.backward(*feedback_state)


class NPBackwardPolicy(BackwardPolicy):
    def __call__(self, target, layers, **kwargs):
        loss_pre = kwargs['loss_pre']
        loss_post = kwargs['loss_post']

        for layer in layers[::-1]:
            layer.backward(loss_pre, loss_post)
