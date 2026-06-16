import torch
from torch import nn

from burstccn.models.networks.base import NetworkBase
from burstccn.models.networks.layers.burstprop_layers import (BurstpropHiddenLayer, BurstpropOutputLayer,
                                                                  BurstpropConvHiddenLayer, BurstpropConvFinalLayer)


class BurstpropLayerBuilder:
    def create_hidden_layer(self, in_features, out_features, next_features, **layer_kwargs):
        return BurstpropHiddenLayer(in_features, out_features,
                                    next_features, **layer_kwargs)

    def create_output_layer(self, in_features, out_features, **layer_kwargs):
        return BurstpropOutputLayer(in_features, out_features, **layer_kwargs)

    def create_hidden_conv_layer(self, in_channels, out_channels, in_shape, kernel_size,
                                 stride, padding, dilation, groups, padding_mode,
                                 next_channels, next_kernel_size, next_stride, **layer_kwargs):
        return BurstpropConvHiddenLayer(in_channels=in_channels,
                                        out_channels=out_channels,
                                        in_shape=in_shape,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        padding_mode=padding_mode,
                                        next_channels=next_channels,
                                        next_kernel_size=next_kernel_size,
                                        next_stride=next_stride,
                                        **layer_kwargs)

    def create_final_conv_layer(self, in_channels, out_channels, in_shape, kernel_size,
                                stride, padding, dilation, groups, padding_mode, next_features, **layer_kwargs):
        return BurstpropConvFinalLayer(in_channels=in_channels,
                                       out_channels=out_channels,
                                       in_shape=in_shape,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       padding_mode=padding_mode,
                                       next_features=next_features,
                                       **layer_kwargs)


class BurstpropBase(NetworkBase):
    def __init__(self, cfg, network_factory, forward_policy, backward_policy):
        super().__init__()
        self.cfg = cfg

        self.n_outputs = cfg.n_outputs
        self.p_baseline = cfg.p_baseline
        self.error_scale = cfg.error_scale
        assert 0 < self.error_scale <= 1.0

        self.W_mode = cfg.W.mode
        assert self.W_mode in ["random_init", "load_init"]

        if self.W_mode == "random_init":
            self.W_scale = cfg.W.scale
        elif self.W_mode == "load_init":
            self.W_load_file = cfg.W.load_file

        assert cfg.Y.mode in ["sym_W", "tied_W", "random_init"]
        assert cfg.rec.mode in ["random_init", "none", "load_init"]

        self.Y_mode = cfg.Y.mode
        self.Y_scale = cfg.Y.scale

        self.rec_mode = cfg.rec.mode
        if self.rec_mode == "random_init":
            self.rec_scale = cfg.rec.scale
        elif self.rec_mode == "load_init":
            self.rec_load_file = cfg.rec.load_file

        self.rec_input = self.rec_mode != 'none'

        self.hidden_activation_function = cfg.hidden_activation_function
        self.output_activation_function = cfg.output_activation_function
        self.output_activity_shift = cfg.get('output_activity_shift', None)

        self.forward_noise = getattr(cfg, "forward_noise", None)
        self.use_teacher = cfg.use_teacher

        self.optimiser_info = self._get_optimiser_info()
        self.network_factory = network_factory
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy

        layer_builder = self._get_layer_builder()
        layer_kwargs = self._get_layer_kwargs()
        self.layers = network_factory.create_layers(layer_builder, layer_kwargs)

        self.loggable_state_types = {'angle_bp', 'angle_fa', 'angle_WY',
                                     'grad_norm', 'grad_norm_bp', 'grad_norm_ratio_bp',
                                     'grad_norm_fa', 'grad_norm_ratio_fa',
                                     'apical_magnitude', 'apical_variance',
                                     'burst_prob_change_magnitude', 'burst_prob_change_variance',
                                     'burst_rate_change_magnitude', 'burst_rate_change_variance',
                                     'event_rate_variance', 'event_rate_derivative_mean',
                                     'event_rate_saturation_factor'}

        self._init_parameters()
        self.apply_weight_constraints()

    def _get_optimiser_info(self):
        optimiser_info = {
            prefix: cfg
            for prefix, cfg in [
                ("W", getattr(self.cfg.W, "optimiser", None)),
                ("Y", getattr(self.cfg.Y, "optimiser", None)),
                ("rec", getattr(self.cfg.rec, "optimiser", None)),
            ]
        }

        self.W_learning = optimiser_info['W'] is not None
        self.Y_learning = optimiser_info['Y'] is not None
        self.rec_learning = optimiser_info['rec'] is not None
        return optimiser_info

    def _get_layer_builder(self):
        layer_builder = BurstpropLayerBuilder()
        return layer_builder

    def _get_layer_kwargs(self):
        hidden_kwargs = dict(p_baseline=self.p_baseline, Y_learning=self.Y_learning,
                             rec_input=self.rec_input, rec_learning=self.rec_learning,
                             activation_function=self.hidden_activation_function)
        output_kwargs = dict(p_baseline=self.p_baseline, error_scale=self.error_scale,
                             activation_function=self.output_activation_function,
                             activity_shift=self.output_activity_shift)

        local_feedback_scales = list(self.cfg.local_feedback_scales) if "local_feedback_scales" in self.cfg else None

        layer_lists = {
            "local_feedback_scale": local_feedback_scales
        }
        n = max(len(v) if v else 0 for v in layer_lists.values())
        hidden_kwargs_list = [dict() for _ in range(n)]

        for key, values in layer_lists.items():
            if not values:
                continue
            for i, value in enumerate(values):
                hidden_kwargs_list[i][key] = value

        layer_kwargs = self.network_factory.build_layer_kwargs(hidden_kwargs=hidden_kwargs, output_kwargs=output_kwargs,
                                                               hidden_kwargs_list=hidden_kwargs_list)

        return layer_kwargs

    def _init_parameters(self):
        self._init_W_parameters()
        self._init_Y_parameters()
        if self.rec_input:
            self._init_rec_parameters()

    def _init_Y_parameters(self):
        if self.Y_mode in ["sym_W", "tied_W"]:
            self._init_Y_parameters_sym()
        elif self.Y_mode == 'random_init':
            self._init_Y_parameters_random()
        elif self.Y_mode == 'load_init':
            self._load_parameters('Y', self.Y_load_file)

    @torch.no_grad()
    def _init_Y_parameters_sym(self):
        with torch.no_grad():
            layers = self.get_layers()
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                layer.Y_weight.copy_(self.Y_scale * next_layer.W_weight)

    @torch.no_grad()
    def _init_Y_parameters_random(self):
        with torch.no_grad():
            layers = self.get_layers()
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                # nn.init.normal_(layer.Y_weight, 0, self.Y_scale)
                nn.init.xavier_normal_(layer.Y_weight, gain=self.W_scale * self.Y_scale)

    def _init_rec_parameters(self):
        if self.rec_mode == "random_init":
            self._init_rec_parameters_random()
        elif self.rec_mode == "load_init":
            self._load_parameters('rec', self.rec_load_file)

    @torch.no_grad()
    def _init_rec_parameters_random(self):
        layers = self.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            nn.init.normal_(layer.rec_weight, 0, self.rec_scale)

    @torch.no_grad()
    def apply_weight_constraints(self):
        layers = self.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            if self.Y_mode == 'tied_W':
                layer.Y_weight.copy_(self.Y_scale * next_layer.W_weight.detach())
