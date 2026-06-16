import torch
from torch import nn

from burstccn.models.networks.base import NetworkBase
from burstccn.models.networks.layers.ann_layers import ANNHiddenLayer, ANNOutputLayer, ANNConvHiddenLayer, \
    ANNConvFinalLayer

class ANNLayerBuilder:
    def create_hidden_layer(self, in_features, out_features, next_features, **layer_kwargs):
        return ANNHiddenLayer(in_features, out_features,
                              next_features, **layer_kwargs)

    def create_output_layer(self, in_features, out_features, **layer_kwargs):
        return ANNOutputLayer(in_features, out_features, **layer_kwargs)

    def create_hidden_conv_layer(self, in_channels, out_channels, in_shape, kernel_size,
                                  stride, padding, dilation, groups, padding_mode,
                                  next_channels, next_kernel_size, next_stride, **layer_kwargs):
        return ANNConvHiddenLayer(in_channels=in_channels,
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
        return ANNConvFinalLayer(in_channels=in_channels,
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


class ANNBase(NetworkBase):
    def __init__(self, cfg, network_factory, forward_policy, backward_policy):
        super().__init__()

        self.cfg = cfg
        self.n_outputs = cfg.n_outputs

        assert cfg.W.mode in ["random_init", "load_init"]
        assert cfg.Y.mode in ["sym_W", "tied_W", "random_init", "load_init", "tied_decay_W"]

        self.W_mode = cfg.W.mode
        if self.W_mode == "random_init":
            self.W_scale = cfg.W.scale
        elif self.W_mode == "load_init":
            self.W_load_file = cfg.W.load_file

        self.Y_mode = cfg.Y.mode
        if self.Y_mode in ["sym_Y", "tied_Y", "sym_W", "tied_W", "random_init", "tied_decay_W"]:
            self.Y_scale = cfg.Y.scale
        elif self.Y_mode == "load_init":
            self.Y_load_file = cfg.Y.load_file

        self.hidden_activation_function = cfg.hidden_activation_function
        self.output_activation_function = cfg.output_activation_function

        self.always_use_backprop = False

        self.forward_noise = getattr(cfg, "forward_noise", None)
        self.use_teacher = cfg.use_teacher

        self.optimiser_info = {
            prefix: cfg
            for prefix, cfg in [
                ("W", getattr(self.cfg.W, "optimiser", None)),
                ("Y", getattr(self.cfg.Y, "optimiser", None)),
            ]
        }
        self.W_learning = self.optimiser_info['W'] is not None
        self.Y_learning = self.optimiser_info['Y'] is not None

        hidden_kwargs = dict(Y_learning=self.Y_learning, activation_function=self.hidden_activation_function)
        output_kwargs = dict(activation_function=self.output_activation_function)
        layer_kwargs = network_factory.build_layer_kwargs(hidden_kwargs=hidden_kwargs, output_kwargs=output_kwargs)

        layer_builder = ANNLayerBuilder()
        self.layers = network_factory.create_layers(layer_builder, layer_kwargs)
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy

        # self.loggable_state_types = {'angle_bp', 'angle_fa', 'angle_WY',
        #                              'grad_norm', 'grad_norm_bp', 'grad_norm_ratio_bp',
        #                              'grad_norm_fa', 'grad_norm_ratio_fa',
        #                              'event_rate_variance', 'event_rate_derivative_mean',
        #                              'event_rate_saturation_factor', 'W_norm', 'Y_norm'}

        self.loggable_state_types = {'angle_bp', 'angle_fa', 'angle_WY',
                                     'grad_norm', 'grad_norm_bp', 'grad_norm_ratio_bp',
                                     'grad_norm_fa', 'grad_norm_ratio_fa',
                                     'event_rate_variance', 'W_norm', 'Y_norm'}

        self._init_parameters()
        self.apply_weight_constraints()

    def _init_parameters(self):
        self._init_W_parameters()
        self._init_Y_parameters()

    @torch.no_grad()
    def _init_Y_parameters(self):
        layers = self.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            if self.Y_mode in ["sym_W", "tied_W", "tied_decay_W"]:
                layer.Y_weight.copy_(next_layer.W_weight)

                # if next_layer.W_weight.shape[0] == 10:
                #     layer.Y_weight.copy_(next_layer.W_weight)
                # else:
                #     import math
                #     theta = math.radians(45.0)
                #
                #     Z = torch.empty_like(next_layer.W_weight)
                #     if next_layer.activation_function == 'sigmoid' or next_layer.activation_function == 'softmax':
                #         nn.init.xavier_normal_(Z, gain=self.W_scale)
                #     elif next_layer.activation_function in ['relu', 'ln_relu', 'relu_ln']:
                #         nn.init.kaiming_normal_(Z, mode='fan_in', nonlinearity='relu')
                #
                #     # Misalign by ~10 degrees in weight space, while keeping Xavier-normal marginals
                #     Y = math.cos(theta) * next_layer.W_weight + math.sin(theta) * Z
                #
                #     layer.Y_weight.copy_(Y)

            elif self.Y_mode == "random_init":
                # nn.init.xavier_normal_(layer.Y_weight, gain=self.W_scale * self.Y_scale)
                if next_layer.activation_function == 'sigmoid' or next_layer.activation_function == 'softmax':
                    nn.init.xavier_normal_(layer.Y_weight, gain=self.W_scale * self.Y_scale)
                    # nn.init.normal_(layer.Y_weight, mean=0, std=0.01)
                elif next_layer.activation_function in ['relu', 'ln_relu', 'relu_ln', 'ms_relu_n', 'ms_softplus_n', 'relu_rms']:
                    nn.init.kaiming_normal_(layer.Y_weight, mode='fan_in', nonlinearity='relu')
                    layer.Y_weight.data *= self.W_scale * self.Y_scale
                    # nn.init.normal_(layer.Y_weight, mean=0, std=0.01)


    @torch.no_grad()
    def apply_weight_constraints(self):
        # layers = self.get_layers()
        # layers[-2].Y_weight.copy_(layers[-1].W_weight.detach())

        if self.Y_mode == 'tied_W':
            layers = self.get_layers()
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                layer.Y_weight.copy_(self.Y_scale * next_layer.W_weight.detach())

        elif self.Y_mode == 'tied_decay_W':
            layers = self.get_layers()
            # alpha = 0.99  # float in [0, 1]
            # alpha = 0.99995  # float in [0, 1]
            alpha = 1 - 1e-4
            for layer, next_layer in zip(layers[:-1], layers[1:]):
                layer.Y_weight.mul_(alpha).add_(self.Y_scale * next_layer.W_weight, alpha=(1.0 - alpha))