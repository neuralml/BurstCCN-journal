import torch
from torch import nn

from burstccn.models.networks.base import NetworkBase
from burstccn.models.networks.layers.edn_layers import EDNHiddenLayer, EDNIdealOutputLayer


class EDNLayerBuilder:
    def create_hidden_layer(self, in_features, out_features, next_features, **layer_kwargs):
        return EDNHiddenLayer(in_features, out_features,
                              next_features, **layer_kwargs)

    def create_output_layer(self, in_features, out_features, **layer_kwargs):
        return EDNIdealOutputLayer(in_features, out_features, **layer_kwargs)


class EDNBase(NetworkBase):
    def __init__(self, cfg, network_factory, forward_policy, backward_policy):
        super().__init__()

        self.cfg = cfg
        self.n_outputs = cfg.n_outputs

        self.lambda_intn = cfg.lambda_intn
        self.lambda_hidden = cfg.lambda_hidden
        self.lambda_output = cfg.lambda_hidden

        assert cfg.W.mode in ["random_init", "load_init"]
        assert cfg.Y.mode in ["sym_W", "tied_W", "random_init", "load_init"]
        assert cfg.pyr_intn.mode in ["sym_W", "tied_W", "random_init", "load_init"]
        assert cfg.intn_pyr.mode in ["sym_Y", "tied_Y", "random_init", "load_init"]

        self.W_mode = cfg.W.mode
        if self.W_mode == "random_init":
            self.W_scale = cfg.W.scale
        elif self.W_mode == "load_init":
            self.W_load_file = cfg.W.load_file

        self.Y_mode = cfg.Y.mode
        if self.Y_mode == "random_init":
            self.Y_scale = cfg.Y.scale
        elif self.Y_mode == "load_init":
            self.Y_load_file = cfg.Y.load_file

        self.pyr_intn_mode = cfg.pyr_intn.mode
        if self.pyr_intn_mode == "random_init":
            self.pyr_intn_scale = cfg.pyr_intn.scale
        elif self.pyr_intn_mode == "load_init":
            self.pyr_intn_load_file = cfg.pyr_intn.load_file

        self.intn_pyr_mode = cfg.intn_pyr.mode
        if self.intn_pyr_mode == "random_init":
            self.intn_pyr_scale = cfg.intn_pyr.scale
        elif self.intn_pyr_mode == "load_init":
            self.intn_pyr_load_file = cfg.intn_pyr.load_file

        self.hidden_activation_function = cfg.hidden_activation_function
        self.output_activation_function = cfg.output_activation_function

        self.forward_noise = getattr(cfg, "forward_noise", None)
        self.use_teacher = cfg.use_teacher

        self.optimiser_info = {
            prefix: cfg
            for prefix, cfg in [
                ("W", getattr(self.cfg.W, "optimiser", None)),
                ("Y", getattr(self.cfg.Y, "optimiser", None)),
                ("pyr_intn", getattr(self.cfg.pyr_intn, "optimiser", None)),
                ("intn_pyr", getattr(self.cfg.intn_pyr, "optimiser", None)),
            ]
        }

        self.W_learning = self.optimiser_info['W'] is not None
        self.Y_learning = self.optimiser_info['Y'] is not None
        self.pyr_intn_learning = self.optimiser_info['pyr_intn'] is not None
        self.intn_pyr_learning = self.optimiser_info['intn_pyr'] is not None

        hidden_kwargs = dict(Y_learning=self.Y_learning, pyr_intn_learning=self.pyr_intn_learning,
                                        intn_pyr_learning=self.intn_pyr_learning, lambda_intn=self.lambda_intn,
                                        lambda_hidden=self.lambda_hidden,
                                        activation_function=self.hidden_activation_function)

        output_kwargs = dict(lambda_output=self.lambda_output,
                                        activation_function=self.output_activation_function)

        layer_kwargs = network_factory.build_layer_kwargs(hidden_kwargs=hidden_kwargs, output_kwargs=output_kwargs)

        layer_builder = EDNLayerBuilder()
        self.layers = network_factory.create_layers(layer_builder, layer_kwargs)
        self.forward_policy = forward_policy
        self.backward_policy = backward_policy

        self.loggable_state_types = {'angle_bp', 'angle_fa', 'angle_WY', 'angle_W_pyr_intn', 'angle_Y_intn_pyr',
                                     'grad_norm', 'grad_norm_bp', 'grad_norm_ratio_bp',
                                     'grad_norm_fa', 'grad_norm_ratio_fa',
                                     'event_rate_variance', 'W_norm', 'Y_norm'
                                     #, 'event_rate_derivative_mean', 'event_rate_saturation_factor'
                                     }

        self._init_parameters()
        self.apply_weight_constraints()

    def _init_parameters(self):
        self._init_W_parameters()
        self._init_Y_parameters()
        self._init_intn_parameters()

    def _init_Y_parameters(self):
        if self.Y_mode in ["sym_W", "tied_W"]:
            self._init_Y_parameters_sym_W()
        elif self.Y_mode == "random_init":
            self._init_Y_parameters_random()

    @torch.no_grad()
    def _init_Y_parameters_sym_W(self):
        for layer, next_layer in zip(self.get_layers()[:-1], self.get_layers()[1:]):
            layer.Y_weight.copy_(next_layer.W_weight)

    @torch.no_grad()
    def _init_Y_parameters_random(self):
        for layer in self.get_layers()[:-1]:
            # nn.init.normal_(layer.Y_weight, 0, self.Y_scale)
            nn.init.xavier_normal_(layer.Y_weight, gain=self.W_scale * self.Y_scale)

    def _init_intn_parameters(self):
        if self.pyr_intn_mode in ["sym_W", "tied_W"]:
            self._init_pyr_intn_parameters_sym_W()
        elif self.pyr_intn_mode == "random_init":
            self._init_pyr_intn_parameters_random()

        if self.intn_pyr_mode in ["sym_Y", "tied_Y"]:
            self._init_intn_pyr_parameters_sym_Y()
        elif self.intn_pyr_mode == "random_init":
            self._init_intn_pyr_parameters_random()

    @torch.no_grad()
    def _init_pyr_intn_parameters_sym_W(self):
        for layer, next_layer in zip(self.get_layers()[:-1], self.get_layers()[1:]):
            layer.pyr_intn_weight.copy_(next_layer.W_weight)
            layer.pyr_intn_bias.copy_(next_layer.W_bias)

    @torch.no_grad()
    def _init_pyr_intn_parameters_random(self):
        for layer in self.get_layers()[:-1]:
            if layer.activation_function == 'sigmoid':
                nn.init.xavier_normal_(layer.pyr_intn_weight, gain=self.pyr_intn_scale)
                nn.init.constant_(layer.pyr_intn_bias, 0)
            elif layer.activation_function == 'relu':
                nn.init.kaiming_normal_(layer.pyr_intn_weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.pyr_intn_bias, 0)
            elif layer.activation_function == 'softmax':
                nn.init.xavier_normal_(layer.pyr_intn_weight, gain=self.W_scale)
                nn.init.constant_(layer.pyr_intn_bias, 0)
            else:
                raise NotImplementedError(layer.activation_function)

            # nn.init.normal_(layer.pyr_intn_weight, 0, self.pyr_intn_scale)
            # nn.init.constant_(layer.pyr_intn_bias, 0)

    @torch.no_grad()
    def _init_intn_pyr_parameters_sym_Y(self):
        for layer in self.get_layers()[:-1]:
            layer.intn_pyr_weight.copy_(layer.Y_weight)

    @torch.no_grad()
    def _init_intn_pyr_parameters_random(self):
        for layer in self.get_layers()[:-1]:
            # nn.init.normal_(layer.intn_pyr_weight, 0, self.intn_pyr_scale)
            nn.init.xavier_normal_(layer.pyr_intn_weight, gain=self.W_scale * self.intn_pyr_scale)


    @torch.no_grad()
    def apply_weight_constraints(self):
        layers = self.get_layers()

        for layer, next_layer in zip(layers[:-1], layers[1:]):
            if self.pyr_intn_mode == 'tied_W':
                layer.pyr_intn_weight.copy_(next_layer.W_weight)
                layer.pyr_intn_bias.copy_(next_layer.W_bias)
            if self.Y_mode == 'tied_W':
                layer.Y_weight.copy_(next_layer.W_weight)
            if self.intn_pyr_mode == 'tied_Y':
                layer.intn_pyr_weight.copy_(layer.Y_weight)
