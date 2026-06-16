import warnings
from abc import abstractmethod

import torch
from torch import nn

from burstccn.models.networks.base import NetworkBase
from burstccn.models.networks.layers.burstccn_layers import BurstCCNHiddenLayer, BurstCCNOutputLayer, \
    BurstCCNConvHiddenLayer, BurstCCNConvFinalLayer
from burstccn.models.networks.layers.burstccn_layers_dales import DalesBurstCCNHiddenLayer, DalesBurstCCNOutputLayer
from burstccn.utils import matrix_factorization


class BurstCCNLayerBuilder:
    def create_hidden_layer(self, in_features, out_features, next_features, **layer_kwargs):
        return BurstCCNHiddenLayer(in_features, out_features,
                                   next_features, **layer_kwargs)

    def create_output_layer(self, in_features, out_features, **layer_kwargs):
        return BurstCCNOutputLayer(in_features, out_features, **layer_kwargs)

    def create_hidden_conv_layer(self, in_channels, out_channels, in_shape, kernel_size,
                                 stride, padding, dilation, groups, padding_mode,
                                 next_channels, next_kernel_size, next_stride, **layer_kwargs):
        return BurstCCNConvHiddenLayer(in_channels=in_channels,
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
        return BurstCCNConvFinalLayer(in_channels=in_channels,
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


class DalesBurstCCNLayerBuilder:
    def create_hidden_layer(self, in_features, out_features, next_features, **layer_kwargs):
        return DalesBurstCCNHiddenLayer(in_features, out_features,
                                        next_features, **layer_kwargs)

    def create_output_layer(self, in_features, out_features, **layer_kwargs):
        return DalesBurstCCNOutputLayer(in_features, out_features, **layer_kwargs)


class BurstCCNBase(NetworkBase):
    def __init__(self, cfg, network_factory, forward_policy, backward_policy):
        super().__init__()

        self.cfg = cfg
        self.n_outputs = cfg.n_outputs

        self.p_baseline = cfg.p_baseline
        self.error_scale = cfg.error_scale
        # assert 0 <= self.error_scale <= 1.0

        assert cfg.W.mode in ["random_init", "load_init"]
        assert cfg.Q.mode in ["sym_Y", "tied_Y", "sym_W", "tied_W", "random_init", "load_init"]
        assert cfg.Y.mode in ["sym_Q", "tied_Q", "sym_W", "tied_W", "random_init", "load_init"]

        self.W_mode = cfg.W.mode
        if self.W_mode == "random_init":
            self.W_scale = cfg.W.scale
        elif self.W_mode == "load_init":
            self.W_load_file = cfg.W.load_file
            self.W_scale = cfg.W.scale  # default assumed

        self.Y_mode = cfg.Y.mode
        if self.Y_mode in ["sym_Q", "tied_Q", "sym_W", "tied_W", "random_init"]:
            self.Y_scale = cfg.Y.scale
        elif self.Y_mode == "load_init":
            self.Y_load_file = cfg.Y.load_file

        self.Q_mode = cfg.Q.mode
        if self.Q_mode in ["sym_Y", "tied_Y", "sym_W", "tied_W", "random_init"]:
            self.Q_scale = cfg.Q.scale
        elif self.Q_mode == "load_init":
            self.Q_load_file = cfg.Q.load_file

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

        # self.loggable_state_types = {'angle_bp', 'angle_fa', 'angle_WY', 'angle_QY', 'grad_norm', 'grad_norm_bp',
        #                              'grad_norm_ratio_bp', 'grad_norm_fa', 'grad_norm_ratio_fa',
        #                              'apical_magnitude', 'apical_variance',
        #                              'burst_prob_change_magnitude', 'burst_prob_change_variance',# 'burst_prob_change_magnitude_top95',
        #                              'burst_rate_change_magnitude', 'burst_rate_change_variance',
        #                              'event_rate_variance', 'event_rate_derivative_mean', 'event_rate_saturation_factor'
        #                              }

        self.loggable_state_types = {'angle_bp', 'angle_fa', 'angle_WY', 'angle_QY', 'grad_norm', 'grad_norm_bp',
                                     'grad_norm_ratio_bp', 'grad_norm_fa', 'grad_norm_ratio_fa',
                                     'apical_magnitude', 'apical_variance', 'apical_max',
                                     'burst_prob_change_magnitude', 'burst_prob_change_variance', 'burst_prob_change_max',
                                     'burst_rate_change_magnitude', 'burst_rate_change_variance', 'burst_rate_change_max',
                                     'event_rate_variance', 'W_norm', 'Y_norm', 'Q_norm', 'grad_norm_Y'}

        self._init_parameters()
        self.apply_weight_constraints()

    @abstractmethod
    def _get_optimiser_info(self):
        pass

    @abstractmethod
    def _get_layer_builder(self):
        pass

    @abstractmethod
    def _get_layer_kwargs(self):
        pass

    def _init_parameters(self):
        self._init_W_parameters()
        self._init_Q_Y_parameters()

    @abstractmethod
    def _init_Q_Y_parameters(self):
        pass

    @abstractmethod
    def _init_Q_parameters_random(self):
        pass

    @abstractmethod
    def _init_Q_parameters_sym_W(self):
        pass

    @abstractmethod
    def _init_Q_parameters_sym_Y(self):
        pass

    @abstractmethod
    def _init_Y_parameters_random(self):
        pass

    @abstractmethod
    def _init_Y_parameters_sym_W(self):
        pass

    @abstractmethod
    def _init_Y_parameters_sym_Q(self):
        pass

    @abstractmethod
    def apply_weight_constraints(self):
        pass


class BurstCCN(BurstCCNBase):
    def _get_optimiser_info(self):
        optimiser_info = {
            prefix: cfg
            for prefix, cfg in [
                ("W", getattr(self.cfg.W, "optimiser", None)),
                ("Y", getattr(self.cfg.Y, "optimiser", None)),
                ("Q", getattr(self.cfg.Q, "optimiser", None)),
            ]
        }
        self.W_learning = optimiser_info['W'] is not None
        self.Y_learning = optimiser_info['Y'] is not None
        self.Q_learning = optimiser_info['Q'] is not None
        return optimiser_info

    def _get_layer_builder(self):
        layer_builder = BurstCCNLayerBuilder()
        return layer_builder

    def _get_layer_kwargs(self):
        y_grad_type = getattr(self.cfg.Y, "grad_type", None)
        q_grad_type = getattr(self.cfg.Q, "grad_type", None)
        store_exc_inh_branch_state = getattr(self.cfg, "store_exc_inh_branch_state", False)

        hidden_kwargs = dict(p_baseline=self.p_baseline, Y_learning=self.Y_learning,
                             Q_learning=self.Q_learning, Y_grad_type=y_grad_type, Q_grad_type=q_grad_type,
                             activation_function=self.hidden_activation_function, store_exc_inh_branch_state=store_exc_inh_branch_state)
        output_kwargs = dict(p_baseline=self.p_baseline, error_scale=self.error_scale,
                             activation_function=self.output_activation_function,
                             activity_shift=self.output_activity_shift)

        local_feedback_scales = list(self.cfg.local_feedback_scales) if "local_feedback_scales" in self.cfg else None
        n_apical_branches = list(self.cfg.n_apical_branches) if "n_apical_branches" in self.cfg else None

        layer_lists = {
            "local_feedback_scale": local_feedback_scales,
            "n_apical_branches": n_apical_branches,
        }
        n = max(len(v) if v else 0 for v in layer_lists.values())
        hidden_kwargs_list = [dict() for _ in range(n)]

        for key, values in layer_lists.items():
            if not values:
                continue
            for i, value in enumerate(values):
                hidden_kwargs_list[i][key] = value

        # hidden_kwargs_list = [dict(local_feedback_scale=local_feedback_scale) for local_feedback_scale in
        #                       local_feedback_scales] if local_feedback_scales is not None else None

        layer_kwargs = self.network_factory.build_layer_kwargs(hidden_kwargs=hidden_kwargs, output_kwargs=output_kwargs,
                                                               hidden_kwargs_list=hidden_kwargs_list)
        return layer_kwargs

    def _init_Q_Y_parameters(self):
        # First pass without cross-dependencies
        if self.Q_mode in ["sym_W", "tied_W"]:
            self._init_Q_parameters_sym_W()
        elif self.Q_mode == 'random_init':
            self._init_Q_parameters_random()
        elif self.Q_mode == 'load_init':
            self._load_parameters('Q', self.Q_load_file)

        if self.Y_mode in ["sym_W", "tied_W"]:
            self._init_Y_parameters_sym_W()
        elif self.Y_mode == 'random_init':
            self._init_Y_parameters_random()
        elif self.Y_mode == 'load_init':
            self._load_parameters('Y', self.Y_load_file)

        # Second pass with cross-dependencies
        if self.Q_mode in ["sym_Y", "tied_Y"]:
            self._init_Q_parameters_sym_Y()
        if self.Y_mode in ["sym_Q", "tied_Q"]:
            self._init_Y_parameters_sym_Q()

    @torch.no_grad()
    def _init_Q_parameters_random(self):
        layers = self.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            # nn.init.normal_(layer.Q_weight, 0, self.Q_scale)
            nn.init.xavier_normal_(layer.Q_weight, gain=self.W_scale * self.Q_scale * next_layer.p_baseline)

    @torch.no_grad()
    def _init_Q_parameters_sym_W(self):
        layers = self.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            layer.Q_weight.copy_(-next_layer.W_weight * next_layer.p_baseline)

    @torch.no_grad()
    def _init_Q_parameters_sym_Y(self):
        layers = self.get_layers()
        for layer in layers[:-1]:
            layer.Q_weight.copy_(-layer.Y_weight * layer.p_baseline)

    @torch.no_grad()
    def _init_Y_parameters_random(self):
        layers = self.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            # nn.init.normal_(layer.Y_weight, 0, self.Y_scale)
            # nn.init.xavier_normal_(layer.Y_weight, gain=self.W_scale * self.Y_scale)
            # if next_layer
            # nn.init.kaiming_normal_(layer.Y_weight, mode='fan_in', nonlinearity='relu')

            if next_layer.activation_function == 'sigmoid':
                nn.init.xavier_normal_(layer.Y_weight, gain=self.W_scale * self.Y_scale)
            elif next_layer.activation_function in ['relu', 'ln_relu', 'relu_ln', 'ms_relu_n', 'ms_softplus_n', 'relu_rms']:
                # nn.init.kaiming_normal_(next_layer.Y_weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(layer.Y_weight, mode='fan_in', nonlinearity='relu')
                if hasattr(self, "W_scale"):
                    layer.Y_weight.data *= self.W_scale * self.Y_scale
            elif next_layer.activation_function == 'softmax':
                nn.init.xavier_normal_(layer.Y_weight, gain=self.W_scale * self.Y_scale)
            else:
                raise NotImplementedError(next_layer.activation_function)


    @torch.no_grad()
    def _init_Y_parameters_sym_W(self):
        layers = self.get_layers()
        for layer, next_layer in zip(layers[:-1], layers[1:]):
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
            #     elif next_layer.activation_function == 'relu':
            #         nn.init.kaiming_normal_(Z, mode='fan_in', nonlinearity='relu')
            #
            #     # Misalign by ~10 degrees in weight space, while keeping Xavier-normal marginals
            #     Y = math.cos(theta) * next_layer.W_weight + math.sin(theta) * Z
            #
            #     layer.Y_weight.copy_(Y)


    @torch.no_grad()
    def _init_Y_parameters_sym_Q(self):
        layers = self.get_layers()
        for layer in layers[:-1]:
            layer.Y_weight.copy_(-layer.Q_weight / layer.p_baseline)

    @torch.no_grad()
    def apply_weight_constraints(self):
        layers = self.get_layers()

        for layer, next_layer in zip(layers[:-1], layers[1:]):

            # QY_midpoint = (layer.p_baseline * layer.Y_weight - layer.Q_weight) / 2
            # layer.Y_weight.copy_(QY_midpoint / layer.p_baseline)
            # layer.Q_weight.copy_(-QY_midpoint)
            # def align():
            #     E = layer.Q_weight + layer.p_baseline * layer.Y_weight
            #     layer.Y_weight += -(1.0 / (2.0 * layer.p_baseline)) * E
            #     layer.Q_weight += -(1.0 / 2.0) * E

            # Y_norm = layer.Y_weight.norm()
            # W_norm = next_layer.W_weight.norm()
            # layer.Y_weight *= W_norm / Y_norm

            if self.Q_mode == 'tied_W':
                layer.Q_weight.copy_(self.Y_scale * -next_layer.p_baseline * next_layer.W_weight.detach())
            if self.Y_mode == 'tied_W':
                layer.Y_weight.copy_(self.Y_scale * next_layer.W_weight.detach())

            if self.Q_mode == 'tied_Y':
                layer.Q_weight.copy_(-layer.Y_weight.detach() * layer.p_baseline)
            if self.Y_mode == 'tied_Q':
                layer.Y_weight.copy_(-layer.Q_weight.detach() / layer.p_baseline)


class DalesBurstCCN(BurstCCNBase):
    def _get_optimiser_info(self):
        optimiser_info = {
            prefix: cfg
            for prefix, cfg in [
                ("W_direct", getattr(self.cfg.W, "optimiser", None)),
                ("W_from_PV", getattr(self.cfg.W, "optimiser", None)),
                ("Y_from_SST1", getattr(self.cfg.Y, "optimiser", None)),
                ("Y_from_SST2", getattr(self.cfg.Y, "optimiser", None)),
                ("Q_direct", getattr(self.cfg.Q, "optimiser", None)),
                ("Q_from_NDNF", getattr(self.cfg.Q, "optimiser", None)),
            ]
        }
        self.W_learning = optimiser_info['W_direct'] is not None or optimiser_info['W_from_PV'] is not None
        self.Y_learning = optimiser_info['Y_from_SST1'] is not None or optimiser_info['Y_from_SST2'] is not None
        self.Q_learning = optimiser_info['Q_direct'] is not None or optimiser_info['Q_from_NDNF'] is not None
        return optimiser_info

    def _get_layer_builder(self):
        layer_builder = DalesBurstCCNLayerBuilder()
        return layer_builder

    def _get_layer_kwargs(self):
        y_grad_type = getattr(self.cfg.Y, "grad_type", None)
        q_grad_type = getattr(self.cfg.Q, "grad_type", None)
        apical_bias_learning = self.cfg.get("apical_bias_learning", False)
        hidden_kwargs = dict(p_baseline=self.p_baseline, Y_learning=self.Y_learning,
                             Q_learning=self.Q_learning, Y_grad_type=y_grad_type, Q_grad_type=q_grad_type,
                             activation_function=self.hidden_activation_function,
                             apical_bias_learning=apical_bias_learning)
        output_kwargs = dict(p_baseline=self.p_baseline, error_scale=self.error_scale,
                             activation_function=self.output_activation_function)

        feedback_bottleneck_sizes = list(self.cfg.feedback_bottleneck_sizes) if "feedback_bottleneck_sizes" in self.cfg else None
        local_feedback_scales = list(self.cfg.local_feedback_scales) if "local_feedback_scales" in self.cfg else None

        layer_lists = {
            "feedback_bottleneck_size": feedback_bottleneck_sizes,
            "local_feedback_scale": local_feedback_scales,
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

    def _init_W_parameters(self):
        if self.W_mode == 'random_init':
            self._init_W_parameters_random()
        elif self.W_mode == 'load_init':
            self._load_parameters('W_direct', self.W_load_file)
            self._load_parameters('W_from_PV', self.W_load_file)
            self._load_parameters('W_to_PV', self.W_load_file)

            #TEMP
            print("WARNING: SUBTRACTING FROM PV WEIGHTS")
            self.layers[0].W_direct_weight.data -= 0.5
            self.layers[0].W_from_PV_weight.data -= 0.5

        else:
            raise ValueError('Unknown weight mode: ' + self.W_mode)

    @torch.no_grad()
    def _init_W_parameters_random(self):
        layers = self.get_layers()
        for layer in layers:
            assert layer.activation_function == 'sigmoid'

            # Initialize effective weight matrix
            effective_W_weight = torch.zeros_like(layer.W_weight)
            nn.init.xavier_normal_(effective_W_weight, gain=self.W_scale)

            # Apply positive and negative masks
            pos_mask_W = effective_W_weight >= 0.0
            neg_mask_W = ~pos_mask_W

            # Assign masked values to the correct paths
            layer.W_direct_weight.masked_scatter_(pos_mask_W, effective_W_weight[pos_mask_W])
            layer.W_from_PV_weight.masked_scatter_(neg_mask_W, -effective_W_weight[neg_mask_W])

            # positive_weight_shift = 0.1
            positive_weight_shift = 0.0
            layer.W_direct_weight.add_(positive_weight_shift)
            layer.W_from_PV_weight.add_(positive_weight_shift)

            # Set interneuron feedback weights to identity
            layer.W_to_PV_weight.copy_(torch.eye(layer.n_PV, device=layer.W_to_PV_weight.device))

            # Initialize bias
            nn.init.constant_(layer.W_direct_bias, 0.0)

            # verify effective_W_weight equals full composite W_weight
            assert torch.allclose(layer.W_weight, effective_W_weight)

    def _init_Q_Y_parameters(self):
        # First pass without cross-dependencies
        if self.Q_mode in ["sym_W", "tied_W"]:
            self._init_Q_parameters_sym_W()
        elif self.Q_mode == 'random_init':
            self._init_Q_parameters_random()
        elif self.Q_mode == 'load_init':
            self._load_parameters('Q_direct', self.Q_load_file)
            self._load_parameters('Q_from_NDNF', self.Q_load_file)
            raise NotImplementedError("need to check the to weights are correct")

        if self.Y_mode in ["sym_W", "tied_W"]:
            self._init_Y_parameters_sym_W()
        elif self.Y_mode == 'random_init':
            self._init_Y_parameters_random()
        elif self.Y_mode == 'load_init':
            self._load_parameters('Y_from_SST1', self.Y_load_file)
            self._load_parameters('Y_from_SST2', self.Y_load_file)
            raise NotImplementedError("need to check the to weights are correct")

        # Second pass with cross-dependencies
        if self.Q_mode in ["sym_Y", "tied_Y"]:
            self._init_Q_parameters_sym_Y()
        if self.Y_mode in ["sym_Q", "tied_Q"]:
            self._init_Y_parameters_sym_Q()

    @torch.no_grad()
    def _init_Q_parameters_random(self):
        layers = self.get_layers()
        effective_Qs = []
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            effective_Q = torch.zeros_like(layer.Q_weight)
            # nn.init.normal_(effective_Q, 0, self.Q_scale)
            nn.init.xavier_normal_(effective_Q, gain=self.W_scale * self.Q_scale * next_layer.p_baseline)
            effective_Qs.append(effective_Q)

        self._init_Q_parameters_from_effective_Qs(effective_Qs)

    @torch.no_grad()
    def _init_Q_parameters_sym_W(self):
        layers = self.get_layers()
        effective_Qs = []
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            effective_Q = -self.Q_scale * next_layer.W_weight * next_layer.p_baseline
            effective_Qs.append(effective_Q)

        self._init_Q_parameters_from_effective_Qs(effective_Qs)

    @torch.no_grad()
    def _init_Q_parameters_sym_Y(self):
        layers = self.get_layers()
        effective_Qs = []
        for layer, next_layer in zip(layers[:-1], layers[1:]):
            # effective_Q = -layer.Y_weight * layer.p_baseline
            effective_Q = -self.Q_scale * layer.effective_Y_weight * layer.p_baseline
            effective_Qs.append(effective_Q)

        self._init_Q_parameters_from_effective_Qs(effective_Qs)

    @torch.no_grad()
    def _init_Q_parameters_from_effective_Qs(self, effective_Qs):
        layers = self.get_layers()
        for layer, effective_Q in zip(layers[:-1], effective_Qs):
            if effective_Q is None:
                continue

            pos_mask_Q = effective_Q >= 0.0
            neg_mask_Q = ~pos_mask_Q
            layer.Q_direct_weight.masked_scatter_(pos_mask_Q, effective_Q[pos_mask_Q])
            layer.Q_from_NDNF_weight.masked_scatter_(neg_mask_Q, -effective_Q[neg_mask_Q])

            positive_weight_shift = 0.0 # Can set to e.g. 2.0
            layer.Q_direct_weight.add_(positive_weight_shift)
            layer.Q_from_NDNF_weight.add_(positive_weight_shift)
            layer.Q_to_NDNF_weight.copy_(torch.eye(layer.n_NDNF, device=layer.Q_to_NDNF_weight.device))

            # assert torch.allclose(layer.Y_weight, effective_Y, atol=1e-05)
            assert torch.allclose(layer.Q_weight, effective_Q, atol=1e-07)

    @torch.no_grad()
    def _init_Y_parameters_random(self):
        layers = self.get_layers()
        effective_Ys = []
        for layer in layers[:-1]:
            effective_Y = torch.zeros_like(layer.Y_weight)
            # nn.init.normal_(effective_Y, 0, self.Y_scale)
            nn.init.xavier_normal_(effective_Y, gain=self.W_scale * self.Y_scale)
            effective_Ys.append(effective_Y)

        self._init_Y_parameters_from_effective_Ys(effective_Ys)

    @torch.no_grad()
    def _init_Y_parameters_sym_W(self):
        layers = self.get_layers()
        effective_Ys = []

        for layer, next_layer in zip(layers[:-1], layers[1:]):
            effective_Y = self.Y_scale * next_layer.W_weight
            effective_Ys.append(effective_Y)

            # from burstccn.bci_rotation_prediction_corrected_dalean import NDNF_BACKWARD_OFFSET
            # Y_weight_shift = 0.1 * -NDNF_BACKWARD_OFFSET if effective_Y.shape[0] == 40 else 0.1*40  * -NDNF_BACKWARD_OFFSET
            # Y_weight_shift = 0
            # effective_Ys.append(effective_Y + Y_weight_shift)

        self._init_Y_parameters_from_effective_Ys(effective_Ys)

    @torch.no_grad()
    def _init_Y_parameters_sym_Q(self):
        raise NotImplementedError

    @torch.no_grad()
    def _init_Y_parameters_from_effective_Ys(self, effective_Ys):
        layers = self.get_layers()
        for layer, effective_Y in zip(layers[:-1], effective_Ys):
            if effective_Y is None:
                continue
            # reduction_factor = 1.0 if effective_Y.shape[0] == 10 else 1.0 # 0.1
            # new_rank = max(1, int(rank_reduction * effective_Y.shape[0]))

            old_rank = layer.next_features
            new_rank = layer.feedback_bottleneck_size

            if new_rank == old_rank:
                effective_Y1 = torch.eye(old_rank, device=effective_Y.device)
                effective_Y2 = effective_Y
            elif 0 < new_rank < old_rank:
                layer.orig_Y_weight = nn.Parameter(effective_Y.clone())

                effective_Y1, effective_Y2 = matrix_factorization(effective_Y, new_rank=new_rank)
                effective_Y = effective_Y1 @ effective_Y2

                effective_Y_norm = effective_Y.norm()
                orig_Y_norm = layer.orig_Y_weight.norm()
                ratio = orig_Y_norm / effective_Y_norm

                effective_Y2 *= ratio

                effective_Y = effective_Y1 @ effective_Y2

                effective_Y_norm = effective_Y.norm()
                orig_Y_norm = layer.orig_Y_weight.norm()
                ratio = orig_Y_norm / effective_Y_norm

                print(f"{effective_Y_norm} / {orig_Y_norm} = {ratio}")

            else:
                raise ValueError(f"Invalid feedback bottleneck size: {layer.feedback_bottleneck_size}")

            # Y1_scale = 1.0 / 25.0
            # Y2_scale = 1.0 / Y1_scale

            Y1_scale = 1.0
            Y2_scale = 1.0

            effective_Y1 *= Y1_scale
            effective_Y2 *= Y2_scale

            pos_mask_Y1 = effective_Y1 >= 0.0
            neg_mask_Y1 = ~pos_mask_Y1
            layer.Y_to_SST1_weight.masked_scatter_(pos_mask_Y1, effective_Y1[pos_mask_Y1])
            layer.Y_to_SST2_weight.masked_scatter_(neg_mask_Y1, -effective_Y1[neg_mask_Y1])

            layer.Y_to_VIP2_weight.masked_scatter_(pos_mask_Y1, effective_Y1[pos_mask_Y1])
            layer.Y_to_VIP1_weight.masked_scatter_(neg_mask_Y1, -effective_Y1[neg_mask_Y1])

            layer.Y_VIP1_to_SST1_weight.copy_(torch.eye(layer.n_VIP1, device=layer.Y_VIP1_to_SST1_weight.device))
            layer.Y_VIP2_to_SST2_weight.copy_(torch.eye(layer.n_VIP2, device=layer.Y_VIP2_to_SST2_weight.device))

            pos_mask_Y2 = effective_Y2 >= 0.0
            neg_mask_Y2 = ~pos_mask_Y2

            layer.Y_from_SST2_weight.masked_scatter_(pos_mask_Y2, effective_Y2[pos_mask_Y2])
            layer.Y_from_SST1_weight.masked_scatter_(neg_mask_Y2, -effective_Y2[neg_mask_Y2])

            positive_weight_shift = 0.0 # Can set to e.g. 2.0
            layer.Y_from_SST1_weight.add_(positive_weight_shift)
            layer.Y_from_SST2_weight.add_(positive_weight_shift)

            layer.sst1_bias.copy_(-((layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) * (
                    (layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) < 0)).sum(dim=0))
            layer.sst2_bias.copy_(-((layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) * (
                    (layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) < 0)).sum(dim=0))

            layer.apical_bias_input.copy_(
                layer.sst1_bias @ layer.Y_from_SST1_weight + layer.sst2_bias @ layer.Y_from_SST2_weight)

    @torch.no_grad()
    def apply_weight_constraints(self):
        @torch.no_grad()
        def _assign_Q_from_effective(layer, effective_Q, shift=0.0):
            # write nonnegative sub-paths deterministically
            pos = torch.clamp(effective_Q, min=0.0)
            neg = torch.clamp(-effective_Q, min=0.0)
            layer.Q_direct_weight.copy_(pos + shift)
            layer.Q_from_NDNF_weight.copy_(neg + shift)
            # keep routing identity
            # layer.Q_to_NDNF_weight.copy_(torch.eye(layer.n_NDNF, device=layer.Q_to_NDNF_weight.device))

        @torch.no_grad()
        def _assign_Y_from_effective(layer, effective_Y, new_rank, shift=0.0):
            """
            Map effective_Y into Y1 (to SST/VIP) and Y2 (from SST) with nonneg constraints.
            If new_rank < full, use your matrix_factorization(..) to make a rank-k approx.
            """
            old_rank = layer.next_features
            if new_rank == old_rank:
                # Y1 = I, Y2 = effective_Y
                Y1 = torch.eye(old_rank, device=effective_Y.device)
                Y2 = effective_Y
            elif 0 < new_rank < old_rank:
                Y1, Y2 = matrix_factorization(effective_Y, new_rank=new_rank, print_details=False)
            else:
                raise ValueError(f"Invalid feedback bottleneck size: {new_rank}")

            # Nonnegative decompositions, deterministic writes (no add_ accumulation)
            Y1_pos = torch.clamp(Y1, min=0.0)
            Y1_neg = torch.clamp(-Y1, min=0.0)
            layer.Y_to_SST1_weight.copy_(Y1_pos + shift)
            layer.Y_to_SST2_weight.copy_(Y1_neg + shift)
            # Mirror into VIP routes as in the init
            layer.Y_to_VIP2_weight.copy_(Y1_pos + shift)
            layer.Y_to_VIP1_weight.copy_(Y1_neg + shift)

            # layer.Y_VIP1_to_SST1_weight.copy_(torch.eye(layer.n_VIP1, device=layer.Y_VIP1_to_SST1_weight.device))
            # layer.Y_VIP2_to_SST2_weight.copy_(torch.eye(layer.n_VIP2, device=layer.Y_VIP2_to_SST2_weight.device))

            Y2_pos = torch.clamp(Y2, min=0.0)
            Y2_neg = torch.clamp(-Y2, min=0.0)
            layer.Y_from_SST2_weight.copy_(Y2_pos + shift)
            layer.Y_from_SST1_weight.copy_(Y2_neg + shift)

            # Update SST biases and apical bias
            layer.sst1_bias.copy_(-((layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) * (
                    (layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) < 0)).sum(dim=0))
            layer.sst2_bias.copy_(-((layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) * (
                    (layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) < 0)).sum(dim=0))
            layer.apical_bias_input.copy_(
                layer.sst1_bias @ layer.Y_from_SST1_weight + layer.sst2_bias @ layer.Y_from_SST2_weight
            )

        layers = self.get_layers()

        for layer in layers:
            layer.W_direct_weight.clamp_(min=0.0)
            layer.W_from_PV_weight.clamp_(min=0.0)
            layer.W_to_PV_weight.clamp_(min=0.0)

        for i, (layer, next_layer) in enumerate(zip(layers[:-1], layers[1:])):
            layer.Q_direct_weight.clamp_(min=0.0)
            layer.Q_from_NDNF_weight.clamp_(min=0.0)

            layer.Y_from_SST1_weight.clamp_(min=0.0)
            layer.Y_from_SST2_weight.clamp_(min=0.0)

            if self.Q_mode == "tied_W":
                eff_Q = -next_layer.W_weight * next_layer.p_baseline
                _assign_Q_from_effective(layer, eff_Q, shift=0.0)
            if self.Y_mode == "tied_W":
                eff_Y = next_layer.W_weight
                _assign_Y_from_effective(layer, eff_Y, new_rank=layer.feedback_bottleneck_size, shift=0.0)

            if self.Q_mode == "tied_Y":
                eff_Q = -layer.Y_weight * layer.p_baseline
                _assign_Q_from_effective(layer, eff_Q, shift=0.0)
            if self.Y_mode == "tied_Q":
                # If/when you define it: tie Y to (a function of) Q
                eff_Y = -layer.Q_weight / layer.p_baseline
                _assign_Y_from_effective(layer, eff_Y, new_rank=layer.feedback_bottleneck_size, shift=0.0)

            # warnings.warn("Not updating layer.apical_bias!")
            # raise NotImplementedError()
            # layer.apical_bias_input.copy_(layer.sst1_bias @ layer.Y_from_SST1_weight + layer.sst2_bias @ layer.Y_from_SST2_weight)

            # layer.apical_bias_input.add_(-0.02 * layer.apic.reshape(layer.apical_bias_input.shape) if hasattr(layer, 'apic') else 0.0)

            # target_sst1_bias = layer.Y_VIP1_to_SST1_weight.item() * (-((layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) * ((layer.Y_to_SST1_weight - layer.Y_to_VIP1_weight) < 0)).sum(dim=0))
            # target_sst2_bias = layer.Y_VIP2_to_SST2_weight.item() * (-((layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) * ((layer.Y_to_SST2_weight - layer.Y_to_VIP2_weight) < 0)).sum(dim=0))
            #
            # layer.sst1_bias.copy_(0.5 * target_sst1_bias + 0.5 * layer.sst1_bias)
            # layer.sst2_bias.copy_(0.5 * target_sst2_bias + 0.5 * layer.sst2_bias)
            #
            # target_apical_bias = layer.sst1_bias @ layer.Y_from_SST1_weight + layer.sst2_bias @ layer.Y_from_SST2_weight
            #
            # layer.apical_bias_input.copy_(0.5 * target_apical_bias + 0.5 * layer.apical_bias_input)


