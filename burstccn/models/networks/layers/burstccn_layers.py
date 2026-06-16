import math
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight, conv2d_input

from burstccn.models.networks.activation_functions import get_activation_function
from burstccn.models.networks.layers.base import LayerBase


class BurstCCNOutputLayer(LayerBase):
    def __init__(self, in_features, out_features, p_baseline, error_scale, activation_function="sigmoid",
                 activity_shift=None):
        super(BurstCCNOutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline = p_baseline
        self.error_scale = error_scale
        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

        self.activity_shift = activity_shift

        self.register_buffer("p", self.p_baseline * torch.ones(self.out_features))

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            assert forward_noise != 0.0
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.soma = F.linear(self.input, self.W_weight, self.W_bias)
        self.e, self.act_ctx = self.act(self.soma)

        return self.e

    def calc_p_sigmoid(self, target):
        if self.activity_shift:
            shifted_e = self.e + self.activity_shift
            return self.p_baseline * ((target - self.e) * (1 - self.e) * ((self.e) / shifted_e) * self.error_scale + 1)
        else:
            return self.p_baseline * ((target - self.e) * (1 - self.e) * self.error_scale + 1)

    def calc_p_softmax(self, target):
        if self.activity_shift:
            shifted_target = target + self.activity_shift
            shifted_e = self.e + self.activity_shift
            return 2.0 * self.p_baseline * torch.sigmoid(
                4.0 * self.p_baseline * self.error_scale * ((shifted_target - shifted_e) / shifted_e))
        else:
            return 2.0 * self.p_baseline * torch.sigmoid(
                4.0 * self.error_scale * (target - self.e) / (self.e + 1e-8))

    def backward(self, target):
        if target is None:
            target = self.e

        if self.activation_function == "sigmoid":
            self.p_t = self.calc_p_sigmoid(target)
        elif self.activation_function == "softmax":
            self.p_t = self.calc_p_softmax(target)
        else:
            raise ValueError(f"Invalid activation function {self.activation_function}")

        if self.activity_shift:
            shifted_e = self.e + self.activity_shift

            self.b = self.p * shifted_e
            self.b_t = self.p_t * shifted_e
        else:
            self.b = self.p * self.e
            self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        update_scale_factor = self.input.size(0) * self.p_baseline * self.error_scale

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        if self.activity_shift:
            b_bias = self.p * self.activity_shift
            return self.b_t - b_bias, self.e, self.error_scale
            # return self.b_t, shifted_e, self.error_scale
        else:
            return self.b_t, self.e, self.error_scale

    def backward_bp(self, target):
        diff = -(target - self.e)
        self.b_input_bp = diff
        self.delta_bp = self.act.backward(diff, self.act_ctx)

        self.W_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / self.input.size(0)
        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, target):
        diff = -(target - self.e)
        self.delta_fa = self.act.backward(diff, self.act_ctx)

        self.W_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / self.input.size(0)
        return self.delta_fa


class BurstCCNHiddenLayer(LayerBase):
    def __init__(self, in_features, out_features, next_features, p_baseline, Y_learning, Q_learning,
                 Y_grad_type, Q_grad_type, activation_function="sigmoid", local_feedback_scale=1.0,
                 n_apical_branches=1, store_exc_inh_branch_state=False):
        super(BurstCCNHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.next_features = next_features

        self.p_baseline = p_baseline
        self.Y_learning = Y_learning
        self.Q_learning = Q_learning
        self.Y_grad_type = Y_grad_type
        self.Q_grad_type = Q_grad_type
        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

        self.local_feedback_scale = local_feedback_scale
        self.n_apical_branches = n_apical_branches

        n_branches = min(self.n_apical_branches, next_features)
        k_per_branch = next_features // n_branches
        k_remainder = next_features % n_branches
        self.apical_branch_sizes = [k_per_branch + 1 if b < k_remainder else k_per_branch for b in range(n_branches)]
        self.store_exc_inh_branch_state = store_exc_inh_branch_state

        self.register_buffer("p", self.p_baseline * torch.ones(self.out_features))

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

        self.Y_weight = nn.Parameter(torch.zeros(next_features, out_features))
        self.Q_weight = nn.Parameter(torch.zeros(next_features, out_features))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.soma = F.linear(self.input, self.W_weight, self.W_bias)
        self.e, self.act_ctx = self.act(self.soma)

        return self.e

    def _act_backward(self, delta, ctx, surrogate=False):
        if surrogate and hasattr(self.act, "backward_surrogate"):
            return self.act.backward_surrogate(delta, ctx)
        return self.act.backward(delta, ctx)

    def calc_p_sigmoid(self):
        k = 2.0 / self.p_baseline
        return 2.0 * self.p_baseline * torch.sigmoid(k * self.apic * (1.0 - self.e))

    def calc_p_relu(self):
        # k = 2.0 / self.p_baseline
        #
        # p_t = self.p_baseline * torch.ones_like(self.apic)
        # non_zero_mask = self.e > 1e-8
        # p_t[non_zero_mask] = 2.0 * self.p_baseline * torch.sigmoid(k * self.apic[non_zero_mask] / self.e[non_zero_mask])
        # return p_t
        k = 2.0 / self.p_baseline
        # shift = 1e-1
        shift = 1e-8

        g = self._act_backward(self.apic, self.act_ctx, surrogate=True)
        p_t = torch.sigmoid(k * g / (self.e + shift))
        # b_t = p_t * (self.e + shift) - shift * self.p_baseline
        b_t = p_t * self.e
        return p_t, b_t

    def calc_p_relu_shift(self):
        k = 2.0 / self.p_baseline
        # p_t = self.p_baseline * torch.ones_like(self.apic)
        shift = 1e-1

        g = self._act_backward(self.apic, self.act_ctx, surrogate=True)
        p_t = torch.sigmoid(k * g / (self.e + shift))
        b_t = p_t * (self.e + shift) - shift * self.p_baseline
        return p_t, b_t

    def check_asserts(self):
        assert not torch.any(torch.isnan(self.Q_input))
        assert not torch.any(torch.isnan(self.Y_input))
        assert not torch.any(torch.isnan(self.apic))
        assert not torch.any(torch.isnan(self.p_t))
        assert self.p_t.max() <= 1.0
        assert self.p_t.min() >= 0.0

    def calc_Y_apic_grad(self, next_burst_rates, next_event_rates):
        Y_inputs = []
        Q_inputs = []
        apic_branches = []
        Y_apic_grads = []

        Y_exc_inputs = []
        Y_inh_inputs = []
        Q_exc_inputs = []
        Q_inh_inputs = []

        n_branches = len(self.apical_branch_sizes)
        k0 = 0
        for branch, branch_size in enumerate(self.apical_branch_sizes):
            k1 = k0 + branch_size

            Y_weight_branch = self.Y_weight[k0:k1, :]
            Q_weight_branch = self.Q_weight[k0:k1, :]

            burst_branch = next_burst_rates[:, k0:k1]
            event_branch = next_event_rates[:, k0:k1]

            Y_input_branch = burst_branch.mm(Y_weight_branch)
            Q_input_branch = event_branch.mm(Q_weight_branch)
            apic_branch = Y_input_branch + Q_input_branch  # (B, N)

            # ### Excitatory/Inhibitory inputs
            if self.store_exc_inh_branch_state:
                Y_pos = (Y_weight_branch > 0).float()
                Y_neg = (Y_weight_branch < 0).float()

                Q_pos = (Q_weight_branch > 0).float()
                Q_neg = (Q_weight_branch < 0).float()

                # Excitatory contributions = only positive weights contribute
                Y_exc_input_branch = burst_branch.mm(Y_weight_branch * Y_pos)
                Q_exc_input_branch = event_branch.mm(Q_weight_branch * Q_pos)

                # Inhibitory contributions = only negative weights contribute
                Y_inh_input_branch = burst_branch.mm(Y_weight_branch * Y_neg)
                Q_inh_input_branch = event_branch.mm(Q_weight_branch * Q_neg)

                Y_exc_inputs.append(Y_exc_input_branch)
                Y_inh_inputs.append(Y_inh_input_branch)
                Q_exc_inputs.append(Q_exc_input_branch)
                Q_inh_inputs.append(Q_inh_input_branch)

            ###

            Y_apic_grad = math.sqrt(self.next_features / branch_size) * burst_branch.transpose(0, 1).mm(apic_branch)

            # norm_burst_branch = burst_branch / torch.linalg.vector_norm(next_event_rates, dim=1, keepdim=True).clamp_min(1e-6)
            # Y_apic_grad = math.sqrt(self.next_features) * math.sqrt(self.next_features / branch_size) * norm_burst_branch.transpose(0, 1).mm(apic_branch)
            # Y_apic_grad = math.sqrt(self.next_features) * (self.next_features / branch_size) * norm_burst_branch.transpose(0, 1).mm(apic_branch)
            # Y_apic_grad = math.sqrt(self.next_features) * norm_burst_branch.transpose(0, 1).mm(apic_branch)

            # apic_grad = burst_branch.transpose(0, 1).mm(apic_branch)

            # norm_burst_branch = burst_branch / torch.linalg.vector_norm(burst_branch, dim=1, keepdim=True).clamp_min(1e-6)
            # Y_apic_grad = math.sqrt(self.next_features / 500) * norm_burst_branch.transpose(0, 1).mm(apic_branch)

            # norm_burst_branch = burst_branch / torch.linalg.vector_norm(next_burst_rates, dim=1, keepdim=True)
            # Y_apic_grad = math.sqrt(self.next_features / branch_size) * norm_burst_branch.transpose(0, 1).mm(apic_branch)

            # norm_burst_branch = burst_branch / (burst_branch.pow(2).sum(dim=1, keepdim=True) + 1e-8)
            # Y_apic_grad = math.sqrt(branch_size / self.next_features) * norm_burst_branch.transpose(0, 1).mm(apic_branch)
            # Y_apic_grad = math.sqrt(self.next_features / branch_size) * norm_burst_branch.transpose(0, 1).mm(apic_branch)
            # Y_apic_grad = math.sqrt(branch_size / self.next_features) * norm_burst_branch.transpose(0, 1).mm(apic_branch)

            Y_inputs.append(Y_input_branch)
            Q_inputs.append(Q_input_branch)
            apic_branches.append(apic_branch)
            Y_apic_grads.append(Y_apic_grad)

            k0 = k1

        if n_branches > 1 and self.store_exc_inh_branch_state:
            # assert self.apic_branch.allclose(torch.stack(apic_branches, dim=1), rtol=1e-4, atol=1e-5)
            self.apic_branch = torch.stack(apic_branches, dim=1)
            # assert self.apic_branch.sum(dim=1).allclose(self.apic, rtol=1e-4, atol=1e-5)

            self.apic = self.apic_branch.sum(dim=1)

            # also keep summed Y/Q for asserts
            self.Y_input_branch = torch.stack(Y_inputs, dim=1)
            self.Q_input_branch = torch.stack(Q_inputs, dim=1)

            self.Y_input = self.Y_input_branch.sum(dim=1)
            self.Q_input = self.Q_input_branch.sum(dim=1)

            self.Y_exc_input_branch = torch.stack(Y_exc_inputs, dim=1)
            self.Q_exc_input_branch = torch.stack(Q_exc_inputs, dim=1)

            self.Y_inh_input_branch = torch.stack(Y_inh_inputs, dim=1)
            self.Q_inh_input_branch = torch.stack(Q_inh_inputs, dim=1)

        Y_apic_grad = torch.vstack(Y_apic_grads)  # * 10 / self.next_features
        return Y_apic_grad

    def calc_Y_kp_grad(self, next_burst_rates, next_event_rates):
        Y_next_delta = -(next_burst_rates - self.p_baseline * next_event_rates)
        Y_kp_grad = Y_next_delta.transpose(0, 1).mm(self.e)
        return Y_kp_grad

    def calc_Y_QY_align_grad(self):
        Y_QY_align_grad = (self.Q_weight + self.Y_weight * self.p_baseline)
        return Y_QY_align_grad

    def calc_Q_kp_grad(self, next_burst_rates, next_event_rates):
        Q_next_delta = (next_burst_rates - self.p_baseline * next_event_rates)
        Q_kp_grad = self.p_baseline * Q_next_delta.transpose(0, 1).mm(self.e)
        return Q_kp_grad

    def backward(self, next_burst_rates, next_event_rates, next_feedback_scale):
        self.Y_input = next_burst_rates.mm(self.Y_weight)
        self.Q_input = next_event_rates.mm(self.Q_weight)

        self.apic = self.Y_input + self.Q_input

        self.apic *= self.local_feedback_scale
        feedback_scale = next_feedback_scale * self.local_feedback_scale

        if self.activation_function == 'sigmoid':
            self.p_t = self.calc_p_sigmoid()
        elif self.activation_function == 'relu':
            self.p_t, self.b_t = self.calc_p_relu()
        elif self.activation_function in ("ln_relu", "relu_ln", "ms_relu_n", "relu_rms"):
            self.p_t, self.b_t = self.calc_p_relu()
        else:
            raise NotImplementedError

        self.b = self.p_baseline * self.e
        if self.activation_function not in ("relu", "ln_relu", "relu_ln", "ms_relu_n", "relu_rms"):
            self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        if self.Y_learning:
            if self.Y_grad_type is None:
                raise ValueError("Y_grad_type must be set when Y_learning is True")
            # self.Y_weight.grad = next_burst_rates.transpose(0, 1).mm(self.apic) / update_scale_factor

            if self.Y_grad_type == 'apic':
                Y_update_scale_factor = self.input.size(0) * self.local_feedback_scale
                Y_apic_grad = self.calc_Y_apic_grad(next_burst_rates, next_event_rates)
                self.Y_weight.grad = Y_apic_grad / Y_update_scale_factor
            elif self.Y_grad_type == 'kp':
                Y_update_scale_factor = self.input.size(0) * self.p_baseline * next_feedback_scale
                Y_kp_grad = self.calc_Y_kp_grad(next_burst_rates, next_event_rates)
                self.Y_weight.grad = Y_kp_grad / Y_update_scale_factor
            else:
                raise ValueError(f"Invalid Y grad type {self.Y_grad_type}")

        if self.Q_learning:
            if self.Q_grad_type is None:
                raise ValueError("Q_grad_type must be set when Q_learning is True")
            if self.Q_grad_type == 'kp':
                Q_update_scale_factor = self.input.size(0) * self.p_baseline * next_feedback_scale
                Q_kp_grad = self.calc_Q_kp_grad(next_burst_rates, next_event_rates)
                self.Q_weight.grad = Q_kp_grad / Q_update_scale_factor
            else:
                raise ValueError(f"Invalid Q grad type {self.Q_grad_type}")

        return self.b_t, self.e, feedback_scale

    def backward_bp(self, b_input_bp):
        self.b_input_bp = b_input_bp
        self.delta_bp = self._act_backward(b_input_bp, self.act_ctx, surrogate=False)

        self.W_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / self.input.size(0)
        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, next_delta_fa):
        g = next_delta_fa.mm(self.Y_weight)
        self.delta_fa = self._act_backward(g, self.act_ctx, surrogate=False)

        self.W_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / self.input.size(0)
        return self.delta_fa


class BurstCCNConvLayer(LayerBase, ABC):
    def __init__(self, in_channels, out_channels,
                 in_shape, kernel_size, stride,
                 padding, dilation, groups,
                 padding_mode, p_baseline, Y_learning, Q_learning,
                 Y_grad_type="kp", Q_grad_type="kp",
                 activation_function="sigmoid",
                 local_feedback_scale=1.0,
                 store_exc_inh_branch_state=False):
        super(BurstCCNConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.p_baseline = p_baseline
        self.Y_learning = Y_learning
        self.Q_learning = Q_learning
        self.Y_grad_type = None if Y_grad_type in (None, "none") else Y_grad_type
        self.Q_grad_type = None if Q_grad_type in (None, "none") else Q_grad_type

        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

        self.local_feedback_scale = local_feedback_scale
        self.store_exc_inh_branch_state = store_exc_inh_branch_state

        self.W_weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size))
        self.W_bias = nn.Parameter(torch.zeros(out_channels))

        out_y = ((in_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1)
        out_x = ((in_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)

        self.out_shape = (out_y, out_x)
        self.register_buffer("p", self.p_baseline * torch.ones(self.out_shape))

    def _forward_reshape(self, e):
        raise NotImplementedError

    def _backward_project_W(self, next_delta):
        return conv2d_input(self.input.shape, self.W_weight, next_delta, self.stride,
                            self.padding, self.dilation, self.groups)

    def _backward_project_Y(self, next_delta):
        raise NotImplementedError

    def _backward_project_Q(self, next_delta):
        raise NotImplementedError

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.soma = F.conv2d(self.input, self.W_weight, self.W_bias, self.stride, self.padding, self.dilation,
                             self.groups)
        self.e, self.act_ctx = self.act(self.soma)

        return self._forward_reshape(self.e)

    def _act_backward(self, delta, ctx, surrogate=False):
        if surrogate and hasattr(self.act, "backward_surrogate"):
            return self.act.backward_surrogate(delta, ctx)
        return self.act.backward(delta, ctx)

    def calc_p_sigmoid(self):
        k = 2.0 / self.p_baseline
        return 2.0 * self.p_baseline * torch.sigmoid(k * self.apic * (1.0 - self.e))

    def calc_p_relu(self):
        # k = 2.0 / self.p_baseline
        #
        # p_t = self.p_baseline * torch.ones_like(self.apic)
        # non_zero_mask = self.e > 1e-8
        # p_t[non_zero_mask] = 2.0 * self.p_baseline * torch.sigmoid(k * self.apic[non_zero_mask] / self.e[non_zero_mask])
        # return p_t
        # shift = 1e-1
        k = 2.0 / self.p_baseline
        # shift = 1e-1
        shift = 1e-8

        g = self._act_backward(self.apic, self.act_ctx, surrogate=True)
        p_t = torch.sigmoid(k * g / (self.e + shift))
        # b_t = p_t * (self.e + shift) - shift * self.p_baseline

        # p_t[(p_t - self.p_baseline).abs() < 0.0005] = self.p_baseline

        # p_t += 0.001 * torch.randn_like(p_t)

        b_t = p_t * self.e
        return p_t, b_t

    def calc_p_relu_shift(self):
        k = 2.0 / self.p_baseline
        # self.p_t = self.p_baseline * torch.ones_like(self.apic)
        shift = 1e-1

        g = self._act_backward(self.apic, self.act_ctx, surrogate=True)
        p_t = torch.sigmoid(k * g / (self.e + shift))
        b_t = p_t * (self.e + shift) - self.p_baseline
        return p_t, b_t

    def calc_Y_apic_grad(self, next_burst_rates, next_event_rates):
        raise NotImplementedError("Apical Y gradients are not implemented for conv layers.")

    def calc_Y_kp_grad(self, next_burst_rates, next_event_rates):
        Y_next_delta = -(next_burst_rates - self.p_baseline * next_event_rates)
        if self.Y_weight.dim() == 4:
            return conv2d_weight(
                input=self.e,
                weight_size=self.Y_weight.shape,
                grad_output=Y_next_delta,
                stride=self.next_stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.Y_weight.dim() == 2:
            return Y_next_delta.t().mm(self.e.view(self.e.size(0), -1))
        raise RuntimeError(f"Unexpected Y_weight dim: {self.Y_weight.dim()}")

    def calc_Q_kp_grad(self, next_burst_rates, next_event_rates):
        Q_next_delta = (next_burst_rates - self.p_baseline * next_event_rates)
        if self.Q_weight.dim() == 4:
            return self.p_baseline * conv2d_weight(
                input=self.e,
                weight_size=self.Q_weight.shape,
                grad_output=Q_next_delta,
                stride=self.next_stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        if self.Q_weight.dim() == 2:
            return self.p_baseline * Q_next_delta.t().mm(self.e.view(self.e.size(0), -1))
        raise RuntimeError(f"Unexpected Q_weight dim: {self.Q_weight.dim()}")

    def backward(self, next_burst_rates, next_event_rates, next_feedback_scale):
        self.Y_input = self._backward_project_Y(next_burst_rates)
        self.Q_input = self._backward_project_Q(next_event_rates)
        self.apic = self.Y_input + self.Q_input

        self.apic *= self.local_feedback_scale
        feedback_scale = next_feedback_scale * self.local_feedback_scale

        if self.activation_function == 'sigmoid':
            self.p_t = self.calc_p_sigmoid()
        elif self.activation_function == 'relu':
            self.p_t, self.b_t = self.calc_p_relu()
        elif self.activation_function in ("ln_relu", "relu_ln", "ms_relu_n", "relu_rms"):
            self.p_t, self.b_t = self.calc_p_relu()

        self.b = self.p_baseline * self.e
        if self.activation_function not in ("relu", "ln_relu", "relu_ln", "ms_relu_n", "relu_rms"):
            self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale

        self.W_weight.grad = conv2d_weight(self.input, self.W_weight.shape, self.delta, self.stride,
                                           self.padding, self.dilation, self.groups) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=[0, 2, 3]) / update_scale_factor

        if self.Y_learning:
            if self.Y_grad_type is None:
                raise ValueError("Y_grad_type must be set when Y_learning is True")
            # self.Y_weight.grad = next_burst_rates.transpose(0, 1).mm(self.apic) / update_scale_factor

            # Y_update_scale_factor = update_scale_factor / self.local_feedback_scale

            if self.Y_grad_type == 'apic':
                raise NotImplementedError ()
                Y_apic_grad = self.calc_Y_apic_grad(next_burst_rates, next_event_rates)
                self.Y_weight.grad = Y_apic_grad / Y_update_scale_factor
            elif self.Y_grad_type == 'kp':
                Y_update_scale_factor = self.input.size(0) * self.p_baseline * next_feedback_scale
                Y_kp_grad = self.calc_Y_kp_grad(next_burst_rates, next_event_rates)
                self.Y_weight.grad = Y_kp_grad / Y_update_scale_factor
            else:
                raise ValueError(f"Invalid Y grad type {self.Y_grad_type}")

        if self.Q_learning:
            if self.Q_grad_type is None:
                raise ValueError("Q_grad_type must be set when Q_learning is True")
            # Q_update_scale_factor = update_scale_factor / self.local_feedback_scale
            #
            # # self.Q_weight.grad = -next_event_rates.transpose(0, 1).mm(self.apic) / update_scale_factor
            # Q_next_delta = (next_burst_rates - self.p_baseline * next_event_rates)
            #
            # # Q_kp_grad = self.p_baseline * Q_next_delta.transpose(0, 1).mm(self.e)
            # Q_kp_grad = self.p_baseline * conv2d_weight(self.e, self.Q_weight.shape, Q_next_delta, self.next_stride,
            #               self.padding, self.dilation, self.groups)
            #
            # self.Q_weight.grad = Q_kp_grad / Q_update_scale_factor

            if self.Q_grad_type == 'kp':
                Q_update_scale_factor = self.input.size(0) * self.p_baseline * next_feedback_scale
                Q_kp_grad = self.calc_Q_kp_grad(next_burst_rates, next_event_rates)
                self.Q_weight.grad = Q_kp_grad / Q_update_scale_factor
            else:
                raise ValueError(f"Invalid Q grad type {self.Q_grad_type}")

        return self.b_t, self.e, feedback_scale

    def backward_bp(self, b_input_bp):
        g = self._backward_reshape(b_input_bp)
        self.b_input_bp = g

        self.delta_bp = self._act_backward(g, self.act_ctx, surrogate=False)

        self.W_weight.grad_bp = conv2d_weight(self.input, self.W_weight.shape, self.delta_bp, self.stride,
                                              self.padding, self.dilation, self.groups) / self.input.size(0)

        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=[0, 2, 3]) / self.input.size(0)

        return self._backward_project_W(self.delta_bp)

    def backward_fa(self, next_delta_fa):
        g = self._backward_project_Y(next_delta_fa)
        self.delta_fa = self._act_backward(g, self.act_ctx, surrogate=False)

        self.W_weight.grad_fa = conv2d_weight(self.input, self.W_weight.shape, self.delta_fa, self.stride,
                                              self.padding, self.dilation, self.groups) / self.input.size(0)

        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=[0, 2, 3]) / self.input.size(0)

        return self.delta_fa


class BurstCCNConvHiddenLayer(BurstCCNConvLayer):
    def __init__(self, *, next_channels, next_kernel_size, next_stride, **base_kwargs):
        super().__init__(**base_kwargs)

        self.next_stride = next_stride
        self.Y_weight = nn.Parameter(torch.zeros(next_channels, self.out_channels // self.groups, *next_kernel_size))
        self.Q_weight = nn.Parameter(torch.zeros(next_channels, self.out_channels // self.groups, *next_kernel_size))

    def _forward_reshape(self, e):
        return e

    def _backward_reshape(self, b_input):
        return b_input

    def _backward_project_Y(self, next_delta):
        return conv2d_input(self.e.shape, self.Y_weight, next_delta, self.next_stride,
                            self.padding, self.dilation, self.groups)

    def _backward_project_Q(self, next_delta):
        return conv2d_input(self.e.shape, self.Q_weight, next_delta, self.next_stride,
                            self.padding, self.dilation, self.groups)


class BurstCCNConvFinalLayer(BurstCCNConvLayer):
    def __init__(self, *, next_features, **base_kwargs):
        super().__init__(**base_kwargs)

        self.out_features = self.out_channels * self.out_shape[0] * self.out_shape[1]
        self.Y_weight = nn.Parameter(torch.zeros(next_features, self.out_features))
        self.Q_weight = nn.Parameter(torch.zeros(next_features, self.out_features))

    def _forward_reshape(self, e):
        output_size = e.size()
        return e.view(output_size[0], -1)

    def _backward_reshape(self, b_input):
        return b_input.reshape(self.e.shape)

    def _backward_project_Y(self, next_delta):
        return next_delta.mm(self.Y_weight).view(self.e.shape)

    def _backward_project_Q(self, next_delta):
        return next_delta.mm(self.Q_weight).view(self.e.shape)
