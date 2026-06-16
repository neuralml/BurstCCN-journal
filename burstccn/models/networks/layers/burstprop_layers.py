from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight, conv2d_input

from burstccn.models.networks.layers.base import LayerBase


global_lr_scale = 1.0
sigmoid_scale = 1.0


class BurstpropOutputLayer(LayerBase):
    def __init__(self, in_features, out_features, p_baseline, error_scale, activation_function="sigmoid",
                 activity_shift=None):
        super(BurstpropOutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline = p_baseline
        self.error_scale = error_scale
        self.activation_function = activation_function
        self.activity_shift = activity_shift

        if self.activation_function == "sigmoid":
            self.f = torch.sigmoid
            self.f_deriv = lambda soma: torch.sigmoid(soma) * (1.0 - torch.sigmoid(soma))
            assert not activity_shift
        elif self.activation_function == "softmax":
            self.f = lambda x: F.softmax(x, dim=1)
            self.f_deriv = lambda soma: torch.ones_like(soma)

        # self.p = self.p_baseline * torch.ones(self.out_features)
        self.register_buffer("p", sigmoid_scale * self.p_baseline * torch.ones(self.out_features))

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            assert forward_noise != 0.0
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.soma = F.linear(self.input, self.W_weight, self.W_bias)
        # self.e = torch.sigmoid(self.soma)
        self.e = self.f(self.soma)
        return self.e

    def backward(self, target):
        if target is None:
            target = self.e

        # self.p = self.p_baseline
        # self.p_t = self.p_baseline * ((target - self.e) * (1 - self.e) * self.error_scale + 1)

        if self.activation_function == "sigmoid":
            self.p_t = self.p_baseline * ((target - self.e) * (1 - self.e) * self.error_scale + 1)
        elif self.activation_function == "softmax":
            # self.p_t = 2.0 * self.p_baseline * torch.sigmoid(4.0 * self.error_scale * (target - self.e) / (self.e + 1e-8))
            if self.activity_shift:
                shifted_target = target + self.activity_shift
                shifted_e = self.e + self.activity_shift
                self.p_t = sigmoid_scale * torch.sigmoid(4.0 * self.p_baseline * self.error_scale * ((shifted_target - shifted_e) / shifted_e))
                # self.p_t = sigmoid_scale * torch.sigmoid(4.0 * self.p_baseline * self.error_scale * ((target - self.e) / shifted_e))
            else:
                self.p_t = sigmoid_scale * torch.sigmoid(4.0 * self.error_scale * (target - self.e) / (self.e + 1e-8))

        if self.activity_shift:
            self.b = self.p * shifted_e
            self.b_t = self.p_t * shifted_e  # - self.p_baseline * 1.0
        else:
            self.b = self.p * self.e
            self.b_t = self.p_t * self.e

        # self.b = self.p * self.e
        # self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        update_scale_factor = self.input.size(0) * self.p_baseline * self.error_scale * global_lr_scale * sigmoid_scale

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        # return self.b_t, self.b, self.error_scale
        if self.activity_shift:
            b_bias = self.p_baseline * self.activity_shift
            # return self.b_t, self.b, self.error_scale * sigmoid_scale
            return self.b_t - b_bias, self.b - b_bias, self.error_scale * sigmoid_scale
        else:
            return self.b_t, self.b, self.error_scale * sigmoid_scale

    def backward_bp(self, target):
        # self.delta_bp = -(target - self.e) * self.e * (1 - self.e)
        if self.activation_function == "softmax":
            self.delta_bp = -(target - self.e)
        else:
            self.delta_bp = -(target - self.e) * self.f_deriv(self.soma)

        self.W_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / self.input.size(0)
        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, target):
        # self.delta_fa = -(target - self.e) * self.e * (1 - self.e)
        # self.delta_fa = -(target - self.e) * self.f_deriv(self.soma)

        if self.activation_function == "softmax":
            self.delta_fa = -(target - self.e)
        else:
            self.delta_fa = -(target - self.e) * self.f_deriv(self.soma)

        self.W_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / self.input.size(0)
        return self.delta_fa


class BurstpropHiddenLayer(LayerBase):
    def __init__(self, in_features, out_features, next_features, p_baseline, Y_learning,
                 rec_input, rec_learning, activation_function="sigmoid", local_feedback_scale=1.0):
        super(BurstpropHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.next_features = next_features

        self.p_baseline = p_baseline

        self.Y_learning = Y_learning
        self.rec_input = rec_input
        self.rec_learning = rec_learning
        self.activation_function = activation_function
        # assert self.activation_function == "sigmoid"
        if self.activation_function == "sigmoid":
            self.f = torch.sigmoid
            self.f_deriv = lambda soma: torch.sigmoid(soma) * (1.0 - torch.sigmoid(soma))
        elif self.activation_function == "relu":
            self.f = torch.relu
            self.f_deriv = lambda soma: (soma > 0).float()

        self.local_feedback_scale = local_feedback_scale
        # self.p = self.p_baseline * torch.ones(self.out_features)
        self.register_buffer("p", self.p_baseline * torch.ones(self.out_features))

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

        self.Y_weight = nn.Parameter(torch.zeros(next_features, out_features))

        if self.rec_input:
            self.rec_weight = nn.Parameter(torch.zeros(out_features, out_features))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.soma = F.linear(self.input, self.W_weight, self.W_bias)
        # self.e = torch.sigmoid(self.soma)
        self.e = self.f(self.soma)

        return self.e

    def backward(self, next_burst_rates_t, next_burst_rates, feedback_scale):
        # next_burst_rates_mean = next_burst_rates.clone().detach()
        # next_burst_rates -= next_burst_rates_mean
        # next_burst_rates_t -= next_burst_rates_mean

        Y_input = next_burst_rates.mm(self.Y_weight)
        Y_input_t = next_burst_rates_t.mm(self.Y_weight)

        local_feedback_scale = self.local_feedback_scale

        # Y_input /= Y_input_std
        # Y_input_t /= Y_input_std
        # feedback_scale /= Y_input_std

        # local_feedback_scale = 1.0
        Y_input *= local_feedback_scale
        Y_input_t *= local_feedback_scale
        feedback_scale *= local_feedback_scale

        # Y_input_mean = Y_input.mean(axis=0)
        # Y_input_mean = Y_input.clone().detach()
        # Y_input -= Y_input_mean
        # Y_input_t -= Y_input_mean

        if self.rec_input:
            # TODO: Is this the correct code based on equations in the paper?
            p_pre = torch.sigmoid(4.0 * Y_input * (1 - self.e))

            # p_pre = torch.sigmoid(Y_input)
            b_pre = p_pre * self.e
            self.apic = Y_input * (1 - self.e) - b_pre.mm(self.rec_weight.T)
            self.apic_t = Y_input_t * (1 - self.e) - b_pre.mm(self.rec_weight.T)

            # apic = (Y_input - b_pre.mm(self.rec_weight.T)) * (1 - self.e)
            # apic_t = (Y_input_t - b_pre.mm(self.rec_weight.T)) * (1 - self.e)

            # # Ideal case
            # if not hasattr(self, "tmp"):
            #     self.tmp = True
            #     print("Running one-time test code")
            #     self.rec_weight.data = torch.eye(self.rec_weight.shape[0], device=self.rec_weight.device)
            #
            # p_pre = Y_input
            # b_pre = p_pre * self.e
            # apic = (Y_input - p_pre.mm(self.rec_weight.T)) * (1 - self.e)
            # apic_t = (Y_input_t - p_pre.mm(self.rec_weight.T)) * (1 - self.e)
            k = 2.0 / self.p_baseline
            self.p = torch.sigmoid(k * self.apic)
            self.p_t = torch.sigmoid(k * self.apic_t)
        else:
            # self.apic = Y_input * (1 - self.e)
            # self.apic_t = Y_input_t * (1 - self.e)
            self.apic = Y_input
            self.apic_t = Y_input_t

            k = 2.0 / self.p_baseline
            if self.activation_function == 'sigmoid':
                self.p = 2.0 * self.p_baseline * torch.sigmoid(k * self.apic * (1.0 - self.e))
                self.p_t = 2.0 * self.p_baseline * torch.sigmoid(k * self.apic_t * (1.0 - self.e))
            elif self.activation_function == 'relu':
                self.p = self.p_baseline * torch.ones_like(self.apic_t)
                self.p_t = self.p_baseline * torch.ones_like(self.apic_t)
                non_zero_mask = self.e > 1e-8
                self.p[non_zero_mask] = sigmoid_scale * 2.0 * self.p_baseline * torch.sigmoid(
                    k * self.apic[non_zero_mask] / self.e[non_zero_mask])
                self.p_t[non_zero_mask] = sigmoid_scale * 2.0 * self.p_baseline * torch.sigmoid(
                    k * self.apic_t[non_zero_mask] / self.e[non_zero_mask])

                assert torch.all(self.e[non_zero_mask] != 0)
                assert not torch.any(torch.isnan(next_burst_rates))
                assert not torch.any(torch.isnan(Y_input))
                assert not torch.any(torch.isnan(self.apic))
                assert not torch.any(torch.isnan(self.p_t))
                assert self.p_t.max() <= sigmoid_scale
                assert self.p_t.min() >= 0.0
            else:
                raise NotImplementedError

        feedback_scale *= sigmoid_scale

        # self.p = torch.sigmoid(4.0 * self.apic)
        # self.p_t = torch.sigmoid(4.0 * self.apic_t)

        # self.p = apic
        # self.p_t = apic_t

        # self.p = torch.sigmoid(0.0 * apic)
        # self.p_t = torch.sigmoid(4.0 * (apic_t - apic))

        # self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))
        self.b = self.p * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale * global_lr_scale

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        if self.Y_learning:
            raise NotImplementedError()

        if self.rec_learning:
            self.rec_weight.grad = -self.apic.transpose(0, 1).mm(b_pre) / (self.input.size(0))

            # # Ideal case
            # self.rec_weight.grad = -(Y_input - p_pre.mm(self.rec_weight.T)).transpose(0, 1).mm(p_pre) / (self.input.size(0))

        return self.b_t, self.b, feedback_scale

    def backward_bp(self, b_input_bp):
        assert getattr(self.W_weight, 'grad_bp', None) is None
        assert getattr(self.W_bias, 'grad_bp', None) is None

        # self.delta_bp = b_input_bp * self.e * (1 - self.e)
        self.delta_bp = b_input_bp * self.f_deriv(self.soma)
        self.W_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / self.input.size(0)
        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, next_delta_fa):
        assert getattr(self.W_weight, 'grad_fa', None) is None
        assert getattr(self.W_bias, 'grad_fa', None) is None

        # self.delta_fa = next_delta_fa.mm(self.Y_weight) * self.e * (1 - self.e)
        self.delta_fa = next_delta_fa.mm(self.Y_weight) * self.f_deriv(self.soma)
        self.W_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / self.input.size(0)
        return self.delta_fa


class BurstpropConvLayer(LayerBase, ABC):
    def __init__(self, in_channels, out_channels,
                 in_shape, kernel_size, stride,
                 padding, dilation, groups,
                 padding_mode, p_baseline, Y_learning, rec_input, rec_learning,
                 activation_function="sigmoid", local_feedback_scale=1.0):
        super(BurstpropConvLayer, self).__init__()

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
        self.rec_input = rec_input
        self.rec_learning = rec_learning
        assert not rec_input

        self.activation_function = activation_function
        if self.activation_function == "sigmoid":
            self.f = torch.sigmoid
            self.f_deriv = lambda soma: torch.sigmoid(soma) * (1.0 - torch.sigmoid(soma))
        elif self.activation_function == "relu":
            self.f = torch.relu
            self.f_deriv = lambda soma: (soma > 0).float()

        self.local_feedback_scale = local_feedback_scale

        self.W_weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size))
        self.W_bias = nn.Parameter(torch.zeros(out_channels))

        out_y = ((in_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1)
        out_x = ((in_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)

        self.out_shape = (out_y, out_x)
        self.register_buffer("p", sigmoid_scale * self.p_baseline * torch.ones(self.out_shape))

    def _forward_reshape(self, e):
        raise NotImplementedError

    def _backward_project_W(self, next_delta):
        return conv2d_input(self.input.shape, self.W_weight, next_delta, self.stride,
                            self.padding, self.dilation, self.groups)

    def _backward_project_Y(self, next_delta):
        raise NotImplementedError

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.soma = F.conv2d(self.input, self.W_weight, self.W_bias, self.stride, self.padding, self.dilation,
                             self.groups)
        # self.e = torch.sigmoid(self.soma)
        self.e = self.f(self.soma)

        return self._forward_reshape(self.e)

    # def backward(self, next_burst_rates, next_event_rates, feedback_scale):
    #     # self.Y_input = next_burst_rates.mm(self.Y_weight)
    #     self.Y_input = self._backward_project_Y(next_burst_rates)
    #     # self.Y_input = ((next_burst_rates + next_event_rates) / 3.0).mm(self.Y_weight)
    #     # self.Q_input = next_event_rates.mm(self.Q_weight)
    #     self.Q_input = self._backward_project_Q(next_event_rates)
    #     self.apic = self.Y_input + self.Q_input
    #
    #     k = 2.0 / self.p_baseline
    #     self.p_t = 2.0 * self.p_baseline * torch.sigmoid(k * self.apic * (1.0 - self.e))
    #
    #     # self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))
    #     self.b = self.p_baseline * self.e
    #     self.b_t = self.p_t * self.e
    #
    #     self.delta = -(self.b_t - self.b)
    #
    #     update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale
    #
    #     # feedback_scale *= 2.0 * self.p_baseline
    #
    #     self.W_weight.grad = conv2d_weight(self.input, self.W_weight.shape, self.delta, self.stride,
    #                                        self.padding, self.dilation, self.groups) / update_scale_factor
    #     # self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor
    #     self.W_bias.grad = torch.sum(self.delta, dim=[0, 2, 3]) / update_scale_factor
    #
    #     if self.Y_learning:
    #         raise NotImplementedError
    #         # self.Y_weight.grad = next_burst_rates.transpose(0, 1).mm(self.apic) / update_scale_factor
    #
    #         # next_burst_event_std_ratio = (next_event_rates * self.p_baseline).std(axis=1) / next_burst_rates.std(axis=1)
    #         # Y_lr = next_burst_event_std_ratio
    #         # Y_lr = (next_burst_event_std_ratio.clone() >= 0.99).float()
    #
    #         # self.Y_weight.grad = next_burst_rates.transpose(0, 1).mm(Y_lr.unsqueeze(1) * self.apic) / update_scale_factor
    #         # self.Y_weight.grad = ((next_burst_rates + next_event_rates) / 3.0).transpose(0, 1).mm(self.apic) / update_scale_factor
    #
    #     return self.b_t, self.e, feedback_scale

    def backward(self, next_burst_rates_t, next_burst_rates, feedback_scale):
        # Y_input = next_burst_rates.mm(self.Y_weight)
        # Y_input_t = next_burst_rates_t.mm(self.Y_weight)
        # next_burst_rates_mean = 0.95 * next_burst_rates.clone().detach()
        # next_burst_rates -= next_burst_rates_mean
        # next_burst_rates_t -= next_burst_rates_mean

        Y_input = self._backward_project_Y(next_burst_rates)
        Y_input_t = self._backward_project_Y(next_burst_rates_t)

        # Y_input_std = Y_input.std()

        local_feedback_scale = self.local_feedback_scale

        # Y_input /= Y_input_std
        # Y_input_t /= Y_input_std
        # feedback_scale /= Y_input_std

        # local_feedback_scale = 1.0
        Y_input *= local_feedback_scale
        Y_input_t *= local_feedback_scale
        feedback_scale *= local_feedback_scale

        # Y_input_mean = Y_input.mean(axis=0)
        # # Y_input_mean = Y_input.clone().detach()
        # Y_input -= Y_input_mean
        # Y_input_t -= Y_input_mean

        if self.rec_input:
            p_pre = torch.sigmoid(4.0 * Y_input * (1 - self.e))

            # p_pre = torch.sigmoid(Y_input)
            b_pre = p_pre * self.e
            self.apic = Y_input * (1 - self.e) - b_pre.mm(self.rec_weight.T)
            self.apic_t = Y_input_t * (1 - self.e) - b_pre.mm(self.rec_weight.T)

            # apic = (Y_input - b_pre.mm(self.rec_weight.T)) * (1 - self.e)
            # apic_t = (Y_input_t - b_pre.mm(self.rec_weight.T)) * (1 - self.e)

            # # Ideal case
            # if not hasattr(self, "tmp"):
            #     self.tmp = True
            #     print("Running one-time test code")
            #     self.rec_weight.data = torch.eye(self.rec_weight.shape[0], device=self.rec_weight.device)
            #
            # p_pre = Y_input
            # b_pre = p_pre * self.e
            # apic = (Y_input - p_pre.mm(self.rec_weight.T)) * (1 - self.e)
            # apic_t = (Y_input_t - p_pre.mm(self.rec_weight.T)) * (1 - self.e)
        else:
            # self.apic = Y_input * (1 - self.e)
            # self.apic_t = Y_input_t * (1 - self.e)

            self.apic = Y_input
            self.apic_t = Y_input_t

            k = 2.0 / self.p_baseline
            if self.activation_function == 'sigmoid':
                self.p = 2.0 * self.p_baseline * torch.sigmoid(k * self.apic * (1.0 - self.e))
                self.p_t = 2.0 * self.p_baseline * torch.sigmoid(k * self.apic_t * (1.0 - self.e))
            elif self.activation_function == 'relu':
                self.p = self.p_baseline * torch.ones_like(self.apic_t)
                self.p_t = self.p_baseline * torch.ones_like(self.apic_t)
                non_zero_mask = self.e > 1e-8
                self.p[non_zero_mask] = sigmoid_scale * 2.0 * self.p_baseline * torch.sigmoid(
                    k * self.apic[non_zero_mask] / self.e[non_zero_mask])
                self.p_t[non_zero_mask] = sigmoid_scale * 2.0 * self.p_baseline * torch.sigmoid(
                    k * self.apic_t[non_zero_mask] / self.e[non_zero_mask])

                assert torch.all(self.e[non_zero_mask] != 0)
                assert not torch.any(torch.isnan(next_burst_rates))
                assert not torch.any(torch.isnan(Y_input))
                assert not torch.any(torch.isnan(self.apic))
                assert not torch.any(torch.isnan(self.p_t))
                assert self.p_t.max() <= sigmoid_scale
                assert self.p_t.min() >= 0.0
            else:
                raise NotImplementedError

        feedback_scale *= sigmoid_scale

        # self.p = torch.sigmoid(4.0 * self.apic)
        # self.p_t = torch.sigmoid(4.0 * self.apic_t)

        # self.p = apic
        # self.p_t = apic_t

        # self.p = torch.sigmoid(0.0 * apic)
        # self.p_t = torch.sigmoid(4.0 * (apic_t - apic))

        # self.p_t = torch.sigmoid(4.0 * self.apic * (1 - self.e))
        self.b = self.p * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale * global_lr_scale

        self.W_weight.grad = conv2d_weight(self.input, self.W_weight.shape, self.delta, self.stride,
                                           self.padding, self.dilation, self.groups) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=[0, 2, 3]) / update_scale_factor

        if self.Y_learning:
            raise NotImplementedError

        if self.rec_input and self.rec_learning:
            raise NotImplementedError
            # self.rec_weight.grad = -self.apic.transpose(0, 1).mm(b_pre) / (self.input.size(0))

            # # Ideal case
            # self.rec_weight.grad = -(Y_input - p_pre.mm(self.rec_weight.T)).transpose(0, 1).mm(p_pre) / (self.input.size(0))

        return self.b_t, self.b, feedback_scale

    def backward_bp(self, b_input_bp):
        assert getattr(self.W_weight, 'grad_bp', None) is None
        assert getattr(self.W_bias, 'grad_bp', None) is None

        # self.delta_bp = self._backward_reshape(b_input_bp) * self.e * (1 - self.e)
        self.delta_bp = self._backward_reshape(b_input_bp) * self.f_deriv(self.soma)
        self.W_weight.grad_bp = conv2d_weight(self.input, self.W_weight.shape, self.delta_bp, self.stride,
                                              self.padding, self.dilation, self.groups) / self.input.size(0)

        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=[0, 2, 3]) / self.input.size(0)

        return self._backward_project_W(self.delta_bp)

    def backward_fa(self, next_delta_fa):
        assert getattr(self.W_weight, 'grad_fa', None) is None
        assert getattr(self.W_bias, 'grad_fa', None) is None

        # self.delta_fa = self._backward_project_Y(next_delta_fa) * self.e * (1 - self.e)
        self.delta_fa = self._backward_project_Y(next_delta_fa) * self.f_deriv(self.soma)

        self.W_weight.grad_fa = conv2d_weight(self.input, self.W_weight.shape, self.delta_fa, self.stride,
                                              self.padding, self.dilation, self.groups) / self.input.size(0)

        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=[0, 2, 3]) / self.input.size(0)

        return self.delta_fa


class BurstpropConvHiddenLayer(BurstpropConvLayer):
    def __init__(self, *, next_channels, next_kernel_size, next_stride, **base_kwargs):
        super().__init__(**base_kwargs)

        self.next_stride = next_stride
        self.Y_weight = nn.Parameter(torch.zeros(next_channels, self.out_channels // self.groups, *next_kernel_size))

    def _forward_reshape(self, e):
        return e

    def _backward_reshape(self, b_input):
        return b_input

    def _backward_project_Y(self, next_delta):
        return conv2d_input(self.e.shape, self.Y_weight, next_delta, self.next_stride,
                            self.padding, self.dilation, self.groups)


class BurstpropConvFinalLayer(BurstpropConvLayer):
    def __init__(self, *, next_features, **base_kwargs):
        super().__init__(**base_kwargs)

        self.out_features = self.out_channels * self.out_shape[0] * self.out_shape[1]
        self.Y_weight = nn.Parameter(torch.zeros(next_features, self.out_features))

    def _forward_reshape(self, e):
        output_size = e.size()
        return e.view(output_size[0], -1)

    def _backward_reshape(self, b_input):
        return b_input.reshape(self.e.shape)

    def _backward_project_Y(self, next_delta):
        return next_delta.mm(self.Y_weight).view(self.e.shape)

