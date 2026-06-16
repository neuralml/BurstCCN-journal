from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_input, conv2d_weight

from burstccn.models.networks.activation_functions import get_activation_function
from burstccn.models.networks.layers.base import LayerBase


class ANNOutputLayer(LayerBase):
    def __init__(self, in_features, out_features, activation_function="sigmoid"):
        super(ANNOutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

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

    def backward(self, target):
        if target is None:
            target = self.e

        diff = -(target - self.e)
        self.delta = self.act.backward(diff, self.act_ctx)

        update_scale_factor = self.input.size(0)

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        return self.delta

    def backward_bp(self, target):
        diff = -(target - self.e)
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


class ANNHiddenLayer(LayerBase):
    def __init__(self, in_features, out_features, next_features, Y_learning, activation_function="sigmoid"):
        super(ANNHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.next_features = next_features

        self.Y_learning = Y_learning
        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

        self.Y_weight = nn.Parameter(torch.zeros(next_features, out_features))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.soma = F.linear(self.input, self.W_weight, self.W_bias)
        self.e, self.act_ctx = self.act(self.soma)

        return self.e

    def backward(self, next_delta):
        g = next_delta.mm(self.Y_weight)

        self.delta = self.act.backward(g, self.act_ctx)

        update_scale_factor = self.input.size(0)

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        if self.Y_learning:
            # raise NotImplementedError
            kp_grad = next_delta.transpose(0, 1).mm(self.e)
            self.Y_weight.grad = kp_grad / update_scale_factor

        return self.delta

    def backward_bp(self, b_input_bp):
        self.delta_bp = self.act.backward(b_input_bp, self.act_ctx)

        self.W_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / self.input.size(0)
        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, next_delta_fa):
        g = next_delta_fa.mm(self.Y_weight)
        self.delta_fa = self.act.backward(g, self.act_ctx)

        self.W_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / self.input.size(0)
        return self.delta_fa


class ANNConvLayer(LayerBase, ABC):
    def __init__(self, in_channels, out_channels,
                       in_shape, kernel_size, stride,
                       padding, dilation, groups,
                       padding_mode, Y_learning, activation_function="sigmoid"):
        super(ANNConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_shape = in_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.Y_learning = Y_learning
        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

        self.W_weight = nn.Parameter(torch.zeros(out_channels, in_channels // groups, *kernel_size))
        self.W_bias = nn.Parameter(torch.zeros(out_channels))

        out_y = ((in_shape[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1)
        out_x = ((in_shape[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1)

        self.out_shape = (out_y, out_x)


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

        self.soma = F.conv2d(self.input, self.W_weight, self.W_bias, self.stride, self.padding, self.dilation, self.groups)
        self.e, self.act_ctx = self.act(self.soma)

        return self._forward_reshape(self.e)

    def backward(self, next_delta):
        g = self._backward_project_Y(next_delta)
        self.delta = self.act.backward(g, self.act_ctx)

        update_scale_factor = self.input.size(0)

        self.W_weight.grad = conv2d_weight(self.input, self.W_weight.shape, self.delta, self.stride,
                                                   self.padding, self.dilation, self.groups) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=[0, 2, 3]) / update_scale_factor

        if self.Y_learning:
            if self.Y_weight.dim() == 4:
                kp_grad = conv2d_weight(
                    input=self.e,
                    weight_size=self.Y_weight.shape,
                    grad_output=next_delta,
                    stride=self.next_stride,
                    padding=self.padding,
                    dilation=self.dilation,
                    groups=self.groups,
                )
            elif self.Y_weight.dim() == 2:
                kp_grad = next_delta.t().mm(self.e.view(self.e.size(0), -1))
            else:
                raise RuntimeError(f"Unexpected Y_weight dim: {self.Y_weight.dim()}")

            self.Y_weight.grad = kp_grad / update_scale_factor

        return self.delta

    def backward_bp(self, b_input_bp):
        g = self._backward_reshape(b_input_bp)

        self.delta_bp = self.act.backward(g, self.act_ctx)

        self.W_weight.grad_bp = conv2d_weight(self.input, self.W_weight.shape, self.delta_bp, self.stride,
                                              self.padding, self.dilation, self.groups) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=[0, 2, 3]) / self.input.size(0)

        return self._backward_project_W(self.delta_bp)

    def backward_fa(self, next_delta_fa):
        g = self._backward_project_Y(next_delta_fa)

        self.delta_fa = self.act.backward(g, self.act_ctx)

        self.W_weight.grad_fa = conv2d_weight(self.input, self.W_weight.shape, self.delta_fa, self.stride,
                                              self.padding, self.dilation, self.groups) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=[0, 2, 3]) / self.input.size(0)

        return self.delta_fa


class ANNConvHiddenLayer(ANNConvLayer):
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


class ANNConvFinalLayer(ANNConvLayer):
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

