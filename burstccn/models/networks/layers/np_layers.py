import torch
import torch.nn as nn
import torch.nn.functional as F

from burstccn.models.networks.activation_functions import get_activation_function
from burstccn.models.networks.layers.base import LayerBase


class NPOutputLayer(LayerBase):
    def __init__(self, in_features, out_features, perturb_sigma, activation_function="sigmoid"):
        super(NPOutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

        self.perturb_sigma = perturb_sigma #0.1# / math.sqrt(self.out_features)

    def forward(self, input, perturb=False):
        if perturb:
            self.input_p = input

            self.soma_p = F.linear(input, self.W_weight, self.W_bias)
            self.perturb = self.perturb_sigma * torch.randn_like(self.soma_p)
            self.soma_p += self.perturb

            self.e_p, self.act_ctx_p = self.act(self.soma_p)

            return self.e_p
        else:
            self.input = input

            self.soma = F.linear(self.input, self.W_weight, self.W_bias)
            self.e, self.act_ctx = self.act(self.soma)

            return self.e

    def backward(self, loss_pre, loss_post):
        # if target is None:
        #     target = self.e

        # diff = -(target - self.e)
        # self.delta = self.act.backward(diff, self.act_ctx)

        loss_diff = (loss_post - loss_pre).view(-1, 1)   # [B, 1]
        self.delta = loss_diff * (self.perturb / (self.perturb_sigma**2))

        update_scale_factor = self.input.size(0)

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input_p) / update_scale_factor
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


class NPHiddenLayer(LayerBase):
    def __init__(self, in_features, out_features, next_features, Y_learning, perturb_sigma, activation_function="sigmoid"):
        super(NPHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.next_features = next_features

        self.Y_learning = Y_learning
        self.activation_function = activation_function
        self.act = get_activation_function(self.activation_function)

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

        self.Y_weight = nn.Parameter(torch.zeros(next_features, out_features))

        self.perturb_sigma = perturb_sigma # 0.1# / math.sqrt(self.out_features)

    def forward(self, input, perturb=False):
        if perturb:
            self.input_p = input

            self.soma_p = F.linear(input, self.W_weight, self.W_bias)
            self.perturb = self.perturb_sigma * torch.randn_like(self.soma_p)
            self.soma_p += self.perturb

            self.e_p, self.act_ctx_p = self.act(self.soma_p)

            return self.e_p
        else:
            self.input = input

            self.soma = F.linear(self.input, self.W_weight, self.W_bias)
            self.e, self.act_ctx = self.act(self.soma)

            return self.e

    def backward(self, loss_pre, loss_post):
        loss_diff = (loss_post - loss_pre).view(-1, 1)  # [B, 1]
        self.delta = loss_diff * (self.perturb / (self.perturb_sigma ** 2))

        update_scale_factor = self.input.size(0)

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input_p) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        if self.Y_learning:
            raise NotImplementedError

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