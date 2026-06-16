import torch
import torch.nn as nn
import torch.nn.functional as F

from burstccn.models.networks.layers.base import LayerBase


class EDNOutputLayer(LayerBase):
    def __init__(self, in_features, out_features, lambda_output, activation_function="sigmoid"):
        super(EDNOutputLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lambda_output = lambda_output
        self.activation_function = activation_function
        assert self.activation_function == "sigmoid"

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, input, forward_noise=None):
        # if forward_noise is not None:
        #     self.input = input + forward_noise * torch.randn_like(input)
        # else:
        self.input = input

        self.pyr_basal = F.linear(self.input, self.W_weight, self.W_bias)
        self.pyr_soma = self.pyr_basal

        self.pyr_basal_rate = torch.sigmoid(self.pyr_basal)
        self.pyr_soma_rate = torch.sigmoid(self.pyr_soma)

        return self.pyr_soma_rate

    def backward(self, target):
        if target is None:
            target_potentials = self.pyr_basal
        else:
            softened_rate = 0.2 + target * 0.6
            target_potentials = torch.log(softened_rate / (1.0 - softened_rate))

        self.pyr_soma_t = self.pyr_basal + self.lambda_output * (target_potentials - self.pyr_basal)
        self.pyr_soma_rate_t = torch.sigmoid(self.pyr_soma_t)

        self.delta = -(self.pyr_soma_rate_t - self.pyr_basal_rate)

        update_scale_factor = self.input.size(0) * self.lambda_output

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        feedback_scale = 1.0

        return self.pyr_soma_t, self.pyr_soma_rate_t, feedback_scale

    def backward_bp(self, target):
        self.delta_bp = -(target - self.pyr_soma_rate) * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
        self.W_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / self.input.size(0)
        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, target):
        self.delta_fa = -(target - self.pyr_soma_rate) * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
        self.W_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / self.input.size(0)
        return self.delta_fa

    def get_state(self, state_key):
        if state_key == 'e':
            return self.get_state('pyr_basal_rate')
        elif state_key == 'p':
            raise NotImplementedError
        else:
            return super().get_state(state_key)


class EDNIdealOutputLayer(EDNOutputLayer):
    def backward(self, target):
        if target is None:
            # target = self.pyr_basal
            self.pyr_soma_t = self.pyr_basal
        else:
            perturbed_output = self.pyr_soma_rate + self.lambda_output * (target - self.pyr_soma_rate) * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
            perturbed_output = torch.clamp(perturbed_output, 1e-6, 1 - 1e-6)
            self.pyr_soma_t = torch.logit(perturbed_output)

        # self.pyr_soma_t = self.pyr_basal + self.lambda_output * (target - self.pyr_soma_rate)
        # self.pyr_soma_rate_t = torch.sigmoid(self.pyr_soma_t)

        self.pyr_soma_rate_t = torch.sigmoid(self.pyr_soma_t)
        self.delta = -(self.pyr_soma_rate_t - self.pyr_basal_rate)

        update_scale_factor = self.input.size(0) * self.lambda_output

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / update_scale_factor

        feedback_scale = self.lambda_output
        return self.pyr_soma_t, self.pyr_soma_rate_t, feedback_scale


class EDNHiddenLayer(LayerBase):
    def __init__(self, in_features, out_features, next_features, lambda_intn, lambda_hidden, Y_learning,
                 pyr_intn_learning, intn_pyr_learning, activation_function="sigmoid"):
        super(EDNHiddenLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.next_features = next_features

        self.lambda_intn = lambda_intn
        self.lambda_hidden = lambda_hidden

        self.Y_learning = Y_learning
        self.pyr_intn_learning = pyr_intn_learning
        self.intn_pyr_learning = intn_pyr_learning

        self.activation_function = activation_function
        assert self.activation_function == "sigmoid"

        self.W_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_bias = nn.Parameter(torch.zeros(out_features))

        self.Y_weight = nn.Parameter(torch.zeros(next_features, out_features))

        self.pyr_intn_weight = nn.Parameter(torch.zeros(next_features, out_features))
        self.pyr_intn_bias = nn.Parameter(torch.zeros(next_features))

        self.intn_pyr_weight = nn.Parameter(torch.zeros(next_features, out_features))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.pyr_basal = F.linear(self.input, self.W_weight, self.W_bias)
        self.pyr_soma = self.pyr_basal

        self.pyr_basal_rate = torch.sigmoid(self.pyr_basal)
        self.pyr_soma_rate = torch.sigmoid(self.pyr_soma)

        self.intn_basal = F.linear(self.pyr_soma_rate, self.pyr_intn_weight, self.pyr_intn_bias)
        self.intn_soma = self.intn_basal

        self.intn_basal_rate = torch.sigmoid(self.intn_basal)
        self.intn_soma_rate = torch.sigmoid(self.intn_soma)

        return self.pyr_soma_rate

    def backward(self, next_layer_pyr_soma_t, next_layer_pyr_soma_rate_t, feedback_scale):
        self.intn_soma = (1 - self.lambda_intn) * self.intn_basal + self.lambda_intn * next_layer_pyr_soma_t
        self.intn_soma_rate = torch.sigmoid(self.intn_soma)

        self.pyr_apical = next_layer_pyr_soma_rate_t.mm(self.Y_weight) - self.intn_soma_rate.mm(self.intn_pyr_weight)

        self.pyr_soma_t = self.pyr_basal + self.lambda_hidden * self.pyr_apical
        self.pyr_soma_rate_t = torch.sigmoid(self.pyr_soma_t)

        self.delta = -(self.pyr_soma_rate_t - self.pyr_basal_rate)

        feedback_scale *= self.lambda_hidden * (1 - self.lambda_intn)
        pyr_update_scale_factor = self.input.size(0) * feedback_scale
        # pyr_intn_update_scale_factor = self.input.size(0) * self.lambda_intn
        pyr_intn_update_scale_factor = self.input.size(0) * feedback_scale
        intn_pyr_update_scale_factor = self.input.size(0)

        self.W_weight.grad = self.delta.transpose(0, 1).mm(self.input) / pyr_update_scale_factor
        self.W_bias.grad = torch.sum(self.delta, dim=0) / pyr_update_scale_factor

        if self.Y_learning:
            raise NotImplementedError
            # self.Y_weight.grad = #

        if self.pyr_intn_learning:
            pyr_intn_delta = self.intn_soma_rate - self.intn_basal_rate

            # # TODO: CHECK THIS
            # pyr_intn_delta = self.intn_soma - self.intn_basal

            self.pyr_intn_weight.grad = -pyr_intn_delta.transpose(0, 1).mm(self.pyr_soma_rate_t) / pyr_intn_update_scale_factor
            self.pyr_intn_bias.grad = torch.sum(-pyr_intn_delta, dim=0) / pyr_intn_update_scale_factor

        if self.intn_pyr_learning:
            self.intn_pyr_weight.grad = -self.intn_soma_rate.transpose(0, 1).mm(self.pyr_apical) / intn_pyr_update_scale_factor

        return self.pyr_soma_t, self.pyr_soma_rate_t, feedback_scale

    def backward_bp(self, b_input_bp):
        assert getattr(self.W_weight, 'grad_bp', None) is None
        assert getattr(self.W_bias, 'grad_bp', None) is None

        self.delta_bp = b_input_bp * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
        self.W_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / self.input.size(0)
        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, next_delta_fa):
        assert getattr(self.W_weight, 'grad_fa', None) is None
        assert getattr(self.W_bias, 'grad_fa', None) is None

        self.delta_fa = next_delta_fa.mm(self.Y_weight) * self.pyr_soma_rate * (1 - self.pyr_soma_rate)
        self.W_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / self.input.size(0)
        self.W_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / self.input.size(0)
        return self.delta_fa

    def get_state(self, state_key):
        if state_key == 'e':
            return self.get_state('pyr_basal_rate')
        elif state_key == 'p':
            raise NotImplementedError
        else:
            return super().get_state(state_key)
