import torch
import torch.nn as nn
import torch.nn.functional as F

from burstccn.models.networks.layers.base import LayerBase
from burstccn.models.networks.activation_functions import get_activation_function


class DalesBurstCCNOutputLayer(LayerBase):
    def __init__(self, in_features, out_features, p_baseline, error_scale, activation_function="sigmoid"):
        super(DalesBurstCCNOutputLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline = p_baseline
        self.error_scale = error_scale
        self.activation_function = activation_function

        assert self.activation_function == "sigmoid"
        self.act = get_activation_function(self.activation_function)

        # self.p = self.p_baseline * torch.ones(self.out_features)
        self.register_buffer("p", self.p_baseline * torch.ones(self.out_features))

        self.W_direct_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_direct_bias = nn.Parameter(torch.zeros(out_features))

        self.n_PV = int(in_features)
        self.W_to_PV_weight = nn.Parameter(torch.zeros(self.n_PV, in_features))
        self.W_from_PV_weight = nn.Parameter(torch.zeros(out_features, self.n_PV))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            assert forward_noise != 0.0
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.pv = F.linear(self.input, self.W_to_PV_weight, bias=None)
        self.excitatory_input = F.linear(self.input, self.W_direct_weight, bias=None)
        self.inhibitory_input = -1 * F.linear(self.pv, self.W_from_PV_weight, bias=None)
        self.bias_input = self.W_direct_bias

        assert torch.all(self.excitatory_input >= 0)
        assert torch.all(self.inhibitory_input <= 0)

        self.soma = self.excitatory_input + self.inhibitory_input + self.bias_input
        self.e, self.act_ctx = self.act(self.soma)
        return self.e
    
    def calc_p_sigmoid(self, target):
        return self.p_baseline * ((target - self.e) * (1 - self.e) * self.error_scale + 1)

    def backward(self, target):
        if target is None:
            target = self.e

        # self.p = self.p_baseline
        # self.p_t = self.p_baseline * ((target - self.e) * (1 - self.e) + 1)
        if self.activation_function == "sigmoid":
            self.p_t = self.calc_p_sigmoid(target)
        else:
            raise NotImplementedError

        self.b = self.p * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        weight_update_scale_factor = self.input.size(0) * self.p_baseline * self.error_scale * 2
        bias_update_scale_factor = self.input.size(0) * self.p_baseline * self.error_scale

        self.W_direct_weight.grad = self.delta.transpose(0, 1).mm(self.input) / weight_update_scale_factor
        self.W_direct_bias.grad = torch.sum(self.delta, dim=0) / bias_update_scale_factor

        self.W_from_PV_weight.grad = -self.delta.transpose(0, 1).mm(self.pv) / weight_update_scale_factor

        return self.b_t, self.e, self.error_scale

    @property
    def W_weight(self):
        def safe_sub(a, b):
            return a - b if a is not None and b is not None else None

        effective_W_weight = self.W_direct_weight - self.W_from_PV_weight

        effective_W_weight.grad = safe_sub(self.W_direct_weight.grad, self.W_from_PV_weight.grad)
        effective_W_weight.grad_bp = safe_sub(
            getattr(self.W_direct_weight, 'grad_bp', None),
            getattr(self.W_from_PV_weight, 'grad_bp', None)
        )
        effective_W_weight.grad_fa = safe_sub(
            getattr(self.W_direct_weight, 'grad_fa', None),
            getattr(self.W_from_PV_weight, 'grad_fa', None)
        )

        return effective_W_weight

    @property
    def W_bias(self):
        return self.W_direct_bias

    def backward_bp(self, target):
        diff = -(target - self.e)
        self.b_input_bp = diff
        self.delta_bp = self.act.backward(diff, self.act_ctx)

        update_scale_factor = self.input.size(0) * 2.0
        self.W_direct_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_direct_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / update_scale_factor

        self.W_from_PV_weight.grad_bp = -self.delta_bp.transpose(0, 1).mm(self.pv) / update_scale_factor

        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, target):
        diff = -(target - self.e)
        self.delta_fa = self.act.backward(diff, self.act_ctx)

        update_scale_factor = self.input.size(0) * 2.0
        self.W_direct_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_direct_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / update_scale_factor

        self.W_from_PV_weight.grad_fa = -self.delta_fa.transpose(0, 1).mm(self.pv) / update_scale_factor

        return self.delta_fa


class DalesBurstCCNHiddenLayer(LayerBase):
    def __init__(self, in_features, out_features, next_features, p_baseline, Y_learning, Q_learning,
                 Y_grad_type, Q_grad_type, activation_function="sigmoid", feedback_bottleneck_size=None,
                 local_feedback_scale=1.0, apical_bias_learning=False):
        super(DalesBurstCCNHiddenLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.next_features = next_features

        self.p_baseline = p_baseline
        self.Y_learning = Y_learning
        self.Q_learning = Q_learning
        self.Y_grad_type = Y_grad_type
        self.Q_grad_type = Q_grad_type
        self.activation_function = activation_function
        self.local_feedback_scale = local_feedback_scale
        self.apical_bias_learning = apical_bias_learning

        assert self.activation_function == "sigmoid"
        self.act = get_activation_function(self.activation_function)

        self.feedback_bottleneck_size = (
            feedback_bottleneck_size if feedback_bottleneck_size is not None else self.next_features
        )
        # self.p = self.p_baseline * torch.ones(self.out_features)
        self.register_buffer("p", self.p_baseline * torch.ones(self.out_features))

        self.W_direct_weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.W_direct_bias = nn.Parameter(torch.zeros(out_features))

        self.n_PV = int(in_features)
        self.W_to_PV_weight = nn.Parameter(torch.zeros(self.n_PV, in_features))
        self.W_from_PV_weight = nn.Parameter(torch.zeros(out_features, self.n_PV))

        self.n_NDNF = int(next_features)
        self.Q_direct_weight = nn.Parameter(torch.zeros(next_features, out_features))
        self.Q_to_NDNF_weight = nn.Parameter(torch.zeros(next_features, self.n_NDNF))
        self.Q_from_NDNF_weight = nn.Parameter(torch.zeros(self.n_NDNF, out_features))

        self.n_VIP1 = int(self.feedback_bottleneck_size)
        self.n_SST1 = int(self.feedback_bottleneck_size)
        self.Y_to_VIP1_weight = nn.Parameter(torch.zeros(next_features, self.n_VIP1))
        self.Y_to_SST1_weight = nn.Parameter(torch.zeros(next_features, self.n_SST1))
        self.Y_VIP1_to_SST1_weight = nn.Parameter(torch.zeros(self.n_VIP1, self.n_SST1))
        self.Y_from_SST1_weight = nn.Parameter(torch.zeros(self.n_SST1, out_features))

        self.n_VIP2 = int(self.feedback_bottleneck_size)
        self.n_SST2 = int(self.feedback_bottleneck_size)
        self.Y_to_VIP2_weight = nn.Parameter(torch.zeros(next_features, self.n_VIP2))
        self.Y_to_SST2_weight = nn.Parameter(torch.zeros(next_features, self.n_SST2))
        self.Y_VIP2_to_SST2_weight = nn.Parameter(torch.zeros(self.n_VIP2, self.n_SST2))
        self.Y_from_SST2_weight = nn.Parameter(torch.zeros(self.n_SST2, out_features))

        self.sst1_bias = nn.Parameter(torch.zeros(self.n_SST1))  # or register_buffer if fixed
        self.sst2_bias = nn.Parameter(torch.zeros(self.n_SST2))
        self.apical_bias_input = nn.Parameter(torch.zeros(out_features))

        # self.Y_weight = nn.Parameter(torch.zeros(next_features, out_features))
        # self.Q_weight = nn.Parameter(torch.zeros(next_features, out_features))

    def forward(self, input, forward_noise=None):
        if forward_noise is not None:
            assert forward_noise != 0.0
            self.input = input + forward_noise * torch.randn_like(input)
        else:
            self.input = input

        self.pv = F.linear(self.input, self.W_to_PV_weight, bias=None)
        self.excitatory_input = F.linear(self.input, self.W_direct_weight, bias=None)
        self.inhibitory_input = -1 * F.linear(self.pv, self.W_from_PV_weight, bias=None)
        self.bias_input = self.W_direct_bias

        # assert torch.all(self.excitatory_input >= 0)
        # assert torch.all(self.inhibitory_input <= 0)

        self.soma = self.excitatory_input + self.inhibitory_input + self.bias_input
        self.e, self.act_ctx = self.act(self.soma)
        return self.e

    def backward(self, next_burst_rates, next_event_rates, feedback_scale):
        self.vip1 = next_burst_rates @ self.Y_to_VIP1_weight
        self.sst1 = next_burst_rates @ self.Y_to_SST1_weight - self.vip1 @ self.Y_VIP1_to_SST1_weight
        # self.sst1_bias = (-(self.Y_to_SST1_weight - self.Y_to_VIP1_weight) * ((self.Y_to_SST1_weight - self.Y_to_VIP1_weight) < 0)).sum(dim=0)

        self.sst1 += self.sst1_bias

        self.vip2 = next_burst_rates @ self.Y_to_VIP2_weight
        self.sst2 = next_burst_rates @ self.Y_to_SST2_weight - self.vip2 @ self.Y_VIP2_to_SST2_weight
        # self.sst2_bias = (-(self.Y_to_SST2_weight - self.Y_to_VIP2_weight) * ((self.Y_to_SST2_weight - self.Y_to_VIP2_weight) < 0)).sum(dim=0)
        self.sst2 += self.sst2_bias

        assert torch.all(self.sst1 >= 0)
        assert torch.all(self.sst2 >= 0)

        # self.apical_bias_input = 0.0 #self.Y_from_SST1_weight.sum(dim=0, keepdim=True) + self.Y_from_SST2_weight.sum(dim=0, keepdim=True)
        # self.apical_bias_input = self.sst1_bias @ self.Y_from_SST1_weight + self.sst2_bias @ self.Y_from_SST2_weight

        self.Y_input = self.apical_bias_input - self.sst1 @ self.Y_from_SST1_weight - self.sst2 @ self.Y_from_SST2_weight

        self.ndnf = next_event_rates @ self.Q_to_NDNF_weight
        # ndnf_for_Q = self.ndnf + getattr(self, "ndnf_backward_offset", 0.0)
        # self.Q_input = next_event_rates @ self.Q_direct_weight - ndnf_for_Q @ self.Q_from_NDNF_weight
        self.Q_input = next_event_rates @ self.Q_direct_weight - self.ndnf @ self.Q_from_NDNF_weight - getattr(self, "ndnf_backward_offset", 0.0)

        self.apic = self.Y_input + self.Q_input

        # if getattr(self, "random_flip_apical", False):
        #     flip_rows = (torch.rand(self.apic.size(0), 1, device=self.apic.device) < 0.1)  # (32,1) bool
        #     sign = torch.where(flip_rows, -1.0, 1.0).to(self.apic.dtype)  # (32,1)
        #     self.apic *= sign  # broadcasts across 500 neurons

        self.apic *= self.local_feedback_scale
        feedback_scale *= self.local_feedback_scale

        if self.activation_function == 'sigmoid':
            self.p_t = self.calc_p_sigmoid()
        else:
            raise NotImplementedError

        self.b = self.p_baseline * self.e
        self.b_t = self.p_t * self.e

        self.delta = -(self.b_t - self.b)

        weight_update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale * 2
        bias_update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale

        self.W_direct_weight.grad = self.delta.transpose(0, 1).mm(self.input) / weight_update_scale_factor
        self.W_direct_bias.grad = torch.sum(self.delta, dim=0) / bias_update_scale_factor

        self.W_from_PV_weight.grad = -self.delta.transpose(0, 1).mm(self.pv) / weight_update_scale_factor

        if self.Y_learning:
            if self.Y_grad_type == 'apic':
                y_weight_update_scale_factor = self.input.size(0) * self.p_baseline * feedback_scale
                y1_grad, y2_grad = self.calc_Y_apic_grads()
                self.Y_from_SST1_weight.grad = y1_grad / y_weight_update_scale_factor
                self.Y_from_SST2_weight.grad = y2_grad / y_weight_update_scale_factor
            else:
                raise ValueError(f"Invalid Y grad type {self.Y_grad_type}")

            # self.apical_bias_input.grad = torch.sum(self.apic, dim=0) / bias_update_scale_factor
            # self.apical_bias_input.data += -0.01 * torch.sum(self.apic, dim=0)

        if self.Q_learning:
            raise NotImplementedError
            self.Q_weight.grad = -next_event_rates.transpose(0, 1).mm(self.apic) / update_scale_factor

        if self.apical_bias_learning:
            vip1_e = (self.p_baseline * next_event_rates) @ self.Y_to_VIP1_weight
            sst1_e = (self.p_baseline * next_event_rates) @ self.Y_to_SST1_weight - vip1_e @ self.Y_VIP1_to_SST1_weight
            sst1_e += self.sst1_bias

            vip2_e = (self.p_baseline * next_event_rates) @ self.Y_to_VIP2_weight
            sst2_e = (self.p_baseline * next_event_rates) @ self.Y_to_SST2_weight - vip2_e @ self.Y_VIP2_to_SST2_weight
            sst2_e += self.sst2_bias

            self.Y_input_e = self.apical_bias_input - sst1_e @ self.Y_from_SST1_weight - sst2_e @ self.Y_from_SST2_weight

            with torch.no_grad():
                #Ablation:
                # self.apical_bias_input.add_(-0.65 * (self.Y_input_e + self.Q_input).sum(axis=0))

                #BCI NDNF offset
                # self.apical_bias_input.add_(-0.01/128 * (self.Y_input_e + self.Q_input).sum(axis=0))
                self.apical_bias_input.add_(-0.01/128 * (self.Y_input_e + self.Q_input).sum(axis=0))
                # self.apical_bias_input.add_(-0.00/128 * (self.Y_input_e + self.Q_input).sum(axis=0))

        return self.b_t, self.e, feedback_scale

    def calc_p_sigmoid(self):
        k = 2.0 / self.p_baseline
        return 2.0 * self.p_baseline * torch.sigmoid(k * self.apic * (1 - self.e))

    def calc_Y_apic_grads(self):
        y1_grad = -(self.sst1 - self.sst1_bias).transpose(0, 1).mm(self.apic)
        y2_grad = -(self.sst2 - self.sst2_bias).transpose(0, 1).mm(self.apic)
        return y1_grad, y2_grad

    @property
    def W_weight(self):
        def safe_sub(a, b):
            return a - b if a is not None and b is not None else None

        effective_W_weight = self.W_direct_weight - self.W_from_PV_weight

        effective_W_weight.grad = safe_sub(self.W_direct_weight.grad, self.W_from_PV_weight.grad)
        effective_W_weight.grad_bp = safe_sub(
            getattr(self.W_direct_weight, 'grad_bp', None),
            getattr(self.W_from_PV_weight, 'grad_bp', None)
        )
        effective_W_weight.grad_fa = safe_sub(
            getattr(self.W_direct_weight, 'grad_fa', None),
            getattr(self.W_from_PV_weight, 'grad_fa', None)
        )

        return effective_W_weight

    @property
    def W_bias(self):
        return self.W_direct_bias

    @property
    def Y_weight(self):
        if hasattr(self, 'orig_Y_weight'):
            return self.orig_Y_weight
        effective_Y_weight = -1 * ((self.Y_to_SST1_weight - self.Y_to_VIP1_weight) @ self.Y_from_SST1_weight + (
                    self.Y_to_SST2_weight - self.Y_to_VIP2_weight) @ self.Y_from_SST2_weight)

        return effective_Y_weight

    @property
    def effective_Y_weight(self):
        effective_Y_weight = -1 * ((self.Y_to_SST1_weight - self.Y_to_VIP1_weight) @ self.Y_from_SST1_weight + (
                self.Y_to_SST2_weight - self.Y_to_VIP2_weight) @ self.Y_from_SST2_weight)

        return effective_Y_weight

    @property
    def Q_weight(self):
        effective_Q_weight = (
                self.Q_direct_weight -
                self.Q_to_NDNF_weight @ self.Q_from_NDNF_weight
        )

        return effective_Q_weight

    def backward_bp(self, b_input_bp):
        # assert getattr(self.W_weight, 'grad_bp', None) is None
        # assert getattr(self.W_bias, 'grad_bp', None) is None

        self.b_input_bp = b_input_bp
        self.delta_bp = self.act.backward(b_input_bp, self.act_ctx)

        update_scale_factor = self.input.size(0) * 2.0

        self.W_direct_weight.grad_bp = self.delta_bp.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_direct_bias.grad_bp = torch.sum(self.delta_bp, dim=0) / update_scale_factor

        self.W_from_PV_weight.grad_bp = -self.delta_bp.transpose(0, 1).mm(self.pv) / update_scale_factor

        return self.delta_bp.mm(self.W_weight)

    def backward_fa(self, next_delta_fa):
        # assert getattr(self.W_weight, 'grad_fa', None) is None
        # assert getattr(self.W_bias, 'grad_fa', None) is None

        g = next_delta_fa.mm(self.Y_weight)
        self.delta_fa = self.act.backward(g, self.act_ctx)

        update_scale_factor = self.input.size(0) * 2.0

        self.W_direct_weight.grad_fa = self.delta_fa.transpose(0, 1).mm(self.input) / update_scale_factor
        self.W_direct_bias.grad_fa = torch.sum(self.delta_fa, dim=0) / update_scale_factor

        self.W_from_PV_weight.grad_fa = -self.delta_fa.transpose(0, 1).mm(self.pv) / update_scale_factor

        return self.delta_fa

    def get_state(self, state_key):
        if state_key == 'sst':
            return torch.hstack([self.sst1, self.sst2])
        elif state_key == 'vip':
            return torch.hstack([self.vip1, self.vip2])
        else:
            return super().get_state(state_key)

