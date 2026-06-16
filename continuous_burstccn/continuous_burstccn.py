
import math
import torch
from torch import nn


class ContinuousBurstCCNNetwork(nn.Module):
    def __init__(self, n_inputs, n_hidden_layers, n_hidden_units, n_outputs, p_baseline, tau_W, lr):
        super().__init__()

        self.layers = nn.ModuleList()
        self.weight_transport = True

        if n_hidden_layers == 0:
            self.layers.append(ContinuousBurstCCNOutputLayer(n_inputs, n_outputs, p_baseline, tau_W=tau_W, lr=lr))
        elif n_hidden_layers == 1:
            self.layers.append(ContinuousBurstCCNHiddenLayer(n_inputs, n_hidden_units, n_outputs, p_baseline, tau_W=tau_W, lr=lr))
            self.layers.append(ContinuousBurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline, tau_W=tau_W, lr=lr))
        else:
            self.layers.append(ContinuousBurstCCNHiddenLayer(n_inputs, n_hidden_units, n_hidden_units, p_baseline, tau_W=tau_W, lr=lr))

            for i in range(1, n_hidden_layers - 1):
                self.layers.append(ContinuousBurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_hidden_units, p_baseline, tau_W=tau_W, lr=lr))

            self.layers.append(ContinuousBurstCCNHiddenLayer(n_hidden_units, n_hidden_units, n_outputs, p_baseline, tau_W=tau_W, lr=lr))
            self.layers.append(ContinuousBurstCCNOutputLayer(n_hidden_units, n_outputs, p_baseline, tau_W=tau_W, lr=lr))

        self.p_baseline = p_baseline

    def prediction_update(self, input_event_rate, dt):
        event_rate = input_event_rate
        # for i, layer in enumerate(self.layers):
        for i in range(len(self.layers)):
            self.layers[i].cache_state()
            event_rate = self.layers[i].feedforward_update(event_rate, dt=dt)

        output_layer = self.layers[-1]
        # next_layer_burst_rate_cache = output_layer.burst_rate_cache
        output_layer.cache_state()
        next_layer_event_rate_cache, next_layer_burst_rate_cache = output_layer.feedback_update(dt=dt)
        # for i, layer in list(enumerate(self.layers))[-2::-1]:
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].cache_state()
            next_layer_event_rate_cache, next_layer_burst_rate_cache = self.layers[i].feedback_update(next_layer_event_rate_cache, next_layer_burst_rate_cache, dt=dt)

    def teaching_update(self, input_event_rate, target, dt):
        event_rate = input_event_rate
        # for i, layer in enumerate(self.layers):
        for i in range(len(self.layers)):
            self.layers[i].cache_state()
            event_rate = self.layers[i].feedforward_update(event_rate, dt=dt)

        output_layer = self.layers[-1]
        output_layer.cache_state()
        next_layer_event_rate_cache, next_layer_burst_rate_cache = output_layer.feedback_update(target=target, dt=dt)

        # for i, layer in list(enumerate(self.layers))[-2::-1]:
        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].cache_state()
            next_layer_event_rate_cache, next_layer_burst_rate_cache = self.layers[i].feedback_update(next_layer_event_rate_cache, next_layer_burst_rate_cache, dt=dt)

        first_hidden_layer = self.layers[0]
        first_hidden_layer.weight_update(input_event_rate, dt=dt)

        # for i, layer in list(enumerate(self.layers))[1:]:
        for i in range(1, len(self.layers)):
            self.layers[i].weight_update(self.layers[i-1].event_rate_cache, dt=dt)

        if self.weight_transport:
            for i in range(len(self.layers) - 1):
                # self.layers[i].weight_Y = next_weight
                # self.layers[i].weight_Q = -self.p_baseline * next_weight
                next_weight = self.layers[i + 1].weight
                self.layers[i].weight_Y.data.copy_(next_weight)
                self.layers[i].weight_Q.data.copy_(-self.p_baseline * next_weight)

    def set_weights(self, layer_weights, layer_biases):
        for layer, weight, bias in zip(self.layers, layer_weights, layer_biases):
            # layer.weight = weight.clone()
            # layer.bias = bias.unsqueeze(1).clone()
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias.unsqueeze(1))

        if self.weight_transport:
            for i in range(len(self.layers) - 1):
                next_weight = self.layers[i + 1].weight
                # self.layers[i].weight_Y = next_weight
                # self.layers[i].weight_Q = -self.p_baseline * next_weight
                self.layers[i].weight_Y.data.copy_(next_weight)
                self.layers[i].weight_Q.data.copy_(-self.p_baseline * next_weight)


class ContinuousBurstCCNOutputLayer(nn.Module):
    def __init__(self, in_features, out_features, p_baseline, tau_W, lr):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        # self.p_baseline = p_baseline

        self.register_buffer('soma_potential', torch.zeros(out_features, 1))
        self.register_buffer('soma_potential_cache', torch.zeros(out_features, 1))
        self.register_buffer('event_rate', torch.zeros(out_features, 1))
        self.register_buffer('event_rate_cache', torch.zeros(out_features, 1))
        # self.register_buffer('event_rate_ma', torch.zeros(out_features, 1))
        # self.register_buffer('event_rate_ma_cache', torch.zeros(out_features, 1))
        self.register_buffer('burst_rate', torch.zeros(out_features, 1))
        self.register_buffer('burst_rate_cache', torch.zeros(out_features, 1))
        # self.register_buffer('burst_rate_ma', torch.zeros(out_features, 1))
        # self.register_buffer('burst_rate_ma_cache', torch.zeros(out_features, 1))
        self.register_buffer('dendritic_potential', torch.zeros(out_features, 1))
        self.register_buffer('dendritic_potential_cache', torch.zeros(out_features, 1))
        self.register_buffer('burst_prob', torch.zeros(out_features, 1))
        self.register_buffer('burst_prob_cache', torch.zeros(out_features, 1))
        self.register_buffer('p_baseline', torch.ones(out_features, 1) * p_baseline)

        # Trainable parameters
        W_init_scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(W_init_scale * torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features, 1))

        self.beta = 1
        # self.lambda_ = 0.05
        self.tau_u = 0.1
        self.tau_z = 0.1
        # self.tau_u = 0.05
        # self.tau_z = 0.05

        self.tau_e = self.tau_z
        self.tau_W = tau_W
        self.lr = lr
        # self.tau_moving_avg = 5.0
        self.e_max = 5.0
        # self.lr_correction = 20.0

        # self.small_vector = torch.ones((out_features, 1)) * 1e-8
        # self.p_baseline = torch.ones((out_features, 1)) * p_baseline

        self.activation_function = torch.sigmoid

        def h_for_sigmoid(x):
            return 1.0 - x

        self.h = h_for_sigmoid

    def cache_state(self):
        self.event_rate_cache.copy_(self.event_rate)
        self.burst_rate_cache.copy_(self.burst_rate)
        self.burst_prob_cache.copy_(self.burst_prob)

        # self.event_rate_ma_cache.copy_(self.event_rate_ma)
        # self.burst_rate_ma_cache.copy_(self.burst_rate_ma)
        # self.burst_prob_ma_cache.copy_(self.burst_prob_ma)

        # self.event_rate_cache = self.event_rate.clone()
        # self.burst_rate_cache = self.burst_rate.detach().clone()
        # self.burst_prob_cache = self.burst_prob.detach().clone()
        #
        # self.event_rate_ma_cache = self.event_rate_ma.detach().clone()
        # self.burst_rate_ma_cache = self.burst_rate_ma.detach().clone()
        # # self.burst_prob_ma_cache = self.burst_prob_ma.detach().clone()

    def feedforward_update(self, input_event_rate, dt):
        # self.soma_potential = (1. - dt / self.tau_z) * self.soma_potential + (dt / self.tau_z) * (self.weight.mm(input_event_rate) + self.bias)
        # self.event_rate = self.activation_function(self.soma_potential)
        # self.burst_rate = self.burst_prob * self.event_rate

        self.soma_potential.mul_(1. - dt / self.tau_z).add_((dt / self.tau_z) * (self.weight.mm(input_event_rate) + self.bias))
        self.event_rate = self.activation_function(self.soma_potential)
        self.burst_rate = self.burst_prob * self.event_rate

        return self.event_rate

    def feedback_update(self, dt, target=None):
        if target is not None:
            # delta_burst_prob = self.p_baseline * torch.tanh((target - self.event_rate_ma) / (self.event_rate_ma + self.small_vector))
            delta_burst_prob = self.p_baseline * (target - self.event_rate) * (1 - self.event_rate)
            self.burst_prob = self.p_baseline + delta_burst_prob
        else:
            self.burst_prob = self.p_baseline
            # self.burst_prob = self.p_baseline.detach().clone()

        self.burst_rate = self.burst_prob * self.event_rate

        # self.event_rate_ma = self.event_rate_ma + (dt / self.tau_moving_avg) * (self.event_rate_cache - self.event_rate_ma)
        # self.burst_rate_ma = self.burst_rate_ma + (dt / self.tau_moving_avg) * (self.burst_rate_cache - self.burst_rate_ma)
        # self.burst_prob_ma = self.burst_rate_ma / self.event_rate_ma

        # self.event_rate_ma.add_((dt / self.tau_moving_avg) * (self.event_rate_cache - self.event_rate_ma))
        # self.burst_rate_ma.add_((dt / self.tau_moving_avg) * (self.burst_rate_cache - self.burst_rate_ma))


        return self.event_rate_cache, self.burst_rate_cache

    def weight_update(self, input_event_rate, dt):

        # H = ((torch.ones(self.out_features, 1) - self.e_max / (self.event_rate + 1e-8 * torch.ones(self.out_features, 1))) * (
        #             self.event_rate > self.e_max * torch.ones(self.out_features, 1))).mm(input_event_rate.t())


        # # self.weight = self.weight + (dt / self.tau_W) * (((self.burst_prob_cache - self.burst_prob_ma_cache) * self.event_rate_cache) * input_event_rate.t() - self.lambda_ * H)
        # self.weight = self.weight + self.lr_correction * (dt / self.tau_W) * (((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache) * input_event_rate.t())
        # self.bias = self.bias + self.lr_correction * (dt / self.tau_W) * ((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache)

        lr_factor = self.lr * (dt / self.tau_W)
        self.weight.add_(lr_factor * ((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache).mm(input_event_rate.t()))
        self.bias.add_(lr_factor * (self.burst_prob_cache - self.p_baseline) * self.event_rate_cache)


class ContinuousBurstCCNHiddenLayer(nn.Module):
    def __init__(self, in_features, out_features, next_features, p_baseline, tau_W, lr):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.p_baseline_value = p_baseline
        self.tau_W = tau_W
        self.lr = lr

        # Buffers: not trainable
        self.register_buffer('soma_potential', torch.zeros(out_features, 1))
        self.register_buffer('soma_potential_cache', torch.zeros(out_features, 1))
        self.register_buffer('event_rate', torch.zeros(out_features, 1))
        self.register_buffer('event_rate_cache', torch.zeros(out_features, 1))
        # self.register_buffer('event_rate_ma', torch.zeros(out_features, 1))
        # self.register_buffer('event_rate_ma_cache', torch.zeros(out_features, 1))
        self.register_buffer('burst_rate', torch.zeros(out_features, 1))
        self.register_buffer('burst_rate_cache', torch.zeros(out_features, 1))
        # self.register_buffer('burst_rate_ma', torch.zeros(out_features, 1))
        # self.register_buffer('burst_rate_ma_cache', torch.zeros(out_features, 1))
        self.register_buffer('dendritic_potential', torch.zeros(out_features, 1))
        self.register_buffer('dendritic_potential_cache', torch.zeros(out_features, 1))
        self.register_buffer('burst_prob', torch.zeros(out_features, 1))
        self.register_buffer('burst_prob_cache', torch.zeros(out_features, 1))
        self.register_buffer('p_baseline', torch.ones(out_features, 1) * p_baseline)

        # Trainable parameters
        W_init_scale = math.sqrt(2.0 / (in_features + out_features))
        self.weight = nn.Parameter(W_init_scale * torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features, 1))

        Y_init_scale = math.sqrt(2.0 / (out_features + next_features))
        self.weight_Y = nn.Parameter(Y_init_scale * torch.randn(next_features, out_features))
        self.weight_Q = nn.Parameter(-p_baseline * self.weight_Y.clone())

        self.beta = 4.0
        # self.lambda_ = 0.05
        self.tau_u = 0.1
        self.tau_z = 0.1
        # self.tau_u = 0.05
        # self.tau_z = 0.05
        self.tau_e = self.tau_z
        # self.tau_moving_avg = 5.0
        self.e_max = 5.0
        # self.lr_correction = 20.0

        # self.weight = math.sqrt(2.0 / (in_features + out_features)) * torch.randn(out_features, in_features)
        # self.bias = torch.zeros((out_features, 1))

        # if weight_transport:
        #     B = W2.t()
        # else:
        #     B = torch.randn(n_hidden, n_classes)
        #     B *= 0.1 * math.sqrt(2.0 / (n_hidden + n_classes))

        # self.weight_Y = math.sqrt(2.0 / (out_features + next_features)) * torch.randn(next_features, out_features)
        # self.weight_Q = -self.p_baseline * math.sqrt(2.0 / (out_features + next_features)) * torch.randn(next_features, out_features)
        # self.weight_Q = -self.p_baseline * self.weight_Y.clone()

        self.activation_function = torch.sigmoid

        def h_for_sigmoid(x):
            return 1.0 - x

        self.h = h_for_sigmoid

    def cache_state(self):
        self.event_rate_cache.copy_(self.event_rate)
        self.burst_rate_cache.copy_(self.burst_rate)
        self.burst_prob_cache.copy_(self.burst_prob)

        # self.event_rate_ma_cache.copy_(self.event_rate_ma)
        # self.burst_rate_ma_cache.copy_(self.burst_rate_ma)
        # self.burst_prob_ma_cache.copy_(self.burst_prob_ma)

        # self.event_rate_cache = self.event_rate.detach().clone()
        # self.burst_rate_cache = self.burst_rate.detach().clone()
        # self.burst_prob_cache = self.burst_prob.detach().clone()
        #
        # self.event_rate_ma_cache = self.event_rate_ma.detach().clone()
        # self.burst_rate_ma_cache = self.burst_rate_ma.detach().clone()
        # # self.burst_prob_ma_cache = self.burst_prob_ma.detach().clone()

    def feedforward_update(self, input_event_rate, dt=0.1):
        # self.soma_potential = (1. - dt / self.tau_z) * self.soma_potential + (dt / self.tau_z) * (self.weight.mm(input_event_rate) + self.bias)
        # self.event_rate = self.activation_function(self.soma_potential)
        # self.burst_rate = self.burst_prob * self.event_rate

        self.soma_potential.mul_(1. - dt / self.tau_z).add_((dt / self.tau_z) * (self.weight.mm(input_event_rate) + self.bias))
        self.event_rate = self.activation_function(self.soma_potential)
        self.burst_rate = self.burst_prob * self.event_rate

        return self.event_rate

    def feedback_update(self, next_layer_event_rate_cache, next_layer_burst_rate_cache, dt=0.1):
        # self.dendritic_potential = (1. - dt / self.tau_u) * self.dendritic_potential + (dt / self.tau_u) * (self.beta * self.h(self.event_rate_cache) * (self.weight_Q.T.mm(next_layer_event_rate_cache) + self.weight_Y.T.mm(next_layer_burst_rate_cache)))
        self.dendritic_potential.mul_(1. - dt / self.tau_u).add_((dt / self.tau_u) * self.beta * self.h(self.event_rate_cache) * (
                        self.weight_Q.T.mm(next_layer_event_rate_cache) + self.weight_Y.T.mm(next_layer_burst_rate_cache)))
        self.burst_prob = torch.sigmoid(self.dendritic_potential)
        self.burst_rate = self.burst_prob * self.event_rate

        # self.event_rate_ma = self.event_rate_ma + (dt / self.tau_moving_avg) * (self.event_rate_cache - self.event_rate_ma)
        # self.burst_rate_ma = self.burst_rate_ma + (dt / self.tau_moving_avg) * (self.burst_rate_cache - self.burst_rate_ma)
        # self.burst_prob_ma = self.burst_rate_ma / self.event_rate_ma

        # self.event_rate_ma.add_((dt / self.tau_moving_avg) * (self.event_rate_cache - self.event_rate_ma))
        # self.burst_rate_ma.add_((dt / self.tau_moving_avg) * (self.burst_rate_cache - self.burst_rate_ma))

        return self.event_rate_cache, self.burst_rate_cache

    def weight_update(self, input_event_rate, dt=0.1):

        # H = ((torch.ones(self.out_features, 1) - self.e_max / (self.event_rate + 1e-8 * torch.ones(self.out_features, 1))) * (
        #             self.event_rate > self.e_max * torch.ones(self.out_features, 1))).mm(input_event_rate.t())

        # self.weight = self.weight + (dt / self.tau_W) * (((self.burst_prob_cache - self.burst_prob_ma_cache) * self.event_rate_cache) * input_event_rate.t() - self.lambda_ * H)
        # self.weight = self.weight + self.lr_correction * (dt / self.tau_W) * (((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache) * input_event_rate.t())
        # self.bias = self.bias + self.lr_correction * (dt / self.tau_W) * (
        #             (self.burst_prob_cache - self.p_baseline) * self.event_rate_cache)

        lr_factor = self.lr * (dt / self.tau_W)

        self.weight.add_(lr_factor * ((self.burst_prob_cache - self.p_baseline) * self.event_rate_cache).mm(input_event_rate.t()))
        self.bias.add_(lr_factor * (self.burst_prob_cache - self.p_baseline) * self.event_rate_cache)


