from burstccn.models.networks.base import FullyConnectedNetworkFactory, ConvNetworkFactory, \
    FullyConnectedForwardPolicy, ConvForwardPolicy, MultiBackwardPolicy
from burstccn.models.networks.burstprop_base import BurstpropBase


class FullyConnectedBurstprop(BurstpropBase):
    def __init__(self, cfg):
        network_factory = FullyConnectedNetworkFactory(cfg)
        forward_policy = FullyConnectedForwardPolicy()
        backward_policy = MultiBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)


class ConvBurstprop(BurstpropBase):
    def __init__(self, cfg):
        network_factory = ConvNetworkFactory(cfg)
        forward_policy = ConvForwardPolicy()
        backward_policy = MultiBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)