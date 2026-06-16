from burstccn.models.networks.base import FullyConnectedNetworkFactory, ConvNetworkFactory, \
    FullyConnectedForwardPolicy, ConvForwardPolicy, MultiBackwardPolicy
from burstccn.models.networks.burstccn_base import BurstCCN, DalesBurstCCN


class FullyConnectedBurstCCN(BurstCCN):
    def __init__(self, cfg):
        network_factory = FullyConnectedNetworkFactory(cfg)
        forward_policy = FullyConnectedForwardPolicy()
        backward_policy = MultiBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)


class ConvBurstCCN(BurstCCN):
    def __init__(self, cfg):
        network_factory = ConvNetworkFactory(cfg)
        forward_policy = ConvForwardPolicy()
        backward_policy = MultiBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)
