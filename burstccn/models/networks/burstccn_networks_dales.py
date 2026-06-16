from burstccn.models.networks.base import FullyConnectedNetworkFactory, \
    FullyConnectedForwardPolicy, MultiBackwardPolicy
from burstccn.models.networks.burstccn_base import DalesBurstCCN


class FullyConnectedDalesBurstCCN(DalesBurstCCN):
    def __init__(self, cfg):
        network_factory = FullyConnectedNetworkFactory(cfg)
        forward_policy = FullyConnectedForwardPolicy()
        backward_policy = MultiBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)

