from burstccn.models.networks.base import FullyConnectedNetworkFactory, NPBackwardPolicy, \
    FullyConnectedNPForwardPolicy
from burstccn.models.networks.np_base import NPBase


class FullyConnectedNP(NPBase):
    def __init__(self, cfg):
        network_factory = FullyConnectedNetworkFactory(cfg)
        forward_policy = FullyConnectedNPForwardPolicy()
        backward_policy = NPBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)