from burstccn.models.networks.base import FullyConnectedNetworkFactory, ConvNetworkFactory, \
    FullyConnectedForwardPolicy, ConvForwardPolicy, SingleBackwardPolicy
from burstccn.models.networks.ann_base import ANNBase


class FullyConnectedANN(ANNBase):
    def __init__(self, cfg):
        network_factory = FullyConnectedNetworkFactory(cfg)
        forward_policy = FullyConnectedForwardPolicy()
        backward_policy = SingleBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)


class ConvANN(ANNBase):
    def __init__(self, cfg):
        network_factory = ConvNetworkFactory(cfg)
        forward_policy = ConvForwardPolicy()
        backward_policy = SingleBackwardPolicy()

        super().__init__(cfg, network_factory, forward_policy, backward_policy)