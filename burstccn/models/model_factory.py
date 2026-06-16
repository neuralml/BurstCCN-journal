import torch

from burstccn.models.networks.ann_networks import FullyConnectedANN, ConvANN
from burstccn.models.networks.base import ManualGradNetwork
from burstccn.models.networks.burstccn_networks import FullyConnectedBurstCCN, ConvBurstCCN
from burstccn.models.networks.burstprop_networks import FullyConnectedBurstprop, ConvBurstprop
# from burstccn.models.networks.autograd_burstccn_networks import AutogradFullyConnectedBurstCCN#, AutogradImageNetConvBurstCCN
from burstccn.models.networks.burstccn_networks_dales import FullyConnectedDalesBurstCCN
from burstccn.models.networks.edn_networks import FullyConnectedEDN
from burstccn.models.networks.np_networks import FullyConnectedNP


class ModelFactory:
    @classmethod
    def create_model(cls, config):
        if config.model_name == 'mnist_ann':
            model = FullyConnectedANN(config)
        elif config.model_name == 'mnist_burstccn':
            model = FullyConnectedBurstCCN(config)
        # elif config.model_name == 'mnist_burstccn_autograd':
        #     model = AutogradFullyConnectedBurstCCN(config)
        elif config.model_name == 'mnist_burstprop':
            model = FullyConnectedBurstprop(config)
        elif config.model_name == 'mnist_edn':
            model = FullyConnectedEDN(config)
        elif config.model_name == 'mnist_burstccn_dales':
            model = FullyConnectedDalesBurstCCN(config)
        elif config.model_name == 'simple_increasing_task_burstccn_dales':
            model = FullyConnectedDalesBurstCCN(config)
        # elif config.model_name == 'imagenet_burstccn':
        #     model = AutogradImageNetConvBurstCCN(config)
        elif config.model_name == 'cifar10_conv_ann':
            model = ConvANN(config)
        elif config.model_name == 'cifar10_conv_burstccn':
            model = ConvBurstCCN(config)
        elif config.model_name == 'cifar10_conv_burstprop':
            model = ConvBurstprop(config)
        elif config.model_name == 'imagenet_conv_ann':
            model = ConvANN(config)
        elif config.model_name == 'imagenet_conv_burstccn':
            model = ConvBurstCCN(config)
        elif config.model_name == 'imagenet_conv_burstprop':
            model = ConvBurstprop(config)
        elif config.model_name == 'mnist_np':
            model = FullyConnectedNP(config)
        else:
            raise ValueError(f"Unsupported model_name: {config.model_name}")

        if isinstance(model, ManualGradNetwork):
            for p in model.parameters():
                p.requires_grad = False

        dtype = getattr(torch, config.dtype)
        model.to(dtype)

        return model
