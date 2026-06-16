import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerBase(nn.Module):
    def get_state(self, state_key):
        return getattr(self, state_key, None)