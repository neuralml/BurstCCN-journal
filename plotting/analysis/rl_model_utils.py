from __future__ import annotations

import copy
import random
from collections import deque

import numpy as np
import torch
from torch import nn
from torchvision import transforms


def phi(x):
    return 0.2 + 0.6 * x


def invphi(x):
    return (x - 0.2) / 0.6


def move_start_location(curr_map, start_state):
    position_vec = start_state[:9].reshape(3, 3)
    agent_position = tuple(np.argwhere(position_vec == 1)[0])

    new_map = copy.deepcopy(curr_map)
    new_map[0] = "F" + curr_map[0][1:]

    map_row_str = new_map[agent_position[0]]
    new_map[agent_position[0]] = (
        map_row_str[:agent_position[1]]
        + "S"
        + map_row_str[agent_position[1] + 1:]
    )
    return new_map


def prep_state(state, env, rgb, curr_map=None):
    state_size = env.observation_space.n
    if rgb:
        return env.render()[0]

    ret = np.eye(state_size)[state]
    if curr_map is not None:
        map_str = "".join(curr_map)
        hole_indices = np.zeros(state_size)
        hole_pos = [pos for pos, char in enumerate(map_str) if char == "H"]
        hole_indices[hole_pos] = 1
        ret = np.concatenate((ret, hole_indices))
    return ret


class DQN(nn.Module):
    def __init__(self, model, state_size, action_size, device, args, mem_size=2000):
        super().__init__()
        self.model = model
        self.args = args
        self.conv = args["use_conv"]
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=mem_size)
        self.rgb = args["rgb"]
        self.transform = transforms.Resize((32, 32)) if self.rgb else None

    def forward(self, x):
        if self.rgb:
            x = x.permute(0, 3, 1, 2)
            x = torch.mean(x, dim=1, keepdim=True)
            x = x / 255.0
            if self.transform:
                x = self.transform(x)

        if self.conv:
            return self.model(x)

        x = x.view(x.size(0), -1)
        return self.model(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.eval()
            with torch.no_grad():
                action_values = self.forward(state)
            self.train()
            return np.argmax(action_values.cpu().data.numpy())

        return random.choice(np.arange(self.action_size))
