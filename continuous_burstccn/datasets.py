import math
import os
import random
import sys
from pathlib import Path
import PIL.Image

import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset

from torchvision.transforms import transforms
import torchvision.transforms.functional as F

from continuous_burstccn_ma import MAContinuousBurstCCNNetwork


class ANN(nn.Module):
    def __init__(self, n_inputs, n_hidden_layers, n_hidden_units, n_outputs):
        super(ANN, self).__init__()

        self.linear_layers = nn.ModuleList()

        if n_hidden_layers == 0:
            self.linear_layers.append(nn.Linear(n_inputs, n_outputs))
            # self.classification_layers.append(nn.Sigmoid())
        else:
            # self.layers.append(ContinuousBurstCCNHiddenLayer(n_inputs, n_hidden_units, n_hidden_units, p_baseline, device))
            self.linear_layers.append(nn.Linear(n_inputs, n_hidden_units))
            # self.classification_layers.append(nn.Sigmoid())

            for i in range(1, n_hidden_layers):
                self.linear_layers.append(nn.Linear(n_hidden_units, n_hidden_units))
                # self.classification_layers.append(nn.Sigmoid())

            self.linear_layers.append(nn.Linear(n_hidden_units, n_outputs))
            # self.classification_layers.append(nn.Sigmoid())


        all_layers = []
        for l in self.linear_layers:
            all_layers.append(l)
            all_layers.append(nn.Sigmoid())

        for l in self.linear_layers:
            # 3x3
            # nn.init.xavier_normal_(l.weight, gain=3.6)
            # 5x5
            # nn.init.xavier_normal_(l.weight, gain=3.5)
            # CatCam
            nn.init.xavier_normal_(l.weight, gain=1.0)
            nn.init.constant_(l.bias, 0)
        # self.layers = nn.Sequential(*all_layers)
        # self.layers.to(device)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        # print(x.shape)

        for layer in self.linear_layers:
            x = layer(x)
            # print(x.shape)
            x = torch.sigmoid(x)
            layer.activity = x

        return x
        # return self.layers(x)

    def set_weights(self, weight_list, bias_list):
        for i, (weights, biases) in enumerate(zip(weight_list, bias_list)):
            self.linear_layers[i].weight.data = weights.detach().clone()
            self.linear_layers[i].bias.data = biases.detach().clone()


# class CatCamMovieDataset(Dataset):
#     def __init__(self, data_dir, movie_index, transform=None, crop_size=(32, 32), downsample_size=None):
#         data_dir = Path(data_dir)
#         movie_name = f"movie{movie_index + 1:02d}"
#         movie_dir = data_dir / movie_name
#         cache_file = data_dir / f"{movie_name}_{crop_size[0]}_{crop_size[1]}.pt"
#
#         self.transform = transform
#         self.crop_size = crop_size
#         self.downsample_size = downsample_size
#
#         if cache_file.exists():
#             # Load cached tensor data
#             self.images = torch.load(cache_file)
#         else:
#             self.image_paths = sorted(movie_dir.glob("*"))
#
#             # Preload and crop all images
#             with PIL.Image.open(self.image_paths[0]) as img:
#                 img = img.convert('L')
#                 if self.transform:
#                     img = self.transform(img)
#                 i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_size)
#                 self.crop_indices = (i, j, h, w)
#
#             self.images = []
#             for path in self.image_paths:
#                 with PIL.Image.open(path) as img:
#                     img = img.convert('L')
#                     if self.transform:
#                         img = self.transform(img)
#                     img = F.crop(img, *self.crop_indices)
#
#                     if self.downsample_size:
#                         img = F.resize(img, self.downsample_size)
#
#                     self.images.append(img)
#
#             # Stack and save as .pt
#             tensor_stack = torch.stack(self.images)
#             torch.save(tensor_stack, cache_file)
#             self.images = tensor_stack
#
#     def __getitem__(self, index):
#         return self.images[index]
#
#     def __len__(self):
#         return len(self.images)


class CatCamMovieDataset(Dataset):
    def __init__(self, data_dir, movie_index, transform=None, crop_size=(32, 32), downsample_factor=None, crop_indices=None, normalised_inputs=False):
        data_dir = Path(data_dir)
        movie_name = f"movie{movie_index + 1:02d}"
        movie_dir = data_dir / movie_name
        cache_file = data_dir / f"{movie_name}_full.pt"

        self.transform = transform
        self.crop_size = crop_size
        self.downsample_factor = downsample_factor
        self.normalised_inputs = normalised_inputs

        self.image_paths = sorted(movie_dir.glob("*"))

        # Load or create full-size image cache (uncropped)
        if cache_file.exists():
            self.full_images = torch.load(cache_file)
        else:
            images = []
            for path in self.image_paths:
                with PIL.Image.open(path) as img:
                    img = img.convert('L')
                    if self.transform:
                        img = self.transform(img)
                    images.append(img)
            self.full_images = torch.stack(images)
            torch.save(self.full_images, cache_file)

        if crop_indices is None:
            sample_img = self.full_images[0]
            i, j, h, w = transforms.RandomCrop.get_params(sample_img, output_size=self.crop_size)
            self.crop_indices = (i, j, h, w)
        else:
            self.crop_indices = crop_indices

        if downsample_factor is not None:
            self.downsample_size = [int(self.crop_size[0]) // downsample_factor,
                                    int(self.crop_size[1]) // downsample_factor]

    def __getitem__(self, index):
        img = self.full_images[index]
        img = F.crop(img, *self.crop_indices)

        if self.downsample_factor is not None:
            img = F.resize(img, self.downsample_size)

        if self.normalised_inputs:
            img = (img - img.mean()) / img.std()

        return img

    def __len__(self):
        return len(self.full_images)


# class CatCamContinuousDataLoader:
#     def __init__(self, data_dir, ann_path, window_size=32):
#         self.data_dir = data_dir
#         self.window_size = window_size
#         self.movie_index = 0
#         self.frame_timestep = 0
#         self.frame_index = 0
#         self.next_frame_index = self.frame_index + 1
#
#         self.transform_train = transforms.Compose([
#             transforms.ToTensor()
#         ])
#
#         self.ann = ANN(self.window_size ** 2, 2, 500, 10)
#         # self.ann = ANN(32 ** 2, 2, 500, 10)
#
#         self.ann.load_state_dict(torch.load(ann_path))
#
#         self.cont_target_net = MAContinuousBurstCCNNetwork(
#                 n_inputs=32 * 32,
#                 n_hidden_layers=2,
#                 n_hidden_units=500,
#                 n_outputs=10,
#                 p_baseline=0.5,
#                 tau_W=0.0,
#                 lr=0.0
#             )
#         self.cont_target_net.to('cpu')
#         self.cont_target_net.set_weights(
#             [layer.weight.clone() for layer in self.ann.linear_layers],
#             [layer.bias.clone() for layer in self.ann.linear_layers]
#         )
#
#         self.moving_average_image = torch.zeros(1, self.window_size * self.window_size)
#         # self.target_moving_average = torch.zeros(1, 10)
#         self.target_delay_steps = 20
#         self.target_delay_buffer = []
#
#         # print(f"Loading initial movie index {self.movie_index}")
#         # self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=(self.window_size*2, self.window_size*2), downsample_size=(self.window_size, self.window_size))
#         self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=(self.window_size, self.window_size))
#
#         self.last_image = self.image_data[0].reshape(1, -1)
#         self.next_image = self.image_data[1].reshape(1, -1)
#         # print(f"Initial last_image: frame 0, next_image: frame 1")
#
#     def next_dataset(self):
#         self.movie_index = (self.movie_index + 1) % 17
#         # print(f"\n=== Switching to movie index {self.movie_index} ===")
#         # self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=(self.window_size*2, self.window_size*2), downsample_size=(self.window_size, self.window_size))
#         self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=(self.window_size, self.window_size))
#
#         self.frame_index = 0
#
#         self.next_image = self.image_data[0].reshape(1, -1)
#         # print(f"Continuing interpolation from previous last_image to next_image from new movie (frame 0)")
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         interp_weight = (self.frame_timestep % 10) * 0.1
#         interp_image = self.next_image * interp_weight + self.last_image * (1 - interp_weight)
#
#         # print(f"\nTimestep {self.frame_timestep}")
#         # print(f"Frame index: {self.frame_index}")
#         # print(f"Interpolation weights. Last (frame): {1 - interp_weight:.2f}, next: {interp_weight:.2f})")
#
#         self.moving_average_image = 0.05 * interp_image + 0.95 * self.moving_average_image
#         # print(f"Updated moving average image (norm): {self.moving_average_image.norm().item():.4f}")
#
#         # Reshape to 32x32
#
#         # Display as image
#         # if self.frame_timestep % 100 == 0:
#         #     img = self.moving_average_image.view(self.window_size, self.window_size)
#         #
#         #     plt.imshow(img, cmap='gray')  # You can change colormap (e.g., 'viridis')
#         #     plt.axis('off')
#         #     plt.title("32x32 image from 1024 input")
#         #     plt.show()
#
#         # target = torch.zeros(1)
#
#         # target = self.ann(self.moving_average_image)
#         #
#         # # self.target_moving_average = 0.05 * target + 0.95 * self.target_moving_average
#         # # Update the delay buffer
#         # self.target_delay_buffer.append(target.detach())
#         # if len(self.target_delay_buffer) > self.target_delay_steps:
#         #     delayed_target = self.target_delay_buffer.pop(0)
#         # else:
#         #     delayed_target = self.target_delay_buffer[0]
#
#         self.cont_target_net.feedforward_only_update(self.moving_average_image.reshape(-1, 1), dt=0.01)
#         delayed_target = self.cont_target_net.layers[-1].event_rate
#
#         self.frame_timestep += 1
#
#         if self.frame_timestep % 10 == 0:
#             self.last_image = self.next_image
#
#             self.frame_index += 1
#             # print(f"--- Step complete: Advancing to next frame ---")
#             # print(f"Updated last_image to previous next_image")
#
#             if self.frame_index + 1 >= len(self.image_data):
#                 # print(f"Reached end of movie frames ({len(self.image_data)}), loading next dataset")
#                 self.next_dataset()
#             else:
#                 self.next_image = self.image_data[self.frame_index + 1].reshape(1, -1)
#                 # print(f"Loaded next_image: frame {self.frame_index + 1}")
#
#         return self.moving_average_image, delayed_target #self.target_moving_average #target

class CatCamContinuousDataLoader:
    def __init__(self, data_dir, ann_path, window_size=32, normalised_inputs=False, downsample_factor=None):
        self.data_dir = data_dir
        self.window_size = window_size
        self.movie_index = 0
        self.frame_timestep = 0
        self.frame_index = 0
        self.next_frame_index = self.frame_index + 1

        self.normalised_inputs = normalised_inputs

        self.transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

        n_hidden_units = 250
        self.ann = ANN(self.window_size ** 2, 2, n_hidden_units, 10)
        # self.ann = ANN(32 ** 2, 2, 500, 10)

        if ann_path:
            self.ann.load_state_dict(torch.load(ann_path))

        self.cont_target_net = MAContinuousBurstCCNNetwork(
                n_inputs=self.window_size * self.window_size,
                n_hidden_layers=2,
                n_hidden_units=n_hidden_units,
                n_outputs=10,
                p_baseline=0.5,
                tau_W=0.0,
                lr=0.0
            )
        self.cont_target_net.to('cpu')
        self.cont_target_net.set_weights(
            [layer.weight.clone() for layer in self.ann.linear_layers],
            [layer.bias.clone() for layer in self.ann.linear_layers]
        )

        # delta_b = torch.tensor(
        #     [-0.1075, -0.2000, 0.0197, 0.1142, -0.1823, 0.0512, -0.1610, -0.4435, -0.6320, -0.0160]
        # ).unsqueeze(1)
        #
        # self.cont_target_net.layers[-1].bias.data.add_(delta_b)

        self.moving_average_image = torch.zeros(1, self.window_size * self.window_size)
        # self.target_moving_average = torch.zeros(1, 10)
        # self.target_delay_steps = 20
        # self.target_delay_buffer = []

        # print(f"Loading initial movie index {self.movie_index}")
        self.downsample_factor = downsample_factor
        if self.downsample_factor is None:
            self.crop_size = (self.window_size, self.window_size)
            initial_crop_indices = (104, 144, 32, 32)
            self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=self.crop_size, crop_indices=initial_crop_indices, normalised_inputs=self.normalised_inputs)
        else:
            self.crop_size = (self.window_size * downsample_factor, self.window_size * downsample_factor)
            initial_crop_indices = (104, 144, self.window_size*downsample_factor, self.window_size*downsample_factor)
            self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=self.crop_size, crop_indices=initial_crop_indices, normalised_inputs=self.normalised_inputs, downsample_factor=downsample_factor)

        self.last_image = self.image_data[0].reshape(1, -1)
        self.next_image = self.image_data[1].reshape(1, -1)
        # print(f"Initial last_image: frame 0, next_image: frame 1")

    def next_dataset(self):
        self.movie_index = (self.movie_index + 1) % 17
        # print(f"\n=== Switching to movie index {self.movie_index} ===")
        # self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=(self.window_size*2, self.window_size*2), downsample_size=(self.window_size, self.window_size))
        # self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train, crop_size=(self.window_size, self.window_size), normalised_inputs=self.normalised_inputs)
        self.image_data = CatCamMovieDataset(self.data_dir, self.movie_index, transform=self.transform_train,
                                             crop_size=self.crop_size,
                                             normalised_inputs=self.normalised_inputs,
                                             downsample_factor=self.downsample_factor)

        self.frame_index = 0

        self.next_image = self.image_data[0].reshape(1, -1)
        # print(f"Continuing interpolation from previous last_image to next_image from new movie (frame 0)")

    def reset(self):
        self.movie_index = 0
        self.frame_index = 0
        self.frame_timestep = 0
        self.next_frame_index = 1
        self.image_data = CatCamMovieDataset(
            self.data_dir, self.movie_index,
            transform=self.transform_train,
            crop_size=(self.window_size, self.window_size)
        )

        self.last_image = self.image_data[0].reshape(1, -1)
        self.next_image = self.image_data[1].reshape(1, -1)
        self.moving_average_image = torch.zeros(1, self.window_size * self.window_size)
        self.target_delay_buffer = []

        self.cont_target_net.reset_state()

    def __iter__(self):
        return self

    def __next__(self):
        interp_weight = (self.frame_timestep % 10) * 0.1
        interp_image = self.next_image * interp_weight + self.last_image * (1 - interp_weight)

        # print(f"\nTimestep {self.frame_timestep}")
        # print(f"Frame index: {self.frame_index}")
        # print(f"Interpolation weights. Last (frame): {1 - interp_weight:.2f}, next: {interp_weight:.2f})")

        self.moving_average_image = 0.05 * interp_image + 0.95 * self.moving_average_image
        # print(f"Updated moving average image (norm): {self.moving_average_image.norm().item():.4f}")

        # Reshape to 32x32

        # Display as image
        # if self.frame_timestep % 100 == 0:
        #     img = self.moving_average_image.view(self.window_size, self.window_size)
        #
        #     plt.imshow(img, cmap='gray')  # You can change colormap (e.g., 'viridis')
        #     plt.axis('off')
        #     plt.title("32x32 image from 1024 input")
        #     plt.show()

        # target = torch.zeros(1)

        # target = self.ann(self.moving_average_image)
        #
        # # self.target_moving_average = 0.05 * target + 0.95 * self.target_moving_average
        # # Update the delay buffer
        # self.target_delay_buffer.append(target.detach())
        # if len(self.target_delay_buffer) > self.target_delay_steps:
        #     delayed_target = self.target_delay_buffer.pop(0)
        # else:
        #     delayed_target = self.target_delay_buffer[0]

        self.cont_target_net.feedforward_only_update(self.moving_average_image.reshape(-1, 1), dt=0.01)
        target = self.cont_target_net.layers[-1].event_rate

        self.frame_timestep += 1

        if self.frame_timestep % 10 == 0:
            self.last_image = self.next_image

            self.frame_index += 1
            # print(f"--- Step complete: Advancing to next frame ---")
            # print(f"Updated last_image to previous next_image")

            if self.frame_index + 1 >= len(self.image_data):
                # print(f"Reached end of movie frames ({len(self.image_data)}), loading next dataset")
                self.next_dataset()
            else:
                self.next_image = self.image_data[self.frame_index + 1].reshape(1, -1)
                # print(f"Loaded next_image: frame {self.frame_index + 1}")

        return self.moving_average_image, target #self.target_moving_average #target


class SinWaveContinuousDataLoader:
    def __init__(self, n_inputs, n_hidden_layers, n_hidden_units, n_outputs, seed):
        self.num_inputs = n_inputs
        self.dt = 0.01

        min_period = 40.0

        self.cosine_freqs = [random.random() * 2 * math.pi / min_period for _ in range(self.num_inputs)]
        self.cosine_phases = [random.random() * (math.pi * 2.0) for _ in range(self.num_inputs)]

        print(self.cosine_freqs)
        print(self.cosine_phases)

        self.timestep = 0

        self.ann = ANN(n_inputs, n_hidden_layers, n_hidden_units, n_outputs)
        model_file = f"saved_models/sin_task_seed{seed}.pt"
        # Load or initialize ANN weights
        if os.path.exists(model_file):
            print(f"Loading ANN weights from {model_file}")
            self.ann.load_state_dict(torch.load(model_file))
        else:
            print(f"Saving ANN weights to {model_file}")
            # for layer in self.ann.linear_layers:
            #     layer.weight.data *= 3

            for layer in self.ann.linear_layers:
                nn.init.xavier_normal_(layer.weight, gain=3.0)
                nn.init.constant_(layer.bias, 0)

            torch.save(self.ann.state_dict(), model_file)

        self.cont_target_net = MAContinuousBurstCCNNetwork(
            n_inputs=3,
            n_hidden_layers=1,
            n_hidden_units=n_hidden_units,
            n_outputs=1,
            p_baseline=0.5,
            tau_W=0.0,
            lr=0.0
        )
        self.cont_target_net.to('cpu')
        self.cont_target_net.set_weights(
            [layer.weight.clone() for layer in self.ann.linear_layers],
            [layer.bias.clone() for layer in self.ann.linear_layers]
        )

    def reset(self):
        print("Resetting the dataset")
        self.timestep = 0

    def __iter__(self):
        return self

    def __next__(self):
        inputs = torch.tensor([math.cos(freq*self.timestep + phase)
                               for freq, phase in zip(self.cosine_freqs, self.cosine_phases)]).reshape(1, -1)
        # targets = self.ann(inputs).detach()

        self.cont_target_net.feedforward_only_update(inputs.reshape(-1, 1), dt=self.dt)
        target = self.cont_target_net.layers[-1].event_rate

        self.timestep += 1

        return inputs, target

