# Code for "Cell-type-specific cortical feedback coordinates hierarchical credit assignment"

<img width="610" height="757" alt="burstccn_schematic" src="https://github.com/user-attachments/assets/df8ce294-aa4f-4311-b8f2-e5f5165a9b44" />

A preprint of our paper is available at https://doi.org/10.64898/2026.06.16.732595
This repository contains the code to run the BurstCCN model variants used in the paper: discrete-time rate-based BurstCCN, Dalean BurstCCN, continuous-time rate-based BurstCCN, and spiking BurstCCN. It also includes the code used to reproduce the figures.

## Requirements

Create the environment with:

```bash
conda env create -f environment.yaml
```

Then activate it with `conda activate burstccn`.

## Training the discrete-time rate-based models

To run the discrete-time rate-based MNIST models, use the commands:

```bash
python train.py --config-name=mnist_burstccn
python train.py --config-name=mnist_burstccn_dales
```

Individual hyperparameters can be overridden on the command line. Weights & Biases is used to log run data and can be configured either in a `wandb:` block inside a config or through the `WANDB_PROJECT` and `WANDB_ENTITY` environment variables.

## Training the continuous-time model

```bash
python continuous_burstccn/train_continuous.py
```

## Training the spiking model

```bash
python spiking_burstccn/run_spiking_burstccn_model.py
```

