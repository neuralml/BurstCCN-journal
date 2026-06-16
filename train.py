import random
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

from burstccn.triggers import Trigger
from burstccn.utils import flatten_dict, unflatten_dict

import wandb

from burstccn.datasets import DatasetFactory
from burstccn.model_inspector import ModelInspector
from burstccn.model_trainers import ModelTrainer
from burstccn.models.model_factory import ModelFactory
from burstccn.wandb_logger import WandbLogger

import signal

from burstccn.wandb_pandas_interface import get_wandb_entity, get_wandb_project

def _handle_sigterm(signum, frame):
    print("Received SIGTERM (likely scancel). Exiting cleanly...", flush=True)
    raise KeyboardInterrupt  # will trigger your try/except/finally

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)  # optional: Ctrl+C same behaviour


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_trigger_event_map_from_config(trigger_cfgs, unique_run_name):
    from hydra.utils import instantiate

    # if trigger_cfgs is None:
    #     trigger_cfgs = []
    #
    # if isinstance(trigger_cfgs, (dict, DictConfig)):
    #     trigger_items = trigger_cfgs.items()
    # else:
    #     trigger_items = [(None, cfg) for cfg in trigger_cfgs]
    #
    # triggers = []
    # for trigger_name, cfg in trigger_items:
    #     instantiate_kwargs = {"unique_run_name": unique_run_name}
    #     if trigger_name is not None:
    #         instantiate_kwargs["trigger_name"] = trigger_name
    #     triggers.append(instantiate(cfg, **instantiate_kwargs))

    triggers = [instantiate(cfg, unique_run_name=unique_run_name) for cfg in trigger_cfgs]

    event_names = [
        "on_epoch_start", "on_epoch_end",
        "on_batch_start", "on_batch_end",
        "on_batch_pre_update"
    ]
    triggers_by_event = {event: [] for event in event_names}

    for trig in triggers:
        for event in event_names:
            try:
                trig_func = getattr(type(trig), event)
                base_func = getattr(Trigger, event)
            except AttributeError:
                continue

            if callable(trig_func) and trig_func is not base_func:
                triggers_by_event[event].append(trig)

    return triggers_by_event


def setup_and_train(config):
    # Your training logic here.
    print("Training config:", config.training)
    print("Training with dataset:", config.dataset)
    print("Training with model:", config.model)
    # print("Training with optimiser:", config.optimiser)

    set_seed(config.training.seed)

    # print(config.training.triggers)

    train_loader, val_loader, test_loader = DatasetFactory.get_dataset(config.dataset,
                                                                       train_batch_size=config.training.batch_size,
                                                                       test_batch_size=config.training.test_batch_size)

    if config.training.require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU required by config but not available.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ModelFactory.create_model(config.model)
    parallel_model = model.to(device)

    if config.training.save_initial_model:
        save_file = "saved_models/test.pt"
        print(f"Saving model: {save_file}")
        torch.save(model.state_dict(), save_file)

    lr_scheduler_context = {"steps_per_epoch": len(train_loader), "total_epochs": config.training.n_epochs}
    optimiser = model.create_optimiser(lr_scheduler_context=lr_scheduler_context)

    wandb_logger = WandbLogger(log_interval=config.training.log_interval)
    model_inspector = ModelInspector(model)

    unique_run_name = wandb.run.name + "-" + wandb.run.id
    # trigger_cfgs = getattr(config.training, "triggers", [])

    trigger_cfgs = getattr(config.training, "triggers", None)
    if isinstance(trigger_cfgs, DictConfig):
        trigger_cfgs = list(trigger_cfgs.values())
    elif trigger_cfgs is None:
        trigger_cfgs = []


    trigger_event_map = build_trigger_event_map_from_config(trigger_cfgs, unique_run_name=unique_run_name)

    model_trainer = ModelTrainer(model, parallel_model, optimiser, train_loader, val_loader, test_loader,
                                 config.dataset.task_type, config.model.loss_type, wandb_logger, model_inspector,
                                 max_stagnant_epochs=config.training.get('max_stagnant_epochs', None),
                                 save_models=config.training.get("save_models", False),
                                 model_output_dir=wandb.run.dir if wandb.run else "./",
                                 trigger_event_map=trigger_event_map,
                                 label_smoothing=config.training.get('label_smoothing', 0.0))

    try:
        model_trainer.train_with_evaluation(config.training.n_epochs)
    except KeyboardInterrupt:
        print("Interrupted. Finishing W&B run and exiting cleanly...")
        wandb.finish(exit_code=1)
    else:
        wandb.finish(exit_code=0)


def save_code():
    import __main__
    artifact = wandb.Artifact("code", type="code")
    artifact.add_dir(f"./burstccn", name="burstccn")
    artifact.add_dir(f"./configs", name="configs")
    artifact.add_file(__main__.__file__)
    wandb.log_artifact(artifact)


@hydra.main(config_path="configs", version_base=None)
def main(cfg: DictConfig):
    if not cfg:
        raise ValueError("You must provide a config name via --config-name.")

    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    flat_cfg = flatten_dict(cfg_dict)

    wandb_cfg = cfg.get("wandb", None)
    project = wandb_cfg.get("project") if wandb_cfg and wandb_cfg.get("project") else get_wandb_project()
    entity = wandb_cfg.get("entity") if wandb_cfg and wandb_cfg.get("entity") else get_wandb_entity()
    group = flat_cfg.get("group", "default")
    flat_cfg["wandb.project"] = project
    if entity is not None:
        flat_cfg["wandb.entity"] = entity

    def init_setup_and_train():
        wandb_kwargs = dict(project=project, group=group, config=flat_cfg)
        if entity:
            wandb_kwargs["entity"] = entity

        with wandb.init(**wandb_kwargs) as run:
            save_code()

            config_dict = unflatten_dict(dict(wandb.config))
            config = OmegaConf.create(config_dict)

            run.name = config.get("run_name", run.name)
            setup_and_train(config)

    init_setup_and_train()


def repeat(value, n):
    return [value] * int(n)

if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("repeat", repeat)
    main()
