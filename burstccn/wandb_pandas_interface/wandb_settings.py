from __future__ import annotations

import os
from typing import Optional

DEFAULT_WANDB_PROJECT = "burstccn"


def get_wandb_entity(default: Optional[str] = None, *, required: bool = False) -> Optional[str]:
    entity = os.getenv("WANDB_ENTITY", default)
    if required and not entity:
        raise RuntimeError("Set WANDB_ENTITY or pass an explicit W&B entity.")
    return entity


def get_wandb_project(default: str = DEFAULT_WANDB_PROJECT) -> str:
    return os.getenv("WANDB_PROJECT", default)


def get_wandb_path(
    project: Optional[str] = None,
    entity: Optional[str] = None,
    *,
    default_project: str = DEFAULT_WANDB_PROJECT,
    require_entity: bool = False,
) -> str:
    resolved_entity = entity if entity is not None else get_wandb_entity(required=require_entity)
    if not resolved_entity:
        raise RuntimeError("Set WANDB_ENTITY or pass an explicit W&B entity.")

    resolved_project = project if project is not None else get_wandb_project(default_project)
    return f"{resolved_entity}/{resolved_project}"
