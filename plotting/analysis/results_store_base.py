from __future__ import annotations

import functools
import os
import pickle
import tempfile
from pathlib import Path
from typing import Literal, Tuple, Callable

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from burstccn.wandb_pandas_interface import WandbPandasInterface, get_wandb_entity, get_wandb_project

CACHE_OVERWRITE_GLOBAL = True


class FileCache:
    def __init__(self, cache_filename):
        self.cache_path = Path(__file__).parent / 'results_caches' / cache_filename
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._safe_load()

    def _safe_load(self):
        if not self.cache_path.exists() or self.cache_path.stat().st_size == 0:
            return {}
        try:
            with self.cache_path.open('rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError):
            # Optional: rename for debugging
            try:
                self.cache_path.rename(self.cache_path.with_suffix(self.cache_path.suffix + '.corrupt'))
            except OSError:
                pass
            return {}

    def get(self, key, default=None):
        return self._cache.get(key, default)

    def set(self, key, value):
        self._cache[key] = value
        self._atomic_dump(self._cache)

    def clear(self):
        self._cache = {}
        try:
            self.cache_path.unlink()
        except FileNotFoundError:
            pass

    def _atomic_dump(self, obj):
        dirpath = self.cache_path.parent
        with tempfile.NamedTemporaryFile('wb', delete=False, dir=dirpath) as tmp:
            tmp_path = Path(tmp.name)
            pickle.dump(obj, tmp, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_path, self.cache_path)


def make_hashable(obj):
    """Recursively convert unhashable types (lists, dicts, sets) to hashable equivalents."""
    if isinstance(obj, (tuple, list)):
        return tuple(make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, set):
        return frozenset(make_hashable(x) for x in obj)
    elif isinstance(obj, np.ndarray):
        return (obj.dtype.str, obj.shape, obj.tobytes())
    else:
        return obj  # Assume hashable


def file_cache_decorator(cache_attr='cache'):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            key_args = make_hashable(args)
            key_kwargs = make_hashable(kwargs)
            key = (func.__name__, key_args, key_kwargs)

            cache = getattr(self, cache_attr)

            if not CACHE_OVERWRITE_GLOBAL:
                cached_result = cache.get(key)
                if cached_result is not None:
                    return cached_result

            result = func(self, *args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper

    return decorator


class ResultsStore:
    def __init__(self, cache_path):
        if cache_path is None:
            self.cache = None
        else:
            self.cache = FileCache(cache_path)


class WandbResultsStore(ResultsStore):
    def __init__(self, project_entity=None, project_name=None, cache_path=None):
        super().__init__(cache_path)
        resolved_entity = project_entity or get_wandb_entity(required=True)
        resolved_project = project_name or get_wandb_project()
        self.wandb_interface = WandbPandasInterface(project_entity=resolved_entity,
                                                    project_name=resolved_project,
                                                    base_data_path=Path(__file__).parent)

    def fetch(self, run_name: str, group: str | None, keys, sort_by=None):
        return self.wandb_interface.get_run_data_(run_name, group, keys, sort_by=sort_by)

    # Convenience: resolve run_name from filters, infer group from results_identifier
    def fetch_by(self, *, keys, sort_by=None, **filters):
        run_filter = self.get_run_filter(**filters)
        return self.fetch(run_filter['run_name'], run_filter.get('group', None), keys, sort_by=sort_by)

    def fetch_and_summarise(self, run_name, group, step_key, data_key=None, data_keys=None, batch_to_epoch=False,
                            final_only=False, per_seed_fn=None, out_key=None):
        if data_keys is None:
            if data_key is None:
                raise ValueError("Either 'data_key' or 'data_keys' must be provided.")
            resolved_data_keys = [data_key]
        else:
            resolved_data_keys = [data_keys] if isinstance(data_keys, str) else list(data_keys)
            if not resolved_data_keys:
                raise ValueError("'data_keys' must contain at least one key.")
            # Keep key order stable while removing duplicates.
            resolved_data_keys = list(dict.fromkeys(resolved_data_keys))

        keys = [step_key, *resolved_data_keys]
        if batch_to_epoch:
            if 'batch/epoch' not in keys:
                keys.append('batch/epoch')

        data = self.fetch(run_name=run_name, group=group, keys=keys)

        if batch_to_epoch:
            if len(resolved_data_keys) == 1:
                data = convert_batch_to_epoch(data,
                                              step_col=step_key,
                                              value_col=resolved_data_keys[0])
            else:
                epoch_col = 'batch/epoch'
                sample_col = 'seed'
                converted = []
                for key in resolved_data_keys:
                    subset_cols = [sample_col, epoch_col, key]
                    if step_key != epoch_col:
                        subset_cols = [step_key, *subset_cols]
                    converted.append(
                        convert_batch_to_epoch(
                            data[subset_cols],
                            step_col=step_key,
                            value_col=key
                        )
                    )
                data = converted[0]
                for converted_df in converted[1:]:
                    data = data.merge(converted_df, on=[epoch_col, sample_col], how='outer')

        if per_seed_fn is not None:
            data = (
                data.groupby("seed", group_keys=False)
                .apply(per_seed_fn)
            )

        if final_only:
            data = data.groupby('seed', as_index=False).tail(1)

        if out_key is None:
            if len(resolved_data_keys) == 1:
                value_col = resolved_data_keys[0]
            else:
                raise ValueError("When multiple 'data_keys' are provided, 'out_key' must also be provided.")
        else:
            value_col = out_key

        return summarise_metric(df=data,
                                step_col=step_key,
                                value_col=value_col,
                                return_arrays=True)


class BurstCCNWandbResultsStore(WandbResultsStore):
    def __init__(self, project_entity=None, project_name='burstccn', cache_path=None):
        super().__init__(project_entity, project_name, cache_path)

        self.EPOCH_KEY = "epoch"
        self.BATCH_EPOCH_KEY = "batch/epoch"

        self.TEST_ERROR_KEY = "epoch/top1_error/test"
        self.BEST_TEST_ERROR_KEY = "epoch/top1_error_best/test"

        self.BATCH_KEY = "batch"
        self.ANGLE_KEYS = {"qy": "batch/angle_QY/global",
                           "fa": "batch/angle_fa/global_hidden",
                           "bp": "batch/angle_bp/global_hidden"
                           }

        self.APICAL_MAGNITUDE_KEY = "batch/apical_magnitude/global"
        self.BURST_PROB_MAGNITUDE_KEY = "batch/burst_prob_change_magnitude/global"


def convert_batch_to_epoch(
        df: pd.DataFrame,
        *,
        step_col: str,  # e.g. "batch" (global step)
        value_col: str,  # metric
        sample_col: str = "seed",
        epoch_col: str = "batch/epoch",
        steps_per_epoch: int | None = None,  # if epoch not already present
        # reduce: str = "mean",  # "mean" | "min" | "max" | "median" | "first"
        reduce: str = "first",  # "mean" | "min" | "max" | "median" | "first"
) -> pd.DataFrame:
    """
    Returns df with one row per (epoch, seed): columns [epoch_col, sample_col, value_col].
    """
    df = df.copy()

    if epoch_col not in df.columns:
        if steps_per_epoch is None:
            raise ValueError(f"'{epoch_col}' not in df and steps_per_epoch is None")
        # epoch index starting at 0; use +1 if you prefer 1-indexed epochs
        df[epoch_col] = (df[step_col] // steps_per_epoch).astype(int)

    first_raw = df.iloc[[0]].copy()
    first_raw[epoch_col] = 0
    df = pd.concat([first_raw, df], ignore_index=True)

    reducers = {
        "mean": "mean",
        "min": "min",
        "max": "max",
        "median": "median",
        "first": "first",
    }
    if reduce not in reducers:
        raise ValueError(f"reduce must be one of {list(reducers)}, got {reduce}")

    out = (
        df.groupby([epoch_col, sample_col], as_index=False)[value_col]
        .agg(reducers[reduce])
    )
    return out



def arrays_from_summary(summary: pd.DataFrame, step_col: str,
                        mean_col: str = "mean", err_col: str = "stderr"):
    x = summary[step_col].to_numpy()
    m = summary[mean_col].to_numpy()
    e = summary[err_col].to_numpy() if err_col in summary.columns else None
    return x, m, e


def summarise_metric(
        df: pd.DataFrame,
        *,
        step_col: str,  # e.g. "batch"
        value_col: str,  # e.g. "batch/angle_fa/global"
        sample_col: str = "seed",  # what counts as an independent replicate ("seed" is often better)
        err: Literal["sem", "std"] = "sem",
        sort: bool = True,
        return_arrays: bool = False
):
    """
    Two-stage aggregation:
      1) average duplicates within each unit at each step
      2) aggregate across units -> mean + stderr/std (+ n)

    Returns tidy DataFrame with columns:
      [step_col, "mean", ("stderr" or "std"), "n"]
    """
    required = [step_col, value_col, sample_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    # Stage 1: collapse duplicates within each unit at each step
    per_unit = (
        df.groupby([step_col, sample_col], as_index=False)[value_col]
        .mean()
    )

    # Stage 2: aggregate across units for each step
    g = per_unit.groupby(step_col)[value_col]
    out = pd.DataFrame({
        step_col: g.mean().index.to_numpy(),
        "mean": g.mean().to_numpy(),
        "n": g.count().to_numpy(),
    })

    if err == "sem":
        err_col = "stderr"
        # SEM = std / sqrt(n); use ddof=1 for sample std; handle n<=1 safely
        std = g.std(ddof=1).to_numpy()
        n = out["n"].to_numpy()
        out[err_col] = std / np.sqrt(np.maximum(n, 1))
    elif err == "std":
        err_col = "std"
        out[err_col] = g.std(ddof=1).to_numpy()
    else:
        raise ValueError("err must be 'sem' or 'std'")

    if sort:
        out = out.sort_values(step_col).reset_index(drop=True)

    if return_arrays:
        return arrays_from_summary(out, step_col=step_col, err_col=err_col)

    return out


def smooth_arrays(x, mean, err=None, sigma: float = 1.0, preserve_first=True):
    m = gaussian_filter1d(mean, sigma=sigma)
    e = gaussian_filter1d(err, sigma=sigma) if err is not None else None

    if preserve_first:
        m[0] = mean[0]  # note: creates a tiny kink at t=0
        e[0] = err[0] if err is not None else None

    return x, m, e


def summarise_final_step(
        df: pd.DataFrame,
        step_col: str,  # e.g. "epoch"
        value_col: str,  # e.g. "epoch/top1_error_best/test"
        per_run_col: str = "run_id",
        err: str = "sem",
):
    """
    Take value at the *final step per run* (max step), then aggregate over runs.
    Returns (mean, err_value).
    """
    idx = df.groupby(per_run_col)[step_col].idxmax()
    last = df.loc[idx, value_col]
    if err == "sem":
        return last.mean(), last.sem(ddof=1)
    elif err == "std":
        return last.mean(), last.std(ddof=1)
    else:
        raise ValueError("err must be 'sem' or 'std'")


def select_at_best_epoch(
        values_df: pd.DataFrame,
        best_epoch_df: pd.DataFrame,
        values_step_col: str,  # e.g. "batch/epoch"
        best_epoch_col: str  # e.g. "epoch/top1_error_best_epoch/test"
) -> pd.DataFrame:
    """
    Keep rows where values_step_col == best_epoch_col for each run_id.
    best_epoch_df must have ['run_id', best_epoch_col].
    """
    merged = values_df.merge(best_epoch_df[["run_id", best_epoch_col]], on="run_id", how="inner")
    out = merged.loc[merged[values_step_col] == merged[best_epoch_col]].copy()

    # If a batch index column exists, sort by it for nice plotting
    batch_index_cols = [c for c in out.columns if c.startswith("batch/") and "batch_index" in c]
    if batch_index_cols:
        out = out.sort_values(by=batch_index_cols)

    return out


def select_at_epoch(
        values_df: pd.DataFrame,
        epoch_to_select: float | int,
        *,
        values_step_col: str
) -> pd.DataFrame:
    out = values_df.loc[values_df[values_step_col] == epoch_to_select].copy()
    return out


def build_per_seed_function(
        *,
        step_key: str,
        out_key: str,
        fn: Callable[[pd.DataFrame], pd.Series],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    if fn is None:
        raise ValueError("'fn' must be provided.")

    def _per_seed(group: pd.DataFrame) -> pd.DataFrame:
        g = group.sort_values(step_key).copy()
        new_vals = fn(g)  # fn receives full per-seed dataframe

        # Ensure we can assign back row-aligned values
        new_vals = pd.Series(new_vals, index=g.index, name=out_key)
        g[out_key] = new_vals
        return g

    return _per_seed
