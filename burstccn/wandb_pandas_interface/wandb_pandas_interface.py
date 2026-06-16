import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm


# These look like W&B *run IDs*, not display names
RUN_ID_BLACKLIST = ['ym30i94t', 'vdtitro5']


# ----------------------------
# Selection model (local + remote)
# ----------------------------
@dataclass(frozen=True)
class RunSelector:
    """Describe which runs you want.

    Works BOTH for remote W&B API filters and for local (cached) metadata filtering.
    """
    names: Optional[Sequence[str]] = None          # display_name(s)
    group: Optional[str] = None                    # W&B group
    required_config_keys: Optional[Sequence[str]] = None
    # only_finished: bool = True
    only_finished: bool = False
    ignore_sweeps: bool = True
    apply_blacklist: bool = True
    additional_filters: Mapping[str, Any] = field(default_factory=dict)

    def to_wandb_filters(self) -> Dict[str, Any]:
        """Translate to W&B API filters."""
        f: Dict[str, Any] = {}
        if self.names:
            f["display_name"] = {"$in": list(self.names)}
        if self.group:
            f["group"] = {"$eq": self.group}
        if self.required_config_keys:
            for k in self.required_config_keys:
                f[f"config.{k}"] = {"$ne": None}
        if self.only_finished:
            f["state"] = {"$eq": "finished"}
        if self.ignore_sweeps:
            f["sweep"] = {"$eq": None}
        if self.apply_blacklist and RUN_ID_BLACKLIST:
            # # IMPORTANT: filter by run *id*, not run name
            # f["id"] = {"$nin": RUN_ID_BLACKLIST}
            f["name"] = {"$nin": RUN_ID_BLACKLIST}
        if self.additional_filters:
            f.update(self.additional_filters)
        return f

    def predicate(self, row: pd.Series) -> bool:
        """Local (cached) filtering against run_metadata rows."""
        if self.names and row.get("display_name") not in self.names:
            return False
        if self.group and row.get("group") != self.group:
            return False
        if self.only_finished and row.get("state") != "finished":
            return False
        if self.ignore_sweeps and pd.notna(row.get("sweep")):
            return False
        if self.apply_blacklist and row.get("run_id") in RUN_ID_BLACKLIST:
            return False
        if self.required_config_keys:
            for k in self.required_config_keys:
                if pd.isna(row.get(f"config.{k}")):
                    return False
        return True


def history_strict(run, keys=None, samples=100000):
    # Normalize keys
    requested_keys = list(keys) if keys is not None else None

    # Pre-check using history_keys when available
    hk = getattr(run, "history_keys", None)
    if callable(hk):  # some versions expose it as a method
        hk = hk()
    logged = set(hk.get("keys", [])) if isinstance(hk, dict) else set(hk or [])

    if requested_keys and logged:  # only enforce if W&B gave us the index
        missing = [k for k in requested_keys if k not in logged]
        if missing:
            raise KeyError(f"W&B run {getattr(run, 'id', '<unknown>')} is missing history keys: {missing}")

    # Fetch
    df = run.history(pandas=True, keys=requested_keys, samples=samples)

    # Fallback if likely truncated
    if len(df) >= samples:
        print(f"[{getattr(run, 'id', '<unknown>')}] Hit {samples} row cap, falling back to scan_history()...")
        it = run.scan_history(keys=requested_keys)
        df = pd.DataFrame(list(tqdm(it, desc=f"Loading history for {getattr(run, 'id', '<unknown>')}")))

    # Post-check (only if keys were requested)
    if requested_keys:
        missing_cols = [k for k in requested_keys if k not in df.columns]
        if missing_cols:
            raise KeyError(f"Requested keys not present in returned history: {missing_cols}")
        if df.empty:
            raise ValueError(f"History is empty for keys {requested_keys} (despite being listed).")

    return df


class WandbPandasInterface:
    def __init__(self, project_entity: str, project_name: str, base_data_path: str):
        self.api = wandb.Api(timeout=30)

        self.project_entity = project_entity
        self.project_name = project_name
        self.rename_episode_keys_backwards_compatible = True

        self.base_data_path = Path(base_data_path) / 'wandb_run_data'
        self.run_data_path = self.base_data_path / f'{project_entity}_{project_name}'
        self.run_data_path.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.run_data_path / 'run_metadata.pkl'

        if self.metadata_file.exists():
            self.run_metadata = pd.read_pickle(self.metadata_file)

            # # Remove cached rows whose display_name contains "x"
            # mask = self.run_metadata["display_name"].astype(str).str.contains(
            #     "imagenet", case=False, na=False
            # )
            # if mask.any():
            #     self.run_metadata = self.run_metadata.loc[~mask].copy()
            #     self.run_metadata.to_pickle(self.metadata_file)
        else:
            self.run_metadata = pd.DataFrame(columns=["run_id", "display_name"]).set_index("run_id", drop=False)

    # ----------------------------
    # Utilities
    # ----------------------------
    @staticmethod
    def parse_data_type(data_type: str) -> Sequence[str]:
        if '/' in data_type:
            return data_type.split('/', 1)
        else:
            return data_type, data_type  # base_key, sub_key

    def get_file_path(self, run_id: str, base_key: str) -> Path:
        return self.run_data_path / f"{run_id}_{base_key}_data.pkl"

    def check_missing_columns(self, run_data: pd.DataFrame, required_columns: List[str]) -> List[str]:
        missing_columns = [col for col in required_columns if col not in run_data.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        return missing_columns

    # (Legacy helper; kept if you still use it elsewhere. Not used by update_metadata.)
    @staticmethod
    def build_filters(run_names=None, group=None, required_config_keys=None, only_finished=True, ignore_sweeps=True,
                      apply_blacklist=True, additional_filters=None) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        if run_names:
            filters["display_name"] = {"$in": run_names}
        if group:
            filters["group"] = {"$eq": group}
        if required_config_keys:
            filters.update({f"config.{key}": {"$ne": None} for key in required_config_keys})
        if only_finished:
            filters["state"] = {"$eq": "finished"}
        if ignore_sweeps:
            filters["sweep"] = {"$eq": None}
        if apply_blacklist and RUN_ID_BLACKLIST:
            filters["id"] = {"$nin": RUN_ID_BLACKLIST}
        if additional_filters:
            filters.update(additional_filters)
        return filters

    # ----------------------------
    # Metadata sync (remote) + cache
    # ----------------------------
    def update_metadata(self, selector: RunSelector) -> None:
        """Fetch metadata from W&B for the given selector and merge into local cache.

        Semantics:
        - Always performs a remote call and upserts rows for the runs matching `selector`.
        - Existing rows are replaced (keep='last') so 'refresh' happens naturally.
        """
        filters = selector.to_wandb_filters()
        runs = self.api.runs(f"{self.project_entity}/{self.project_name}", filters=filters)

        new_rows: List[Dict[str, Any]] = []
        for run in runs:
            run_id = run.id
            metadata: Dict[str, Any] = {
                "run_id": run_id,
                "display_name": getattr(run, "display_name", None),
                "state": getattr(run, "state", None),
                "sweep": getattr(run, "sweep", None),
                "group": getattr(run, "group", None),  # store group for local filtering
                "host": (getattr(run, "metadata", {}) or {}).get("host"),  # safe get
            }
            for config_key, config_value in (getattr(run, "config", {}) or {}).items():
                metadata[f"config.{config_key}"] = config_value
            new_rows.append(metadata)

        if new_rows:
            new_df = pd.DataFrame(new_rows).set_index("run_id", drop=False)
            self.run_metadata = pd.concat([self.run_metadata, new_df], ignore_index=False)
            # Upsert semantics: keep the last version for each run_id
            self.run_metadata = self.run_metadata[~self.run_metadata.index.duplicated(keep='last')]
            self.run_metadata.to_pickle(self.metadata_file)
            print(f"Metadata updated and saved to {self.metadata_file}")

    # ----------------------------
    # Local-first selection
    # ----------------------------
    def ensure_metadata(self, selector: RunSelector, policy: str = "if_missing") -> None:
        """Control when to call W&B:
        - 'never': don't sync, rely on local cache only
        - 'if_missing': sync only if local cache has no matches
        - 'always': sync every time (refresh)
        """
        local_matches = self.run_metadata[self.run_metadata.apply(selector.predicate, axis=1)]

        if policy == "always":
            self.update_metadata(selector)
        elif policy == "if_missing" and local_matches.empty:
            self.update_metadata(selector)
        # 'never' does nothing

    def run_ids(self, selector: RunSelector, metadata_policy: str = "if_missing") -> List[str]:
        """Get run_ids matching selector, optionally syncing metadata first."""
        self.ensure_metadata(selector, policy=metadata_policy)
        matches = self.run_metadata[self.run_metadata.apply(selector.predicate, axis=1)]
        if matches.empty:
            target = f"group='{selector.group}'" if selector.group else f"names={selector.names}"
            raise ValueError(f"No runs found for {selector} in local metadata.")
        return matches.index.tolist()

    # Back-compat thin wrappers (optional to keep)
    def get_run_ids_by_name(self, run_name: str) -> List[str]:
        selector = RunSelector(names=[run_name])
        return self.run_ids(selector, metadata_policy="never")  # local-only

    def list_groups(self) -> List[str]:
        if "group" not in self.run_metadata.columns:
            return []
        vals = self.run_metadata["group"].dropna().unique()
        return sorted(vals)

    # ----------------------------
    # History download (key-aware)
    # ----------------------------
    def get_wandb_run_data(self, run_id: str, keys: Optional[Sequence[str]] = None,
                           modify_for_backwards_compatibility: bool = True) -> pd.DataFrame:
        """Download run history. If keys is provided, request only those columns."""
        run = self.api.run(f"{self.project_entity}/{self.project_name}/{run_id}")

        if modify_for_backwards_compatibility:
            wandb_keys = self._rename_keys_backwards_compatible(keys)
        else:
            wandb_keys = keys

        # wandb_run_data = run.history(pandas=True, keys=list(keys) if keys else None, samples=100000)
        wandb_run_data = history_strict(run=run, keys=list(wandb_keys) if wandb_keys else None, samples=100000)

        # (Optional) Apply your back-compat tweaks here if needed:
        if modify_for_backwards_compatibility:
            # if 'episode' in wandb_run_data.columns and 'avg_test_score' in wandb_run_data.columns:
            #     wandb_run_data['log_episode'] = wandb_run_data['episode'].where(
            #         wandb_run_data['avg_test_score'].notna()
            #     )
            rename_map = {
                old: new for old, new in zip(wandb_keys, keys) if old in wandb_run_data.columns
            }
            wandb_run_data = wandb_run_data.rename(columns=rename_map)

        return wandb_run_data

    def _rename_keys_backwards_compatible(self, new_keys):
        if not new_keys:
            return new_keys
        if not self.rename_episode_keys_backwards_compatible:
            return new_keys
        if any(k.startswith("episode/") for k in new_keys):
            return [k.removeprefix("episode/") for k in new_keys]
        return new_keys

    # ----------------------------
    # Helpers for history alignment & verification
    # ----------------------------
    @staticmethod
    def _align_cache_to_remote(cache: pd.DataFrame, remote_base: pd.DataFrame, base_key: str) -> pd.DataFrame:
        """Make cache rows match remote base_key values (extend/truncate), preserving cache columns."""
        if base_key in cache.columns:
            other = [c for c in cache.columns if c != base_key]
            return remote_base[[base_key]].merge(cache[[base_key] + other], on=base_key, how="left")
        return remote_base[[base_key]].copy()

    @staticmethod
    def _verify_columns_equal(cache: pd.DataFrame, remote: pd.DataFrame, base_key: str, cols: List[str],
                              rtol: float = 1e-7, atol: float = 1e-12) -> None:
        """Verify overlapping rows (by base_key) match for given cols. Raise on mismatch."""
        if not cols or base_key not in cache.columns:
            return
        to_cmp = [base_key] + [c for c in cols if c in remote.columns and c in cache.columns]
        if len(to_cmp) <= 1:
            return
        merged = cache[to_cmp].merge(remote[to_cmp], on=base_key, how="inner",
                                     suffixes=("_cache", "_remote"))

        cache_keys = set(cache[base_key].dropna())
        remote_keys = set(remote[base_key].dropna())
        extra_cache = cache_keys - remote_keys
        if extra_cache:
            raise AssertionError(
                f"Remote missing {len(extra_cache)} {base_key} values present in cache "
                f"(e.g. {next(iter(extra_cache))})."
            )

        for c in cols:
            if c not in remote.columns or c not in cache.columns:
                continue
            a = merged[f"{c}_cache"].to_numpy()
            b = merged[f"{c}_remote"].to_numpy()
            mask = ~(pd.isna(a) & pd.isna(b))
            a, b = a[mask], b[mask]
            if a.size == 0:
                continue
            if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
                if not np.allclose(a, b, rtol=rtol, atol=atol):
                    raise AssertionError(f"Verification failed for '{c}': numeric values differ.")
            else:
                if not (a == b).all():
                    raise AssertionError(f"Verification failed for '{c}': non-numeric values differ.")

    # ----------------------------
    # Cache run histories (per-run implementation, 3 modes)
    # ----------------------------
    def _update_run_data_for_run(self, run_id: str, data_type_or_types, data_update: str = "missing") -> None:
        """
        Modes:
          - "force":   Fetch base_key + ALL requested cols; verify overlap; overwrite requested cols.
          - "missing": If all present, do nothing. If missing, fetch base_key + ONLY missing cols; no verify of existing; write missing.
          - "none":    Do nothing (offline; reading will error if missing).
        """
        assert data_update in {"force", "missing", "none"}, "data_update must be 'force'|'missing'|'none'"

        data_types = [data_type_or_types] if isinstance(data_type_or_types, str) else list(data_type_or_types)

        # Group requested data_types by base_key
        base_key_map: Dict[str, List[str]] = {}
        for dt in data_types:
            bk, _ = self.parse_data_type(dt)
            base_key_map.setdefault(bk, []).append(dt)

        # Inspect cache for each base_key
        run_data_files = {bk: self.get_file_path(run_id, bk) for bk in base_key_map}
        missing_by_bk: Dict[str, List[str]] = {}
        cache_by_bk: Dict[str, pd.DataFrame] = {}

        for bk, fp in run_data_files.items():
            if fp.exists():
                cache = pd.read_pickle(fp)
                cache_by_bk[bk] = cache
                missing_by_bk[bk] = [dt for dt in base_key_map[bk] if dt not in cache.columns]
            else:
                cache_by_bk[bk] = pd.DataFrame()
                missing_by_bk[bk] = list(base_key_map[bk])

        if data_update == "none":
            return  # offline, no fetch/write

        # Decide which base_keys to fetch
        if data_update == "force":
            bks_to_fetch = list(base_key_map.keys())
        else:  # "missing"
            bks_to_fetch = [bk for bk, missing in missing_by_bk.items() if missing]

        if not bks_to_fetch:
            return  # nothing to do for this run

        # Build union of keys to request once for this run
        keys_needed = set()
        for bk in bks_to_fetch:
            keys_needed.add(bk)
            if data_update == "force":
                keys_needed.update(base_key_map[bk])           # all requested cols (for verify)
            else:  # "missing"
                keys_needed.update(missing_by_bk[bk])          # only missing cols

        wb = self.get_wandb_run_data(run_id, keys=sorted(keys_needed))

        # Verify + write per base_key
        for bk in bks_to_fetch:
            fp = run_data_files[bk]
            cache = cache_by_bk[bk]

            if bk not in wb.columns:
                raise KeyError(f"{bk} not found in wandb data for run {run_id}")

            remote = wb[wb[bk].notna()].copy()
            cache_aligned = self._align_cache_to_remote(cache, remote[[bk]], bk)

            if data_update == "force":
                # verify overlap for requested cols already present in cache
                existing_req = [c for c in base_key_map[bk] if c in cache.columns and c in wb.columns]
                if existing_req:
                    self._verify_columns_equal(cache_aligned, remote, bk, existing_req)
                targets = list(base_key_map[bk])  # overwrite all requested
            else:  # "missing"
                targets = list(missing_by_bk[bk])  # write only missing

            if not targets:
                continue

            out = cache_aligned.copy()
            for dt in targets:
                if dt in remote.columns:
                    out[dt] = remote[dt].values
                else:
                    print(f"[{run_id}] sub_data_type '{dt}' not found among columns for base '{bk}'.")
            out.to_pickle(fp)
            print(f"Updated {fp} for run_id '{run_id}'.")

    # Selector wrapper (resolve run_ids then update each)
    def update_run_data(self, selector: RunSelector, data_type_or_types, data_update: str = "missing",
                        policy: str = "if_missing") -> List[str]:
        """Resolve run_ids from selector (respecting metadata sync policy) and update each run's cache."""
        run_ids = self.run_ids(selector, metadata_policy=policy)
        for rid in run_ids:
            self._update_run_data_for_run(rid, data_type_or_types, data_update=data_update)
        return run_ids

    # Back-compat by-id wrapper (optional to keep)
    def update_run_data_by_id(self, run_id_or_ids, data_type_or_types, data_update: str = "missing") -> None:
        run_ids = [run_id_or_ids] if isinstance(run_id_or_ids, str) else list(run_id_or_ids)
        for rid in run_ids:
            self._update_run_data_for_run(rid, data_type_or_types, data_update=data_update)

    # ----------------------------
    # Public read APIs
    # ----------------------------
    def get_run_data_by_id(self, run_id: str, data_types, update: bool = True, data_update: str = "missing") -> pd.DataFrame:
        if update:
            self._update_run_data_for_run(run_id, data_types, data_update=data_update)

        if isinstance(data_types, str):
            data_types = [data_types]

        base_keys = set()
        all_required_columns: List[str] = []
        for data_type in data_types:
            base_key, _ = self.parse_data_type(data_type)
            base_keys.add(base_key)
            all_required_columns.append(data_type)

        if len(base_keys) != 1:
            raise ValueError(f"Multiple base keys detected: {base_keys}. Expected only one.")

        base_key = base_keys.pop()
        file_path = self.get_file_path(run_id, base_key)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file '{file_path}' not found for run_id '{run_id}'.")

        run_data = pd.read_pickle(file_path)
        required_columns = list(set([base_key] + all_required_columns))
        self.check_missing_columns(run_data, required_columns)

        return run_data[required_columns]

    def get_run_data_(self, run_name, group, data_types, sort_by: Optional[str] = None, policy: str = "if_missing"):
        selector = RunSelector(names=[run_name], group=group)
        return self.get_run_data(selector, data_types, sort_by, policy)

    def get_run_data(self, selector: RunSelector, data_types, sort_by: Optional[str] = None, policy: str = "if_missing") -> pd.DataFrame:
        """Unified convenience: selector -> run_ids -> collect -> sort."""
        run_ids = self.run_ids(selector, metadata_policy=policy)

        if isinstance(data_types, str):
            data_types = [data_types]

        base_keys = {self.parse_data_type(dt)[0] for dt in data_types}
        if len(base_keys) != 1:
            raise ValueError(f"Multiple base keys detected: {base_keys}. Expected only one.")
        base_key = base_keys.pop()

        frames: List[pd.DataFrame] = []
        for rid in run_ids:
            df = self.get_run_data_by_id(rid, data_types)  # defaults to data_update="missing"
            df['run_id'] = rid

            if "config.training.seed" in self.run_metadata.columns:
                seed_value = self.run_metadata.loc[rid, "config.training.seed"]
                if pd.notna(seed_value):
                    df['seed'] = seed_value

            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)

        if sort_by:
            if isinstance(sort_by, str):
                sort_by = [sort_by]
            combined = combined.sort_values(by=sort_by).reset_index(drop=True)
        elif base_key in combined.columns:
            combined = combined.sort_values(by=base_key).reset_index(drop=True)

        return combined
    # # Warm histories once with verification of overlapping columns
    # interface.update_run_data(sel, ["epoch/top1_error/test"], data_update="missing", policy="if_missing")
    #
    # # Read combined data (local-first; will only fetch if missing columns)
    # df = interface.get_run_data(sel, ["epoch/top1_error/test"])
    # print(df.head())
    #
    # # # Example: ensure metadata for a set of names under a group, then warm cache, then fetch
    # # sel = RunSelector(names=[f"mnist_ann_fa_{i+1}h" for i in range(8)], group='fa_runs')
    # #
    # # # Warm metadata (remote if missing)
    # # interface.ensure_metadata(sel, policy="if_missing")
    # #
    # # # Warm histories once with verification of overlapping columns
    # # interface.update_run_data(sel, ["epoch/top1_error/test"], data_update="missing", policy="if_missing")
    # #
    # # # Read combined data (local-first; will only fetch if missing columns)
    # # df = interface.get_run_data(sel, ["epoch/top1_error/test"])
    # # print(df.head())
