import re
from pathlib import Path

import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import procrustes
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS, TSNE

from plotting.analysis.results_store_base import file_cache_decorator, ResultsStore
from burstccn.wandb_pandas_interface import get_wandb_path


class MNISTRepresentationResultsStore(ResultsStore):
    def __init__(self, cache_path='mnist_representation_results.pkl'):
        super().__init__(cache_path)

        self.base_path = Path(__file__).resolve().parent / "mnist_representation_data" / "all"

        self.model_filename_templates = {
            "ann": "mnist_ann_fa_4h-{run_id}.h5",
            "np": "mnist_np_4h-{run_id}.h5",
            "burstccn": "mnist_burstccn_fa_Y_learning_4h-{run_id}.h5",
            # "burstccn_shift": "mnist_burstccn_fa_Y_learning_4h_shift-{run_id}.h5",
            # "burstccn_Y_block_trained": "mnist_burstccn_fa_without_teacher_updates_4h-{run_id}.h5",
            "burstccn_Y_block_trained": "mnist_burstccn_fa_Y_learning_hybrid_4h-{run_id}.h5",
            "burstccn_QY_tied": "mnist_burstccn_fa_4h-{run_id}.h5",
            "burstprop": "mnist_burstprop_fa_4h-{run_id}.h5",
            "edn": "mnist_edn_fa_fb_learning_4h-{run_id}.h5",
        }

        self._organise_files_into_seed_folders()
        self._balanced_subset_indices_cache = {}

    def _get_seed_for_run_id(self, run_id: str):
        import wandb
        if not hasattr(self, 'api'): self.api = wandb.Api()
        runs = self.api.runs(get_wandb_path(require_entity=True), filters={"name": {"$eq": run_id}})
        if len(runs) == 0:
            raise ValueError(f"No W&B run found for run_id={run_id}")
        if len(runs) > 1:
            print(f"[WARN] Multiple runs found for {run_id}; using first.")
        return runs[0].config["training.seed"]

    def _match_model_and_run_id(self, fname):
        for model_name, template in self.model_filename_templates.items():
            pattern = re.escape(template).replace(r"\{run_id\}", r"(?P<run_id>[^\.]+)")
            match = re.fullmatch(pattern, fname)
            if match:
                return model_name, match.group("run_id")
        return None, None

    def _organise_files_into_seed_folders(self):
        for file_path in self.base_path.glob("*.h5"):
            model_name, run_id = self._match_model_and_run_id(file_path.name)
            if model_name is None:
                continue
            seed = self._get_seed_for_run_id(run_id)
            target_dir = self.base_path / f"seed_{seed}"
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path.rename(target_dir / file_path.name)

    def _file(self, model_name, seed=None):
        pattern = self.model_filename_templates[model_name].format(run_id="*")
        if seed is not None:
            matches = sorted((self.base_path / f"seed_{seed}").glob(pattern))
            if matches:
                return matches[0]

        matches = sorted(self.base_path.glob(f"seed_*/{pattern}"))
        if matches:
            return matches[0]
        root_matches = sorted(self.base_path.glob(pattern))
        if root_matches:
            return root_matches[0]

        raise FileNotFoundError(f"No file found for model_name={model_name}, seed={seed}")

    def get_model_names(self):
        return list(self.model_filename_templates.keys())

    def balanced_indices(self, max_per_class, epoch, seed=None):
        key = (max_per_class, epoch, seed)
        if key in self._balanced_subset_indices_cache:
            return self._balanced_subset_indices_cache[key]

        ref_file = self._file("ann", seed=seed)

        with h5py.File(ref_file, "r") as f:
            labels = f[f"epoch_{epoch}_targets_with_teacher"][:]

        labels = np.asarray(labels).squeeze()

        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]

        starts = np.searchsorted(sorted_labels, np.arange(10), side="left")
        ends = np.searchsorted(sorted_labels, np.arange(10), side="right")

        parts = [order[s:e][:max_per_class] for s, e in zip(starts, ends)]
        idx = np.concatenate(parts)

        self._balanced_subset_indices_cache[key] = idx
        return idx

    def get_data(self, model_name, epoch, data_type, **kwargs):
        seed = kwargs.pop("seed", None)
        file_path = self._file(model_name, seed=seed)

        layer_idx = kwargs.pop("layer_idx", None)
        data_type, layer_idx = ("inputs", None) if data_type == "e" and layer_idx == 0 else (data_type, layer_idx)

        with h5py.File(file_path, "r") as f:
            if layer_idx is None:
                key = f"epoch_{epoch}_{data_type}_with_teacher"
            else:
                key = f"epoch_{epoch}_layer_{layer_idx}_{data_type}_with_teacher"

            if key not in f:
                print(f"\n[H5 DEBUG] Key not found: {key}")
                print(f"[H5 DEBUG] Available keys in {file_path.name}:")
                for k in f.keys():
                    print("   ", k)
                raise KeyError(key)

            x = f[key][:]

        if 'max_per_class' in kwargs:
            idx = self.balanced_indices(max_per_class=kwargs["max_per_class"], epoch=epoch, seed=seed)
            x = x[idx]

        if x.ndim > 1:
            x = x.reshape(x.shape[0], -1)

        return x

    # @file_cache_decorator()
    def rdv(self, **kwargs):
        x = self.get_data(**kwargs)
        return pdist(x, metric="correlation")

    def rdvs_for_layers(self, layer_indices, **kwargs):
        seeds = kwargs.pop("seeds", [1])
        rdvs = []
        for seed in seeds:
            rdvs.append(np.vstack([self.rdv(layer_idx=layer_idx, seed=seed, **kwargs) for layer_idx in layer_indices]))

        return np.mean(rdvs, axis=0)
        # return np.vstack([self.rdv(layer_idx=layer_idx, **kwargs) for layer_idx in layer_indices])

    def rdm_for_models_layers(self, models, layer_indices, **kwargs):
        rows = np.vstack([
            self.rdvs_for_layers(layer_indices, model_name=model_name, **kwargs)
            for model_name in models
        ])
        return squareform(pdist(rows, metric="correlation"))

    @file_cache_decorator()
    def mds_for_model_layers(self, models, layer_indices, **kwargs):
        rdm = self.rdm_for_models_layers(models=models, layer_indices=layer_indices, **kwargs)
        coords = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=0,
            eps=1e-9,
            normalized_stress=False,
        ).fit_transform(rdm)

        return coords

    @file_cache_decorator()
    def tsne_for_model(self, model_name, ref_model_name='ann', random_state=1, **kwargs):
        tsne = TSNE(n_components=2, random_state=random_state)
        activity = self.get_data(model_name=model_name, **kwargs)
        tsne_data = tsne.fit_transform(activity)

        if model_name != ref_model_name:
            ref_activity = self.get_data(model_name=ref_model_name, **kwargs)
            ref_tsne_data = TSNE(n_components=2, random_state=random_state).fit_transform(ref_activity)
            _, tsne_data, _ = procrustes(ref_tsne_data, tsne_data)

        return tsne_data

    def mds_coords(self, models, layer_indices, random_state=1, **kwargs):
        rdm = self.rdm_for_models_layers(models=models, layer_indices=layer_indices, **kwargs)
        return MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=random_state,
            eps=1e-9,
            normalized_stress=False,
        ).fit_transform(rdm)

    def compare_activities_across_models(
            self,
            models=None,
            *,
            epoch=0,
            layer_idx=0,
            data_type="e",
            max_per_class=None,
            atol=1e-8,
            rtol=1e-8,
            print_mismatches=5,
    ):
        """
        Compare a given (epoch, layer_idx, data_type) across models.

        - If atol=rtol=0: exact equality check for numeric arrays (still allows NaNs only if identical pattern).
        - If atol/rtol > 0: uses np.allclose.
        Prints basic stats and a few mismatch locations if it fails.
        """
        if models is None:
            models = list(self.results_map.keys())
        else:
            models = list(models)

        # Load reference
        ref_model = models[0]
        ref = self.get_data(
            model_name=ref_model,
            layer_idx=layer_idx,
            epoch=epoch,
            data_type=data_type,
            max_per_class=max_per_class,
        )

        def stats(x):
            x = np.asarray(x)
            return dict(
                shape=x.shape,
                dtype=str(x.dtype),
                min=float(np.nanmin(x)),
                max=float(np.nanmax(x)),
                mean=float(np.nanmean(x)),
                std=float(np.nanstd(x)),
                nan=int(np.isnan(x).sum()) if np.issubdtype(x.dtype, np.floating) else 0,
            )

        print(f"\n[COMPARE] epoch={epoch} layer_idx={layer_idx} data_type={data_type} max_per_class={max_per_class}")
        print(f"[REF] {ref_model}: {stats(ref)}")

        ok_all = True
        for m in models[1:]:
            x = self.get_data(
                model_name=m,
                layer_idx=layer_idx,
                epoch=epoch,
                data_type=data_type,
                max_per_class=max_per_class,
            )

            same_shape = (x.shape == ref.shape)
            if not same_shape:
                ok_all = False
                print(f"[FAIL] {m}: shape mismatch {x.shape} vs {ref.shape}")
                print(f"       stats: {stats(x)}")
                continue

            if atol == 0.0 and rtol == 0.0:
                equal = np.array_equal(x, ref)
            else:
                equal = np.allclose(x, ref, atol=atol, rtol=rtol, equal_nan=True)

            if equal:
                print(f"[OK]   {m}: matches {ref_model}")
            else:
                ok_all = False
                print(f"[FAIL] {m}: differs from {ref_model}")
                print(f"       stats: {stats(x)}")

                # Show a few mismatch indices + values
                # Works for any shape.
                diff_mask = ~(x == ref) if (atol == 0.0 and rtol == 0.0) else ~(
                    np.isclose(x, ref, atol=atol, rtol=rtol, equal_nan=True))
                mismatch_idx = np.argwhere(diff_mask)

                print(f"       mismatches: {mismatch_idx.shape[0]}")
                for idx in mismatch_idx[:print_mismatches]:
                    idx = tuple(idx)
                    print(f"         idx={idx}  ref={ref[idx]}  {m}={x[idx]}")

        if ok_all:
            print("[COMPARE] All models match.")
        else:
            print("[COMPARE] Some models differ.")

        return ok_all
