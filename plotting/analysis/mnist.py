from pathlib import Path

import h5py

from plotting.analysis.results_store_base import WandbResultsStore, ResultsStore


def tag(x) -> str:
    return str(x).replace(".", "p")


class MNISTResultsStore(WandbResultsStore):
    def __init__(self):
        super().__init__(cache_path='mnist')

        self.EPOCH_KEY = "epoch"
        self.BATCH_EPOCH_KEY = "batch/epoch"

        self.TEST_ERROR_KEY = "epoch/top1_error/test"
        self.BEST_TEST_ERROR_KEY = "epoch/top1_error_best/test"

        self.BATCH_KEY = "batch"
        self.ANGLE_KEYS = {"qy": "batch/angle_QY/global",
                           "fa": "batch/angle_fa/global_hidden",
                           "bp": "batch/angle_bp/global_hidden"
                           # "bp": "batch/angle_bp/global_hidden_average"
                           }

        self.APICAL_MAGNITUDE_KEY = "batch/apical_magnitude/global"
        self.BURST_PROB_MAGNITUDE_KEY = "batch/burst_prob_change_magnitude/global"

    def get_group_params(self, group):
        GROUP_PARAM_SETS = {
            "Y_learning_branches": dict(
                n_branches=[1, 2, 5, 10, 15],
            ),
            "Y_learning_forward_noise": dict(
                noise=["null", 0.05, 0.1, 0.2, 0.4],
            ),
            "Y_learning_error_scale": dict(
                error_scale=[0.0, 0.25, 0.5, 0.75, 1.0],
            ),
            "fa_performance": dict(
                # model_types=["ann", "burstccn_0q", "burstccn_Y_block_trained", "burstccn_Y_learning_noise",
                #             "burstprop", "edn", "edn_pred_tied"],
                # model_types=['ann', 'burstccn_Y_learning_noise', 'burstccn_Y_block_trained', 'burstprop', 'edn'],
                # model_types=['ann', 'burstccn_Y_learning_noise', 'burstccn_Y_block_trained', 'burstccn_QY_tied', 'burstprop', 'edn'],
                # model_types=['ann', 'burstccn_Y_learning', 'burstccn_Y_block_trained', 'burstccn_QY_tied', 'burstprop', 'edn'],
                model_types=['ann', 'burstccn_Y_learning', 'burstccn_Y_block_trained', 'burstccn_QY_tied'],
                n_hidden_layers=[1, 2, 3, 4, 5, 6, 7, 8],
            ),
            "tied_performance": dict(
                model_types=["ann", "burstccn_0q", "burstprop", "edn", "edn_pred_tied"],
                n_hidden_layers=[1, 2, 3, 4, 5, 6, 7, 8],
            ),
            "fa_performance_block_training": dict(
                n_block_batches=[0, 100, 300, 500, 1000, 1500],
            ),
            "fa_performance_with_without_Y_learning": dict(
                # Y_lrs=[0.0, 0.001]
                Y_lrs = [0.0, 0.004]
            ),
            "fa_performance_branches": dict(
                n_branches=[1, 2, 5, 10, 15],
            )
        }

        return GROUP_PARAM_SETS[group]

    def get_wandb_run_name(self, group, **kwargs):
        if group == "Y_learning_no_teacher":
            run_name = "mnist_burstccn_Y_learning_rand_rand_no_teacher_error_scale0p5"
        elif group == "Y_learning_branches":
            n_branches = kwargs['n_branches']
            run_name = f"mnist_burstccn_Y_learning_rand_rand_with_teacher_nbranches{n_branches}"
        elif group == "Y_learning_forward_noise":
            noise = kwargs['noise']
            run_name = f"mnist_burstccn_Y_learning_rand_rand_with_teacher_noise{tag(noise)}"
        elif group == "Y_learning_error_scale":
            error_scale = kwargs['error_scale']
            run_name = f"mnist_burstccn_Y_learning_rand_rand_with_teacher_error_scale{tag(error_scale)}"
        elif group == "fa_performance":
            model_type = kwargs['model_type']
            n_hidden_layers = kwargs['n_hidden_layers']

            model_type_dict = {"ann": f"mnist_ann_fa_{n_hidden_layers}h",
                               # "burstccn": f"mnist_burstccn_fa_Y_noise_sym_Y_learning_{n_hidden_layers}h",
                               # "burstccn": f"mnist_burstccn_fa_Y_learning_{n_hidden_layers}h",
                               "burstccn_0q": f"mnist_burstccn_fa_{n_hidden_layers}h",
                               # "burstccn_two_phase": f"mnist_burstccn_fa_Y_noise_sym_Y_learning_Y_two_phases_{n_hidden_layers}h",
                               # "burstccn_block_train": f"mnist_burstccn_fa_Y_noise_sym_Y_learning_Y_block_training_{n_hidden_layers}h",

                               # "burstccn_Y_block_trained": f"mnist_burstccn_fa_block_trained_{n_hidden_layers}h",
                               # "burstccn_Y_learning_noise": f"mnist_burstccn_fa_Y_learning_{n_hidden_layers}h_noise",
                               # "burstccn_QY_tied": f"mnist_burstccn_fa_{n_hidden_layers}h",

                               "burstccn_Y_block_trained": f"mnist_burstccn_fa_Y_learning_hybrid_{n_hidden_layers}h",
                               "burstccn_Y_learning": f"mnist_burstccn_fa_Y_learning_{n_hidden_layers}h",
                               "burstccn_QY_tied": f"mnist_burstccn_fa_{n_hidden_layers}h",

                               "burstprop": f"mnist_burstprop_fa_{n_hidden_layers}h",
                               # "edn": f"mnist_edn_fa_fb_noise_sym_fb_learning_{n_hidden_layers}h",
                               "edn": f"mnist_edn_fa_fb_learning_{n_hidden_layers}h",
                               "edn_pred_tied": f"mnist_edn_fa_pred_tied_{n_hidden_layers}h"
                               }
            run_name = model_type_dict[model_type]

        elif group == 'tied_performance':
            model_type = kwargs['model_type']
            n_hidden_layers = kwargs['n_hidden_layers']

            model_type_dict = {"ann": f"mnist_ann_tied_{n_hidden_layers}h",
                               "burstccn_0q": f"mnist_burstccn_tied_{n_hidden_layers}h",
                               "burstprop": f"mnist_burstprop_tied_{n_hidden_layers}h",
                               "edn": f"mnist_edn_tied_{n_hidden_layers}h",
                               "edn_pred_tied": f"mnist_edn_tied_pred_tied_{n_hidden_layers}h",
                               }
            run_name = model_type_dict[model_type]
        elif group == 'fa_performance_block_training':
            n_block_batches = kwargs['n_block_batches']
            run_name = f"mnist_burstccn_fa_without_teacher_updates_blockB{n_block_batches}"
        elif group == 'fa_performance_with_without_Y_learning':
            Y_lr = kwargs['Y_lr']
            run_name = f"mnist_burstccn_fa_Y_noise_sym_Y_learning_lr{tag(Y_lr)}"
        elif group == 'fa_performance_branches':
            n_branches = kwargs['n_branches']
            run_name = f"mnist_burstccn_fa_Y_noise_sym_Y_learning_nbranches{n_branches}"
            # run_name = f"mnist_burstccn_fa_Y_learning_nbranches{n_branches}"
        else:
            raise ValueError(f"Invalid group: {group}")

        return run_name

    def get_wandb_group_name(self, group, **kwargs):
        group_dict = {
            'Y_learning_no_teacher': 'Y_learning_error_scale_tmp_no_teacher',
            # 'Y_learning_branches': 'Y_learning_apical_branches_tmp2',
            'Y_learning_branches': 'Y_learning_branches_feb1',
            # 'Y_learning_forward_noise': 'Y_learning_forward_noise_tmp2',
            'Y_learning_forward_noise': 'Y_learning_forward_noise_feb1',
            # 'Y_learning_error_scale': 'Y_learning_error_scale_tmp2',
            'Y_learning_error_scale': 'Y_learning_error_scale_feb1',
            # 'fa_performance': 'fa_runs',
            'fa_performance': 'mnist_fa_feb2',
            # 'tied_performance': '',
            # 'fa_performance_block_training': 'without_teacher_updates_tmp3',
            # 'fa_performance_block_training': 'without_teacher_updates_tmp4',
            'fa_performance_block_training': 'fa_Y_learning_hybrid_feb1',
            # 'fa_performance_with_without_Y_learning': 'fa_Y_learning_tmp5',
            'fa_performance_with_without_Y_learning': 'fa_Y_learning_feb1',
            # 'fa_performance_branches': 'fa_Y_learning_tmp3'
            'fa_performance_branches': 'fa_Y_learning_branches_tmp4'
        }

        return group_dict[group]

    def get_wandb_run_filter(self, group, **kwargs):
        wandb_run_name = self.get_wandb_run_name(group, **kwargs)
        wandb_group = self.get_wandb_group_name(group, **kwargs)

        return {"run_name": wandb_run_name,
                "group": wandb_group}


class MNISTApicalActivityResultsStore(ResultsStore):
    def __init__(self, h5_path, neuron_index=4, branch_index=0):
        super().__init__(cache_path='mnist_apical')
        self.h5_path = h5_path
        self.neuron_index = neuron_index
        self.branch_index = branch_index

        self._loaded = False
        self._tensors = {}

    # -------------------------
    # Internal: lazy load
    # -------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return

        keys = [
            # total
            "epoch_0_layer_2_Y_input_branch_no_teacher",
            "epoch_200_layer_2_Y_input_branch_no_teacher",
            "epoch_0_layer_2_Q_input_branch_no_teacher",
            "epoch_200_layer_2_Q_input_branch_no_teacher",
            # decomposed
            "epoch_0_layer_2_Y_exc_input_branch_no_teacher",
            "epoch_0_layer_2_Y_inh_input_branch_no_teacher",
            "epoch_0_layer_2_Q_exc_input_branch_no_teacher",
            "epoch_0_layer_2_Q_inh_input_branch_no_teacher",
            "epoch_200_layer_2_Y_exc_input_branch_no_teacher",
            "epoch_200_layer_2_Y_inh_input_branch_no_teacher",
            "epoch_200_layer_2_Q_exc_input_branch_no_teacher",
            "epoch_200_layer_2_Q_inh_input_branch_no_teacher",
        ]

        with h5py.File(self.h5_path, "r") as f:
            for k in keys:
                self._tensors[k] = f[k][()]

        self._loaded = True

    def _slice(self, tensor, n_examples):
        bi = self.branch_index
        ni = self.neuron_index
        n = min(n_examples, tensor.shape[0])
        return tensor[:n, bi, ni]

    # -------------------------
    # Public API
    # -------------------------

    def get_Q_Y_inputs(self, n_examples, when):
        """
        Returns (Q, Y) arrays of shape (n_examples,).
        `when` must be "before" or "after".
        """
        self._ensure_loaded()

        if when == "before":
            Y_key = "epoch_0_layer_2_Y_input_branch_no_teacher"
            Q_key = "epoch_0_layer_2_Q_input_branch_no_teacher"
        elif when == "after":
            Y_key = "epoch_200_layer_2_Y_input_branch_no_teacher"
            Q_key = "epoch_200_layer_2_Q_input_branch_no_teacher"
        else:
            raise ValueError(f"when must be 'before' or 'after', got {when!r}")

        Y = self._slice(self._tensors[Y_key], n_examples)
        Q = self._slice(self._tensors[Q_key], n_examples)
        return Q, Y

    def get_exc_inh_inputs(self, n_examples, when):
        """
        Returns (exc, inh, total) arrays of shape (n_examples,).
        """
        self._ensure_loaded()

        if when == "before":
            Y_exc_key = "epoch_0_layer_2_Y_exc_input_branch_no_teacher"
            Y_inh_key = "epoch_0_layer_2_Y_inh_input_branch_no_teacher"
            Q_exc_key = "epoch_0_layer_2_Q_exc_input_branch_no_teacher"
            Q_inh_key = "epoch_0_layer_2_Q_inh_input_branch_no_teacher"
        elif when == "after":
            Y_exc_key = "epoch_200_layer_2_Y_exc_input_branch_no_teacher"
            Y_inh_key = "epoch_200_layer_2_Y_inh_input_branch_no_teacher"
            Q_exc_key = "epoch_200_layer_2_Q_exc_input_branch_no_teacher"
            Q_inh_key = "epoch_200_layer_2_Q_inh_input_branch_no_teacher"
        else:
            raise ValueError(f"when must be 'before' or 'after', got {when!r}")

        Y_exc = self._slice(self._tensors[Y_exc_key], n_examples)
        Y_inh = self._slice(self._tensors[Y_inh_key], n_examples)
        Q_exc = self._slice(self._tensors[Q_exc_key], n_examples)
        Q_inh = self._slice(self._tensors[Q_inh_key], n_examples)

        exc = Y_exc + Q_exc
        inh = Y_inh + Q_inh
        total = exc + inh
        return exc, inh, total



