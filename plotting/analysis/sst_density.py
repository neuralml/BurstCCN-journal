import numpy as np
import pandas as pd

from pathlib import Path
from plotting.analysis.results_store_base import file_cache_decorator, WandbResultsStore, BurstCCNWandbResultsStore, \
    ResultsStore


class SSTDensityDataResultsStore(ResultsStore):
    def __init__(self):
        super().__init__(cache_path='sst_density_data')
        self.base_path = Path(__file__).resolve().parent / "sst_density_data"

    @file_cache_decorator()
    def get_sst_density_data(self, sorted=True):
        excel_file = self.base_path / 'mmc3.xlsx'
        data = pd.read_excel(excel_file)
        labels = ['AIp', 'ILA', 'TEa', 'PL', 'ECT', 'ORBm', 'PERI', 'AIv', 'GU', 'RSPagl', 'VISC', 'AUDv', 'VISpm', 'AId', 'VISam', 'ORBvl', 'RSPv', 'ACAv', 'PTLp', 'VISpl', 'ACAd', 'VISl', 'MOs', 'VISp', 'AUDp', 'AUDd', 'RSPd', 'MOp', 'SSp-ll', 'VISal', 'SSp-tr', 'SSs', 'SSp-ul', 'ORBl', 'SSp-bfd', 'SSp-m', 'SSp-n']

        df_filtered = data[data['Unnamed: 0'].isin(labels)].copy()
        df_filtered['Unnamed: 0'] = pd.Categorical(df_filtered['Unnamed: 0'], categories=labels, ordered=True)
        df_filtered.sort_values(by='Unnamed: 0', inplace=True)

        mean_between_columns = df_filtered[['SST (N = 5), male', 'SST (N = 5), female']].mean(axis=1)
        errors = df_filtered[['Unnamed: 8', 'Unnamed: 10']].mean(axis=1)

        if sorted:
            sorted_indices = np.argsort(mean_between_columns)
            labels = np.array(labels)[sorted_indices]
            mean_between_columns = np.array(mean_between_columns)[sorted_indices]
            errors = np.array(errors)[sorted_indices]

        return labels, mean_between_columns, errors

class SSTDensityTrainingResultsStore(BurstCCNWandbResultsStore):
    def __init__(self):
        super().__init__(cache_path='sst_density')

    def get_group_params(self, group):
        GROUP_PARAM_SETS = {
            "sst_bottleneck": dict(
                modes=['equal', 'decreasing', 'increasing'],
            ),
        }

        return GROUP_PARAM_SETS[group]

    def get_wandb_run_name(self, group, **kwargs):
        if group == "sst_bottleneck":
            mode = kwargs['mode']
            run_name = f"mnist_burstccn_hrch_{mode}"
        else:
            raise ValueError(f"Invalid group: {group}")

        return run_name

    def get_wandb_group_name(self, group, **kwargs):
        group_dict = {
            'sst_bottleneck': 'sst_bottleneck',
        }

        return group_dict[group]

    def get_wandb_run_filter(self, group, **kwargs):
        wandb_run_name = self.get_wandb_run_name(group, **kwargs)
        wandb_group = self.get_wandb_group_name(group, **kwargs)

        return {"run_name": wandb_run_name,
                "group": wandb_group}


if __name__ == "__main__":
    res = SSTDensityDataResultsStore()
    labels, means, errors = res.get_sst_density_data(sorted=True)
    print("Loaded SST density data:", len(labels), "regions")





