from pathlib import Path

from matplotlib import pyplot as plt

from plotting.utils import init_global_matplotlib_constants


def plot_single(plotter, plot_fn, figsize=(4, 3), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    plot_fn(plotter, ax, **kwargs)
    fig.tight_layout()
    return fig, ax


def run_plots(
    plotter_cls,
    registry,
    plot_names=None,
    *,
    init=True,
    show=True,
    default_figsize=(4, 3),
    save_dir=None,
    save_ext="pdf",
    savefig_kwargs=None,
):
    """
    names:
        None -> run all plots in registry
        str  -> run one plot
        list/tuple of str -> run subset
    returns:
        dict: {name: (fig, ax)}

    optional saving:
        save_dir: directory where each plot is saved as <plot_name>.<save_ext>
    """
    if registry is None:
        registry = registry

    if init:
        init_global_matplotlib_constants()

    # Normalize names
    if plot_names is None:
        names_to_run = list(registry.keys())
    elif isinstance(plot_names, str):
        names_to_run = [plot_names]
    else:
        names_to_run = list(plot_names)

    out = {}
    plotter = plotter_cls()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for name in names_to_run:
        spec = registry[name]
        fn = spec["fn"]
        figsize = spec.get("figsize", default_figsize)
        plot_kwargs = spec.get("kwargs", {})
        fig, ax = plot_single(plotter, fn, figsize=figsize, **plot_kwargs)
        out[name] = (fig, ax)

        if save_dir is not None:
            fig.savefig(save_dir / f"{name}.{save_ext}", **(savefig_kwargs or {}))

    if show:
        plt.show()

    return out
