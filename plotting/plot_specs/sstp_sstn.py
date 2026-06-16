from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotAxStore, PlotElemDetails, PlotElemStore


class SSTpSSTnAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("data_cue_bars", PlotAxDetails(
            # y_label="fluo (z-sc)",
            # y_label="SST cue resp.\n(z-score)",
            y_label="SST cue\nresponse (z-score)",
            y_lims=(0.0, 32.0),
            y_ticks=(0.0, 10.0, 20.0, 30.0),
        ))

        self.add("data_reward_delta_bars", PlotAxDetails(
            # y_label="Δ fluo (z-sc)",
            # y_label="SST reward resp.\n(z-score)",
            y_label="SST reward\nresponse (z-score)",
        ))

        self.add("model_cue_bars", PlotAxDetails(
            # y_label="SST activity",
            # y_label="SST cue resp.",
            y_label="SST cue\nresponse",
            y_lims=(0.4, None),
        ))

        self.add("model_error_delta_bars", PlotAxDetails(
            # y_label="Δ SST activity",
            # y_label="SST error resp",
            y_label="SST error\nresponse",
        ))


class SSTpSSTnElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("sstp", PlotElemDetails(
            display_name="SSTp",
            line_colour="#274294",
        ))

        self.add("sstn", PlotElemDetails(
            display_name="SSTn",
            line_colour="#54A8DE",
        ))
