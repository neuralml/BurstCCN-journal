from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotAxStore, PlotElemStore, PlotElemDetails


class VectorisedInstructiveAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        common_y_ticks = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

        self.add("data", PlotAxDetails(
            x_label="SD residual (z-score)",
            y_label="Fraction of neurons",
            x_lims=(-1.65, 1.65),
            x_ticks=(-1.5, 0.0, 1.5),
            y_ticks=common_y_ticks,
            y_lims=(-0.01, 1.02),
        ))

        self.add("model", PlotAxDetails(
            x_label="Burst probability",
            y_label="Fraction of neurons",
            x_lims=(0.15, 0.85),
            y_ticks=common_y_ticks,
            y_lims=(-0.01, 1.02),
        ))

        self.add("model_zscore", PlotAxDetails(
            x_label="Burst probability (z-score)",
            y_label="Fraction of neurons",
            x_lims=(-5.5, 5.5),
            x_ticks=(-5.0, 0.0, 5.0),
            y_ticks=common_y_ticks,
            y_lims=(-0.01, 1.02),
        ))

        self.add("data_bar", PlotAxDetails(
            y_label="SD residual (z-score)",
            x_lims=(0.5, 2.5),
            y_lims=(-0.2, 0.1),
            y_ticks=(-0.2, -0.1, 0.0, 0.1),
        ))

        self.add("model_bar", PlotAxDetails(
            y_label=r"$\Delta$ Burst probability",
            x_lims=(0.5, 2.5),
            y_lims=(-0.05, 0.05),
        ))


class VectorisedInstructiveElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("p_plus", PlotElemDetails(
            display_name="P+",
            line_colour="red",
        ))

        self.add("p_minus", PlotElemDetails(
            display_name="P-",
            line_colour="blue",
        ))
