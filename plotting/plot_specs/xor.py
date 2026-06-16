from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotElemDetails, PlotAxStore, PlotElemStore
from plotting.plot_specs.plot_specs_base import PlotColours  # if you still want these constants


class XORAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("event_rate", PlotAxDetails(
            y_label="Event\nrate (Hz)",
            x_ticks=tuple(i * 8 for i in range(5)),
            x_tick_labels=(),
            # included_elems=("output_event_rate",),
        ))

        self.add("burst_probability", PlotAxDetails(
            y_label="Burst\nprobability (%)",
            x_ticks=tuple(i * 8 for i in range(5)),
            x_tick_labels=(),
            # included_elems=("output_burst_probability", "baseline_burst_probability"),
        ))

        self.add("weight_changes", PlotAxDetails(
            y_label=r"$\Delta \mathbf{W}$ [$\times10^{3}$]",
            x_ticks=tuple(i * 8 for i in range(5)),
            x_tick_labels=(),
            x_tick_between_labels=[(0, 0), (0, 1), (1, 0), (1, 1)],
            x_tick_between_labels_fontweight="bold",
            y_lims=(-5.5, 5.5),
            # included_elems=("hidden1_delta_W", "hidden2_delta_W"),
        ))


class XORElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        # Output event rate (was styled by "event_rate" in the old dict)
        self.add("event_rate", PlotElemDetails(
            display_name=r"$\mathbf{e}$",
            line_colour=PlotColours.EVENT,
        ))

        # Output burst probability (was styled by "burst_probability" in the old dict)
        self.add("burst_probability", PlotElemDetails(
            display_name=r"$\mathbf{p}$",
            line_colour=PlotColours.BURST_PROB,
        ))

        # Optional moving-average burst probability (not included by default in axes, but kept)
        self.add("ma_burst_probability", PlotElemDetails(
            display_name=r"$\overline{\mathbf{p}}$",
            # line_colour="#f76825",
            line_colour=PlotColours.BURST_PROB_BASELINE,
            line_style="--",
        ))

        self.add("baseline_burst_probability", PlotElemDetails(
            display_name=r"$\mathbf{p}^b$",
            # line_colour="#f76825",
            line_colour=PlotColours.BURST_PROB_BASELINE,
            line_style="--",
        ))

        self.add("hidden1_delta_W", PlotElemDetails(
            display_name=r"$\mathbf{W}_{hid1}$",
            # line_colour="purple",
            line_colour=PlotColours.WEIGHT_CHANGE
        ))

        self.add("hidden2_delta_W", PlotElemDetails(
            display_name=r"$\mathbf{W}_{hid2}$",
            # line_colour="darkgreen",
            line_colour=PlotColours.WEIGHT_CHANGE2
        ))
