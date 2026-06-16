from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotElemDetails, PlotAxStore, PlotElemStore, PlotColours


class PlasticityRuleAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("delta_W", PlotAxDetails(
            y_label=r"$\Delta \mathbf{W}$",
            y_lims=(-0.15, 0.45),
            included_elems=("delta_W",),
        ))

        self.add("burst_probability", PlotAxDetails(
            y_label="Burst\nprobability (%)",
            y_lims=(0.0, 100.0),
            included_elems=("burst_probability",),
        ))

        self.add("event_burst_rates", PlotAxDetails(
            x_label="Pre/post firing rate (Hz)",
            y_label="Rate (Hz)",
            y_lims=(0.0, None),
            x_ticks=(0, 20, 40, 60, 80),
            included_elems=("event_rate", "burst_rate"),
        ))


class PlasticityRuleElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("event_rate", PlotElemDetails(
            # display_name=r"$\mathbf{e}$",
            display_name="event",
            # line_colour="#0975b3",
            line_colour=PlotColours.EVENT,
        ))

        self.add("burst_rate", PlotElemDetails(
            # display_name=r"$\mathbf{b}$",
            display_name="burst",
            line_colour=PlotColours.BURST,
        ))

        self.add("burst_probability", PlotElemDetails(
            # display_name=r"$\mathbf{p}$",
            # line_colour="#9c2321",
            line_colour=PlotColours.BURST_PROB,
        ))

        self.add("baseline_burst_probability", PlotElemDetails(
            # display_name=r"$\mathbf{p}^b$",
            display_name=r"baseline ($p^b$)",
            # line_colour="#9c2321",
            line_colour=PlotColours.BURST_PROB_BASELINE,
            line_style="--",
        ))

        self.add("delta_W", PlotElemDetails(
            display_name=r"$\Delta \mathbf{W}$",
            # line_colour="gray",
            # line_colour=PlotColours.WEIGHT_CHANGE,
            line_colour='black',
        ))
