from plotting.plot_specs.plot_specs_base import PlotElemDetails, PlotColours, PlotAxDetails, PlotAxStore, PlotElemStore


class StationaryTargetSpikingAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("event_burst_rates", PlotAxDetails(
            y_label="Rate (Hz)",
            y_lims=(0.0, 12.0),
            y_ticks=(0.0, 4.0, 8.0, 12.0),
        ))

        self.add("output_burst_probability", PlotAxDetails(
            y_label="Burst prob. (%)",
            y_lims=(0.0, 80.0),
            y_ticks=(0.0, 25.0, 50.0, 75.0),
        ))

        self.add("output_spike_trains", PlotAxDetails(
            remove_spines=True
        ))

        self.add("input_currents", PlotAxDetails(
            y_label="Input\ncurrent (nA)",
            y_lims=(-0.85, 0.85),
            y_ticks=(-0.75, 0.0, 0.75),
        ))

        self.add("dendritic_potentials", PlotAxDetails(
            y_label="Dendritic\npotential (mV)",
            y_lims=(-86.4705882353, -30.0),
            y_ticks=(-80.0, -70.0, -50.0, -30.0),
        ))

        self.add("input_weight", PlotAxDetails(
            x_label="Time (s)",
            y_label="Input weight",
            y_lims=(0.0, 1.5),
            x_lims=(10000, 90002),
            x_tick_frequency=10000,
        ))


class StationaryTargetSpikingElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("output_event_rates", PlotElemDetails(display_name="event", line_colour=PlotColours.EVENT))
        self.add("output_burst_rates", PlotElemDetails(display_name="burst", line_colour=PlotColours.BURST))
        self.add("target_rates", PlotElemDetails(display_name="target", line_colour="black", line_style="--"))
        self.add("output_burst_prob", PlotElemDetails(display_name=r"$p$", line_colour=PlotColours.BURST_PROB))
        # self.add("output_burst_prob_baseline", PlotElemDetails(display_name=r"$\mathbf{p}_b$", line_colour=PlotColours.BURST_PROB, line_style="--"))
        self.add("output_burst_prob_baseline", PlotElemDetails(display_name=r"baseline ($p^b$)", line_colour=PlotColours.BURST_PROB_BASELINE, line_style="--"))
        self.add("input_events", PlotElemDetails(display_name="events", line_colour=PlotColours.EVENT))
        self.add("input_bursts", PlotElemDetails(display_name="bursts", line_colour=PlotColours.BURST))
        self.add("Y_input_current", PlotElemDetails(display_name=r"$\mathrm{Y}_{\mathrm{input}}$", line_colour=PlotColours.BURST))
        self.add("Q_input_current", PlotElemDetails(display_name=r"$\mathrm{Q}_{\mathrm{input}}$", line_colour=PlotColours.EVENT))
        self.add("zero_input_current_line", PlotElemDetails(line_colour="black", line_style="--"))
        self.add("dendritic_potentials", PlotElemDetails(line_colour="black"))
        self.add("dendritic_reversal_potential", PlotElemDetails(display_name="reversal potential", line_colour="gray", line_style="--"))
        self.add("input_weight", PlotElemDetails(display_name="input weight", line_colour="black"))
