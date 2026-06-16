from dataclasses import replace

from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotElemDetails, PlotAxStore, PlotElemStore, PlotColours


class ContinuousBurstCCNAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()
        self._init_base()
        self._active_condition = None

    def _init_base(self):
        self._axes = {}

        self.add("loss", PlotAxDetails(
            x_label="Time (s)",
            y_label="Loss (MSE)",
            y_lims=(1e-4, 1e-1),
            included_elems=("loss",),
        ))

        self.add("outputs_targets", PlotAxDetails(
            x_label="Time (s)",
            y_label="Event rate (Hz)",
            included_elems=("before_event_rate", "after_event_rate", "target_event_rate"),
        ))

        self.add("burst_probability", PlotAxDetails(
            x_label="Time (s)",
            y_label="Burst\nprobability (%)",
            included_elems=("before_burst_prob", "after_burst_prob"),
        ))

    def _apply_condition_overrides(self, condition: str):
        # Optional: avoid re-applying
        if condition == self._active_condition:
            return

        self._init_base()
        self._active_condition = condition

        if condition == "sin_wave":
            self.set("outputs_targets", replace(self.get("outputs_targets"), x_ticks=(0, 5, 10), y_lims=(0.0, 1.0)))
            # self.set("burst_probability", replace(self.get("burst_probability"), x_ticks=(0, 5, 10), y_lims=(0.4, 0.65)))
            self.set("burst_probability", replace(self.get("burst_probability"), x_ticks=(0, 5, 10), y_lims=(40, 65)))
        elif condition == "catcam":
            self.set("outputs_targets", replace(self.get("outputs_targets"), x_ticks=(0, 10, 20, 30), legend_location="lower right"))
            self.set("burst_probability", replace(self.get("burst_probability"), x_ticks=(0, 10, 20, 30), legend_location="lower right"))
        else:
            raise ValueError(f"Invalid condition")


class ContinuousBurstCCNElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("loss", PlotElemDetails(
            display_name="Loss",
            line_colour="black",
            # sigma=1.0,
        ))

        self.add("before_event_rate", PlotElemDetails(
            display_name="before",
            line_colour=PlotColours.EVENT,
            line_style="dashed",
        ))

        self.add("after_event_rate", PlotElemDetails(
            display_name="after",
            line_colour=PlotColours.EVENT,
        ))

        self.add("target_event_rate", PlotElemDetails(
            display_name="target",
            line_colour="black",
        ))

        self.add("before_burst_prob", PlotElemDetails(
            display_name="before",
            line_colour=PlotColours.BURST_PROB,
            line_style="dashed",
        ))

        self.add("after_burst_prob", PlotElemDetails(
            display_name="after",
            line_colour=PlotColours.BURST_PROB,
        ))
