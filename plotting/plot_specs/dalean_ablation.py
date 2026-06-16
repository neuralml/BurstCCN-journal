from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotAxStore, PlotElemDetails, PlotElemStore


class DaleanAblationAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("ablation_data", PlotAxDetails(
            # y_label=r"pairwise $\dfrac{RPS_{post} - RPS_{pre}}{RPS_{pre}}$",
            # y_lims=(-0.55, 1.25),
            # y_label=r"pairwise $\dfrac{RPS_{post} - RPS_{pre}}{RPS_{pre}}$ (%)",
            # y_label=r"Relative change (\%)",#: $\dfrac{RPS_{post} - RPS_{pre}}{RPS_{pre}}$",
            y_label="Relative change in\nsynaptic strength (%)",
            y_lims=(-75, 175),

        ))

        self.add("ablation_model", PlotAxDetails(
            # y_label=r"Weight change ($\Delta \mathbf{W}$)",
            # y_label=r"Relative weight change (%)", #($\Delta \mathbf{W} / \mathbf{W}_{\text{init}}$)",
            # y_lims=(-0.55, 1.75),
            y_label="Relative change in\nsynaptic weight (%)",
        ))

        self.add("ablation_model_event_rate_change", PlotAxDetails(
            y_label="Change in\nevent rate",
        ))

        self.add("ablation_model_burst_probability_change", PlotAxDetails(
            y_label=r"Initial $\mathbf{p} - \mathbf{p}^b$",
        ))


class DaleanAblationElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("none", PlotElemDetails(
            display_name="control",
            line_colour="#999999",
        ))
        self.add("PV", PlotElemDetails(
            display_name="PV",
            line_colour="#85caa0",
        ))
        self.add("VIP", PlotElemDetails(
            display_name="VIP",
            line_colour="#f58d73",
        ))
        self.add("SST", PlotElemDetails(
            display_name="SST",
            line_colour="#7d9ed2",
        ))
        self.add("NDNF", PlotElemDetails(
            display_name="NDNF",
            line_colour="#d6a8cd",
        ))

        self.add("PV-like", PlotElemDetails(
            display_name="PV-like",
            line_colour="#85caa0",
        ))
        self.add("VIP-like", PlotElemDetails(
            display_name="VIP-like",
            line_colour="#f58d73",
        ))
        self.add("SST-like", PlotElemDetails(
            display_name="SST-like",
            line_colour="#7d9ed2",
        ))
        self.add("NDNF-like", PlotElemDetails(
            display_name="NDNF-like",
            line_colour="#d6a8cd",
        ))
