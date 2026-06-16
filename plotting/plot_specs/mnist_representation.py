from plotting.plot_specs.plot_specs_base import PlotAxStore, PlotElemStore, PlotAxDetails, PlotElemDetails, PlotColours, \
    PlotLabels


class MNISTRepresentationAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("MDS", PlotAxDetails(
            x_label="MDS Dimension 1",
            y_label="MDS Dimension 2",
            x_ticks=(),
            y_ticks=(),
        ))

        self.add("TSNE", PlotAxDetails(
            x_label="TSNE Dimension 1",
            y_label="TSNE Dimension 2",
            x_ticks=(),
            y_ticks=(),
        ))


class MNISTRepresentationElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("np", PlotElemDetails(
            display_name="np",
            line_colour=PlotColours.NODE_PERTURBATION,
            marker_style='o'
        ))

        self.add("ann", PlotElemDetails(
            display_name="ann-fa",
            line_colour=PlotColours.ANN,
            marker_style='o',
            marker_face_colour="none",
            # line_style='--',
            line_style=(0, (5, 5)),
            zorder=50,
            marker_zorder=51,
        ))

        self.add("burstccn", PlotElemDetails(
            display_name=PlotLabels.BURSTCCN_ONLINE,
            line_colour=PlotColours.BURSTCCN,
            marker_style='o'
        ))

        self.add("burstccn_QY_tied", PlotElemDetails(
            # display_name=r"burstccn ($\mathbf{Q}$$\mathbf{Y}$-sym)",
            display_name=PlotLabels.BURSTCCN_QY_TIED,
            line_colour=PlotColours.BURSTCCN_0Q,
            marker_style='o',
            zorder=49,
            marker_zorder=50.5,
        ))

        self.add("burstccn_Y_block_trained", PlotElemDetails(
            display_name=PlotLabels.BURSTCCN_HYBRID,
            line_colour=PlotColours.BURSTCCN_HYBRID,
            marker_style='o'
        ))

        self.add("burstprop", PlotElemDetails(
            display_name="burstprop",
            line_colour=PlotColours.BURSTPROP,
            marker_style='o'
        ))

        self.add("edn", PlotElemDetails(
            display_name="edn",
            line_colour=PlotColours.EDN,
            marker_style='o'
        ))
