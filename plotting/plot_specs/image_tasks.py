from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotElemDetails, PlotAxStore, PlotElemStore, PlotColours, \
    PlotLabels


class ImageTaskAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("top1_test_performance", PlotAxDetails(
            x_label="Epoch",
            y_label="Top-1 Test error (%)",
        ))

        self.add("top5_test_performance", PlotAxDetails(
            x_label="Epoch",
            y_label="Top-5 Test error (%)",
        ))

        self.add("top5_test_performance_imagenet", PlotAxDetails(
            x_label="Epoch",
            y_label="Top-5 Test error (%)",
            x_ticks=(0, 10, 20, 30, 40, 50),
        ))

        self.add("BP_align", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.BP_ALIGNMENT,
        ))

        self.add("BP_align_imagenet", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.BP_ALIGNMENT,
            x_ticks=(0, 10, 20, 30, 40, 50),
        ))

        self.add("WY_align", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.WY_ALIGNMENT,
        ))

        self.add("WY_align_imagenet", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.WY_ALIGNMENT,
            x_ticks=(0, 10, 20, 30, 40, 50),
        ))


class ImageTaskElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("ann", PlotElemDetails(
            display_name="ann",
            line_colour=PlotColours.ANN,
        ))

        self.add("burstccn", PlotElemDetails(
            display_name="burstccn",
            line_colour=PlotColours.BURSTCCN,
        ))

        self.add("burstccn_QY_tied", PlotElemDetails(
            display_name=r"burstccn ($\mathbf{Q}$$\mathbf{Y}$-sym)",
            line_colour=PlotColours.BURSTCCN_0Q,
        ))

        self.add("burstprop", PlotElemDetails(
            display_name="burstprop",
            line_colour=PlotColours.BURSTPROP,
        ))

        self.add("fa", PlotElemDetails(
            display_name="random fixed",
            # line_style=(0, (4, 1.5, 1.2, 1.5)),
            # line_style="densely dotted",
            line_style=(0, (1, 1)),
        ))

        self.add("kp", PlotElemDetails(
            # display_name="random learned",
            display_name="random plastic",
            # line_style="long dash with offset",
            line_style=(5, (10, 3)),
            # line_style="--",
        ))

        self.add("tied", PlotElemDetails(
            display_name="symmetric plastic",
            line_style="-",
        ))
