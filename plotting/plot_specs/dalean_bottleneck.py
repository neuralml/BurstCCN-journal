from plotting.plot_specs.plot_specs_base import (
    PlotAxDetails,
    PlotAxStore,
    PlotElemDetails,
    PlotElemStore,
    PlotColours,
)


class DaleanBottleneckAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("equal_bottleneck", PlotAxDetails(
            x_label="# SST",
            y_label="Test error (%)",
            x_lims=(0.0, 22.0),
        ))

        self.add("reduced_bottleneck", PlotAxDetails(
            x_label="Reduced SST area",
            y_label="Test error (%)",
            y_lims=(13.5, 15.0),
        ))


class DaleanBottleneckElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("test_error_curve", PlotElemDetails(
            line_colour=PlotColours.DEFAULT,
            line_style="-",
            marker_style="o",
        ))

        self.add("sst50_ref", PlotElemDetails(
            display_name="#SST = 50",
            line_colour="gray",
            line_style="--",
        ))
