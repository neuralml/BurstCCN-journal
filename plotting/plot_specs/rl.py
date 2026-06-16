from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotAxStore, PlotColours, PlotElemDetails, PlotElemStore


class RLAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("avg_test_score", PlotAxDetails(
            x_label=r"Episode ($\times 10^3$)",
            y_label="Test score",
            x_lims=(0.0, None),
            legend_location="lower right",
        ))

        self.add("decoding_rl_variables", PlotAxDetails(
            y_label=r"$R^2$ score",
            legend_location="upper right",
        ))

        self.add("decoding_task_variables", PlotAxDetails(
            y_label="Decoding accuracy",
        ))


class RLElemDetailsStore(PlotElemStore):
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

        self.add("fa", PlotElemDetails(
            display_name="random fixed",
            line_style=(0, (1, 1)),
        ))

        self.add("kp", PlotElemDetails(
            display_name="random learned",
            line_style=(5, (10, 3)),
        ))

        self.add("tied", PlotElemDetails(
            display_name="symmetric learned",
            line_style="-",
        ))

        self.add("somatic_potentials", PlotElemDetails(
            display_name="Somatic\npotentials",
            line_colour="#009e9d",
        ))

        self.add("event_rates", PlotElemDetails(
            display_name="Event\nrates",
            line_colour=PlotColours.EVENT,
        ))

        self.add("burst_rates", PlotElemDetails(
            display_name="Burst\nrates",
            line_colour=PlotColours.BURST,
        ))

        self.add("apical_potentials", PlotElemDetails(
            display_name="Apical\npotentials",
            line_colour="#7b3f99",
        ))

        self.add("burst_probabilities", PlotElemDetails(
            display_name="Burst\nprobabilities",
            line_colour=PlotColours.BURST_PROB,
        ))

        self.add("delta_b", PlotElemDetails(
            display_name="Burst rate\nchange",
            line_colour="#DE85B0",
        ))

        self.add("Q_value_prediction_errors", PlotElemDetails(
            # display_name="Q-value\npred. errors",
            display_name="action-value\npred. errors",
        ))

        self.add("action", PlotElemDetails(
            display_name="action",
        ))

        self.add("agent_location", PlotElemDetails(
            display_name="agent\nlocation",
        ))

        self.add("relative_hole_locations", PlotElemDetails(
            display_name="ice hole\nlocations",
        ))

        self.add("is_safe_action", PlotElemDetails(
            display_name="safe\naction",
        ))

        self.add("distance_to_goal", PlotElemDetails(
            display_name="goal\ndistance",
        ))

        self.add("action_Q_value", PlotElemDetails(
            # display_name="action\nQ-value",
            display_name="action\nvalue",
        ))
