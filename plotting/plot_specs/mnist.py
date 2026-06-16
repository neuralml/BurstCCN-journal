from plotting.plot_specs.plot_specs_base import PlotAxDetails, PlotElemDetails, PlotAxStore, PlotElemStore, PlotColours, \
    PlotLabels


class MNISTApicalActivityAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("QY_scatter", PlotAxDetails(
            x_label=r"$\text{Q}_{\mathrm{input}}$",
            y_label=r"$\text{Y}_{\mathrm{input}}$",
        ))

        self.add("exc_inh_inputs", PlotAxDetails(
            x_label="Example",
            y_label="Input",
        ))



class MNISTApicalActivityElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("QY_scatter", PlotElemDetails(
            marker_colour='grey',
            alpha=0.9,
        ))

        self.add("QY_scatter_equal_line", PlotElemDetails(
            display_name=r"$\text{Y}_{\mathrm{input}} = -\text{Q}_{\mathrm{input}}$",
            line_colour='black',
            line_style="--"
        ))

        self.add("exc_input", PlotElemDetails(
            display_name='Excitatory',
            marker_style='o',
            line_colour='red'
        ))

        self.add("inh_input", PlotElemDetails(
            display_name='Inhibitory',
            marker_style='o',
            line_colour='blue'
        ))

        self.add("total_input", PlotElemDetails(
            display_name='Total',
            marker_style='o',
            line_colour='grey'
        ))


class MNISTAxDetailsStore(PlotAxStore):
    def __init__(self):
        super().__init__()

        self.add("QY_across_branches", PlotAxDetails(
            x_label="Iterations",
            y_label=PlotLabels.QY_ALIGNMENT,
            y_scale="log",
            y_lims=[1e-1, 100]
        ))

        self.add("QY_across_noise", PlotAxDetails(
            x_label="Iterations",
            y_label=PlotLabels.QY_ALIGNMENT,
            y_scale="log",
            y_lims=[1e-1, 100]
        ))

        self.add("QY_across_error_scale", PlotAxDetails(
            x_label="Iterations",
            y_label=PlotLabels.QY_ALIGNMENT,
            y_scale="log",
            y_lims=[1e-2, 100]
        ))




        self.add("Y_learning_apical_magnitude", PlotAxDetails(
            x_label="Iterations",
            y_label=PlotLabels.APICAL_MAGNITUDE,
            y_scale="log",
            y_lims=[1e-6, 1],
            x_ticks=[0, 1e5, 2e5, 3e5, 4e5]
        ))


        self.add("Y_learning_burst_prob_magnitude", PlotAxDetails(
            x_label="Iterations",
            y_label=PlotLabels.BURST_PROB_CHANGE_MAGNITUDE,
            y_scale="log",
            y_lims=[1e-6, 1],
            x_ticks=[0, 1e5, 2e5, 3e5, 4e5]
        ))

        self.add("Y_learning_angle_fa", PlotAxDetails(
            x_label="Iterations",
            y_label=PlotLabels.FA_ALIGNMENT,
            y_scale="log",
            y_lims=[1e-2, 100],
            x_ticks=[0, 1e5, 2e5, 3e5, 4e5]
        ))



        self.add("with_without_Y_learning_test_performance", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.TEST_ERROR,
        ))

        self.add("with_without_Y_learning_QY_align", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.QY_ALIGNMENT,
        ))

        self.add("with_without_Y_learning_BP_align", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.BP_ALIGNMENT,
        ))

        self.add("with_without_Y_learning_FA_align", PlotAxDetails(
            x_label="Epoch",
            y_label=PlotLabels.FA_ALIGNMENT,
        ))


        self.add("across_apical_branches_test_error", PlotAxDetails(
            x_label="# Apical branches",
            y_label="Test error (%)",
            # x_ticks=(1, 2, 5, 10, 15)
        ))

        self.add("across_apical_branches_QY_align", PlotAxDetails(
            x_label="# Apical branches",
            y_label=PlotLabels.QY_ALIGNMENT,
            # x_ticks=(1, 2, 5, 10, 15)
        ))

        self.add("across_apical_branches_FA_align", PlotAxDetails(
            x_label="# Apical branches",
            y_label=PlotLabels.FA_ALIGNMENT,
            # x_ticks=(1, 2, 5, 10, 15)
        ))

        self.add("across_apical_branches_BP_align", PlotAxDetails(
            x_label="# Apical branches",
            y_label=PlotLabels.BP_ALIGNMENT,
            # x_ticks=(1, 2, 5, 10, 15)
        ))



        self.add("without_teacher_blocks_test_error", PlotAxDetails(
            # x_label="# without-teacher updates",
            x_label="Realignment / online updates",
            y_label="Test error (%)",
            # x_ticks=(0, 100, 300, 500, 1000, 1500)
        ))

        self.add("without_teacher_blocks_QY_align", PlotAxDetails(
            # x_label="# without-teacher updates",
            x_label="Realignment / online updates",
            y_label=PlotLabels.QY_ALIGNMENT,
            # x_ticks=(0, 100, 300, 500, 1000, 1500)
        ))

        self.add("without_teacher_blocks_BP_align", PlotAxDetails(
            # x_label="# without-teacher updates",
            x_label="Realignment / online updates",
            y_label=PlotLabels.BP_ALIGNMENT,
            # x_ticks=(0, 100, 300, 500, 1000, 1500)
        ))

        self.add("without_teacher_blocks_FA_align", PlotAxDetails(
            # x_label="# without-teacher updates",
            x_label="Realignment / online updates",
            y_label=PlotLabels.FA_ALIGNMENT,
            # x_ticks=(0, 100, 300, 500, 1000, 1500)
        ))


        self.add("network_depth_test_errors", PlotAxDetails(
            x_label="# Hidden areas",
            y_label="Test error (%)",
            # y_scale="log",
            # y_ticks=[2, 3, 4, 6, 10],
            # y_tick_labels=[2, 3, 4, 6, 10]
            y_ticks=[1.5, 2, 2.5, 3.0, 3.5],
            y_tick_labels=[1.5, 2, 2.5, 3.0, 3.5]
        ))

        self.add("network_depth_BP_align", PlotAxDetails(
            x_label="# Hidden areas",
            y_label=PlotLabels.BP_ALIGNMENT,
        ))

        self.add("network_depth_FA_align", PlotAxDetails(
            x_label="# Hidden areas",
            y_label=PlotLabels.FA_ALIGNMENT,
        ))


class MNISTElemDetailsStore(PlotElemStore):
    def __init__(self):
        super().__init__()

        self.add("Y_only_learning_metric", PlotElemDetails(
            line_colour='black',
            smoothing_alpha=0.25
        ))


        self.add("Y_learning_on", PlotElemDetails(
            # line_colour='#05792e',
            line_colour=PlotColours.BURSTCCN,
            display_name='on'
        ))

        self.add("Y_learning_off", PlotElemDetails(
            line_colour='black',
            display_name='off'
        ))

        self.add("across_apical_branches_metric", PlotElemDetails(
            line_colour=PlotColours.BURSTCCN,
            marker_style="o"
        ))

        self.add("without_teacher_blocks_metric", PlotElemDetails(
            line_colour=PlotColours.BURSTCCN_HYBRID,
            marker_style="o"
        ))

        self.add("ann", PlotElemDetails(
            display_name="ann-fa",
            line_colour=PlotColours.ANN,
            # line_style='--',
            line_style=(0, (5, 5)),
            marker_face_colour="none",
            zorder=50,
            marker_zorder=51,
        ))



        self.add("burstccn", PlotElemDetails(
            display_name="burstccn",
            line_colour=PlotColours.BURSTCCN,
        ))

        self.add("burstccn_QY_tied", PlotElemDetails(
            # display_name=r"burstccn (QY-sym)",
            display_name=PlotLabels.BURSTCCN_QY_TIED,
            line_colour=PlotColours.BURSTCCN_0Q,
            zorder=49,
            marker_zorder=50.5,
        ))

        self.add("burstccn_Y_block_trained", PlotElemDetails(
            # display_name="burstccn (online-offline)",
            display_name=PlotLabels.BURSTCCN_HYBRID,
            line_colour=PlotColours.BURSTCCN_HYBRID,
        ))

        # Variants that intentionally share styling with base versions
        self.add("burstccn_Y_learning", PlotElemDetails(
            display_name=PlotLabels.BURSTCCN_ONLINE,
            line_colour=PlotColours.BURSTCCN,
        ))

        # self.add("burstccn_Y_learning_noise", PlotElemDetails(
        #     display_name="burstccn (online)",
        #     line_colour=PlotColours.BURSTCCN,
        # ))

        self.add("burstprop", PlotElemDetails(
            display_name="burstprop",
            line_colour=PlotColours.BURSTPROP,
        ))

        self.add("edn", PlotElemDetails(
            display_name="edn",
            line_colour=PlotColours.EDN,
        ))

        self.add("edn_pred_tied", PlotElemDetails(
            display_name="edn (pred tied)",
            line_colour=PlotColours.EDN,
        ))

