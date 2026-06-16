import warnings
from pathlib import Path

import matplotlib

from matplotlib import pyplot as plt, patches
from matplotlib.transforms import Bbox


class PanelNode:
    def __init__(self, name, inner_pad=(0, 0, 0, 0), remove_x_axis=False, remove_y_axis=False):
        self.name = name
        self.inner_pad = inner_pad   # (left, bottom, right, top)
        self.bbox = None
        self.remove_x_axis = remove_x_axis
        self.remove_y_axis = remove_y_axis


class ContainerNode:
    def __init__(self, children, layout="row", name=None, padding=(0, 0, 0, 0),
                 weights=None, spacings=None):
        self.name = name  # Optional: give this container a name so you can label it
        self.children = children  # List of ContainerNode or PanelNode
        self.layout = layout      # "row", "column"
        self.padding = padding    # outer padding for this container
        self.bbox = None          # Set during layout
        self.weights = weights    # List of floats
        self.spacings = spacings  # List of spacings between children (len = n - 1)


class FigureManager:
    def __init__(self, fig, margin_inch=(0.45, 0.05, 0.05, 0.45)):
        self.fig = fig
        self.margin_inch = margin_inch
        self.panels = {}

        self.scale_factor = matplotlib.rcParams['font.size'] / 15.0

        self.label_kwargs = {
            'fontfamily': 'arial',
            'fontsize': 32 * self.scale_factor,
            'ha': 'left',
            'va': 'top',
            'fontweight': 'bold'
        }
        self.label_offset = (-0.25 * self.scale_factor, 0.35 * self.scale_factor)

        self.warn_if_pycharm_big_fig()

    def warn_if_pycharm_big_fig(self, limit_px=1000):
        if "interagg" not in matplotlib.get_backend().lower():
            return

        w_px, h_px = self.fig.get_size_inches() * self.fig.dpi
        if w_px >= limit_px or h_px >= limit_px:
            warnings.warn(
                f"Figure is {w_px:.0f}×{h_px:.0f}px; "
                "PyCharm SciView may fail. Try lowering figure.dpi.",
                RuntimeWarning,
                stacklevel=2,
            )


    def _inches_to_fig_coords(self, inches):
        """Convert inch or 4-tuple of inches to figure coords."""
        w_inch, h_inch = self.fig.get_size_inches()
        if isinstance(inches, (int, float)):
            return (inches / w_inch, inches / h_inch)
        else:
            l, b, r, t = inches
            return (
                l / w_inch,
                b / h_inch,
                r / w_inch,
                t / h_inch
            )

    def add_panel(self, name, left, bottom, right, top,
                  pad_left=0, pad_bottom=0, pad_right=0, pad_top=0):
        """
        Define a panel using layout coords (0–1), remapped within asymmetric inch-based margins.
        Padding is also specified in inches and converted to figure coords.
        """
        # Convert margin to figure coords
        l_in, b_in, r_in, t_in = self.margin_inch
        l, b, r, t = self._inches_to_fig_coords((l_in, b_in, r_in, t_in))

        # Remap from layout space to figure coords within margins
        layout_left = l + (1 - l - r) * left
        layout_bottom = b + (1 - b - t) * bottom
        layout_right = l + (1 - l - r) * right
        layout_top = b + (1 - b - t) * top

        panel_bbox = Bbox.from_extents(layout_left, layout_bottom, layout_right, layout_top)

        # Convert pad inches to figure coords
        pad_l, pad_b, pad_r, pad_t = self._inches_to_fig_coords((pad_left, pad_bottom, pad_right, pad_top))

        # Apply padding to define padded bbox
        padded_bbox = Bbox.from_extents(
            layout_left - pad_l,
            layout_bottom - pad_b,
            layout_right + pad_r,
            layout_top + pad_t
        )

        self.panels[name] = {
            "bbox": panel_bbox,
            "padded_bbox": padded_bbox,
            "ax": None
        }
        return panel_bbox

    def create_axes(self, name, sharex=None, sharey=None):
        """Create an Axes inside a named panel region, with optional axis sharing."""
        bbox = self.panels[name]["bbox"]
        width = bbox.width
        height = bbox.height

        # Create the axes, allowing axis sharing
        ax = self.fig.add_axes([bbox.x0, bbox.y0, width, height],
                               sharex=sharex, sharey=sharey)

        self.panels[name]["ax"] = ax
        return ax

    def insert_pdf(self, name, pdf_file, scale=1.0, x=0.0, y=0.0, align_x="left", align_y="bottom"):
        """Register a panel to have a PDF placed into it instead of an Axes."""
        self.panels[name]["pdf"] = {
            "filename": pdf_file,
            "scale": scale,
            "x": x,
            "y": y,
            "align_x": align_x,
            "align_y": align_y,
        }

    def add_relative_overlay_pdf(self, parent_name, name, pdf_file, x, y, size,
                                 scale=1.0, align_x="left", align_y="bottom"):
        """
        Add a PDF overlay inside an existing panel using relative coordinates.

        Parameters
        ----------
        parent_name : str
            Name of an existing panel/container whose bbox is used as the reference frame.
        name : str
            Name for the overlay panel.
        pdf_file : str
            Path relative to `pdf_resources`.
        x, y : float
            Relative placement position inside the parent bbox.
        size : float
            Absolute overlay box size in inches. The same size is used for width and
            height; PDF fitting still uses the min dimension internally.
        """
        parent_bbox = self.panels[parent_name]["bbox"]
        fig_w_inch, fig_h_inch = self.fig.get_size_inches()
        size_x = size / fig_w_inch
        size_y = size / fig_h_inch
        overlay_bbox = Bbox.from_extents(
            parent_bbox.x0 + x * parent_bbox.width,
            parent_bbox.y0 + y * parent_bbox.height,
            parent_bbox.x0 + x * parent_bbox.width + size_x,
            parent_bbox.y0 + y * parent_bbox.height + size_y,
        )

        self.panels[name] = {
            "bbox": overlay_bbox,
            "padded_bbox": overlay_bbox,
            "ax": None
        }
        self.insert_pdf(name, pdf_file, scale=scale, align_x=align_x, align_y=align_y)

    def add_label(self, name, label_text):
        """Add a label (e.g. 'a', 'b') at the top-left of the padded panel region using inch-based offset."""
        padded_bbox = self.panels[name]["padded_bbox"]
        fig_width, fig_height = self.fig.get_size_inches()

        # Convert inch offset to figure coordinates
        dx_inch, dy_inch = self.label_offset  # now in inches
        dx_fig = dx_inch / fig_width
        dy_fig = dy_inch / fig_height

        label_x = padded_bbox.x0 + dx_fig
        label_y = padded_bbox.y1 + dy_fig

        print(label_x, label_y, label_text)
        self.fig.text(label_x, label_y, label_text, transform=self.fig.transFigure, **self.label_kwargs)

    def add_spanning_title(
            self,
            panel_name,
            title_text,
            text_pad=0.10,
            line_pad=0.08,
            fontsize=16,
            y_shift=0.0,
            text_kwargs=None,
            line_kwargs=None,
    ):
        """Add a centered title with split horizontal rules above a named panel or container."""
        if panel_name not in self.panels:
            raise KeyError(f"Unknown panel/container: {panel_name}")

        panel = self.panels[panel_name]
        bbox = panel["bbox"]
        padded_bbox = panel.get("padded_bbox", bbox)
        fig_width, fig_height = self.fig.get_size_inches()
        label_dx_inch, label_dy_inch = self.label_offset

        label_top_y = padded_bbox.y1 + (label_dy_inch / fig_height)
        left_edge_x = padded_bbox.x0 + (label_dx_inch / fig_width)
        text_y = label_top_y + ((text_pad + y_shift) * self.scale_factor / fig_height)
        center_x = bbox.x0 + 0.5 * bbox.width

        title_style = {
            "ha": "center",
            "va": "center",
            "fontsize": fontsize,
            "fontfamily": "arial",
        }
        if text_kwargs:
            title_style.update(text_kwargs)

        text = self.fig.text(center_x, text_y, title_text, transform=self.fig.transFigure, **title_style)
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        text_bbox = text.get_window_extent(renderer=renderer).transformed(self.fig.transFigure.inverted())
        text_y += 0.5 * (fontsize / 72.0) / fig_height
        text.set_y(text_y)
        line_y = text_y
        line_gap = line_pad * self.scale_factor / fig_width
        cap_drop = 0.15 * self.scale_factor / fig_height

        rule_style = {
            "color": "#A7A9AB",
            "linewidth": 3.0,
        }
        if line_kwargs:
            rule_style.update(line_kwargs)

        if left_edge_x < text_bbox.x0 - line_gap:
            self.fig.lines.append(
                plt.Line2D(
                    [left_edge_x, text_bbox.x0 - line_gap],
                    [line_y, line_y],
                    transform=self.fig.transFigure,
                    **rule_style,
                )
            )
        if text_bbox.x1 + line_gap < bbox.x1:
            self.fig.lines.append(
                plt.Line2D(
                    [text_bbox.x1 + line_gap, bbox.x1],
                    [line_y, line_y],
                    transform=self.fig.transFigure,
                    **rule_style,
                )
            )
        self.fig.lines.append(
            plt.Line2D(
                [left_edge_x, left_edge_x],
                [line_y, line_y - cap_drop],
                transform=self.fig.transFigure,
                **rule_style,
            )
        )
        self.fig.lines.append(
            plt.Line2D(
                [bbox.x1, bbox.x1],
                [line_y, line_y - cap_drop],
                transform=self.fig.transFigure,
                **rule_style,
            )
        )

    def add_panel_group_title(self, panel_names, title_text, text_pad=0.04, fontsize=14, underline=True, fontfamily="arial"):
        """Add a centered title over one or more panels."""
        missing_panels = [name for name in panel_names if name not in self.panels]
        if missing_panels:
            raise KeyError(f"Unknown panel/container(s): {missing_panels}")

        bboxes = [self.panels[name]["bbox"] for name in panel_names]
        padded_bboxes = [
            self.panels[name].get("padded_bbox", self.panels[name]["bbox"])
            for name in panel_names
        ]
        _, fig_height = self.fig.get_size_inches()

        left = min(bbox.x0 for bbox in bboxes)
        right = max(bbox.x1 for bbox in bboxes)
        top = max(bbox.y1 for bbox in padded_bboxes)

        text = self.fig.text(
            0.5 * (left + right),
            top + (text_pad * self.scale_factor / fig_height),
            title_text,
            transform=self.fig.transFigure,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontfamily=fontfamily,
        )
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        text_bbox = text.get_window_extent(renderer=renderer).transformed(self.fig.transFigure.inverted())
        if underline:
            underline_y = text_bbox.y0 - (0.01 * self.scale_factor / fig_height)
            self.fig.lines.append(
                plt.Line2D(
                    [text_bbox.x0, text_bbox.x1],
                    [underline_y, underline_y],
                    transform=self.fig.transFigure,
                    color="black",
                    linewidth=1.2,
                )
                )

    def add_panel_group_title_with_superscript(
        self,
        panel_names,
        title_text,
        superscript_text,
        text_pad=0.04,
        fontsize=14,
        underline=True,
        fontfamily="arial",
        superscript_fontfamily=None,
        superscript_fontsize=None,
        superscript_raise=0.35,
        superscript_dx=0.0,
    ):
        """Add a centered title with a separately rendered superscript."""
        missing_panels = [name for name in panel_names if name not in self.panels]
        if missing_panels:
            raise KeyError(f"Unknown panel/container(s): {missing_panels}")

        bboxes = [self.panels[name]["bbox"] for name in panel_names]
        padded_bboxes = [
            self.panels[name].get("padded_bbox", self.panels[name]["bbox"])
            for name in panel_names
        ]
        _, fig_height = self.fig.get_size_inches()

        left = min(bbox.x0 for bbox in bboxes)
        right = max(bbox.x1 for bbox in bboxes)
        top = max(bbox.y1 for bbox in padded_bboxes)
        y = top + (text_pad * self.scale_factor / fig_height)

        superscript_fontfamily = superscript_fontfamily or fontfamily
        superscript_fontsize = superscript_fontsize or (0.72 * fontsize)

        base_text = self.fig.text(
            0.0,
            0.0,
            title_text,
            transform=self.fig.transFigure,
            ha="left",
            va="bottom",
            fontsize=fontsize,
            fontfamily=fontfamily,
            alpha=0.0,
        )
        superscript_text_obj = self.fig.text(
            0.0,
            0.0,
            superscript_text,
            transform=self.fig.transFigure,
            ha="left",
            va="bottom",
            fontsize=superscript_fontsize,
            fontfamily=superscript_fontfamily,
            alpha=0.0,
        )

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        base_bbox = base_text.get_window_extent(renderer=renderer).transformed(self.fig.transFigure.inverted())
        superscript_bbox = superscript_text_obj.get_window_extent(renderer=renderer).transformed(self.fig.transFigure.inverted())

        raise_amount = (superscript_raise * fontsize / 72.0) / fig_height
        x_center = 0.5 * (left + right)

        base_text.set_position((x_center, y))
        superscript_text_obj.set_position((x_center + superscript_dx, y + raise_amount))
        base_text.set_alpha(1.0)
        superscript_text_obj.set_alpha(1.0)

        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        base_bbox = base_text.get_window_extent(renderer=renderer).transformed(self.fig.transFigure.inverted())
        superscript_bbox = superscript_text_obj.get_window_extent(renderer=renderer).transformed(self.fig.transFigure.inverted())

        if underline:
            text_left = min(base_bbox.x0, superscript_bbox.x0)
            text_right = max(base_bbox.x1, superscript_bbox.x1)
            underline_y = min(base_bbox.y0, superscript_bbox.y0) - (0.01 * self.scale_factor / fig_height)
            self.fig.lines.append(
                plt.Line2D(
                    [text_left, text_right],
                    [underline_y, underline_y],
                    transform=self.fig.transFigure,
                    color="black",
                    linewidth=1.2,
                )
            )

    def add_shared_xlabel(self, axes, label, **text_kwargs):
        """Place one x-label centered under a group of axes by attaching it to the left axis."""
        left_ax = axes[0]
        left_ax.set_xlabel(label, **text_kwargs)
        for ax in axes[1:]:
            ax.set_xlabel("")

        self.fig.canvas.draw()
        bboxes = [ax.get_position() for ax in axes]
        left = min(bbox.x0 for bbox in bboxes)
        right = max(bbox.x1 for bbox in bboxes)
        pair_center = 0.5 * (left + right)
        left_bbox = left_ax.get_position()
        label_x = (pair_center - left_bbox.x0) / left_bbox.width
        left_ax.xaxis.label.set_x(label_x)

    def resolve_layout(self, root_node, bbox=None):
        if bbox is None:
            # Use full area within margins
            l_in, b_in, r_in, t_in = self.margin_inch
            l, b, r, t = self._inches_to_fig_coords((l_in, b_in, r_in, t_in))
            bbox = Bbox.from_extents(l, b, 1 - r, 1 - t)

        if isinstance(root_node, PanelNode):
            # Shrink bbox by inner padding (IN INCHES, convert to figure coords)
            l_inch, b_inch, r_inch, t_inch = root_node.inner_pad
            l_inch *= self.scale_factor
            b_inch *= self.scale_factor
            r_inch *= self.scale_factor
            t_inch *= self.scale_factor

            fig_w_inch, fig_h_inch = self.fig.get_size_inches()

            # fig_w_inch *= self.scale_factor
            # fig_h_inch *= self.scale_factor

            l = l_inch / fig_w_inch
            r = r_inch / fig_w_inch
            b = b_inch / fig_h_inch
            t = t_inch / fig_h_inch

            panel_bbox = Bbox.from_extents(
                bbox.x0 + l,
                bbox.y0 + b,
                bbox.x1 - r,
                bbox.y1 - t,
            )
            self.panels[root_node.name] = {
                "bbox": panel_bbox,
                "padded_bbox": bbox,
                "ax": None,
                "remove_x_axis": root_node.remove_x_axis,
                "remove_y_axis": root_node.remove_y_axis
            }
            root_node.bbox = panel_bbox

        elif isinstance(root_node, ContainerNode):
            # Apply outer padding
            l, b, r, t = root_node.padding
            inner_bbox = Bbox.from_extents(
                bbox.x0 + bbox.width * l,
                bbox.y0 + bbox.height * b,
                bbox.x1 - bbox.width * r,
                bbox.y1 - bbox.height * t
            )
            root_node.bbox = inner_bbox

            # 🔷 Register container by name (like a panel) — for labeling/debug
            if root_node.name:
                self.panels[root_node.name] = {
                    "bbox": inner_bbox,
                    "padded_bbox": bbox,
                    "ax": None
                }

            n = len(root_node.children)
            # ROW layout
            if root_node.layout == "row":
                n = len(root_node.children)
                weights = root_node.weights or [1] * n
                spacings = root_node.spacings or [0.0] * (n - 1)
                total_weight = sum(weights)
                total_spacing = sum(spacings)
                usable_width = inner_bbox.width - total_spacing

                x = inner_bbox.x0
                for i, (child, weight) in enumerate(zip(root_node.children, weights)):
                    w = usable_width * (weight / total_weight)
                    child_right = x + w
                    child_bbox = Bbox.from_extents(x, inner_bbox.y0, child_right, inner_bbox.y1)
                    self.resolve_layout(child, child_bbox)
                    if i < len(spacings):
                        x = child_right + spacings[i]

            # COLUMN layout (same idea, for height)

            elif root_node.layout == "column":
                weights = root_node.weights or [1] * n
                spacings = root_node.spacings or [0.0] * (n - 1)
                container_name = root_node.name or "<unnamed>"

                assert len(weights) == n, (
                    f"Container {container_name}: expected {n} weights, got {len(weights)}"
                )
                assert len(spacings) == n - 1, (
                    f"Container {container_name}: expected {n-1} spacings, got {len(spacings)}"
                )

                total_weight = sum(weights)
                total_spacing = sum(spacings)
                usable_height = inner_bbox.height - total_spacing

                # Start from top, go downward
                y_top = inner_bbox.y1
                for i, (child, weight) in enumerate(zip(root_node.children, weights)):
                    h = usable_height * (weight / total_weight)
                    bottom = y_top - h
                    child_bbox = Bbox.from_extents(inner_bbox.x0, bottom, inner_bbox.x1, y_top)
                    self.resolve_layout(child, child_bbox)

                    if i < len(spacings):
                        y_top = bottom - spacings[i]  # move down to next block

    def draw_debug(self, show_label=False, show_tightbbox=True):
        fig = self.fig
        fig.canvas.draw()  # Only once
        renderer = fig.canvas.get_renderer()

        for panel_name in self.panels:
            panel = self.panels[panel_name]

            bbox = panel["bbox"]
            padded_bbox = panel["padded_bbox"]
            ax = panel.get("ax", None)

            # Draw red box for panel bbox
            rect = patches.Rectangle(
                (bbox.x0, bbox.y0),
                bbox.width,
                bbox.height,
                transform=fig.transFigure,
                fill=False,
                color='red',
                linewidth=1.5,
                linestyle='-',
                zorder=1000,
                label='panel bbox'
            )
            fig.patches.append(rect)

            # Draw blue dashed box for padded bbox
            padded_rect = patches.Rectangle(
                (padded_bbox.x0, padded_bbox.y0),
                padded_bbox.width,
                padded_bbox.height,
                transform=fig.transFigure,
                fill=False,
                color='blue',
                linewidth=1.5,
                linestyle='--',
                zorder=1000,
                label='padded bbox'
            )
            fig.patches.append(padded_rect)

            # Optional: draw tight bbox if ax exists
            if show_tightbbox and ax is not None:
                tight_bbox = ax.get_tightbbox(renderer)
                tight_bbox_fig = tight_bbox.transformed(fig.transFigure.inverted())

                tb_rect = patches.Rectangle(
                    (tight_bbox_fig.x0, tight_bbox_fig.y0),
                    tight_bbox_fig.width,
                    tight_bbox_fig.height,
                    transform=fig.transFigure,
                    fill=False,
                    color='green',
                    linewidth=1.5,
                    linestyle=':',
                    zorder=1000,
                    label='tight bbox'
                )
                fig.patches.append(tb_rect)

            if show_label:
                cx = bbox.x0 + bbox.width / 2
                cy = bbox.y0 + bbox.height / 2
                fig.text(cx, cy, f"{panel_name}", transform=fig.transFigure,
                         ha='center', va='center', fontsize=10, color='black',
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    def remove_axes(self):
        # Loop over panels
        for name, panel in self.panels.items():
            ax = panel.get("ax")
            if ax is None:
                continue

            if panel.get("remove_x_axis", False):
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")

            if panel.get("remove_y_axis", False):
                ax.tick_params(labelleft=False)
                ax.set_ylabel("")

    def finalise_figure(self, output_filename, show=True, draw_debug=False):
        if draw_debug:
            self.draw_debug()

        self.remove_axes()

        # Draw and save base figure
        output_path = Path("figure_pdfs") / output_filename
        pdf_resource_dir = Path("pdf_resources")

        self.fig.canvas.draw_idle()
        self.fig.savefig(output_path)

        if show:
            # self.fig.show()
            print("backend:", matplotlib.get_backend())
            print("interactive:", plt.isinteractive())
            print("fignum_exists:", plt.fignum_exists(self.fig.number))
            print("num_axes:", len(self.fig.axes))
            self.fig.canvas.draw_idle()
            plt.show()

        # 🔹 Check if any panels have PDFs
        has_embedded_pdfs = any("pdf" in panel for panel in self.panels.values())

        if has_embedded_pdfs:
            from pdfrw import PdfReader, PdfWriter, PageMerge

            # Load the saved base PDF
            base_pdf = PdfReader(str(output_path))
            base_page = base_pdf.pages[0]
            fig_width, fig_height = self.fig.get_size_inches()

            # Loop over panels with registered PDFs
            for name, panel in self.panels.items():
                if "pdf" not in panel:
                    continue

                pdf_info = panel["pdf"]
                pdf_path = pdf_resource_dir / pdf_info["filename"]
                overlay_pdf = PdfReader(str(pdf_path))
                overlay_page = overlay_pdf.pages[0]

                # Compute location in figure space (in points)
                bbox = panel["bbox"]
                x_pts = bbox.x0 * fig_width * 72
                y_pts = bbox.y0 * fig_height * 72
                width_pts = bbox.width * fig_width * 72
                height_pts = bbox.height * fig_height * 72

                # Load and scale the overlay
                overlay_merge = PageMerge()
                overlay_merge.add(overlay_page)
                overlay = overlay_merge[0]
                if overlay is None:
                    raise RuntimeError(f"Failed to load overlay page for panel '{name}'.")

                orig_w, orig_h = overlay.w, overlay.h
                box_scale = min(width_pts / orig_w, height_pts / orig_h)
                scale = pdf_info.get("scale", box_scale)
                scale *= box_scale

                # Resize
                overlay.scale(scale)

                align_x = pdf_info.get("align_x", "left")
                align_y = pdf_info.get("align_y", "bottom")

                if align_x == "center":
                    x_pts += 0.5 * (width_pts - overlay.w)
                elif align_x == "right":
                    x_pts += width_pts - overlay.w
                elif align_x != "left":
                    raise ValueError(f"Unsupported align_x: {align_x}")

                if align_y == "center":
                    y_pts += 0.5 * (height_pts - overlay.h)
                elif align_y == "top":
                    y_pts += height_pts - overlay.h
                elif align_y != "bottom":
                    raise ValueError(f"Unsupported align_y: {align_y}")

                x_pts += pdf_info["x"] * width_pts
                y_pts += pdf_info["y"] * height_pts

                overlay.x = x_pts
                overlay.y = y_pts

                PageMerge(base_page).add(overlay).render()

            PdfWriter(str(output_path), trailer=base_pdf).write()
        output_path = Path(output_path).resolve()
        print(output_path.as_uri())  # -> file:///C:/Users/.../mnist_full_Y_learning.pdf
