import numpy as np
import matplotlib as mpl

from matplotlib import patches
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter


def init_global_matplotlib_constants():
    # mpl.use('Agg')
    base_fontsize = 15
    # plt.setp(ax.spines.values(), linewidth=1.5)

    scale = mpl.rcParams['font.size'] / base_fontsize

    # mpl.rcParams["font.family"] = "Open Sans"
    mpl.rcParams['font.size'] = base_fontsize * scale
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['axes.facecolor'] = 'none'
    mpl.rcParams['axes.linewidth'] = 1.5 * scale
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False

    mpl.rcParams['xtick.major.width'] = 1.5 * scale
    mpl.rcParams['ytick.major.width'] = 1.5 * scale
    mpl.rcParams['xtick.minor.width'] = 1.5 * scale
    mpl.rcParams['ytick.minor.width'] = 1.5 * scale

    # mpl.rcParams['legend.facecolor'] = 'white'
    # mpl.rcParams['legend.fontsize'] = 13.5 * scale

    mpl.rcParams['figure.dpi'] = 600
    mpl.rcParams['savefig.dpi'] = 600
    # mpl.rcParams['figure.autolayout'] = True  # or use tight_layout automatically

    mpl.rcParams['lines.linewidth'] = 1.5 * scale
    mpl.rcParams['grid.linewidth'] = 0.8 * scale
    mpl.rcParams['lines.markersize'] = 6 * scale

    # mpl.rcParams['legend.borderpad'] = 0.4 * scale
    # mpl.rcParams['legend.labelspacing'] = 0.5 * scale
    # mpl.rcParams['legend.handlelength'] = 2.0 * scale

    mpl.rcParams['legend.facecolor'] = 'none'
    mpl.rcParams['legend.fontsize'] = 15 * scale
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['legend.fancybox'] = False
    mpl.rcParams['legend.edgecolor'] = 'none'
    mpl.rcParams['legend.framealpha'] = 0.0
    mpl.rcParams['legend.borderpad'] = 0.25 * scale
    mpl.rcParams['legend.labelspacing'] = 0.35 * scale
    mpl.rcParams['legend.handlelength'] = 2.2 * scale
    mpl.rcParams['legend.handletextpad'] = 0.45 * scale


def smooth_ema(arr, smoothing_alpha):
    arr = np.asarray(arr, dtype=float)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = smoothing_alpha * arr[i] + (1 - smoothing_alpha) * out[i - 1]
    return out


def plot_line(ax, x, y, yerr=None, smoothing_alpha=None, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)
    if yerr is not None:
        yerr = np.asarray(yerr)
        yerr = np.nan_to_num(yerr, nan=0.0)

    key_map = {
        "line_colour": "color",
        "display_name": "label",
        "line_style": "linestyle",
        "line_width": "linewidth",
        "marker_style": "marker",
        "marker_face_colour": "markerfacecolor",
        "marker_size": "markersize",
        "zorder": "zorder"
    }

    if smoothing_alpha != None:
        y = smooth_ema(y, smoothing_alpha=smoothing_alpha)
        yerr = smooth_ema(yerr, smoothing_alpha=smoothing_alpha) if yerr is not None else None

    # Only include keys that are in the map
    plot_kwargs = {
        key_map[k]: v for k, v in kwargs.items() if k in key_map
    }

    marker_zorder = kwargs.get("marker_zorder")
    marker_style = plot_kwargs.get("marker")

    if marker_zorder is not None and marker_style is not None:
        line_kwargs = dict(plot_kwargs)
        marker_kwargs = {
            k: v for k, v in plot_kwargs.items()
            if k in ["color", "marker", "markerfacecolor", "markersize"]
        }
        marker_kwargs.update({
            "linestyle": "None",
            "label": "_nolegend_",
            "zorder": marker_zorder,
        })
        lines = ax.plot(x, y, **line_kwargs)
        ax.plot(x, y, **marker_kwargs)
    else:
        lines = ax.plot(x, y, **plot_kwargs)

    if yerr is not None:
        fill_kwargs = {
            k: v
            for k, v in plot_kwargs.items()
            if k not in ["linestyle", "label", "marker", "markerfacecolor", "markersize", "linewidth"]
        }
        ax.fill_between(x, y - yerr, y + yerr, alpha=0.2, edgecolor="none", linewidth=0, **fill_kwargs)

    assert len(lines) == 1
    return lines[0]


def plot_scatter(ax, x, y, **kwargs):
    x = np.asarray(x)
    y = np.asarray(y)

    key_map = {
        "display_name": "label",
        "marker_colour": "color",
        "marker_style": "marker",
        "zorder": "zorder",
        "alpha": "alpha"
    }

    # Only include keys that are in the map
    plot_kwargs = {
        key_map[k]: v for k, v in kwargs.items() if k in key_map
    }

    points = ax.scatter(x, y, **plot_kwargs)
    return points

def circle_ticklabel(ax, tick_label, radius_in_pixels=10):
    fig = ax.figure

    # Create the circle in display coords; initial dummy position
    circle = patches.Circle(
        (0, 0),
        radius_in_pixels,
        transform=None,  # => interpret as display/pixel coords
        fill=False,
        color='black',
        linewidth=1.25,
    )
    fig.add_artist(circle)

    def on_draw(event):
        tick_labels = ax.get_xticklabels()
        label = next((lbl for lbl in tick_labels if lbl.get_text() == str(tick_label)), None)
        if label is None:
            # If we don't find a matching label, just exit
            return

        # Get bounding box in display coords with the current renderer
        label_bbox = label.get_window_extent(renderer=event.renderer)
        cx = 0.5 * (label_bbox.x0 + label_bbox.x1)
        cy = 0.5 * (label_bbox.y0 + label_bbox.y1)
        circle.center = (cx, cy + 0.05 * label_bbox.height)
        circle.radius = 0.5 * max(label_bbox.width, label_bbox.height) * 1.35

    # Connect the draw_event
    cid = fig.canvas.mpl_connect('draw_event', on_draw)

    # Optional: do an initial draw *once*, outside the callback
    fig.canvas.draw()


def add_random_update_line(ax):
    ax.axhline(90.0, 0, 1.0, color='grey', linestyle='--')

    xmax = ax.get_xlim()[1]
    ax.text(xmax, 91, "random update (90°)",
            fontname="Consolas",
            color='grey',
            fontsize=12,
            va='bottom',
            ha='right')


def add_line_aligned_label(ax, y, label, above=True,
                           x_offset=0.0,
                           y_offset=0.0,
                           color='black',
                           fontsize=12,
                           fontname='Consolas'):
    """
    Adds a text label aligned to a horizontal line at y in data units.

    Parameters:
    - y: The y-value of the line
    - label: Text to display
    - above: Whether to place the label above or below the line
    - offset: Vertical offset in data units
    """
    y_text = y + y_offset if above else y - y_offset
    x_text = ax.get_xlim()[1] + x_offset

    ax.text(x_text, y_text, label,
            fontname=fontname,
            color=color,
            fontsize=fontsize,
            va='bottom' if above else 'top',
            ha='right')


def strip_axis(ax):
    ax.set_facecolor('none')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_border(ax, color='red', linewidth=1.5):
    fig = ax.figure

    rect = patches.Rectangle(
        (0, 0), 1, 1,
        transform=fig.transFigure,
        edgecolor=color,
        facecolor='none',
        linewidth=linewidth,
    )
    fig.patches.append(rect)

    def update_on_draw(event):
        renderer = event.renderer
        bbox = ax.get_tightbbox(renderer)
        bbox_in_inches = bbox.transformed(fig.dpi_scale_trans.inverted())

        fig_width, fig_height = fig.get_size_inches()
        left = bbox_in_inches.x0 / fig_width
        bottom = bbox_in_inches.y0 / fig_height
        width = bbox_in_inches.width / fig_width
        height = bbox_in_inches.height / fig_height

        rect.set_bounds(left, bottom, width, height)

    fig.canvas.mpl_connect('draw_event', update_on_draw)
    fig.canvas.draw()


def setup_axis(ax, **kwargs):
    if "x_label" in kwargs:
        ax.set_xlabel(kwargs["x_label"])

    if "y_label" in kwargs:
        ax.set_ylabel(kwargs["y_label"])

    # if "ax_label" in kwargs:
    #     add_ax_label(fig, ax, kwargs["ax_label"])

    if "x_lims" in kwargs:
        ax.set_xlim(kwargs["x_lims"])

    if "y_scale" in kwargs:
        ax.set_yscale(kwargs["y_scale"])

    if "y_lims" in kwargs:
        ymin, ymax = kwargs["y_lims"]
        current_ymin, current_ymax = ax.get_ylim()
        ax.set_ylim(
            ymin if ymin is not None else current_ymin,
            ymax if ymax is not None else current_ymax
        )

    if "x_ticks" in kwargs:
        ax.xaxis.set_ticks(kwargs["x_ticks"])

    if "y_ticks" in kwargs:
        ax.yaxis.set_ticks(kwargs["y_ticks"])

    if "x_tick_labels" in kwargs:
        ax.set_xticklabels(kwargs["x_tick_labels"])

    if "y_tick_labels" in kwargs:
        ax.set_yticklabels(kwargs["y_tick_labels"])

    if "x_tick_frequency" in kwargs:
        start, end = ax.dataLim.intervalx
        offset = kwargs.get("x_tick_offset", 0)
        ticks = np.arange(start, end, kwargs["x_tick_frequency"]) + offset
        ax.xaxis.set_ticks(ticks)

    if "x_tick_between_labels" in kwargs:
        font_weight = kwargs.get("x_tick_between_labels_fontweight", "normal")  # use "normal" as a default
        major_ticks = ax.get_xticks()
        midpoints = (major_ticks[:-1] + major_ticks[1:]) / 2

        ax.set_xticks(midpoints, minor=True)
        ax.set_xticklabels(kwargs["x_tick_between_labels"], minor=True, fontweight=font_weight)
        ax.tick_params(axis='x', which='minor', length=0, labelsize=10)



    if kwargs.get("remove_axes", False):
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(bottom=False, labelbottom=False,
                       left=False, labelleft=False)

    if kwargs.get("twin_ax", False):
        ax.spines["right"].set_visible(True)

    if "ax_colour" in kwargs:
        colour = kwargs["ax_colour"]
        ax.spines['right'].set_color(colour)
        ax.tick_params(colors=colour)
        ax.yaxis.label.set_color(colour)


def transform_rect_to_fig_coords(fig, ax, rect_coords):
    """
    Transforms a rectangle defined in axis data coordinates to figure coordinates.

    Parameters:
    - fig: matplotlib figure
    - ax: the axes in which the rectangle is defined
    - rect_coords: 2x2 np.array of [(x0, y0), (x1, y1)]

    Returns:
    - bottom-left corner (x, y) in figure coords
    - width, height in figure coords
    """
    pixel_coords = ax.transData.transform(rect_coords)
    fig_coords = fig.transFigure.inverted().transform(pixel_coords)
    (x0, y0), (x1, y1) = fig_coords
    x, y = min(x0, x1), min(y0, y1)
    width, height = abs(x1 - x0), abs(y1 - y0)
    return (x, y), width, height


def draw_vertical_rect_across_axes(fig, x_low, x_high, ax_low, ax_high, colour):
    """
    Draws a vertical rectangle from ax_low to ax_high at the given x-range in data coordinates.

    Parameters:
    - fig: The matplotlib Figure object
    - x_low, x_high: Data x-limits for the rectangle
    - ax_low, ax_high: The lower and upper axes to span
    - facecolor: The fill color of the rectangle
    """
    rect_top = np.array([(x_low, ax_high.get_ylim()[0]), (x_high, ax_high.get_ylim()[1])])
    rect_bottom = np.array([(x_low, ax_low.get_ylim()[0]), (x_high, ax_low.get_ylim()[1])])

    top_loc, top_w, top_h = transform_rect_to_fig_coords(fig, ax_high, rect_top)
    bot_loc, bot_w, bot_h = transform_rect_to_fig_coords(fig, ax_low, rect_bottom)

    height = top_loc[1] - bot_loc[1] + top_h

    rect = patches.Rectangle(bot_loc, bot_w, height, facecolor=colour,
                             zorder=-1, transform=fig.transFigure)

    fig.patches.append(rect)


def add_vertical_span_across_axes(fig, ax_low, ax_high, x_low, x_high, colour='lightgray', zorder=-10):
    """
    Add a vertical span behind all axes, updating dynamically if axes resize.
    """
    rect = patches.Rectangle((0, 0), 1, 1, transform=fig.transFigure,
                             facecolor=colour, edgecolor=None, zorder=zorder)
    fig.patches.append(rect)

    def update_rect(event=None):
        trans_low = ax_low.transData
        trans_high = ax_high.transData
        inv_fig = fig.transFigure.inverted()

        # Convert x and y bounds from data to figure coordinates
        x0_fig, y0_fig = inv_fig.transform(trans_low.transform((x_low, ax_low.get_ylim()[0])))
        x1_fig, y1_fig = inv_fig.transform(trans_high.transform((x_high, ax_high.get_ylim()[1])))

        rect.set_bounds(x0_fig, y0_fig, x1_fig - x0_fig, y1_fig - y0_fig)

    fig.canvas.mpl_connect("draw_event", update_rect)
    update_rect()

    return rect


def add_vertical_line_across_axes(fig, ax_low, ax_high, x, colour='lightgray', linestyle='--', zorder=-10):
    """
    Add a vertical dashed line behind all axes, updating dynamically if axes resize.
    """
    line = mlines.Line2D([], [], color=colour, linestyle=linestyle, zorder=zorder, transform=fig.transFigure)
    fig.lines.append(line)

    def update_line(event=None):
        trans_low = ax_low.transData
        trans_high = ax_high.transData
        inv_fig = fig.transFigure.inverted()

        # Convert x from data to figure coordinates for both low and high axes
        x_fig_low, y_fig_low = inv_fig.transform(trans_low.transform((x, ax_low.get_ylim()[0])))
        x_fig_high, y_fig_high = inv_fig.transform(trans_high.transform((x, ax_high.get_ylim()[1])))

        # Update line data in figure coordinates
        line.set_data([x_fig_low, x_fig_high], [y_fig_low, y_fig_high])
        line.set_transform(fig.transFigure)

    fig.canvas.mpl_connect("draw_event", update_line)
    update_line()

    return line


def rescale_ticks(ax, order, axis='x', relabel_axis=True):
    axis_obj = ax.xaxis if axis == 'x' else ax.yaxis

    def scale_fmt(value, pos):
        return f"{value / 10**order:g}"

    axis_obj.set_major_formatter(FuncFormatter(scale_fmt))

    if relabel_axis:
        if axis == 'x':
            base = ax.get_xlabel() or ""
            if rf"($\times10^{order}$)" not in base:
                ax.set_xlabel(base + rf" ($\times10^{order}$)")
        else:
            base = ax.get_ylabel() or ""
            if rf"($\times10^{order}$)" not in base:
                ax.set_ylabel(base + rf" ($\times10^{order}$)")
