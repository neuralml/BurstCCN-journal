from dataclasses import dataclass, asdict, fields
from enum import Enum
from typing import Optional, Tuple, Literal, List, Union


class PlotColours(str, Enum):
    DEFAULT = "#000000"

    EVENT = '#0975b3'
    BURST = '#e39700'
    BURST_PROB = '#9e2423'
    BURST_PROB_BASELINE = '#E89998'
    # WEIGHT_CHANGE = 'gray'
    # WEIGHT_CHANGE2 = 'silver'
    WEIGHT_CHANGE = 'purple'
    WEIGHT_CHANGE2 = '#BA54A0'

    ANN = '#000000'
    BURSTCCN = '#6881D8'
    BURSTCCN_0Q = '#2ec4ff'
    BURSTCCN_HYBRID = 'purple'
    BURSTPROP = '#B84C3E'
    EDN = '#50B47B'
    NODE_PERTURBATION = '#A684C7'
    # ANN = '#000000'
    # BURSTCCN = '#6881D8'
    # BURSTCCN_HYBRID = '#7A3CFF'
    # BURSTCCN_0Q = '#C000FF'
    # BURSTPROP = '#B84C3E'
    # EDN = '#50B47B'
    # NODE_PERTURBATION = '#8C6D5A'

    # LAYER1 = '#7b9fe0'
    # LAYER2 = '#f2a541'
    # LAYER3 = '#c95f5f'
    # LAYER4 = '#6bcf8e'

    LAYER1 = '#159c76'
    LAYER2 = '#4bb89f'  # lighter/clearer green-teal
    LAYER3 = '#ff8742'  # orange
    LAYER4 = '#ff1e4f'

    INCREASING = '#346ABB'
    DECREASING = '#B03FCC'
    EQUAL = '#ff4d04' # was "#808080"

    @staticmethod
    def from_layer_index(index):
        return getattr(PlotColours, f"LAYER{index+1}")


class PlotLabels(str, Enum):
    EPOCH = "Epoch"
    EXAMPLE = "Example"
    ITERATIONS = "Iterations"

    TEST_ERROR = "Test error (%)"

    # QY_ALIGNMENT = "QY alignment (deg)"
    # WY_ALIGNMENT = "WY alignment (deg)"
    # BP_ALIGNMENT = "BP alignment (deg)"
    # FA_ALIGNMENT = "FA alignment (deg)"

    QY_ALIGNMENT = "QY angle (deg)"
    WY_ALIGNMENT = "WY angle (deg)"
    BP_ALIGNMENT = "ANN-BP angle (deg)"
    FA_ALIGNMENT = "ANN-FA angle (deg)"

    # QY_ALIGNMENT = r"$\theta\left(\mathbf{Q},\mathbf{Y}\right)$ (deg)"
    # WY_ALIGNMENT = r"$\theta\left(\mathbf{W},\mathbf{Y}\right)$ (deg)"
    # BP_ALIGNMENT = r"$\theta\left(\Delta\mathbf{W}, \Delta\mathbf{W}^{\mathrm{ANN\text{-}FA}}\right)$ (deg)"
    # FA_ALIGNMENT = r"$\theta\left(\Delta\mathbf{W}, \Delta\mathbf{W}^{\mathrm{ANN\text{-}BP}}\right)$ (deg)"

    # APICAL_MAGNITUDE = "Apical magnitude"# r'Mean apical $|\mathbf{u}|$' #
    # APICAL_MAGNITUDE = r"$|\text{apical potential}|$"
    APICAL_MAGNITUDE = r"Mean $|\mathbf{u}|$"
    # BURST_PROB_CHANGE_MAGNITUDE = "Burst probability\nchange magnitude"# "Burst prob change magnitude"
    # BURST_PROB_CHANGE_MAGNITUDE = r"$|\Delta \text{burst prob}|$"
    BURST_PROB_CHANGE_MAGNITUDE = r"Mean $|\mathbf{p} - \mathbf{p}^b|$"

    BURSTCCN_ONLINE = "burstccn (online)"
    BURSTCCN_HYBRID = "burstccn (hybrid)"
    BURSTCCN_QY_TIED = r"burstccn ($\mathbf{QY}$-sym)"

    def __str__(self):
        return self.value



@dataclass(frozen=True)
class PlotAxDetails:
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    title: Optional[str] = None

    x_lims: Optional[Tuple[float, float]] = None
    y_lims: Optional[Tuple[float, float]] = None
    x_ticks: Optional[Tuple[float, ...]] = None
    y_ticks: Optional[Tuple[float, ...]] = None
    y_tick_labels: Optional[Tuple[str, ...]] = None

    remove_spines: bool = False
    show_grid: bool = False
    y_scale: Optional[str] = None

    x_tick_frequency: Optional[int] = None
    y_tick_frequency: Optional[int] = None
    # add: tick_params, log scales, etc as needed

    x_tick_labels: Optional[Tuple[str, ...]] = None
    x_tick_between_labels: Optional[List[Tuple[int, int]]] = None
    x_tick_between_labels_fontweight: Optional[str] = None

    legend_location: Optional[str] = None

    included_elems: Optional[Tuple[str, ...]] = None


    def to_kwargs(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}


@dataclass(frozen=True)
class PlotElemDetails:
    # applies to line/scatter/bar etc
    display_name: Optional[str] = None
    alpha: Optional[float] = None
    zorder: Optional[int] = None

    # line-ish
    line_colour: Optional[str] = None
    line_style: Optional[str] = None
    line_width: Optional[float] = None

    # marker-ish
    marker_colour: Optional[str] = None
    marker_face_colour: Optional[str] = None
    marker_style: Optional[str] = None
    marker_size: Optional[float] = None
    marker_zorder: Optional[Union[int, float]] = None

    # bar-ish / patch-ish
    edge_color: Optional[str] = None

    kind: Optional[Literal["line", "scatter", "bar", "patch"]] = None

    smoothing_alpha: Optional[float] = None

    def to_kwargs(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if getattr(self, f.name) is not None}


class PlotAxStore:
    def __init__(self):
        self._axes: dict[str, PlotAxDetails] = {}

    def add(self, key: str, details: PlotAxDetails):
        if key in self._axes:
            raise KeyError(f"Axis {key!r} already defined")
        self._axes[key] = details
        return self  # optional chaining

    def set(self, key: str, details: PlotAxDetails):
        self._axes[key] = details
        return self

    def get(self, key: str) -> PlotAxDetails:
        try:
            return self._axes[key]
        except KeyError as e:
            raise KeyError(f"Unknown axis key {key!r}. Known: {list(self._axes)}") from e


class PlotElemStore:
    def __init__(self):
        self._elems: dict[str, PlotElemDetails] = {}

    def add(self, key: str, details: PlotElemDetails):
        if key in self._elems:
            raise KeyError(f"Elem {key!r} already defined")
        self._elems[key] = details
        return self

    def set(self, key: str, details: PlotAxDetails):
        self._axes[key] = details
        return self

    def get(self, key: str) -> PlotElemDetails:
        try:
            return self._elems[key]
        except KeyError as e:
            raise KeyError(f"Unknown elem key {key!r}. Known: {list(self._elems)}") from e
