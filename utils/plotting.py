"""This module contains common plotting code."""
import datetime
import os
from typing import TYPE_CHECKING, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

mpl.rcParams["axes.labelsize"] = 16
mpl.rcParams["xtick.top"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.labelsize"] = 13
mpl.rcParams["ytick.labelsize"] = 13
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.major.width"] = 0.8  # major tick width in points
mpl.rcParams["xtick.minor.width"] = 0.8  # minor tick width in points
mpl.rcParams["xtick.major.size"] = 7.0  # major tick size in points
mpl.rcParams["xtick.minor.size"] = 4.0  # minor tick size in points
mpl.rcParams["xtick.major.pad"] = 1.5  # distance to major tick label in points
mpl.rcParams["xtick.minor.pad"] = 1.4  # distance to the minor tick label in points
mpl.rcParams["ytick.major.width"] = 0.8  # major tick width in points
mpl.rcParams["ytick.minor.width"] = 0.8  # minor tick width in points
mpl.rcParams["ytick.major.size"] = 7.0  # major tick size in points
mpl.rcParams["ytick.minor.size"] = 4.0  # minor tick size in points
mpl.rcParams["ytick.major.pad"] = 1.5  # distance to major tick label in points
mpl.rcParams["ytick.minor.pad"] = 1.4  # distance to the minor tick label in points

########################################################
# Set common plot parameters
VSIZE: Final = 11  # inches
# aspect ratio width / height
ASPECT_RATIO_SINGLE: Final = 4.0 / 3.0

PLOT_PNG: Final = False
PNG_DPI: Final = 200

STD_ANN_X: Final = 0.80
STD_ANN_Y: Final = 0.95

STD_CMAP: Final = mpl.cm.get_cmap("plasma")

TOP_LINE_STD_ANN: Final = ""

# Define Chance of Showers style elements
C0: Final = "#012169"
C1: Final = "#993399"
MPL_C0: Final = "#1f77b4"
MPL_C1: Final = "#ff7f0e"
C_GREY: Final = "#7f7f7f"
C_GREEN: Final = "#A1B70D"
MS_FLOW_0: Final = "bowtie"
MS_FLOW_1: Final = "bowtie-open"
MC_FLOW_0: Final = C0
MC_FLOW_1: Final = C1
MARKER_SIZE_LARGE: Final = 12
MARKER_SIZE_SMALL: Final = 6


########################################################
# setup my own large number formatter for convenience and tweakability
def my_large_num_formatter(value: float, *, e_precision: int = 3) -> str:
    """Format large numbers.

    Args:
        value: Input value.
        e_precision: Precision passed to e formatter.

    Returns:
        Formated value.
    """
    if value > 1000:
        return f"{value:.{e_precision}e}"
    return f"{value:.0f}"


########################################################
def make_epoch_bins(
    dt_start: datetime.date, dt_stop: datetime.date, bin_size_seconds: int
) -> np.ndarray:
    """Make Unix epoch bins between endpoints, like linspace.

    Args:
        dt_start: Start datetime.
        dt_stop: Stop datetime.
        bin_size_seconds: Bin size in seconds.

    Returns:
        Numpy array of bin edges in epoch seconds between dt_start and dt_stop.
    """
    bin_min = dt_start.timestamp()  # type: ignore[attr-defined]
    bin_max = dt_stop.timestamp()  # type: ignore[attr-defined]
    nbins = int(round((bin_max - bin_min) / bin_size_seconds))
    return np.linspace(bin_min, bin_max, nbins + 1)


########################################################
def date_ann(dt_start: datetime.date | None, dt_stop: datetime.date | None) -> str:
    """Generate date range.

    Args:
        dt_start: Data start date.
        dt_stop: Data end date.

    Returns:
        Date range.
    """
    if not (
        isinstance(dt_start, (datetime.datetime, datetime.date))
        and isinstance(dt_stop, (datetime.datetime, datetime.date))
    ):
        return ""

    if dt_start.month == 1 and dt_start.day == 1 and dt_stop.month == 12 and dt_stop.day == 31:
        # interval is a whole number of years
        if dt_start.year == dt_stop.year:
            return str(dt_start.year)
        return f"{dt_start.year} - {dt_stop.year}"
    return f"{dt_start.strftime('%Y-%m-%d')} - {dt_stop.strftime('%Y-%m-%d')}"


########################################################
def ann_text_std(
    dt_start: datetime.date | None,
    dt_stop: datetime.date | None,
    *,
    ann_text_std_add: str | None = None,
    ann_text_hard_coded: str | None = None,
    gen_date: bool = False,
) -> str:
    """Generate standard annotation.

    Args:
        dt_start: Data start date.
        dt_stop: Data end date.
        ann_text_std_add: Text to add to the standard annotation.
        ann_text_hard_coded: Any hard coded text to add.
        gen_date: If date range should be generated.

    Returns:
        Standard annotation.
    """
    element_1 = ""
    if TOP_LINE_STD_ANN != "":
        element_1 = f"{TOP_LINE_STD_ANN}\n"

    date_ann_str = date_ann(dt_start, dt_stop)
    element_2 = ""
    if date_ann_str != "":
        element_2 = f"{date_ann_str}\n"

    element_3 = ""
    if gen_date:
        element_3 = f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d')}\n"  # noqa: DTZ005

    element_4 = ""
    if ann_text_hard_coded is not None and ann_text_hard_coded != "":
        element_4 = f"{ann_text_hard_coded}\n"

    element_5 = ""
    if ann_text_std_add is not None and ann_text_std_add != "":
        element_5 = f"{ann_text_std_add}"

    ann_str = f"{element_1}{element_2}{element_3}{element_4}{element_5}"

    return ann_str.strip("\n")


########################################################
def _setup_vars(
    ann_texts_in: list[dict] | None, x_axis_params: dict | None, y_axis_params: dict | None
) -> tuple[list[dict], dict, dict]:
    """Setup variables.

    Args:
        ann_texts_in: Input annotations.
        x_axis_params: X axis parameters.
        y_axis_params: Y axis parameters.

    Returns:
        Cleaned variables.
    """
    ann_texts = []
    if ann_texts_in is not None:
        ann_texts = list(ann_texts_in)
    if not isinstance(x_axis_params, dict):
        x_axis_params = {}
    if not isinstance(y_axis_params, dict):
        y_axis_params = {}
    return ann_texts, x_axis_params, y_axis_params


########################################################
def set_ax_limits(
    _ax: mpl.axes.Axes, x_axis_params: dict, y_axis_params: dict, *, allow_max_mult: bool = False
) -> None:
    """Set the axes limits.

    Args:
        _ax: Axes object.
        x_axis_params: X axis parameters.
        y_axis_params: Y axis parameters.
        allow_max_mult: Allow multiplying the y maximum to auto scale the y axis.
    """
    x_min_auto, x_max_auto = _ax.get_xlim()
    if (x_min := x_axis_params.get("min")) is None:
        x_min = x_min_auto
    if (x_max := x_axis_params.get("max")) is None:
        x_max = x_max_auto

    y_min_auto, y_max_auto = _ax.get_ylim()
    y_max_mult = None
    if allow_max_mult:
        y_max_mult = y_axis_params.get("max_mult")
    if (y_min := y_axis_params.get("min")) is None:
        y_min = y_min_auto

    if y_max_mult is not None:
        y_max = y_min + y_max_mult * (y_max_auto - y_min)
    elif (y_max := y_axis_params.get("max")) is None:
        y_max = y_max_auto

    _ax.set_xlim(x_min, x_max)
    _ax.set_ylim(y_min, y_max)

    if (x_ticks := x_axis_params.get("ticks")) is not None:
        _ax.set_xticks(x_ticks)
    if (y_ticks := y_axis_params.get("ticks")) is not None:
        _ax.set_yticks(y_ticks)


########################################################
def clean_ax(
    _ax: mpl.axes.Axes, x_axis_params: dict, y_axis_params: dict, *, turn_off_axes: bool = False
) -> None:
    """Clean the axes.

    Args:
        _ax: Axes object.
        x_axis_params: X axis parameters.
        y_axis_params: Y axis parameters.
        turn_off_axes: Turn off axes.
    """
    if turn_off_axes:
        _ax.axis("off")
    else:
        x_label = x_axis_params.get("axis_label", "")
        if x_label != "" and x_axis_params.get("units") is not None:
            x_label = f"{x_label} [{x_axis_params['units']}]"
        _ax.set_xlabel(x_label)
        y_label = y_axis_params.get("axis_label", "")
        if y_label != "" and y_axis_params.get("units") is not None:
            y_label = f"{y_label} [{y_axis_params['units']}]"
        _ax.set_ylabel(y_label)
        _ax.xaxis.label.set_size(20)
        _ax.yaxis.label.set_size(20)
        _ax.xaxis.set_tick_params(labelsize=15)
        _ax.yaxis.set_tick_params(labelsize=15)

        if x_axis_params.get("ticks") is not None:
            _ax.set_xticks(x_axis_params["ticks"])

        if x_axis_params.get("log", False):
            _ax.set_xscale("log")

        if y_axis_params.get("log", False):
            _ax.set_yscale("log")


########################################################
def draw_legend(fig: mpl.figure.Figure, leg_objects: list, legend_params: dict | None) -> None:
    """Draw the legend from objects.

    Example parameter values:
      legend_params = {"fontsize": None, "bbox_to_anchor": None, "loc": None, "ncol": None, "borderaxespad": None}

    Args:
        fig: figure object.
        leg_objects: List of legend objects.
        legend_params: Legend parameters.
    """
    if len(leg_objects) > 0:
        if not isinstance(legend_params, dict):
            legend_params = {}
        leg = fig.legend(
            leg_objects,
            [ob.get_label() for ob in leg_objects],
            fontsize=legend_params.get("fontsize", 18),
            bbox_to_anchor=legend_params.get("bbox_to_anchor", (0.7, 0.65, 0.2, 0.2)),
            loc=legend_params.get("loc", "upper center"),
            ncol=legend_params.get("ncol", 1),
            borderaxespad=legend_params.get("borderaxespad", 0.0),
        )
        leg.get_frame().set_edgecolor("none")
        leg.get_frame().set_facecolor("none")


########################################################
def ann_and_save(
    _fig: mpl.figure,
    ann_texts: list[dict],
    plot_inline: bool,  # noqa: FBT001
    m_path: str,
    fname: str,
    tag: str,
    *,
    ann_text_origin_x: float = STD_ANN_X,
    ann_text_origin_y: float = STD_ANN_Y,
    forced_text_size: int | None = None,
) -> None:
    """Annotate and save the plot.

    Args:
        _fig: Figure object.
        ann_texts: List of annotation dictionaries.
        plot_inline: Display plot inline in a notebook, or save to file.
        m_path: Path output directory for saved plots.
        fname: Plot output file name.
        tag: Tag to append to file name.
        ann_text_origin_x: Annotation origin in x on pad.
        ann_text_origin_y: Annotation origin in y on pad.
        forced_text_size: Override given text sizes per annotation and use this value for all annotations instead.
    """
    if ann_texts is not None:
        for text in ann_texts:
            if forced_text_size is not None:
                text_size = forced_text_size
            else:
                text_size = text.get("size", 18)

            plt.figtext(
                ann_text_origin_x + text.get("x", 0.0),
                ann_text_origin_y + text.get("y", 0.0),
                text.get("label", "MISSING"),
                ha=text.get("ha", "left"),
                va="top",
                size=text_size,
                backgroundcolor="white",
            )

    _fig.tight_layout()
    if not plot_inline:
        os.makedirs(m_path, exist_ok=True)
        if PLOT_PNG:
            _fig.savefig(f"{m_path}/{fname}{tag}.png", dpi=PNG_DPI)
        _fig.savefig(f"{m_path}/{fname}{tag}.pdf")
        plt.close("all")


########################################################
def _process_hist_binning(
    binning: dict | None,
    hist_values: list[float] | np.ndarray | pd.Series,
    *,
    current_bin_min: float | None = None,
    current_bin_max: float | None = None,
) -> tuple[list[float], int, str]:
    """Process binning dict for histograms.

    Args:
        binning: Binning parameters.
        hist_values: Values for filling histogram
        current_bin_min: Minimum bin value.
        current_bin_max: Maximum bin value.

    Raises:
        ValueError: Bad configuration.

    Returns:
        cleaned binning parameter variables.
    """
    if binning is None:
        binning = {"nbins": 10}

    bin_edges = binning.get("bin_edges", [])
    nbins = binning.get("nbins")
    bin_size = binning.get("bin_size")
    bin_size_str_fmt = binning.get("bin_size_str_fmt", ".2f")

    if current_bin_min is not None:
        bin_min = min(current_bin_min, min(hist_values))  # pylint: disable=nested-min-max
    else:
        bin_min = min(hist_values)

    if current_bin_max is not None:
        bin_max = max(current_bin_max, max(hist_values))  # pylint: disable=nested-min-max
    else:
        bin_max = max(hist_values)

    if isinstance(bin_edges, (list, np.ndarray)) and len(bin_edges) >= 2:
        # possibly variable size bins from bin_edges
        nbins = len(bin_edges) - 1
        bin_edges = np.array(bin_edges)
        if bin_size is not None:
            bin_size_str = f"{bin_size:{bin_size_str_fmt}}"
        else:
            bin_size_str = "Variable"
    elif bin_size is not None and bin_size > 0.0:
        # fixed bin_size
        nbins = int(round((bin_max - bin_min) / bin_size))
        bin_edges = np.linspace(bin_min, bin_max, nbins + 1)
        bin_size = (bin_max - bin_min) / nbins
        bin_size_str = f"{bin_size:{bin_size_str_fmt}}"
    elif nbins is not None and nbins > 0:
        # fixed number of bins
        bin_edges = np.linspace(bin_min, bin_max, nbins + 1)
        bin_size = (bin_max - bin_min) / nbins
        bin_size_str = f"{bin_size:{bin_size_str_fmt}}"
    else:
        print(binning)
        raise ValueError("Can not work with this binning dict!")

    return bin_edges, nbins, bin_size_str


########################################################
def plot_hists(  # noqa: C901 pylint: disable=too-many-locals, too-many-function-args, too-many-statements
    hist_dicts: list[dict],
    *,
    m_path: str,
    fname: str = "hist",
    tag: str = "",
    dt_start: datetime.date | None = None,
    dt_stop: datetime.date | None = None,
    plot_inline: bool = False,
    ann_text_std_add: str | None = None,
    ann_texts_in: list[dict] | None = None,
    binning: dict | None = None,
    x_axis_params: dict | None = None,
    y_axis_params: dict | None = None,
    legend_params: dict | None = None,
    reference_lines: list[dict] | None = None,
) -> None:
    """Plot histograms.

    Example parameter values:
      Standard hist_dicts = [{"values": , "weights": None, "label": None, "histtype": "step", "stacked": False, "density": False, "c": None, "lw": 2}]
      Precomputed hist_dicts = [{"hist_data": {"bin_edges": [], "hist": []}, ...}] AND binning = {"use_hist_data": True}
      Bar graph hist_dicts = [{"plot_via_bar": False, "fill": True, "ec": None, "ls": None, "label_values": False}]
      ann_texts_in = [{"label": "Hello", "x": 0.0, "y": 0.0, "ha": "center", "size": 18}]
      binning = {"nbins": 10}, or {"bin_edges": [0,1,2]}, or {"bin_size": 100}, as well as {"bin_size_str_fmt": ".2f"}
      x_axis_params = {"axis_label": None, "min": None, "max": None, "units": "", "log": False}
      y_axis_params = {"axis_label": None, "min": None, "max": None, "max_mult": None, "log": False, "show_bin_size": True}
      legend_params = {"fontsize": None, "bbox_to_anchor": None, "loc": None, "ncol": None, "borderaxespad": None}
      reference_lines = [{"label": None, "orientation": "v", "value": 100.0, "c": "c0", "lw": 2, "ls": "-"}]

    Args:
        hist_dicts: List of histogram dictionaries.
        m_path: Path output directory for saved plots.
        fname: Plot output file name.
        tag: Tag to append to file name.
        dt_start: Data start date.
        dt_stop: Data end date.
        plot_inline: Display plot inline in a notebook, or save to file.
        ann_text_std_add: Text to add to the standard annotation.
        ann_texts_in: List of annotation dictionaries.
        binning: Binning parameters.
        x_axis_params: X axis parameters.
        y_axis_params: Y axis parameters.
        legend_params: Legend parameters.
        reference_lines: List of reference line dicts.

    Raises:
        ValueError: Bad configuration.
    """
    ann_texts, x_axis_params, y_axis_params = _setup_vars(
        ann_texts_in, x_axis_params, y_axis_params
    )

    if binning is None:
        binning = {"nbins": 10}
    x_bin_edges = binning.get("bin_edges", [])

    for i_hist_dict, hist_dict in enumerate(hist_dicts):
        if len(hist_dict.get("values", [])) > 0:
            x_values = hist_dict["values"]
        elif len(hist_dict.get("hist_data", {}).get("bin_edges", [])) > 0:
            x_values = hist_dict["hist_data"]["bin_edges"]

            if x_bin_edges == [] and binning.get("use_hist_data", False):
                x_bin_edges = list(x_values)

        else:
            raise ValueError(
                "Should not end up here, re-evaluate your hist_dicts and binning inputs!"
            )

        if i_hist_dict == 0:
            x_bin_min = min(x_values)
            x_bin_max = max(x_values)
        else:
            x_bin_min = min(x_bin_min, min(x_values))  # pylint: disable=nested-min-max
            x_bin_max = max(x_bin_max, max(x_values))  # pylint: disable=nested-min-max

    binning["bin_edges"] = x_bin_edges
    x_bin_edges, x_nbins, x_bin_size_str = _process_hist_binning(
        binning, x_values, current_bin_min=x_bin_min, current_bin_max=x_bin_max
    )

    leg_objects = []

    fig, ax = plt.subplots(num=fname)
    fig.set_size_inches(ASPECT_RATIO_SINGLE * VSIZE, VSIZE)

    for hist_dict in hist_dicts:
        if len(hist_dict.get("values", [])) > 0:
            _hist = hist_dict["values"]
            _bins = x_bin_edges
            _weights = hist_dict.get("weights")
        elif len(hist_dict.get("hist_data", {}).get("bin_edges", [])) > 0:
            # results are already binned, so fake the input by giving 1 count to the middle of each bin, then multiplying by the appropriate weight
            _bin_edges = hist_dict["hist_data"]["bin_edges"]
            if list(x_bin_edges) != list(_bin_edges):
                print(
                    "Warning this hist_data dict does not have the same bin edges as the first, not expected! Will try to continue but bins are not going to line up and may be beyond the axis range"
                )
                print(hist_dict)

            _hist = []
            for ibin in range(len(_bin_edges) - 1):
                _bin_min = _bin_edges[ibin]
                _bin_max = _bin_edges[ibin + 1]
                _hist.append(_bin_min + 0.5 * (_bin_max - _bin_min))

            _bins = _bin_edges
            _weights = hist_dict["hist_data"]["hist"]

        if not hist_dict.get("plot_via_bar", False):
            _plotted_hist, _plotted_bin_edges, _plotted_patches = ax.hist(
                _hist,
                bins=_bins,
                weights=_weights,
                label=hist_dict.get("label"),
                histtype=hist_dict.get("histtype", "step"),
                stacked=hist_dict.get("stacked", False),
                density=hist_dict.get("density", False),
                log=y_axis_params.get("log", False),
                color=hist_dict.get("c"),
                linewidth=hist_dict.get("lw", 2),
            )

            _label = _plotted_patches[0].get_label()
            if _label is not None and _label != "":
                leg_objects.append(_plotted_patches[0])

        else:
            # plot via ax.bar instead of ax.hist - better for some use cases with variable bins
            if TYPE_CHECKING:
                assert isinstance(_weights, list)  # noqa: SCS108 # nosec assert_used

            if len(_bins) - 1 != len(_weights):
                raise ValueError("Need to write numpy code to histogram the values")

            _nbins = len(_bins) - 1
            x_axis_labels = []
            x_axis_ticks = np.arange(_nbins)
            for i in range(_nbins):
                upper_inequality = "$<$"
                if i == x_nbins - 1:
                    upper_inequality = r"$\leq$"
                x_axis_labels.append(
                    r"{low} $\leq$ {var} {upper_inequality} {high}".format(  # pylint: disable=consider-using-f-string
                        low=my_large_num_formatter(_bins[i], e_precision=0),
                        high=my_large_num_formatter(_bins[i + 1], e_precision=0),
                        var=x_axis_params.get("axis_label", "Binned Variable"),
                        upper_inequality=upper_inequality,
                    )
                )

            hist_bin_values = np.array(_weights)
            if hist_dict.get("density", False):
                hist_bin_values = np.divide(hist_bin_values, float(sum(hist_bin_values)))

            _label = hist_dict.get("label")
            ax.bar(
                x_axis_ticks,
                hist_bin_values,
                width=0.5,
                label=_label,
                log=y_axis_params.get("log", False),
                color=hist_dict.get("c"),
                linewidth=hist_dict.get("lw", 2),
                fill=hist_dict.get("fill", True),
                edgecolor=hist_dict.get("ec"),
                ls=hist_dict.get("ls"),
            )
            if _label is not None:
                handles, labels = ax.get_legend_handles_labels()
                leg_objects.append(handles[labels.index(_label)])

            ax.set_xticklabels(x_axis_labels, rotation=45, ha="right")
            ax.set_xticks(x_axis_ticks)
            ax.tick_params(axis="x", which="both", length=0)

            if hist_dict.get("label_values", False):
                rects = ax.patches
                for rect, label in zip(rects, hist_bin_values, strict=True):
                    height = rect.get_height()
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        height,
                        my_large_num_formatter(label, e_precision=0),
                        ha="center",
                        va="bottom",
                    )

    # plot reference lines
    if reference_lines is not None:
        for reference_line in reference_lines:
            _label = reference_line.get("label")
            _kwargs = {
                "label": _label,
                "color": reference_line.get("c", "black"),
                "linewidth": reference_line.get("lw", 2),
                "linestyle": reference_line.get("ls", "-"),
            }

            if reference_line["orientation"] == "v":
                line_2d = ax.axvline(x=reference_line["value"], **_kwargs)
            elif reference_line["orientation"] == "h":
                line_2d = ax.axhline(y=reference_line["value"], **_kwargs)
            else:
                raise ValueError(f"Bad orientation= {reference_line.get('orientation')}!")

            if _label is not None and _label != "":
                leg_objects.append(line_2d)

    y_label = y_axis_params.get("axis_label", "$N$")
    x_units = x_axis_params.get("units")
    if y_axis_params.get("show_bin_size", True):
        if x_units is not None:
            y_label = f"{y_label} / {x_bin_size_str} [{x_units}]"
        else:
            y_label = f"{y_label} / {x_bin_size_str}"

    y_axis_params["axis_label"] = y_label

    clean_ax(ax, x_axis_params, y_axis_params)
    set_ax_limits(ax, x_axis_params, y_axis_params, allow_max_mult=True)
    draw_legend(fig, leg_objects, legend_params)

    ann_texts.append(
        {
            "label": ann_text_std(dt_start, dt_stop, ann_text_std_add=ann_text_std_add),
            "ha": "center",
        }
    )
    ann_and_save(fig, ann_texts, plot_inline, m_path, fname, tag)


########################################################
def plot_2d_hist(  # noqa: C901  pylint: disable=too-many-locals
    x_values: list[float] | np.ndarray | pd.Series,
    y_values: list[float] | np.ndarray | pd.Series,
    *,
    m_path: str,
    fname: str = "hist_2d",
    tag: str = "",
    dt_start: datetime.date | None = None,
    dt_stop: datetime.date | None = None,
    plot_inline: bool = False,
    ann_text_std_add: str | None = None,
    ann_texts_in: list[dict] | None = None,
    binning: dict | None = None,
    x_axis_params: dict | None = None,
    y_axis_params: dict | None = None,
    z_axis_params: dict | None = None,
    legend_params: dict | None = None,
    reference_lines: list[dict] | None = None,
) -> None:
    """Plot 2D histograms.

    Example parameter values:
      ann_texts_in = [{"label": "Hello", "x": 0.0, "y": 0.0, "ha": "center", "size": 18}]
      binning = {"x": 1d_binning_dict, "y": 1d_binning_dict}, where 1d_binning_dict is a plot_hist binning dictionary
      x_axis_params = {"is_datetime": False, "axis_label": None, "min": None, "max": None, "units": "", "log": False}
      y_axis_params = {"is_datetime": False, "axis_label": None, "min": None, "max": None, "units": "", "log": False}
      z_axis_params = {"axis_label": None, "min": None, "max": None, "norm": None, "show_bin_size": True, "density": False}
      legend_params = {"fontsize": None, "bbox_to_anchor": None, "loc": None, "ncol": None, "borderaxespad": None}
      reference_lines = [{"label": None, "orientation": "v", "value": 100.0, "c": "c0", "lw": 2, "ls": "-"}]

    Args:
        x_values: X values for filling histogram.
        y_values: Y values for filling histogram.
        m_path: Path output directory for saved plots.
        fname: Plot output file name.
        tag: Tag to append to file name.
        dt_start: Data start date.
        dt_stop: Data end date.
        plot_inline: Display plot inline in a notebook, or save to file.
        ann_text_std_add: Text to add to the standard annotation.
        ann_texts_in: List of annotation dictionaries.
        binning: Binning parameters, for x and y axes.
        x_axis_params: X axis parameters.
        y_axis_params: Y axis parameters.
        z_axis_params: Z axis parameters.
        legend_params: Legend parameters.
        reference_lines: List of reference line dicts.

    Raises:
        ValueError: Bad configuration.
    """
    ann_texts, x_axis_params, y_axis_params = _setup_vars(
        ann_texts_in, x_axis_params, y_axis_params
    )
    if not isinstance(z_axis_params, dict):
        z_axis_params = {}

    z_norm = z_axis_params.get("norm")
    if z_norm == "log":
        z_norm = mpl.colors.LogNorm()
    elif z_norm is not None:
        raise ValueError("Unknown Norm!")

    from_datetime_to_epoch = np.vectorize(datetime.datetime.timestamp)
    from_epoch_to_datetime = np.vectorize(datetime.datetime.fromtimestamp)

    if x_axis_params.get("is_datetime", False):
        x_values = from_datetime_to_epoch(np.array(x_values))
    if y_axis_params.get("is_datetime", False):
        y_values = from_datetime_to_epoch(np.array(y_values))

    if binning is None:
        binning = {"x": {"nbins": 10}, "y": {"nbins": 10}}
    x_bin_edges, _, x_bin_size_str = _process_hist_binning(binning["x"], x_values)
    y_bin_edges, _, y_bin_size_str = _process_hist_binning(binning["y"], y_values)

    leg_objects = []

    fig, ax = plt.subplots(num=fname)
    fig.set_size_inches(ASPECT_RATIO_SINGLE * VSIZE, VSIZE)

    _plotted_hist, _plotted_x_edges, _plotted_y_edges, _plotted_image = ax.hist2d(
        x_values,
        y_values,
        bins=[x_bin_edges, y_bin_edges],
        density=z_axis_params.get("density", False),
        cmin=z_axis_params.get("min"),
        cmax=z_axis_params.get("max"),
        norm=z_norm,
    )

    z_label = z_axis_params.get("axis_label", "$N$")
    x_units = x_axis_params.get("units")
    y_units = y_axis_params.get("units")
    if z_axis_params.get("show_bin_size", True):
        x_unit_part = ""
        if x_units is not None:
            x_unit_part = f" [{x_units}]"
        y_unit_part = ""
        if y_units is not None:
            y_unit_part = f" [{y_units}]"
        z_label = f"{z_label} / {x_bin_size_str}{x_unit_part} x {y_bin_size_str}{y_unit_part}"

    fig.colorbar(_plotted_image, ax=ax, label=z_label, cmap=STD_CMAP)

    clean_ax(ax, x_axis_params, y_axis_params)
    set_ax_limits(ax, x_axis_params, y_axis_params)

    # plot reference lines
    if reference_lines is not None:
        for reference_line in reference_lines:
            _label = reference_line.get("label")
            _kwargs = {
                "label": _label,
                "color": reference_line.get("c", "black"),
                "linewidth": reference_line.get("lw", 2),
                "linestyle": reference_line.get("ls", "-"),
            }

            if reference_line["orientation"] == "v":
                line_2d = ax.axvline(x=reference_line["value"], **_kwargs)
            elif reference_line["orientation"] == "h":
                line_2d = ax.axhline(y=reference_line["value"], **_kwargs)
            else:
                raise ValueError(f"Bad orientation= {reference_line.get('orientation')}!")

            if _label is not None and _label != "":
                leg_objects.append(line_2d)

    if x_axis_params.get("is_datetime", False):
        epoch_xticks = ax.get_xticks()
        datetime_xticks = from_epoch_to_datetime(epoch_xticks)
        if x_axis_params.get("tick_format", False):
            datetime_xticks = [_.strftime(x_axis_params["tick_format"]) for _ in datetime_xticks]
        ax.set_xticks(epoch_xticks, datetime_xticks)
        fig.autofmt_xdate()

    if y_axis_params.get("is_datetime", False):
        epoch_yticks = ax.get_yticks()
        datetime_yticks = from_epoch_to_datetime(epoch_yticks)
        if y_axis_params.get("tick_format", False):
            datetime_yticks = [_.strftime(y_axis_params["tick_format"]) for _ in datetime_yticks]
        ax.set_yticks(epoch_yticks, datetime_yticks)
        fig.autofmt_ydate()

    draw_legend(fig, leg_objects, legend_params)

    ann_texts.append(
        {
            "label": ann_text_std(dt_start, dt_stop, ann_text_std_add=ann_text_std_add),
            "ha": "center",
        }
    )
    ann_and_save(
        fig, ann_texts, plot_inline, m_path, fname, tag, ann_text_origin_x=STD_ANN_X - 0.12
    )


########################################################
def plot_chance_of_showers_timeseries(  # pylint: disable=too-many-locals
    dfp_in: pd.DataFrame,
    x_axis_params: dict,
    y_axis_params: dict,
    *,
    z_axis_params: dict | None = None,
    m_path: str = ".",
    fname: str = "ts",
    tag: str = "",
    dt_start: datetime.date | None = None,
    dt_stop: datetime.date | None = None,
    plot_inline: bool = True,
    ann_text_std_add: str | None = None,
    ann_texts_in: list[dict] | None = None,
    reference_lines: list[dict] | None = None,
    standard_flow_legend: bool = True,
) -> None:
    """Plot time series for the Chance of Showers project.

    Example parameter values:
    x_axis_params={"col": "datetime", "axis_label": "Datetime", "hover_label": "Date: %{x:" + DATETIME_FMT + "}", "type": "date", "min": dt_start, "max": dt_stop, "rangeselector_buttons": True, "rangeslider": True, "fig_width": None}
    y_axis_params={"col": "pressure", "axis_label": "Pressure", "hover_label": "Pressure: %{y:d}", "mode": "lines+markers", "hoverformat": "d", "fig_height": 500}
    z_axis_params={"col": "flow", "hover_label": "Flow: %{customdata:df}", "C0_condition": 1},
    reference_lines=[ {"orientation": "h", "value": 0, "c": "black"}]

    Args:
        dfp_in: Pandas dataframe containing the needed data.
        x_axis_params: X axis parameters.
        y_axis_params: Y axis parameters.
        z_axis_params: Z axis parameters.
        m_path: Path output directory for saved plots.
        fname: Plot output file name.
        tag: Tag to append to file name.
        dt_start: Data start date.
        dt_stop: Data end date.
        plot_inline: Display plot inline in a notebook, or save to file.
        ann_text_std_add: Text to add to the standard annotation.
        ann_texts_in: List of annotation dictionaries.
        reference_lines: List of reference line dicts.
        standard_flow_legend: Use standard "No Flow"/"Had Flow" legend entries, instead of trace legend entries.

    Raises:
        ValueError: Bad configuration.
    """
    if z_axis_params is None:
        z_axis_params = {}

    x_col = x_axis_params["col"]
    y_col = y_axis_params["col"]
    z_col = z_axis_params.get("col")

    if z_col is None:
        dfp = dfp_in[[x_col, y_col]].copy()
        dfp["ms"] = "circle"
        dfp["mc"] = C0
    else:
        dfp = dfp_in[[x_col, y_col, z_col]].copy()
        dfp["ms"] = dfp[z_col].apply(
            lambda x: MS_FLOW_0 if x != z_axis_params.get("C0_condition", 1) else MS_FLOW_1
        )
        dfp["mc"] = dfp[z_col].apply(
            lambda x: MC_FLOW_0 if x != z_axis_params.get("C0_condition", 1) else MC_FLOW_1
        )

    y_trace = {
        "x": dfp[x_col],
        "y": dfp[y_col],
        "type": "scatter",
        "mode": y_axis_params.get("mode", "lines+markers"),
        "marker": {
            "color": dfp["mc"],
            "size": MARKER_SIZE_SMALL,
            "line": {
                "width": 1.5,
                "color": dfp["mc"],
            },
            "symbol": dfp["ms"],
        },
        "line": {"width": 1.0},
        "showlegend": not standard_flow_legend,
    }

    hovertemplate = (
        f"{x_axis_params.get('hover_label', '')}<br>{y_axis_params.get('hover_label', '')}"
    )

    if z_col is not None:
        y_trace["customdata"] = dfp[z_col]
        if z_axis_params.get("hover_label") is not None:
            hovertemplate += f"<br>{z_axis_params['hover_label']}<extra></extra>"

    y_trace["hovertemplate"] = hovertemplate

    trace_layout = {
        "xaxis": {
            "title": x_axis_params.get("axis_label", ""),
            "zeroline": False,
            "type": x_axis_params.get("type", "date"),
            "gridcolor": "lightgrey",
            "range": [x_axis_params.get("min"), x_axis_params.get("max")],
        },
        "yaxis": {
            "title": y_axis_params.get("axis_label", ""),
            "zeroline": False,
            "hoverformat": y_axis_params.get("hoverformat", "d"),
            "gridcolor": "lightgrey",
        },
        "colorway": [C0],
        "plot_bgcolor": "white",
        "font": {
            "color": C_GREY,
            "size": 14,
        },
        "showlegend": True,
        "legend": {
            "orientation": "h",
            "xanchor": "right",
            "yanchor": "bottom",
            "x": 1.0,
            "y": 1.0,
        },
        "margin": {
            "t": 30,
            "b": 45,
            "l": 10,
            "r": 10,
        },
        "width": x_axis_params.get("fig_width"),
        "height": y_axis_params.get("fig_height", 500),
        "autosize": True,
    }

    rangeselector_buttons = x_axis_params.get("rangeselector_buttons", True)

    if (isinstance(rangeselector_buttons, bool) and rangeselector_buttons) or (
        isinstance(rangeselector_buttons, list) and 0 < len(rangeselector_buttons)
    ):
        trace_layout["xaxis"]["rangeselector"] = {"buttons": []}
        known_buttons = {
            "15m": {"count": 15, "label": "15m", "step": "minute", "stepmode": "todate"},
            "1h": {"count": 1, "label": "1h", "step": "hour", "stepmode": "todate"},
            "12h": {"count": 12, "label": "12h", "step": "hour", "stepmode": "todate"},
            "1d": {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
            "1w": {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
            "1m": {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
            "6m": {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
            "YTD": {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
            "1y": {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
            "all": {"step": "all"},
        }
        if isinstance(rangeselector_buttons, bool) and rangeselector_buttons:
            rangeselector_buttons = list(known_buttons.keys())

        if TYPE_CHECKING:
            assert isinstance(rangeselector_buttons, list)  # noqa: SCS108 # nosec assert_used

        for rangeselector_button in rangeselector_buttons:
            if rangeselector_button in known_buttons:
                trace_layout["xaxis"]["rangeselector"]["buttons"].append(
                    known_buttons[rangeselector_button]
                )
            else:
                raise ValueError(f"Unknown {rangeselector_button = }!")

    if x_axis_params.get("rangeslider", True):
        trace_layout["xaxis"]["rangeslider"] = {
            "visible": True,
            "bordercolor": "lightgrey",
            "borderwidth": 1,
            "thickness": 0.05,
        }

    if reference_lines is not None:
        trace_layout["shapes"] = []
        for reference_line in reference_lines:
            if reference_line["orientation"] == "v":
                shape_coords = {
                    "xref": "x",
                    "x0": reference_line["value"],
                    "x1": reference_line["value"],
                    "yref": "paper",
                    "y0": 0,
                    "y1": 1,
                }
            elif reference_line["orientation"] == "h":
                shape_coords = {
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "yref": "y",
                    "y0": reference_line["value"],
                    "y1": reference_line["value"],
                }
            else:
                raise ValueError(f"Bad orientation= {reference_line.get('orientation')}!")

            final_shape = {
                "type": "line",
                **shape_coords,
                "line": {
                    "color": reference_line.get("c", C_GREY),
                    "width": reference_line.get("lw", 1.5),
                    "dash": reference_line.get("ls", "dash"),
                },
            }

            trace_layout["shapes"].append(final_shape)

    fig = go.Figure()
    fig.add_trace(go.Scattergl(y_trace))
    fig.update_layout(trace_layout)

    # traces for legend entries
    if standard_flow_legend and z_col is not None:
        legend_entry_trace_flow_0 = {
            "x": [None],
            "y": [None],
            "name": "No Flow",
            "type": "scatter",
            "mode": "markers",
            "marker": {
                "size": MARKER_SIZE_LARGE,
                "line": {
                    "width": 1.5,
                    "color": MC_FLOW_0,
                },
                "symbol": MS_FLOW_0,
                "color": MC_FLOW_0,
            },
        }
        legend_entry_trace_flow_1 = {
            "x": [None],
            "y": [None],
            "name": "Had Flow",
            "type": "scatter",
            "mode": "markers",
            "marker": {
                "size": MARKER_SIZE_LARGE,
                "line": {
                    "width": 1.5,
                    "color": MC_FLOW_1,
                },
                "symbol": MS_FLOW_1,
                "color": MC_FLOW_1,
            },
        }
        fig.add_traces(
            [go.Scatter(legend_entry_trace_flow_0), go.Scatter(legend_entry_trace_flow_1)]
        )

    if (
        dt_start is not None
        or dt_stop is not None
        or ann_text_std_add is not None
        or ann_texts_in is not None
    ):
        raise ValueError(
            "Have not written annotation function yet!",
            dt_start,
            dt_stop,
            ann_text_std_add,
            ann_texts_in,
        )

    if plot_inline:
        fig.show()
    else:
        raise ValueError("Have not written save function yet!", m_path, fname, tag)
