"""This module contains common plotting code."""
import datetime
import os
from typing import TYPE_CHECKING, Final

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

TOP_LINE_STD_ANN: Final = ""


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
    x_min = x_axis_params.get("min")
    x_max = x_axis_params.get("max")
    if x_min is None:
        x_min = x_min_auto
    if x_max is None:
        x_max = x_max_auto

    y_min_auto, y_max_auto = _ax.get_ylim()
    y_min = y_axis_params.get("min")
    y_max = y_axis_params.get("max")
    y_max_mult = None
    if allow_max_mult:
        y_max_mult = y_axis_params.get("max_mult")
    if y_min is None:
        y_min = y_min_auto

    if y_max_mult is not None:
        y_max = y_min + y_max_mult * (y_max_auto - y_min)
    elif y_max is None:
        y_max = y_max_auto

    _ax.set_xlim(x_min, x_max)
    _ax.set_ylim(y_min, y_max)


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
        _ax.set_xlabel(x_axis_params.get("axis_label", ""))
        _ax.set_ylabel(y_axis_params.get("axis_label", ""))
        _ax.xaxis.label.set_size(20)
        _ax.yaxis.label.set_size(20)
        _ax.xaxis.set_tick_params(labelsize=15)
        _ax.yaxis.set_tick_params(labelsize=15)

        if x_axis_params.get("log", False):
            _ax.set_xscale("log")

        if y_axis_params.get("log", False):
            _ax.set_yscale("log")


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
) -> None:
    """Plot histograms.

    Example parameter values:
      Standard hist_dict = {'values': , 'weights': None, 'label': None, 'histtype': 'step', 'stacked': False, 'density': False, 'c': None, 'lw': 2}
      Precomputed hist_dict = {'hist_data': {'bin_edges': [], 'hist': []}, ...} AND binning = {'use_hist_data': True}
      Bar graph hist_dict = {'plot_via_bar': False, 'fill': True, 'ec': None, 'ls': None, 'label_values': False}
      ann_texts_in = {"label": "Hello", "x": 0.0, "y": 0.0, "ha": "center", "size": 18}
      binning = {"nbins": 10}, or {"bin_edges": [0,1,2]}, or {"bin_size": 100}, as well as {"bin_size_str_fmt": ".2f"}
      x_axis_params = {'axis_label': None, 'min': None, 'max': None, 'units': '', 'log': False}
      y_axis_params = {'axis_label': None, 'min': None, 'max': None, 'max_mult': None, 'log': False, 'show_bin_size': True}

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

    Raises:
        ValueError: Bad configuration.
    """
    ann_texts, x_axis_params, y_axis_params = _setup_vars(
        ann_texts_in, x_axis_params, y_axis_params
    )
    x_axis_params["axis_label"] = x_axis_params.get("axis_label", "Bins")

    if binning is None:
        binning = {"nbins": 10}

    bin_edges = binning.get("bin_edges", [])
    nbins = binning.get("nbins")
    bin_size = binning.get("bin_size")
    bin_size_str_fmt = binning.get("bin_size_str_fmt", ".2f")

    for i_hist_dict, hist_dict in enumerate(hist_dicts):
        if len(hist_dict.get("values", [])) > 0:
            _values = hist_dict["values"]
        elif len(hist_dict.get("hist_data", {}).get("bin_edges", [])) > 0:
            _values = hist_dict["hist_data"]["bin_edges"]

            if bin_edges == [] and binning.get("use_hist_data", False):
                bin_edges = list(_values)

        else:
            raise ValueError(
                "Should not end up here, re-evaluate your hist_dicts and binning inputs!"
            )

        if i_hist_dict == 0:
            _bin_min = min(_values)
            _bin_max = max(_values)
        else:
            _bin_min = min(_bin_min, min(_values))  # pylint: disable=nested-min-max
            _bin_max = max(_bin_max, max(_values))  # pylint: disable=nested-min-max

    _bin_min = binning.get("min", _bin_min)
    _bin_max = binning.get("max", _bin_max)

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
        nbins = int(round((_bin_max - _bin_min) / bin_size))
        bin_edges = np.linspace(_bin_min, _bin_max, nbins + 1)
        bin_size = (_bin_max - _bin_min) / nbins
        bin_size_str = f"{bin_size:{bin_size_str_fmt}}"
    elif nbins is not None and nbins > 0:
        # fixed number of bins
        bin_edges = np.linspace(_bin_min, _bin_max, nbins + 1)
        bin_size = (_bin_max - _bin_min) / nbins
        bin_size_str = f"{bin_size:{bin_size_str_fmt}}"
    else:
        print(binning)
        raise ValueError("Can not work with this binning dict!")

    leg_objects = []

    fig, ax = plt.subplots(num=fname)
    fig.set_size_inches(ASPECT_RATIO_SINGLE * VSIZE, VSIZE)

    for hist_dict in hist_dicts:
        if len(hist_dict.get("values", [])) > 0:
            _hist = hist_dict["values"]
            _bins = bin_edges
            _weights = hist_dict.get("weights")
        elif len(hist_dict.get("hist_data", {}).get("bin_edges", [])) > 0:
            # results are already binned, so fake the input by giving 1 count to the middle of each bin, then multiplying by the appropriate weight
            _bin_edges = hist_dict["hist_data"]["bin_edges"]
            if list(bin_edges) != list(_bin_edges):
                print(
                    "Warning this hist_data dict does not have the same bin edges as the first, not expected! Will try to continue but bins are not going to line up and may be beyond the axis range"
                )
                print(hist_dict)

            _hist = []
            for ibin in range(len(_bin_edges) - 1):
                bin_min = _bin_edges[ibin]
                bin_max = _bin_edges[ibin + 1]
                _hist.append(bin_min + 0.5 * (bin_max - bin_min))

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
                if i == nbins - 1:
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

    y_label = y_axis_params.get("axis_label", "$N$")
    if y_axis_params.get("show_bin_size", True):
        y_label = f"{y_label} / {bin_size_str}"

    x_units = x_axis_params.get("units", "")
    if x_units is not None and x_units != "":
        y_label = f"{y_label} [{x_units}]"

    y_axis_params["axis_label"] = y_label

    clean_ax(ax, x_axis_params, y_axis_params)
    set_ax_limits(ax, x_axis_params, y_axis_params, allow_max_mult=True)

    if len(leg_objects) > 0:
        leg = fig.legend(
            leg_objects,
            [ob.get_label() for ob in leg_objects],
            fontsize=18,
            bbox_to_anchor=(0.7, 0.65, 0.2, 0.2),
            loc="upper center",
            ncol=1,
            borderaxespad=0.0,
        )
        leg.get_frame().set_edgecolor("none")
        leg.get_frame().set_facecolor("none")

    ann_texts.append(
        {
            "label": ann_text_std(dt_start, dt_stop, ann_text_std_add=ann_text_std_add),
            "ha": "center",
        }
    )
    ann_and_save(fig, ann_texts, plot_inline, m_path, fname, tag)
