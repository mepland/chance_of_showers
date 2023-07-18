"""This module contains common plotting code."""
import matplotlib as mpl  # pylint: disable=import-error
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
import matplotlib.pyplot as plt  # noqa: E402 pylint: disable=import-error


########################################################
# Correctly setup types, kagrs, and docstring descriptions later if continuing to use this function
def plot_func(plot_objs_in, col_x, col_y, *, fig_size=(6, 5)) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
    """Plot scatter and linear fit.

    # noqa: DAR101
    Args:
       plot_objs_in: List of objects to plot.
       col_x: X axis label.
       col_y: Y axis label.
       fig_size: Figure size in inches.
    """
    plot_objs = dict(plot_objs_in)
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)

    for k, plot_obj in plot_objs.items():
        if plot_obj["type"] == "scatter":
            _ = ax.plot(
                plot_obj["x"],
                plot_obj["y"],
                marker=plot_obj["ms"],
                linestyle=plot_obj.get("ls", "None"),
                color=plot_obj["c"],
                label=plot_obj["label"],
            )
            plot_objs[k]["leg_object"] = _[0]  # pylint: disable=unnecessary-dict-index-lookup

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for k, plot_obj in plot_objs.items():
        if plot_obj["type"] == "linear_fit":
            x_min = plot_obj.get("x_min", xlim[0])
            x_max = plot_obj.get("x_max", xlim[1])
            x = np.linspace(x_min, x_max, 3)
            y = np.array(plot_obj["beta_0"] + plot_obj["beta_1"] * x)
            _ = ax.plot(
                x,
                y,
                marker="None",
                linestyle=plot_obj.get("ls", "-"),
                color=plot_obj["c"],
                label=plot_obj["label"],
            )
            plot_objs[k]["leg_object"] = _[0]  # pylint: disable=unnecessary-dict-index-lookup

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(col_x)
    ax.set_ylabel(col_y)

    leg_objects = [v["leg_object"] for v in plot_objs.values()]
    leg = ax.legend(
        leg_objects,
        [ob.get_label() for ob in leg_objects],
        fontsize=10,
        bbox_to_anchor=(1.02, 1),
        ncol=1,
        borderaxespad=0.0,
    )
    leg.get_frame().set_edgecolor("none")
    leg.get_frame().set_facecolor("none")

    fig.tight_layout()