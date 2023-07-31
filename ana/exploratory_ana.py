# %%
"""Analysis notebook."""
# %% [markdown]
# # Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %matplotlib inline

# import datetime
# import glob
# import natsort
# import pprint
# import zoneinfo
# import numpy as np
import os
import sys
from typing import Final

import pandas as pd
import plotly.graph_objects as go
from hydra import compose, initialize
from IPython.display import display

# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.ar_model import ar_select_order
# from statsmodels.graphics.api import qqplot


sys.path.append(os.path.dirname(os.path.realpath("")))
from utils.plotting import plot_hists  # noqa: E402 # pylint: disable=import-error
from utils.shared_functions import (  # noqa: E402 # pylint: disable=import-error
    normalize_pressure_value,
)

# %%
initialize(version_base=None, config_path="..")
cfg = compose(config_name="config")

OBSERVED_PRESSURE_MIN: Final = cfg["general"]["observed_pressure_min"]
OBSERVED_PRESSURE_MAX: Final = cfg["general"]["observed_pressure_max"]

PACKAGE_PATH: Final = cfg["general"]["package_path"]
SAVED_DATA_RELATIVE_PATH: Final = cfg["etl"]["saved_data_relative_path"]

DATE_FMT: Final = cfg["general"]["date_fmt"]
TIME_FMT: Final = cfg["general"]["time_fmt"]
FNAME_DATETIME_FMT: Final = cfg["general"]["fname_datetime_fmt"]
DATETIME_FMT: Final = f"{DATE_FMT} {TIME_FMT}"

LOCAL_TIMEZONE_STR: Final = cfg["general"]["local_timezone"]

# if LOCAL_TIMEZONE_STR not in zoneinfo.available_timezones():
#     AVAILABLE_TIMEZONES: Final = "\n".join(list(zoneinfo.available_timezones()))
#     raise ValueError(f"Unknown {LOCAL_TIMEZONE_STR = }, choose from:\n{AVAILABLE_TIMEZONES}")

# UTC_TIMEZONE: Final = zoneinfo.ZoneInfo("UTC")

# %%
# marker_styles = ['o', '^', 'v', '+']

# %%
# # https://stackoverflow.com/a/59866006
# from IPython.display import display, HTML

# def force_show_all(dfp):
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
#         display(HTML(dfp.to_html()))

# %%
# force_show_all(dfp_data)

# %% [markdown]
# ***
# # Load Data

# %%
FNAME_PARQUET: Final = "data_2023-04-27-03-00-04_to_2023-07-28-23-50-00.parquet"

F_PARQUET: Final = os.path.expanduser(
    os.path.join(
        PACKAGE_PATH,
        SAVED_DATA_RELATIVE_PATH,
        FNAME_PARQUET,
    )
)

# %%
dfp_data = pd.read_parquet(F_PARQUET)

# %%
dfp_data["mean_pressure_value_normalized"] = dfp_data["mean_pressure_value"].apply(
    normalize_pressure_value, args=(OBSERVED_PRESSURE_MIN, OBSERVED_PRESSURE_MAX)
)

dfp_data = dfp_data.sort_values(["datetime_utc"], ascending=[True]).reset_index(drop=True)

# %%
display(dfp_data)

# %%
print(dfp_data.dtypes)

# %%
dfp_data[["mean_pressure_value", "mean_pressure_value_normalized"]].describe()

# %% [markdown]
# ***
# # Explore the Data

# %%
dt_start = dfp_data["datetime_local"].min()
dt_stop = dfp_data["datetime_local"].max()

# %%
C0 = "#012169"
C1 = "#993399"
C_GREY = "#7f7f7f"
# C_YELLOW = "#FFD960"
# C_ORANGE = "#E89923"

MS_FLOW_0 = "bowtie"
MS_FLOW_1 = "bowtie-open"
# MS_FLOW_0 = "circle"
# MS_FLOW_1 = "circle-open"
MC_FLOW_0 = C0
MC_FLOW_1 = C1
MARKER_SIZE_LARGE = 12
MARKER_SIZE_SMALL = 6

# %%
dfp_data["ms"] = dfp_data["had_flow"].apply(lambda x: MS_FLOW_0 if x != 1 else MS_FLOW_1)
dfp_data["mc"] = dfp_data["had_flow"].apply(lambda x: MC_FLOW_0 if x != 1 else MC_FLOW_1)

# %%
mean_trace = {
    "x": dfp_data["datetime_local"],
    "y": dfp_data["mean_pressure_value"],
    "customdata": dfp_data["had_flow"],
    "type": "scatter",
    "mode": "lines+markers",
    "marker": {
        "color": dfp_data["mc"],
        "size": MARKER_SIZE_SMALL,
        "line": {
            "width": 1.5,
            "color": dfp_data["mc"],
        },
        "symbol": dfp_data["ms"],
    },
    "line": {"width": 1.0},
    "showlegend": False,
    "hovertemplate": "1 Min Sample: %{x:%Y-%m-%d %H:%M:%S}<br>"
    + "Mean Pressure: %{y:d}<br>"
    + "Had Flow: %{customdata:df}"
    + "<extra></extra>",
}

# %%
# flow null traces for legend entries
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


# %%
mean_layout = {
    "xaxis": {
        "title": LOCAL_TIMEZONE_STR,
        "zeroline": False,
        "rangeselector": {
            "buttons": [
                {"count": 15, "label": "15m", "step": "minute", "stepmode": "todate"},
                {"count": 1, "label": "1h", "step": "hour", "stepmode": "todate"},
                {"count": 12, "label": "12h", "step": "hour", "stepmode": "todate"},
                {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
                {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
                {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                {"step": "all"},
            ],
        },
        "rangeslider": {
            "visible": True,
            "bordercolor": "lightgrey",
            "borderwidth": 1,
            "thickness": 0.05,
        },
        "type": "date",
        "gridcolor": "lightgrey",
        "range": [dt_start, dt_stop],
    },
    "yaxis": {
        "title": "Mean Pressure",
        "zeroline": False,
        "hoverformat": "d",
        "gridcolor": "lightgrey",
    },
    "colorway": [C0],
    "showlegend": True,
    "legend": {
        "orientation": "h",
        "xanchor": "right",
        "yanchor": "bottom",
        "x": 1.0,
        "y": 1.0,
    },
    "shapes": [
        {
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": OBSERVED_PRESSURE_MAX,
            "y1": OBSERVED_PRESSURE_MAX,
            "line": {
                "color": C_GREY,
                "width": 1.5,
                "dash": "dash",
            },
        },
        {
            "type": "line",
            "xref": "paper",
            "x0": 0,
            "x1": 1,
            "yref": "y",
            "y0": OBSERVED_PRESSURE_MIN,
            "y1": OBSERVED_PRESSURE_MIN,
            "line": {
                "color": C_GREY,
                "width": 1.5,
                "dash": "dash",
            },
        },
    ],
    "margin": {
        "t": 30,
        "b": 45,
        "l": 10,
        "r": 10,
    },
    "height": 500,
    "autosize": True,
    "font": {"color": C_GREY, "size": 14},
    "plot_bgcolor": "white",
}

# %%
fig = go.Figure()
fig.add_trace(go.Scattergl(mean_trace))
fig.add_traces([go.Scatter(legend_entry_trace_flow_0), go.Scatter(legend_entry_trace_flow_1)])
fig.update_layout(mean_layout)
fig.show()

# %%
hist_dicts = [
    {
        "values": dfp_data.loc[dfp_data["had_flow"] != 1, "mean_pressure_value"].values,
        "label": "No Flow",
        "density": True,
        "c": MC_FLOW_0,
        "lw": 2,
    },
    {
        "values": dfp_data.loc[dfp_data["had_flow"] == 1, "mean_pressure_value"].values,
        "label": "Had Flow",
        "density": True,
        "c": MC_FLOW_1,
        "lw": 2,
    },
]

plot_hists(
    hist_dicts,
    m_path=".",
    fname="mean_pressure_value_density",
    tag="",
    dt_start=dt_start,
    dt_stop=dt_stop,
    plot_inline=True,
    binning={"bin_size": 100},
    x_axis_params={
        "axis_label": "Mean Pressure",
        "min": None,
        "max": None,
        "units": "",
        "log": False,
    },
    y_axis_params={
        "axis_label": "Density",
        "min": None,
        "max": None,
        "max_mult": None,
        "log": True,
        "show_bin_size": True,
    },
)

# %%
