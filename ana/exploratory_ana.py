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
from hydra import compose, initialize
from IPython.display import display

# import statsmodels.api as sm
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.ar_model import ar_select_order
# from statsmodels.graphics.api import qqplot

# import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.realpath("")))
# from utils.plotting import plot_func
from utils.shared_functions import (  # noqa: E402 # pylint: disable=import-error
    normalize_pressure_value,
)

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

# LOCAL_TIMEZONE_STR: Final = cfg["general"]["local_timezone"]

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

# %%
display(dfp_data)

# %%
print(dfp_data.dtypes)

# %%
dfp_data[["mean_pressure_value", "mean_pressure_value_normalized"]].describe()

# %% [markdown]
# ***
# # Explore the Data

# %% [markdown]
# ### Minute Time Series

# %% [raw]
# dfp_data.index = pd.DatetimeIndex(dfp_data["datetime_est"]).tz_localize(None).to_period("T")

# %% [raw]
# # convert to DatetimeIndex, with T (minute) fequency
# # create null rows between min and max datetime if they do not exist
# # dfp.index = pd.DatetimeIndex(dfp['datetime_est']).tz_localize(None)
# # dfp = dfp.asfreq('T')

# %% [raw]
# dfp_data

# %% [raw]
# dfp_data.loc[dfp_data["had_flow"].isnull()]

# %% [raw]
# plot_objs_ts = {}
# plot_objs_ts["minutes"] = {
#     "type": "scatter",
#     "x": dfp_data["datetime_est"],
#     "y": dfp_data["mean_pressure_value"],
#     "c": f"C0",
#     "ms": ".",
#     "ls": "",
#     "label": None,
# }

# %% [raw]
# plot_func(plot_objs_ts, "Minute", "Mean Pressure Value", fig_size=(12, 8))

# %% [raw]
# fig = go.Figure()
#
# fig.add_trace(go.Scatter(x=dfp_data["datetime_est"], y=dfp_data["mean_pressure_value"]))
#
# # Add range slider
# fig.update_layout(
#     xaxis=dict(
#         rangeselector=dict(
#             buttons=list(
#                 [
#                     dict(count=1, label="1h", step="hour", stepmode="todate"),
#                     dict(count=12, label="12h", step="hour", stepmode="todate"),
#                     dict(count=1, label="1d", step="day", stepmode="backward"),
#                     dict(count=7, label="1w", step="day", stepmode="backward"),
#                     dict(count=1, label="1m", step="month", stepmode="backward"),
#                     dict(count=6, label="6m", step="month", stepmode="backward"),
#                     dict(count=1, label="YTD", step="year", stepmode="todate"),
#                     dict(count=1, label="1y", step="year", stepmode="backward"),
#                     dict(step="all"),
#                 ]
#             )
#         ),
#         rangeslider=dict(visible=True),
#         type="date",
#     )
# )
#
#
# fig.update_layout(
#     xaxis_title="Date",
#     yaxis_title="Pressure DAQ Value",
# )
#
# fig.update_xaxes(minor=dict(ticks="inside", showgrid=True))
#
#
# fig.show()
