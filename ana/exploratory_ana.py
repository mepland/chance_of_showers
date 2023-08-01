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


sys.path.append(os.path.dirname(os.path.realpath("")))
from utils.plotting import (  # noqa: E402 # pylint: disable=import-error
    MC_FLOW_0,
    MC_FLOW_1,
    MPL_C0,
    MPL_C1,
    plot_chance_of_showers_timeseries,
    plot_hists,
)
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

dfp_data = pd.read_parquet(F_PARQUET)

dfp_data["mean_pressure_value_normalized"] = dfp_data["mean_pressure_value"].apply(
    normalize_pressure_value, args=(OBSERVED_PRESSURE_MIN, OBSERVED_PRESSURE_MAX)
)

dfp_data = dfp_data.sort_values(["datetime_utc"], ascending=[True]).reset_index(drop=True)

# %%
print(dfp_data.dtypes)

# %%
display(dfp_data)

# %%
dfp_data[["mean_pressure_value", "mean_pressure_value_normalized"]].describe()

# %%
dt_start_local = dfp_data["datetime_local"].min()
dt_stop_local = dfp_data["datetime_local"].max()
print(f"{dt_start_local=}, {dt_stop_local=}")

# %% [markdown]
# ***
# # Explore the Data

# %%
plot_chance_of_showers_timeseries(
    dfp_data,
    x_axis_params={
        "col": "datetime_local",
        "axis_label": LOCAL_TIMEZONE_STR,
        "hover_label": "1 Min Sample: %{x:" + DATETIME_FMT + "}",
        "min": dt_start_local,
        "max": dt_stop_local,
    },
    y_axis_params={
        "col": "mean_pressure_value",
        "axis_label": "Mean Pressure",
        "hover_label": "Mean Pressure: %{y:d}",
    },
    z_axis_params={
        "col": "had_flow",
        "hover_label": "Had Flow: %{customdata:df}",
    },
    reference_lines=[
        {"orientation": "h", "value": OBSERVED_PRESSURE_MIN, "c": MPL_C0},
        {"orientation": "h", "value": OBSERVED_PRESSURE_MAX, "c": MPL_C1},
    ],
)

# %%
hist_dicts = [
    {
        "values": dfp_data.loc[dfp_data["had_flow"] != 1, "mean_pressure_value"].values,
        "label": "No Flow",
        "density": True,
        "c": MC_FLOW_0,
    },
    {
        "values": dfp_data.loc[dfp_data["had_flow"] == 1, "mean_pressure_value"].values,
        "label": "Had Flow",
        "density": True,
        "c": MC_FLOW_1,
    },
]

plot_hists(
    hist_dicts,
    m_path=".",
    fname="mean_pressure_value_density",
    tag="",
    dt_start=dt_start_local,
    dt_stop=dt_stop_local,
    plot_inline=True,
    binning={
        "bin_size": 100,
    },
    x_axis_params={
        "axis_label": "Mean Pressure",
    },
    y_axis_params={
        "axis_label": "Density",
        "log": True,
    },
    reference_lines=[
        {
            "label": f"Normalized 0% = {OBSERVED_PRESSURE_MIN:d}",
            "orientation": "v",
            "value": OBSERVED_PRESSURE_MIN,
            "c": "C0",
            "ls": "--",
        },
        {
            "label": f"Normalized 100% = {OBSERVED_PRESSURE_MAX:d}",
            "orientation": "v",
            "value": OBSERVED_PRESSURE_MAX,
            "c": "C1",
            "ls": ":",
        },
    ],
)
