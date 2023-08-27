# %%
"""Analysis notebook."""
# %% [markdown]
# # Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %matplotlib inline
# pylint: disable=wrong-import-order

import datetime
import os
import sys

# import natsort
# import pprint
# import numpy as np
import zoneinfo
from typing import Final

import pandas as pd
from darts import TimeSeries
from darts.utils.missing_values import fill_missing_values, missing_values_ratio
from hydra import compose, initialize
from IPython.display import display

sys.path.append(os.path.dirname(os.path.realpath("")))
from utils.plotting import (  # noqa: E402 # pylint: disable=import-error
    C_GREEN,
    C_RED,
    MC_FLOW_0,
    MC_FLOW_1,
    MPL_C0,
    MPL_C1,
    make_epoch_bins,
    plot_2d_hist,
    plot_chance_of_showers_time_series,
    plot_hists,
)
from utils.shared_functions import (  # noqa: E402 # pylint: disable=import-error
    normalize_pressure_value,
    rebin_chance_of_showers_time_series,
)

# %%
initialize(version_base=None, config_path="..")
cfg = compose(config_name="config")

TRAINABLE_START_DATETIME_LOCAL: Final = cfg["ana"]["trainable_start_datetime_local"]
TRAINABLE_END_DATETIME_LOCAL: Final = cfg["ana"]["trainable_end_datetime_local"]

OBSERVED_PRESSURE_MIN: Final = cfg["general"]["observed_pressure_min"]
OBSERVED_PRESSURE_MAX: Final = cfg["general"]["observed_pressure_max"]

PACKAGE_PATH: Final = cfg["general"]["package_path"]
SAVED_DATA_RELATIVE_PATH: Final = cfg["etl"]["saved_data_relative_path"]

DATE_FMT: Final = cfg["general"]["date_fmt"]
TIME_FMT: Final = cfg["general"]["time_fmt"]
FNAME_DATETIME_FMT: Final = cfg["general"]["fname_datetime_fmt"]
DATETIME_FMT: Final = f"{DATE_FMT} {TIME_FMT}"

LOCAL_TIMEZONE_STR: Final = cfg["general"]["local_timezone"]

if LOCAL_TIMEZONE_STR not in zoneinfo.available_timezones():
    AVAILABLE_TIMEZONES: Final = "\n".join(list(zoneinfo.available_timezones()))
    raise ValueError(f"Unknown {LOCAL_TIMEZONE_STR = }, choose from:\n{AVAILABLE_TIMEZONES}")

# UTC_TIMEZONE: Final = zoneinfo.ZoneInfo("UTC")
LOCAL_TIMEZONE: Final = zoneinfo.ZoneInfo(LOCAL_TIMEZONE_STR)

RANDOM_SEED: Final = cfg["general"]["random_seed"]

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
FNAME_PARQUET: Final = "data_2023-04-27-03-00-04_to_2023-08-27-20-12-00.parquet"

# %%
F_PARQUET: Final = os.path.expanduser(
    os.path.join(
        PACKAGE_PATH,
        SAVED_DATA_RELATIVE_PATH,
        FNAME_PARQUET,
    )
)

dfp_data = pd.read_parquet(F_PARQUET)

# normalize pressure, clip values to between 0 and 1
dfp_data["mean_pressure_value_normalized"] = dfp_data["mean_pressure_value"].apply(
    normalize_pressure_value,
    observed_pressure_min=OBSERVED_PRESSURE_MIN,
    observed_pressure_max=OBSERVED_PRESSURE_MAX,
    clip=True,
)

# create local datetime columns
dfp_data["datetime_local"] = dfp_data["datetime_utc"].dt.tz_convert(LOCAL_TIMEZONE)

# columns all in the same day or week
dt_common = datetime.datetime(year=2023, month=1, day=1, tzinfo=LOCAL_TIMEZONE)
dfp_data["datetime_local_same_day"] = dfp_data.apply(
    lambda row: row["datetime_local"].replace(
        year=dt_common.year, month=dt_common.month, day=dt_common.day
    ),
    axis=1,
)

dfp_data["datetime_local_same_week"] = dfp_data.apply(
    lambda row: row["datetime_local"].replace(
        year=dt_common.year, month=dt_common.month, day=row["datetime_local"].isoweekday()
    ),
    axis=1,
)

dfp_data = dfp_data[
    [
        "datetime_local",
        "mean_pressure_value",
        "mean_pressure_value_normalized",
        "had_flow",
        "datetime_local_same_day",
        "datetime_local_same_week",
        # "datetime_utc",
        # "had_flow_original",
        # "fname",
    ]
]

# %%
print(dfp_data.dtypes)

# %%
display(dfp_data)

# %%
dfp_data[["mean_pressure_value", "mean_pressure_value_normalized"]].describe()

# %%
dt_start_local = dfp_data["datetime_local"].min()
dt_stop_local = dfp_data["datetime_local"].max()
print(f"{dt_start_local = }, {dt_stop_local = }")

# %% [markdown]
# ***
# # Training Data Prep

# %%
time_bin_size = datetime.timedelta(minutes=1)
# time_bin_size = datetime.timedelta(minutes=10)

y_bin_edges = None  # pylint: disable=invalid-name
# y_bin_edges = [-float("inf"), 0.6, 0.8, 0.9, 1.0]

n_prediction_steps = (7 * 24 * 60 * 60) // time_bin_size.seconds

# %%
dfp_trainable = dfp_data[["datetime_local", "mean_pressure_value_normalized", "had_flow"]]
dfp_trainable = dfp_trainable.loc[
    (TRAINABLE_START_DATETIME_LOCAL <= dfp_trainable["datetime_local"])
    & (dfp_trainable["datetime_local"] < TRAINABLE_END_DATETIME_LOCAL)
]
dfp_trainable = dfp_trainable.rename(
    columns={"datetime_local": "ds", "mean_pressure_value_normalized": "y"}
)
dfp_trainable["ds"] = dfp_trainable["ds"].dt.tz_localize(None)

dfp_trainable = rebin_chance_of_showers_time_series(
    dfp_trainable,
    "ds",
    "y",
    time_bin_size=time_bin_size,
    other_cols_to_agg_dict={"had_flow": "max"},
    y_bin_edges=y_bin_edges,
)

dart_series_trainable = TimeSeries.from_dataframe(dfp_trainable, "ds", "y", freq=time_bin_size)

frac_missing = missing_values_ratio(dart_series_trainable)
if 0 < frac_missing:
    print(f"Missing {frac_missing:.2%} of values, filling via interpolation")
    dart_series_trainable = fill_missing_values(dart_series_trainable)

dart_series_train, dart_series_val = dart_series_trainable.split_before(
    0.75
)  # TODO configure as date via conf.yaml
dfp_train = dart_series_train.pd_dataframe()
dfp_val = dart_series_val.pd_dataframe()

# %%
print(dfp_val.index.min())  # TODO

# %% [raw]
# display(dfp_train.head(5))
# display(dfp_train.tail(1))
# display(dfp_val.head(1))
# display(dfp_val.tail(5))

# %% [markdown]
# ***
# # Darts Modeling

# %%
import torch  # noqa: E402

if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    raise UserWarning("CUDA IS NOT AVAILABLE!")

from darts.models import NBEATSModel  # noqa: E402

# %% [markdown]
# ## N-BEATS

# %%
model_nbeats = NBEATSModel(input_chunk_length=10, output_chunk_length=5, random_state=RANDOM_SEED)

# %%
model_nbeats.fit(dart_series_train, val_series=dart_series_val)

# %%
prediction = model_nbeats.predict(n_prediction_steps, num_samples=1)

# %%
prediction.values()


# %%
print(min(prediction.values()), max(prediction.values()))


# %% [markdown]
# ***
# # Prophet Modeling

# %%
import prophet  # noqa: E402

# %%
model_prophet = prophet.Prophet(growth="flat")
_ = model_prophet.add_country_holidays(country_name="US")
_ = model_prophet.add_regressor("had_flow", mode="multiplicative")

# %%
_ = model_prophet.fit(dfp_train)

# %%
dfp_prophet_future = model_prophet.make_future_dataframe(
    periods=n_prediction_steps, freq=time_bin_size
)
dfp_prophet_future = pd.merge(
    dfp_prophet_future, dfp_train[["ds", "had_flow"]], on="ds", how="left"
)
dfp_prophet_future["had_flow"] = dfp_prophet_future["had_flow"].fillna(0)

dfp_predict = model_prophet.predict(dfp_prophet_future)

# %%
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     display(dfp_predict.dtypes)

# %%
display(dfp_predict.tail(2))

# %%
_fig_predict = model_prophet.plot(dfp_predict)

# %%
# The plotly version is quite slow as it does not use go.Scattergl as in plot_chance_of_showers_time_series(),
# instead using go.Figure(data=data, layout=layout). See:
# https://github.com/facebook/prophet/blob/main/python/prophet/plot.py

# prophet.plot.plot_plotly(model_prophet, dfp_predict)

# %%
_fig_components = model_prophet.plot_components(dfp_predict)

# %%
# The plotly version is not working. See:
# https://github.com/facebook/prophet/pull/2461

# prophet.plot.plot_components_plotly(model_prophet, dfp_predict)

# %% [markdown]
# ***
# # Explore the Data

# %% [markdown]
# ## Time Series of All Raw ADC Pressure Values

# %%
plot_chance_of_showers_time_series(
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
        "axis_label": "Mean Pressure [Raw ADC]",
        "hover_label": "Mean Pressure [Raw ADC]: %{y:d}",
    },
    z_axis_params={
        "col": "had_flow",
        "hover_label": "Had Flow: %{customdata:df}",
    },
    reference_lines=[
        {"orientation": "h", "value": OBSERVED_PRESSURE_MIN, "c": MPL_C0},
        {"orientation": "h", "value": OBSERVED_PRESSURE_MAX, "c": MPL_C1},
        {"orientation": "v", "value": TRAINABLE_START_DATETIME_LOCAL, "c": C_GREEN, "lw": 2},
        {"orientation": "v", "value": TRAINABLE_END_DATETIME_LOCAL, "c": C_RED, "lw": 2},
    ],
)

# %% [markdown]
# ## Histogram of All Raw ADC Pressure Values
# Helps to set min and max pressure for normalization.

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
        "bin_size_str_fmt": ".0f",
    },
    x_axis_params={
        "axis_label": "Mean Pressure",
        "units": "Raw ADC",
    },
    y_axis_params={
        "axis_label": "Density",
        "log": True,
    },
    legend_params={
        "bbox_to_anchor": (0.72, 0.72, 0.2, 0.2),
        "box_color": "white",
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

# %% [markdown]
# ## Time Series of All Normalized Pressure Values

# %%
plot_chance_of_showers_time_series(
    dfp_data,
    x_axis_params={
        "col": "datetime_local",
        "axis_label": LOCAL_TIMEZONE_STR,
        "hover_label": "1 Min Sample: %{x:" + DATETIME_FMT + "}",
        "min": dt_start_local,
        "max": dt_stop_local,
    },
    y_axis_params={
        "col": "mean_pressure_value_normalized",
        "axis_label": "Mean Pressure %",
        "hover_label": "Mean Pressure: %{y:.2%}",
    },
    z_axis_params={
        "col": "had_flow",
        "hover_label": "Had Flow: %{customdata:df}",
    },
    reference_lines=[
        {"orientation": "v", "value": TRAINABLE_START_DATETIME_LOCAL, "c": C_GREEN, "lw": 2},
        {"orientation": "v", "value": TRAINABLE_END_DATETIME_LOCAL, "c": C_RED, "lw": 2},
    ],
)

# %% [markdown]
# ## 2D Histogram of All Normalized Pressure Values - Same Day

# %%
plot_2d_hist(
    dfp_data["datetime_local_same_day"],
    100 * dfp_data["mean_pressure_value_normalized"],
    m_path=".",
    fname="mean_pressure_value_normalized_vs_time_of_day",
    tag="",
    dt_start=dt_start_local,
    dt_stop=dt_stop_local,
    plot_inline=True,
    binning={
        "x": {
            "bin_edges": make_epoch_bins(
                dt_common, dt_common + datetime.timedelta(days=1), 15 * 60
            ),
            "bin_size": "15 [Minutes]",
            "bin_size_str_fmt": "",
        },
        "y": {
            "bin_size": 5,
            "bin_size_str_fmt": ".0f",
        },
    },
    x_axis_params={
        "is_datetime": True,
        "axis_label": f"Time of Day [{LOCAL_TIMEZONE_STR}]",
        "ticks": make_epoch_bins(dt_common, dt_common + datetime.timedelta(days=1), 2 * 60 * 60),
        "tick_format": TIME_FMT,
    },
    y_axis_params={
        "min": -2,
        "axis_label": "Mean Pressure",
        "units": "%",
    },
    z_axis_params={
        "axis_label": "Density",
        "norm": "log",
        "density": True,
    },
)

# %% [markdown]
# ## 2D Histogram of All Normalized Pressure Values - Same Week

# %%
plot_2d_hist(
    dfp_data["datetime_local_same_week"],
    100 * dfp_data["mean_pressure_value_normalized"],
    m_path=".",
    fname="mean_pressure_value_normalized_vs_time_of_week",
    tag="",
    dt_start=dt_start_local,
    dt_stop=dt_stop_local,
    plot_inline=True,
    binning={
        "x": {
            "bin_edges": make_epoch_bins(
                dt_common, dt_common + datetime.timedelta(days=7), 60 * 60
            ),
            "bin_size": "1 [Hours]",
            "bin_size_str_fmt": "",
        },
        "y": {
            "bin_size": 5,
            "bin_size_str_fmt": ".0f",
        },
    },
    x_axis_params={
        "is_datetime": True,
        "axis_label": f"Time of Week [{LOCAL_TIMEZONE_STR}]",
        "ticks": make_epoch_bins(dt_common, dt_common + datetime.timedelta(days=7), 12 * 60 * 60),
        "tick_format": f"%A {TIME_FMT}",
    },
    y_axis_params={
        "min": -2,
        "axis_label": "Mean Pressure",
        "units": "%",
    },
    z_axis_params={
        "axis_label": "Density",
        "norm": "log",
        "density": True,
    },
)
