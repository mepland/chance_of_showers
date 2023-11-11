# %%
"""Analysis notebook."""
# %% [markdown]
# # Setup

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# %load_ext tensorboard

# %%
# %matplotlib inline
# pylint: disable=wrong-import-order

import datetime
import os
import pprint
import sys

# import natsort
# import numpy as np
import zoneinfo
from typing import Final

import pandas as pd
from hydra import compose, initialize
from IPython.display import display

sys.path.append(os.path.dirname(os.path.realpath("")))
from utils.plotting import (  # noqa: E402 # pylint: disable=import-error
    C_GREEN,
    C_GREY,
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
    create_datetime_component_cols,
    normalize_pressure_value,
)
from utils.TSModelWrappers import (  # noqa: E402 # pylint: disable=import-error
    NBEATSModelWrapper,
    TSModelWrapper,
    run_bayesian_opt,
)

# %%
initialize(version_base=None, config_path="..")
cfg = compose(config_name="config")

TRAINABLE_START_DATETIME_LOCAL: Final = cfg["ana"]["trainable_start_datetime_local"]
TRAINABLE_END_DATETIME_LOCAL: Final = cfg["ana"]["trainable_end_datetime_local"]
TRAINABLE_VAL_FRACTION: Final = cfg["ana"]["trainable_val_fraction"]

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

DT_TRAINABLE_START_DATETIME_LOCAL: Final = datetime.datetime.strptime(
    TRAINABLE_START_DATETIME_LOCAL, DATETIME_FMT
).replace(tzinfo=LOCAL_TIMEZONE)
DT_TRAINABLE_END_DATETIME_LOCAL: Final = datetime.datetime.strptime(
    TRAINABLE_END_DATETIME_LOCAL, DATETIME_FMT
).replace(tzinfo=LOCAL_TIMEZONE)

# Use first 1-TRAINABLE_VAL_FRACTION of trainable days for train, last TRAINABLE_VAL_FRACTION for val, while rounding to the day using pandas.Timedelta
# https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html
DT_VAL_START_DATETIME_LOCAL: Final = (
    DT_TRAINABLE_START_DATETIME_LOCAL
    + pd.Series(
        (1 - TRAINABLE_VAL_FRACTION)
        * (DT_TRAINABLE_END_DATETIME_LOCAL - DT_TRAINABLE_START_DATETIME_LOCAL)
    )
    .dt.round("1d")
    .iloc[0]
)

RANDOM_SEED: Final = cfg["general"]["random_seed"]

# %%
MODELS_PATH: Final = os.path.expanduser(
    os.path.join(
        PACKAGE_PATH,
        "ana",
        "models",
    )
)
os.makedirs(MODELS_PATH, exist_ok=True)

# %% [markdown]
# ***
# # Load Data

# %%
FNAME_PARQUET: Final = "data_2023-04-27-03-00-04_to_2023-09-25-16-01-00.parquet"

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

dfp_data = create_datetime_component_cols(
    dfp_data, datetime_col="datetime_local", date_fmt=DATE_FMT, time_fmt=TIME_FMT
)

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
        "day_of_week_int",
        "day_of_week_frac",
        "time_of_day",
        "time_of_day_frac",
        "is_holiday",
        # "datetime_utc",
        # "had_flow_original",
        # "fname",
    ]
]

# %%
print(dfp_data.dtypes)

# %%
with pd.option_context("display.max_rows", 5, "display.max_columns", None):
    display(dfp_data)

# %%
dfp_data[["mean_pressure_value", "mean_pressure_value_normalized"]].describe()

# %%
dt_start_local = dfp_data["datetime_local"].min()
dt_stop_local = dfp_data["datetime_local"].max()
print(f"{dt_start_local = },\n{dt_stop_local  = }")

# %% [markdown]
# ## Evergreen Training Data Prep

# %%
dfp_trainable_evergreen = dfp_data[["datetime_local", "mean_pressure_value_normalized", "had_flow"]]
dfp_trainable_evergreen = dfp_trainable_evergreen.loc[
    (TRAINABLE_START_DATETIME_LOCAL <= dfp_trainable_evergreen["datetime_local"])
    & (dfp_trainable_evergreen["datetime_local"] < TRAINABLE_END_DATETIME_LOCAL)
]
dfp_trainable_evergreen = dfp_trainable_evergreen.rename(
    columns={"datetime_local": "ds", "mean_pressure_value_normalized": "y"}
)
dfp_trainable_evergreen["ds"] = dfp_trainable_evergreen["ds"].dt.tz_localize(None)


# %% [raw]
# with pd.option_context("display.max_rows", 5, "display.max_columns", None):
#     display(dfp_trainable_evergreen)

# %% [markdown]
# ## Non-Darts Data Prep

# %%
dfp_train = dfp_trainable_evergreen.loc[
    dfp_trainable_evergreen["ds"] < DT_VAL_START_DATETIME_LOCAL.replace(tzinfo=None)
]
dfp_val = dfp_trainable_evergreen.loc[
    DT_VAL_START_DATETIME_LOCAL.replace(tzinfo=None) <= dfp_trainable_evergreen["ds"]
]

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

# %%
PARENT_WRAPPER: Final = TSModelWrapper(
    dfp_trainable_evergreen=dfp_trainable_evergreen,
    dt_val_start_datetime_local=DT_VAL_START_DATETIME_LOCAL,
    work_dir_base=MODELS_PATH,
    random_state=RANDOM_SEED,
    date_fmt=DATE_FMT,
    time_fmt=TIME_FMT,
    fname_datetime_fmt=FNAME_DATETIME_FMT,
    local_timezone=LOCAL_TIMEZONE,
)
# print(PARENT_WRAPPER)

# %% [markdown]
# ## Bayesian Optimization

# %%
BAYESIAN_OPT_WORK_DIR_NAME: Final = "bayesian_optimization"
tensorboard_logs = os.path.join(
    PARENT_WRAPPER.work_dir_base,
    BAYESIAN_OPT_WORK_DIR_NAME,
)
# print(tensorboard_logs)

# %%
# %tensorboard --logdir $tensorboard_logs

# %%
optimal_values, optimizer = run_bayesian_opt(
    parent_wrapper=PARENT_WRAPPER,
    model_wrapper_class=NBEATSModelWrapper,
    n_iter=200,
    enable_progress_bar=True,
    max_time_per_model=datetime.timedelta(minutes=20),
    display_memory_usage=True,
    bayesian_opt_work_dir_name=BAYESIAN_OPT_WORK_DIR_NAME,
)

# %%
pprint.pprint(optimal_values)

# %%
# raise UserWarning("Stopping Here")

# %% [markdown]
# ## N-BEATS

# %%
model_wrapper = NBEATSModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"input_chunk_length_in_minutes": 10, "rebin_y": True},
)
# print(model_wrapper)

# %%
configurable_hyperparams = model_wrapper.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %% [markdown]
# ### Training

# %%
model_wrapper.set_enable_progress_bar_and_max_time(enable_progress_bar=True, max_time=None)
val_loss = -model_wrapper.train_model()
print(f"{val_loss = }")

# %%
print(model_wrapper)

# %%
tensorboard_logs = os.path.join(model_wrapper.work_dir_base, model_wrapper.model_name, "logs")  # type: ignore[arg-type]
print(tensorboard_logs)

# %%
# %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ## AutoARIMA

# %% [raw]
# hyperpar_fixed_AutoARIMA = {
#     "start_p": 2,
#     "d": None,
#     "start_q": 2,
#     "max_p": 10,
#     "max_d": 2,
#     "max_q": 10,
#     "start_P": 1,
#     "D": None,
#     "start_Q": 1,
#     "max_P": 2,
#     "max_D": 1,
#     "max_Q": 2,
#     "max_order": None,
#     "m": 1,
#     "seasonal": True,
#     "stationary": False,
#     "information_criterion": "aic",
#     "alpha": 0.05,
#     "test": "kpss",
#     "seasonal_test": "ocsb",
#     "stepwise": True,
#     "n_jobs": -1,
#     "trend": None,
#     "method": "lbfgs",
#     "maxiter": 100,
#     "error_action": "trace",
#     "trace": False,
#     "out_of_sample_size": 0,
#     "scoring": "mse",
#     "with_intercept": "auto",
# }

# %% [markdown]
# ***
# # Prophet Modeling

# %%
# Hyperparams - Rework these!
time_bin_size = datetime.timedelta(minutes=5)

prediction_time_size = datetime.timedelta(hours=1.5)
n_prediction_steps = prediction_time_size.seconds // time_bin_size.seconds


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
        {"orientation": "v", "value": DT_VAL_START_DATETIME_LOCAL, "c": C_GREY, "lw": 2},
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
        {"orientation": "v", "value": DT_VAL_START_DATETIME_LOCAL, "c": C_GREY, "lw": 2},
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
