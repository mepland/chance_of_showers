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
import pathlib
import pprint
import shutil
import sys
import warnings
import zoneinfo
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import pandas as pd
from hydra import compose, initialize
from IPython.display import Image, display

sys.path.append(str(pathlib.Path.cwd().parent))

# pylint: disable=import-error
from utils.shared_functions import (
    create_datetime_component_cols,
    normalize_pressure_value,
)

# isort: off
from utils.TSModelWrapper import TSModelWrapper
from utils.bayesian_opt import run_bayesian_opt

from utils.ProphetWrapper import ProphetWrapper
from utils.NBEATSModelWrapper import NBEATSModelWrapper
from utils.NHiTSModelWrapper import NHiTSModelWrapper
from utils.TCNModelWrapper import TCNModelWrapper

from utils.plotting import (
    C_GREEN,
    C_GREY,
    C_RED,
    MC_FLOW_0,
    MC_FLOW_1,
    MPL_C0,
    MPL_C1,
    make_epoch_bins,
    save_ploty_to_html,
    plot_prophet,
    plot_2d_hist,
    plot_chance_of_showers_time_series,
    plot_hists,
)

# pylint: enable=import-error
# isort: on
# pylint: disable=unreachable

# %%
initialize(version_base=None, config_path="..")
cfg = compose(config_name="config")

TRAINABLE_START_DATETIME_LOCAL: Final = cfg["ana"]["trainable_start_datetime_local"]
TRAINABLE_END_DATETIME_LOCAL: Final = cfg["ana"]["trainable_end_datetime_local"]
TRAINABLE_VAL_FRACTION: Final = cfg["ana"]["trainable_val_fraction"]

OBSERVED_PRESSURE_MIN: Final = cfg["general"]["observed_pressure_min"]
OBSERVED_PRESSURE_MAX: Final = cfg["general"]["observed_pressure_max"]

PACKAGE_PATH: Final = pathlib.Path(cfg["general"]["package_path"]).expanduser()
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

START_OF_CRON_HEARTBEAT_MONITORING: Final = cfg["daq"]["start_of_cron_heartbeat_monitoring"]
DT_START_OF_CRON_HEARTBEAT_MONITORING: Final = datetime.datetime.strptime(
    START_OF_CRON_HEARTBEAT_MONITORING, DATETIME_FMT
).replace(tzinfo=LOCAL_TIMEZONE)

TS_LABEL: Final = f"Timestamp [{LOCAL_TIMEZONE_STR}]"

# %%
MODELS_PATH: Final = PACKAGE_PATH / "ana" / "models"
MODELS_PATH.mkdir(parents=True, exist_ok=True)

OUTPUTS_PATH: Final = PACKAGE_PATH / "ana" / "outputs"
OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

MEDIA_PATH: Final = PACKAGE_PATH / "media"

# %%
PLOT_INLINE: Final = False

# %% [markdown]
# ***
# # Load Data

# %%
FNAME_PARQUET: Final = "data_2023-04-27-03-00-04_to_2023-12-30-18-52-00.parquet"

# %%
F_PARQUET: Final = PACKAGE_PATH / SAVED_DATA_RELATIVE_PATH / FNAME_PARQUET

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

minutes_observed = dfp_data.index.size
minutes_possible = int((dt_stop_local - dt_start_local).total_seconds() / 60.0)

minutes_observed_since_start_of_cron_heartbeat_monitoring = dfp_data.loc[
    DT_START_OF_CRON_HEARTBEAT_MONITORING <= dfp_data["datetime_local"]
].index.size
minutes_possible_since_start_of_cron_heartbeat_monitoring = int(
    (dt_stop_local - DT_START_OF_CRON_HEARTBEAT_MONITORING).total_seconds() / 60.0
)

print(
    f"""
{dt_start_local = }
{dt_stop_local  = }

The DAQ system recorded {1 - (minutes_possible - minutes_observed)/minutes_possible:.1%} of possible data points overall,
and {1 - (minutes_possible_since_start_of_cron_heartbeat_monitoring - minutes_observed_since_start_of_cron_heartbeat_monitoring)/minutes_possible_since_start_of_cron_heartbeat_monitoring:.3%} since implementing the cron job heartbeat monitoring.
"""
)

# %%
actual_min_mean_pressure_value_with_flow = dfp_data.loc[dfp_data["had_flow"] == 1][
    "mean_pressure_value"
].min()
print(
    f"""Config {OBSERVED_PRESSURE_MIN = }
Actual Min Mean Pressure with Flow = {actual_min_mean_pressure_value_with_flow}

% Difference = {(OBSERVED_PRESSURE_MIN-actual_min_mean_pressure_value_with_flow)/OBSERVED_PRESSURE_MIN:.1%}
"""
)

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
import torch

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
# ## Prophet

# %%
# raise UserWarning("Stopping Here")

# %%
import prophet
from darts.models.forecasting.prophet_model import Prophet as darts_Prophet

# %%
model_wrapper_Prophet = ProphetWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 20, "rebin_y": False},
)
model_wrapper_Prophet.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_Prophet)

# %%
configurable_hyperparams = model_wrapper_Prophet.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %% [markdown]
# ### Training

# %%
val_loss = -model_wrapper_Prophet.train_model()
print(f"{val_loss = }")

# %% [markdown]
# ### Prophet Diagnostic Plots

# %%
n_prediction_steps, time_bin_size = model_wrapper_Prophet.get_n_prediction_steps_and_time_bin_size()

if TYPE_CHECKING:
    assert isinstance(  # noqa: SCS108 # nosec assert_used
        model_wrapper_Prophet.model, darts_Prophet
    )
model_prophet = model_wrapper_Prophet.model.model

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
# display(dfp_predict.tail(5))

# %% [markdown]
# #### Predictions

# %%
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    _fig_predict = model_prophet.plot(dfp_predict)

plot_prophet(
    _fig_predict,
    m_path=OUTPUTS_PATH / "prophet",
    fname="prophet_predict",
    tag="",
    x_axis_params_list=[{"axis_label": TS_LABEL}],
    y_axis_params_list=[{"axis_label": "Mean Pressure", "min": 0, "max": 1.2}],
    legend_params={
        "bbox_to_anchor": (0.1, 0.0, 0.2, 0.2),
        "box_color": "white",
    },
)

# %%
if PLOT_INLINE:
    Image(filename=OUTPUTS_PATH / "prophet" / "prophet_predict.png")

# %%
# The plotly version can be quite slow as it does not use go.Scattergl as in plot_chance_of_showers_time_series(),
# instead using go.Figure(data=data, layout=layout). See:
# https://github.com/facebook/prophet/blob/main/python/prophet/plot.py

fig_prophet_predict = prophet.plot.plot_plotly(model_prophet, dfp_predict)

save_ploty_to_html(
    fig_prophet_predict,
    m_path=OUTPUTS_PATH / "prophet",
    fname="prophet_predict",
    tag="",
)

# %%
if PLOT_INLINE:
    fig_prophet_predict.show()

# %% [markdown]
# #### Components

# %%
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    _fig_components = model_prophet.plot_components(dfp_predict)

plot_prophet(
    _fig_components,
    m_path=OUTPUTS_PATH / "prophet",
    fname="prophet_components",
    tag="",
    x_axis_params_list=[
        {"axis_label": TS_LABEL},
        {"axis_label": TS_LABEL},
        {"axis_label": "Day of Week"},
        {"axis_label": "Time of Day"},
        {"axis_label": TS_LABEL},
    ],
    y_axis_params_list=[
        {"axis_label": "Trend"},
        {"axis_label": "Holidays"},
        {"axis_label": "Weekly"},
        {"axis_label": "Daily"},
        {"axis_label": "Had Flow"},
    ],
)

# %%
if PLOT_INLINE:
    Image(filename=OUTPUTS_PATH / "prophet" / "prophet_components.png")

# %%
fig_prophet_components = prophet.plot.plot_components_plotly(model_prophet, dfp_predict)

save_ploty_to_html(
    fig_prophet_components,
    m_path=OUTPUTS_PATH / "prophet",
    fname="prophet_components",
    tag="",
)

# %%
if PLOT_INLINE:
    fig_prophet_components.show()

# %% [markdown]
# #### Individual Components

# %%
_fig_component_weekly, _ax_component_weekly = plt.subplots()

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    prophet.plot.plot_seasonality(model_prophet, "weekly", ax=_ax_component_weekly)

plot_prophet(
    _fig_component_weekly,
    m_path=OUTPUTS_PATH / "prophet",
    fname="prophet_component_weekly",
    tag="",
    x_axis_params_list=[{"axis_label": "Day of Week"}],
    y_axis_params_list=[{"axis_label": "Weekly Component"}],
)

# %%
if PLOT_INLINE:
    Image(filename=OUTPUTS_PATH / "prophet" / "prophet_component_weekly.png")

# %%
_fig_component_daily, _ax_component_daily = plt.subplots()

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )
    prophet.plot.plot_seasonality(model_prophet, "daily", ax=_ax_component_daily)

plot_prophet(
    _fig_component_daily,
    m_path=OUTPUTS_PATH / "prophet",
    fname="prophet_component_daily",
    tag="",
    x_axis_params_list=[{"axis_label": "Hour of Day"}],
    y_axis_params_list=[{"axis_label": "Daily Component"}],
)

# %%
if PLOT_INLINE:
    Image(filename=OUTPUTS_PATH / "prophet" / "prophet_component_daily.png")

# %% [markdown]
# ## N-BEATS

# %%
# raise UserWarning("Stopping Here")

# %%
model_wrapper_NBEATS = NBEATSModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"input_chunk_length_in_minutes": 10, "rebin_y": True},
)
model_wrapper_NBEATS.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_NBEATS)

# %%
configurable_hyperparams = model_wrapper_NBEATS.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %% [markdown]
# ### Training

# %%
print(model_wrapper_NBEATS)

# %%
model_wrapper_NBEATS.set_enable_progress_bar_and_max_time(enable_progress_bar=True, max_time=None)
val_loss = -model_wrapper_NBEATS.train_model()
print(f"{val_loss = }")

# %%
print(model_wrapper_NBEATS)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_NBEATS.work_dir, model_wrapper_NBEATS.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ##  N-HiTS

# %%
# raise UserWarning("Stopping Here")

# %%
model_wrapper_NHiTS = NHiTSModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"input_chunk_length_in_minutes": 10, "rebin_y": True},
)
model_wrapper_NHiTS.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_NHiTS)

# %%
configurable_hyperparams = model_wrapper_NHiTS.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %% [markdown]
# ### Training

# %%
print(model_wrapper_NHiTS)

# %%
model_wrapper_NHiTS.set_enable_progress_bar_and_max_time(enable_progress_bar=True, max_time=None)
val_loss = -model_wrapper_NHiTS.train_model()
print(f"{val_loss = }")

# %%
print(model_wrapper_NHiTS)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_NHiTS.work_dir, model_wrapper_NHiTS.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ## TCN

# %%
# raise UserWarning("Stopping Here")

# %%
model_wrapper_TCN = TCNModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"input_chunk_length_in_minutes": 10, "rebin_y": True},
)
model_wrapper_TCN.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_TCN)

# %%
configurable_hyperparams = model_wrapper_TCN.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %% [markdown]
# ### Training

# %%
print(model_wrapper_TCN)

# %%
model_wrapper_TCN.set_enable_progress_bar_and_max_time(enable_progress_bar=True, max_time=None)
val_loss = -model_wrapper_TCN.train_model()
print(f"{val_loss = }")

# %%
print(model_wrapper_TCN)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_TCN.work_dir, model_wrapper_TCN.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ## AutoARIMA

# %%
# raise UserWarning("Stopping Here")

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
# # Bayesian Optimization

# %%
# raise UserWarning("Stopping Here")

# %%
BAYESIAN_OPT_WORK_DIR_NAME: Final = "bayesian_optimization"
tensorboard_logs = pathlib.Path(PARENT_WRAPPER.work_dir_base, BAYESIAN_OPT_WORK_DIR_NAME)
# print(tensorboard_logs)

# %%
# %tensorboard --logdir $tensorboard_logs

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

# %% [markdown]
# ***
# # Explore the Data

# %%
# raise UserWarning("Stopping Here")

# %% [markdown]
# ## Time Series of All Raw ADC Pressure Values

# %%
plot_chance_of_showers_time_series(
    dfp_data,
    x_axis_params={
        "col": "datetime_local",
        "axis_label": TS_LABEL,
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
        {
            "orientation": "v",
            "value": TRAINABLE_START_DATETIME_LOCAL,
            "name": "Train Start",
            "c": C_GREEN,
            "lw": 2,
        },
        {
            "orientation": "v",
            "value": DT_VAL_START_DATETIME_LOCAL,
            "name": "Val Start",
            "c": C_GREY,
            "lw": 2,
        },
        {
            "orientation": "v",
            "value": TRAINABLE_END_DATETIME_LOCAL,
            "name": "Val End",
            "c": C_RED,
            "lw": 2,
        },
    ],
    plot_inline=PLOT_INLINE,
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
    m_path=OUTPUTS_PATH,
    fname="mean_pressure_value_density",
    tag="",
    dt_start=dt_start_local,
    dt_stop=dt_stop_local,
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
        "bbox_to_anchor": (0.73, 0.72, 0.2, 0.2),
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

# %%
if PLOT_INLINE:
    Image(filename=OUTPUTS_PATH / "mean_pressure_value_density.png")

# %% [markdown]
# ## Time Series of All Normalized Pressure Values

# %%
plot_chance_of_showers_time_series(
    dfp_data,
    x_axis_params={
        "col": "datetime_local",
        "axis_label": TS_LABEL,
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
        {
            "orientation": "v",
            "value": TRAINABLE_START_DATETIME_LOCAL,
            "name": "Train Start",
            "c": C_GREEN,
            "lw": 2,
        },
        {
            "orientation": "v",
            "value": DT_VAL_START_DATETIME_LOCAL,
            "name": "Val Start",
            "c": C_GREY,
            "lw": 2,
        },
        {
            "orientation": "v",
            "value": TRAINABLE_END_DATETIME_LOCAL,
            "name": "Val End",
            "c": C_RED,
            "lw": 2,
        },
    ],
    plot_inline=PLOT_INLINE,
    save_html=False,  # 24 MB
    m_path=OUTPUTS_PATH,
    fname="mean_pressure_value_normalized_all_data",
    tag="",
)

# %% [markdown]
# ## 2D Histogram of All Normalized Pressure Values - Same Day

# %%
plot_2d_hist(
    dfp_data["datetime_local_same_day"],
    100 * dfp_data["mean_pressure_value_normalized"],
    m_path=OUTPUTS_PATH,
    fname="mean_pressure_value_normalized_vs_time_of_day",
    tag="",
    dt_start=dt_start_local,
    dt_stop=dt_stop_local,
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
        "axis_label": "Mean Pressure",
        "units": "%",
    },
    z_axis_params={
        "axis_label": "Density",
        "norm": "log",
        "density": True,
    },
)

# %%
if PLOT_INLINE:
    Image(filename=OUTPUTS_PATH / "mean_pressure_value_normalized_vs_time_of_day.png")

# %% [markdown]
# ## 2D Histogram of All Normalized Pressure Values - Same Week

# %%
plot_2d_hist(
    dfp_data["datetime_local_same_week"],
    100 * dfp_data["mean_pressure_value_normalized"],
    m_path=OUTPUTS_PATH,
    fname="mean_pressure_value_normalized_vs_time_of_week",
    tag="",
    dt_start=dt_start_local,
    dt_stop=dt_stop_local,
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
        "axis_label": "Mean Pressure",
        "units": "%",
    },
    z_axis_params={
        "axis_label": "Density",
        "norm": "log",
        "density": True,
    },
)

# %%
if PLOT_INLINE:
    Image(filename=OUTPUTS_PATH / "mean_pressure_value_normalized_vs_time_of_week.png")

# %% [markdown]
# ## Time Series of Selected Pressure Values - For Web

# %%
dt_plotly_web_selection_start = datetime.datetime(year=2023, month=11, day=1, tzinfo=LOCAL_TIMEZONE)
dt_plotly_web_selection_end = datetime.datetime(year=2023, month=12, day=1, tzinfo=LOCAL_TIMEZONE)

dfp_plotly_web_selection = dfp_data.loc[
    (dt_plotly_web_selection_start <= dfp_data["datetime_local"])
    & (dfp_data["datetime_local"] <= dt_plotly_web_selection_end)
]

# %% [markdown]
# ## Raw

# %%
plot_chance_of_showers_time_series(
    dfp_plotly_web_selection,
    x_axis_params={
        "col": "datetime_local",
        "axis_label": TS_LABEL,
        "hover_label": "1 Min Sample: %{x:" + DATETIME_FMT + "}",
        "min": dt_plotly_web_selection_start,
        "max": dt_plotly_web_selection_end,
        "rangeselector_buttons": [
            "10m",
            "15m",
            "1h",
            "12h",
            "1d",
            "1w",
            "1m",
            "all",
        ],
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
        {"orientation": "h", "value": OBSERVED_PRESSURE_MIN, "name": "Normalized 0%", "c": MPL_C0},
        {
            "orientation": "h",
            "value": OBSERVED_PRESSURE_MAX,
            "name": "Normalized 100%",
            "c": MPL_C1,
        },
    ],
    plot_inline=PLOT_INLINE,
    save_html=True,
    m_path=OUTPUTS_PATH,
    fname="mean_pressure_value_selected_data",
    tag="",
)


# %% [markdown]
# ## Normalized

# %%
plot_chance_of_showers_time_series(
    dfp_plotly_web_selection,
    x_axis_params={
        "col": "datetime_local",
        "axis_label": TS_LABEL,
        "hover_label": "1 Min Sample: %{x:" + DATETIME_FMT + "}",
        "min": dt_plotly_web_selection_start,
        "max": dt_plotly_web_selection_end,
        "rangeselector_buttons": [
            "10m",
            "15m",
            "1h",
            "12h",
            "1d",
            "1w",
            "1m",
            "all",
        ],
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
    plot_inline=PLOT_INLINE,
    save_html=True,
    m_path=OUTPUTS_PATH,
    fname="mean_pressure_value_normalized_selected_data",
    tag="",
)

# %% [markdown]
# ***
# # Save outputs to `/media/ana_outputs`

# %%
# raise UserWarning("Stopping Here")

# %%
_ = shutil.copytree(
    OUTPUTS_PATH, MEDIA_PATH / "ana_outputs", dirs_exist_ok=True, copy_function=shutil.copy
)
