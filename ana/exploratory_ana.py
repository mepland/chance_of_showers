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

import datetime as dt
import pathlib
import pprint
import shutil
import sys
import warnings
from typing import TYPE_CHECKING, Final

import matplotlib.pyplot as plt
import pandas as pd
from hydra import compose, initialize
from IPython.display import Image, display

sys.path.append(str(pathlib.Path.cwd().parent))

from utils.bayesian_opt import load_best_points, write_search_results
from utils.shared_functions import (
    create_datetime_component_cols,
    get_local_timezone_from_cfg,
    normalize_pressure_value,
    write_secure_pickle,
)

# isort: off
from TSModelWrappers.TSModelWrapper import TSModelWrapper

# Prophet
from TSModelWrappers.ProphetWrapper import ProphetWrapper

# PyTorch NN Models
from TSModelWrappers.NBEATSModelWrapper import NBEATSModelWrapper
from TSModelWrappers.NHiTSModelWrapper import NHiTSModelWrapper
from TSModelWrappers.TCNModelWrapper import TCNModelWrapper
from TSModelWrappers.TransformerModelWrapper import TransformerModelWrapper
from TSModelWrappers.TFTModelWrapper import TFTModelWrapper
from TSModelWrappers.TSMixerModelWrapper import TSMixerModelWrapper
from TSModelWrappers.DLinearModelWrapper import DLinearModelWrapper
from TSModelWrappers.NLinearModelWrapper import NLinearModelWrapper
from TSModelWrappers.TiDEModelWrapper import TiDEModelWrapper
from TSModelWrappers.RNNModelWrapper import RNNModelWrapper
from TSModelWrappers.BlockRNNModelWrapper import BlockRNNModelWrapper

# Statistical Models
from TSModelWrappers.AutoARIMAWrapper import AutoARIMAWrapper
from TSModelWrappers.BATSWrapper import BATSWrapper
from TSModelWrappers.TBATSWrapper import TBATSWrapper
from TSModelWrappers.FourThetaWrapper import FourThetaWrapper
from TSModelWrappers.StatsForecastAutoThetaWrapper import StatsForecastAutoThetaWrapper
from TSModelWrappers.FFTWrapper import FFTWrapper
from TSModelWrappers.KalmanForecasterWrapper import KalmanForecasterWrapper
from TSModelWrappers.CrostonWrapper import CrostonWrapper

# Regression Models
from TSModelWrappers.LinearRegressionModelWrapper import LinearRegressionModelWrapper
from TSModelWrappers.RandomForestWrapper import RandomForestWrapper
from TSModelWrappers.LightGBMModelWrapper import LightGBMModelWrapper
from TSModelWrappers.XGBModelWrapper import XGBModelWrapper
from TSModelWrappers.CatBoostModelWrapper import CatBoostModelWrapper

# Naive Models
from TSModelWrappers.NaiveMeanWrapper import NaiveMeanWrapper
from TSModelWrappers.NaiveSeasonalWrapper import NaiveSeasonalWrapper
from TSModelWrappers.NaiveDriftWrapper import NaiveDriftWrapper
from TSModelWrappers.NaiveMovingAverageWrapper import NaiveMovingAverageWrapper

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

# isort: on

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
LOCAL_TIMEZONE, LOCAL_TIMEZONE_STR = get_local_timezone_from_cfg(cfg)

DT_TRAINABLE_START_DATETIME_LOCAL: Final = dt.datetime.strptime(
    TRAINABLE_START_DATETIME_LOCAL, DATETIME_FMT
).replace(tzinfo=LOCAL_TIMEZONE)
DT_TRAINABLE_END_DATETIME_LOCAL: Final = dt.datetime.strptime(
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
DT_START_OF_CRON_HEARTBEAT_MONITORING: Final = dt.datetime.strptime(
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


def display_image(fname: pathlib.Path, *, plot_inline: bool = PLOT_INLINE) -> None:
    """Show image from local file in jupyter.

    Args:
        fname (pathlib.Path): Path to image file.
        plot_inline (bool): Display plot, or not. (Default value = PLOT_INLINE)
    """
    if plot_inline:
        display(Image(filename=fname))


# %% [markdown]
# ***
# # Load Data

# %%
FNAME_PARQUET: Final = "data_2023-04-27-03-00-04_to_2024-05-11-18-46-00.parquet"

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

# normalize pressure, unclipped
dfp_data["mean_pressure_value_normalized_unclipped"] = dfp_data["mean_pressure_value"].apply(
    normalize_pressure_value,
    observed_pressure_min=OBSERVED_PRESSURE_MIN,
    observed_pressure_max=OBSERVED_PRESSURE_MAX,
    clip=False,
)

# create local datetime columns
dfp_data["datetime_local"] = dfp_data["datetime_utc"].dt.tz_convert(LOCAL_TIMEZONE)

dfp_data = create_datetime_component_cols(
    dfp_data, datetime_col="datetime_local", date_fmt=DATE_FMT, time_fmt=TIME_FMT
)

# columns all in the same day or week
dt_common = dt.datetime(year=2023, month=1, day=1, tzinfo=LOCAL_TIMEZONE)
dfp_data["datetime_local_same_day"] = dfp_data.apply(
    lambda row: row["datetime_local"].replace(
        year=dt_common.year, month=dt_common.month, day=dt_common.day
    ),
    axis=1,
)

dfp_data["datetime_local_same_week"] = dfp_data.apply(
    lambda row: row["datetime_local"].replace(
        year=dt_common.year,
        month=dt_common.month,
        day=dt_common.day
        + (
            delta_days + 7
            if (delta_days := row["datetime_local"].dayofweek - dt_common.weekday()) < 0
            else delta_days
        ),
    ),
    axis=1,
)

dfp_data = dfp_data[
    [
        "datetime_local",
        "mean_pressure_value",
        "mean_pressure_value_normalized",
        "mean_pressure_value_normalized_unclipped",
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
dfp_data[
    [
        "mean_pressure_value",
        "mean_pressure_value_normalized",
        "mean_pressure_value_normalized_unclipped",
    ]
].describe()

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
dfp_trainable_evergreen = dfp_data[
    [
        "datetime_local",
        "mean_pressure_value_normalized",
        "mean_pressure_value_normalized_unclipped",
        "had_flow",
    ]
]
dfp_trainable_evergreen = dfp_trainable_evergreen.loc[
    (TRAINABLE_START_DATETIME_LOCAL <= dfp_trainable_evergreen["datetime_local"])
    & (dfp_trainable_evergreen["datetime_local"] < TRAINABLE_END_DATETIME_LOCAL)
]
dfp_trainable_evergreen = dfp_trainable_evergreen.rename(
    columns={
        "datetime_local": "ds",
        "mean_pressure_value_normalized": "y",
        "mean_pressure_value_normalized_unclipped": "y_unclipped",
    }
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
# # Setup Darts Wrapper

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


# %%
import torch

if torch.cuda.is_available():
    print("CUDA is available")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    msg = "CUDA IS NOT AVAILABLE!"  # pylint: disable=invalid-name
    raise UserWarning(msg)

# %%
# raise UserWarning("Stopping Here")

# %% [markdown]
# ***
# # Darts Modeling

# %% [markdown]
# ## Prophet

# %%
import prophet
from darts.models.forecasting.prophet_model import Prophet as darts_Prophet

# %%
model_wrapper_Prophet = ProphetWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 20},
)
model_wrapper_Prophet.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_Prophet)

# %%
configurable_hyperparams = model_wrapper_Prophet.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_Prophet.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_Prophet)

# %% [markdown]
# ### Prophet Diagnostic Plots

# %%
n_prediction_steps, time_bin_size = model_wrapper_Prophet.get_n_prediction_steps_and_time_bin_size()

if TYPE_CHECKING:
    assert isinstance(model_wrapper_Prophet.model, darts_Prophet)  # noqa: SCS108 # nosec: B101

model_prophet = model_wrapper_Prophet.model.model

dfp_prophet_future = model_prophet.make_future_dataframe(
    periods=n_prediction_steps, freq=time_bin_size
)
dfp_prophet_future = dfp_prophet_future.merge(dfp_train[["ds", "had_flow"]], on="ds", how="left")
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
display_image(OUTPUTS_PATH / "prophet" / "prophet_predict.png")

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

_x_axis_params_list = [
    {"axis_label": TS_LABEL},
    {"axis_label": TS_LABEL},
    {"axis_label": "Day of Week"},
    {"axis_label": "Time of Day"},
]
_y_axis_params_list = [
    {"axis_label": "Trend"},
    {"axis_label": "Holidays"},
    {"axis_label": "Weekly"},
    {"axis_label": "Daily"},
]

if "had_flow" in model_wrapper_Prophet.chosen_hyperparams["covariates"]:  # type: ignore[index]
    _x_axis_params_list.append({"axis_label": TS_LABEL})
    _y_axis_params_list.append({"axis_label": "Had Flow"})

plot_prophet(
    _fig_components,
    m_path=OUTPUTS_PATH / "prophet",
    fname="prophet_components",
    tag="",
    x_axis_params_list=_x_axis_params_list,
    y_axis_params_list=_y_axis_params_list,
)

# %%
display_image(OUTPUTS_PATH / "prophet" / "prophet_components.png")

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
display_image(OUTPUTS_PATH / "prophet" / "prophet_component_weekly.png")

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
display_image(OUTPUTS_PATH / "prophet" / "prophet_component_daily.png")

# %% [markdown]
# ## PyTorch NN Models

# %% [markdown]
# ### N-BEATS

# %%
model_wrapper_NBEATS = NBEATSModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_NBEATS.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_NBEATS)

# %%
configurable_hyperparams = model_wrapper_NBEATS.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_NBEATS.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_NBEATS)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_NBEATS.work_dir, model_wrapper_NBEATS.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### N-HiTS

# %%
model_wrapper_NHiTS = NHiTSModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_NHiTS.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_NHiTS)

# %%
configurable_hyperparams = model_wrapper_NHiTS.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_NHiTS.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_NHiTS)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_NHiTS.work_dir, model_wrapper_NHiTS.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### TCN

# %%
model_wrapper_TCN = TCNModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_TCN.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_TCN)

# %%
configurable_hyperparams = model_wrapper_TCN.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_TCN.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_TCN)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_TCN.work_dir, model_wrapper_TCN.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### Transformer

# %%
model_wrapper_Transformer = TransformerModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_Transformer.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_Transformer)

# %%
configurable_hyperparams = model_wrapper_Transformer.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_Transformer.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_Transformer)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_Transformer.work_dir, model_wrapper_Transformer.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### TFT

# %%
model_wrapper_TFT = TFTModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_TFT.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_TFT)

# %%
configurable_hyperparams = model_wrapper_TFT.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_TFT.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_TFT)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_TFT.work_dir, model_wrapper_TFT.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### TSMixer

# %%
model_wrapper_TSMixer = TSMixerModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_TSMixer.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_TSMixer)

# %%
configurable_hyperparams = model_wrapper_TSMixer.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_TSMixer.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_TSMixer)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_TSMixer.work_dir, model_wrapper_TSMixer.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### D-Linear

# %%
model_wrapper_DLinear = DLinearModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_DLinear.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_DLinear)

# %%
configurable_hyperparams = model_wrapper_DLinear.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_DLinear.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_DLinear)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_DLinear.work_dir, model_wrapper_DLinear.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### N-Linear

# %%
model_wrapper_NLinear = NLinearModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_NLinear.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_NLinear)

# %%
configurable_hyperparams = model_wrapper_NLinear.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_NLinear.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_NLinear)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_NLinear.work_dir, model_wrapper_NLinear.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### TiDE

# %%
model_wrapper_TiDE = TiDEModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_TiDE.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_TiDE)

# %%
configurable_hyperparams = model_wrapper_TiDE.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_TiDE.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_TiDE)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_TiDE.work_dir, model_wrapper_TiDE.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### RNN
# `models = ["RNN", "LSTM", "GRU"]`

# %%
model_wrapper_RNN = RNNModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
    model="RNN",
)
model_wrapper_RNN.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_RNN)

# %%
configurable_hyperparams = model_wrapper_RNN.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_RNN.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_RNN)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_RNN.work_dir, model_wrapper_RNN.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ### BlockRNN
# `models = ["RNN", "LSTM", "GRU"]`

# %%
model_wrapper_BlockRNN = BlockRNNModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
    model="RNN",
)
model_wrapper_BlockRNN.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_BlockRNN)

# %%
configurable_hyperparams = model_wrapper_BlockRNN.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_BlockRNN.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_BlockRNN)

# %%
tensorboard_logs = pathlib.Path(
    model_wrapper_BlockRNN.work_dir, model_wrapper_BlockRNN.model_name, "logs"  # type: ignore[arg-type]
)
print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ## Statistical Models

# %% [markdown]
# ### AutoARIMA

# %%
model_wrapper_AutoARIMA = AutoARIMAWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={
        "time_bin_size_in_minutes": 10,
        # "m_AutoARIMA": 0,  # Runs extremely slow...
    },
)
model_wrapper_AutoARIMA.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_AutoARIMA)

# %%
configurable_hyperparams = model_wrapper_AutoARIMA.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_AutoARIMA.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_AutoARIMA)

# %% [markdown]
# ### BATS

# %%
model_wrapper_BATS = BATSWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_BATS.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_BATS)

# %%
configurable_hyperparams = model_wrapper_BATS.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_BATS.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_BATS)

# %% [markdown]
# ### TBATS

# %%
model_wrapper_TBATS = TBATSWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_TBATS.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_TBATS)

# %%
configurable_hyperparams = model_wrapper_TBATS.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_TBATS.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_TBATS)

# %% [markdown]
# ### FourTheta

# %%
model_wrapper_FourTheta = FourThetaWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_FourTheta.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_FourTheta)

# %%
configurable_hyperparams = model_wrapper_FourTheta.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_FourTheta.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_FourTheta)

# %% [markdown]
# ### StatsForecastAutoTheta

# %%
model_wrapper_StatsForecastAutoTheta = StatsForecastAutoThetaWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={
        "time_bin_size_in_minutes": 10,
        "season_length_StatsForecastAutoTheta": 0,
    },
)
model_wrapper_StatsForecastAutoTheta.set_work_dir(
    work_dir_relative_to_base=pathlib.Path("local_dev")
)
# print(model_wrapper_StatsForecastAutoTheta)

# %%
configurable_hyperparams = model_wrapper_StatsForecastAutoTheta.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_StatsForecastAutoTheta.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_StatsForecastAutoTheta)

# %% [markdown]
# ### FFT

# %%
model_wrapper_FFT = FFTWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_FFT.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_FFT)

# %%
configurable_hyperparams = model_wrapper_FFT.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_FFT.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_FFT)

# %% [markdown]
# ### KalmanForecaster

# %%
model_wrapper_KalmanForecaster = KalmanForecasterWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_KalmanForecaster.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_KalmanForecaster)

# %%
configurable_hyperparams = model_wrapper_KalmanForecaster.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_KalmanForecaster.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_KalmanForecaster)

# %% [markdown]
# ### Croston
# `versions = ["classic", "optimized", "sba"]`
#
# Do not use `"tsb"` as `alpha_d` and `alpha_p` must be set

# %%
model_wrapper_Croston = CrostonWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
    version="optimized",
)
model_wrapper_Croston.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_Croston)

# %%
configurable_hyperparams = model_wrapper_Croston.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_Croston.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_Croston)

# %% [markdown]
# ## Regression Models

# %% [markdown]
# ### LinearRegressionModel

# %%
model_wrapper_LinearRegressionModel = LinearRegressionModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_LinearRegressionModel.set_work_dir(
    work_dir_relative_to_base=pathlib.Path("local_dev")
)
# print(model_wrapper_LinearRegressionModel)

# %%
configurable_hyperparams = model_wrapper_LinearRegressionModel.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_LinearRegressionModel.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_LinearRegressionModel)

# %% [markdown]
# ### RandomForest

# %%
model_wrapper_RandomForest = RandomForestWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_RandomForest.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_RandomForest)

# %%
configurable_hyperparams = model_wrapper_RandomForest.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_RandomForest.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_RandomForest)

# %% [markdown]
# ### LightGBMModel

# %%
model_wrapper_LightGBMModel = LightGBMModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_LightGBMModel.verbose = -1  # Silence [LightGBM] [Info] messages
model_wrapper_LightGBMModel.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_LightGBMModel)

# %%
configurable_hyperparams = model_wrapper_LightGBMModel.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_LightGBMModel.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_LightGBMModel)

# %% [markdown]
# ### XGBModel

# %%
model_wrapper_XGBModel = XGBModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_XGBModel.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_XGBModel)

# %%
configurable_hyperparams = model_wrapper_XGBModel.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_XGBModel.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_XGBModel)

# %% [markdown]
# ### CatBoostModel

# %%
model_wrapper_CatBoostModel = CatBoostModelWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_CatBoostModel.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))
# print(model_wrapper_CatBoostModel)

# %%
configurable_hyperparams = model_wrapper_CatBoostModel.get_configurable_hyperparams()
pprint.pprint(configurable_hyperparams)

# %%
loss_val, metrics_val = model_wrapper_CatBoostModel.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
print(model_wrapper_CatBoostModel)

# %% [markdown]
# ## Naive Models

# %% [markdown]
# ### NaiveMean

# %%
model_wrapper_NaiveMean = NaiveMeanWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_NaiveMean.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))

loss_val, metrics_val = model_wrapper_NaiveMean.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
# print(model_wrapper_NaiveMean)

# %% [markdown]
# ### NaiveSeasonal

# %%
model_wrapper_NaiveSeasonal = NaiveSeasonalWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_NaiveSeasonal.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))

loss_val, metrics_val = model_wrapper_NaiveSeasonal.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
# print(model_wrapper_NaiveSeasonal)

# %% [markdown]
# ### NaiveDrift

# %%
model_wrapper_NaiveDrift = NaiveDriftWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_NaiveDrift.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))

loss_val, metrics_val = model_wrapper_NaiveDrift.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
# print(model_wrapper_NaiveDrift)

# %% [markdown]
# ### NaiveMovingAverage

# %%
model_wrapper_NaiveMovingAverage = NaiveMovingAverageWrapper(
    TSModelWrapper=PARENT_WRAPPER,
    variable_hyperparams={"time_bin_size_in_minutes": 10},
)
model_wrapper_NaiveMovingAverage.set_work_dir(work_dir_relative_to_base=pathlib.Path("local_dev"))

loss_val, metrics_val = model_wrapper_NaiveMovingAverage.train_model()
print(f"metrics_val = {pprint.pformat(metrics_val)}")

# %%
# print(model_wrapper_NaiveMovingAverage)

# %% [markdown]
# ***
# # Bayesian Optimization

# %%
# raise UserWarning("Stopping Here")

# %% [markdown]
# ## Setup

# %%
BAYESIAN_OPT_WORK_DIR_NAME: Final = "bayesian_optimization"

# %% [markdown]
# ### Create inputs for `bayesian_opt_runner.py`

# %%
PARENT_WRAPPER_PATH: Final = MODELS_PATH / BAYESIAN_OPT_WORK_DIR_NAME / "parent_wrapper.pickle"
if not PARENT_WRAPPER_PATH.is_file():
    if "PARENT_WRAPPER" not in globals():
        print("PARENT_WRAPPER not defined, can not write pickle!")
    else:
        write_secure_pickle(PARENT_WRAPPER, PARENT_WRAPPER_PATH)

# %% [markdown]
# ## Show TensorBoard Logs

# %%
# tensorboard_logs = pathlib.Path(PARENT_WRAPPER.work_dir_base, BAYESIAN_OPT_WORK_DIR_NAME)
# print(tensorboard_logs)

# %%
# # %tensorboard --logdir $tensorboard_logs

# %% [markdown]
# ## Review Best Results

# %%
dfp_best_points, dfp_runs_dict = load_best_points(MODELS_PATH / BAYESIAN_OPT_WORK_DIR_NAME)

# %%
# with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
#     display(dfp_best_points)

# %%
best_model = dfp_best_points["generic_model_name"].iloc[0]
print(f"{best_model = }")

# %%
# with pd.option_context("display.max_rows", None, "display.max_colwidth", None):
#     display(dfp_runs_dict[best_model])

# %% [markdown]
# ### Write results to xlsx

# %%
f_excel = MODELS_PATH / BAYESIAN_OPT_WORK_DIR_NAME / "search_results.xlsx"
write_search_results(f_excel, dfp_best_points, dfp_runs_dict)

# %% [markdown]
# ***
# # Explore the Data

# %%
# raise UserWarning("Stopping Here")

# %% [markdown]
# ## Quick look at morning commute shower for past 30 days

# %%
dt_recent_mornings_start = dt_stop_local - dt.timedelta(days=30)
dt_recent_mornings_end = dt_stop_local

dfp_recent_mornings = dfp_data.loc[
    (dt_recent_mornings_start <= dfp_data["datetime_local"])
    & (dfp_data["datetime_local"] <= dt_recent_mornings_end)
    & (7 <= dfp_data["datetime_local"].dt.hour)
    & (dfp_data["datetime_local"].dt.hour <= 9)
    # Monday=0, Sunday=6
    & dfp_data["day_of_week_int"].isin([1, 2, 3])
]

print(
    f'Mean pressure during morning commute shower for past 30 days = {100*dfp_recent_mornings["mean_pressure_value_normalized"].mean():.2f} ± {dfp_recent_mornings["mean_pressure_value_normalized"].std():.2%}'
)

# %%
plot_2d_hist(
    dfp_recent_mornings["datetime_local_same_week"],
    100 * dfp_recent_mornings["mean_pressure_value_normalized"],
    m_path=OUTPUTS_PATH,
    fname="mean_pressure_value_normalized_vs_time_of_week",
    tag="_recent_mornings",
    dt_start=dt_recent_mornings_start,
    dt_stop=dt_recent_mornings_end,
    binning={
        "x": {
            "bin_edges": make_epoch_bins(
                dt_common, dt_common + dt.timedelta(days=7), 60 * 60
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
        "ticks": make_epoch_bins(dt_common, dt_common + dt.timedelta(days=7), 12 * 60 * 60),
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
display_image(OUTPUTS_PATH / "mean_pressure_value_normalized_vs_time_of_week_recent_mornings.png")

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
        "values": dfp_data.loc[dfp_data["had_flow"] != 1, "mean_pressure_value"].to_numpy(),
        "label": "No Flow",
        "density": True,
        "c": MC_FLOW_0,
    },
    {
        "values": dfp_data.loc[dfp_data["had_flow"] == 1, "mean_pressure_value"].to_numpy(),
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
display_image(OUTPUTS_PATH / "mean_pressure_value_density.png")

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
                dt_common, dt_common + dt.timedelta(days=1), 15 * 60
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
        "ticks": make_epoch_bins(dt_common, dt_common + dt.timedelta(days=1), 2 * 60 * 60),
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
display_image(OUTPUTS_PATH / "mean_pressure_value_normalized_vs_time_of_day.png")

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
                dt_common, dt_common + dt.timedelta(days=7), 60 * 60
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
        "ticks": make_epoch_bins(dt_common, dt_common + dt.timedelta(days=7), 12 * 60 * 60),
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
display_image(OUTPUTS_PATH / "mean_pressure_value_normalized_vs_time_of_week.png")

# %% [markdown]
# ## Time Series of Selected Pressure Values - For Web

# %%
dt_plotly_web_selection_start = dt.datetime(year=2023, month=11, day=1, tzinfo=LOCAL_TIMEZONE)
dt_plotly_web_selection_end = dt.datetime(year=2023, month=12, day=1, tzinfo=LOCAL_TIMEZONE)

dfp_plotly_web_selection = dfp_data.loc[
    (dt_plotly_web_selection_start <= dfp_data["datetime_local"])
    & (dfp_data["datetime_local"] <= dt_plotly_web_selection_end)
]

# %% [markdown]
# ### Raw

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
# ### Normalized

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
