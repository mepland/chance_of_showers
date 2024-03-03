"""Setup Bayesian optimization."""

import datetime
import gc
import json
import os
import pathlib
import platform
import pprint
import re
import signal
import traceback
import warnings
from contextlib import suppress
from csv import writer
from types import FrameType  # noqa: TC003
from typing import TYPE_CHECKING, Final, TypeAlias

import bayes_opt
import humanize
import numpy as np
import pandas as pd
import psutil
import torch
import xlsxwriter
from bayes_opt.event import DEFAULT_EVENTS, Events
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.util import load_logs

# isort: off
from utils.TSModelWrapper import TSModelWrapper, BAD_TARGET, METRICS_KEYS

# Prophet
from utils.ProphetWrapper import ProphetWrapper  # noqa: TC001

# PyTorch NN Models
from utils.NBEATSModelWrapper import NBEATSModelWrapper  # noqa: TC001
from utils.NHiTSModelWrapper import NHiTSModelWrapper  # noqa: TC001
from utils.TCNModelWrapper import TCNModelWrapper  # noqa: TC001
from utils.TransformerModelWrapper import TransformerModelWrapper  # noqa: TC001
from utils.TFTModelWrapper import TFTModelWrapper  # noqa: TC001
from utils.DLinearModelWrapper import DLinearModelWrapper  # noqa: TC001
from utils.NLinearModelWrapper import NLinearModelWrapper  # noqa: TC001
from utils.TiDEModelWrapper import TiDEModelWrapper  # noqa: TC001
from utils.RNNModelWrapper import RNNModelWrapper  # noqa: TC001
from utils.BlockRNNModelWrapper import BlockRNNModelWrapper  # noqa: TC001

# Statistical Models
from utils.AutoARIMAWrapper import AutoARIMAWrapper  # noqa: TC001
from utils.BATSWrapper import BATSWrapper  # noqa: TC001
from utils.TBATSWrapper import TBATSWrapper  # noqa: TC001
from utils.FourThetaWrapper import FourThetaWrapper  # noqa: TC001
from utils.StatsForecastAutoThetaWrapper import StatsForecastAutoThetaWrapper  # noqa: TC001
from utils.FFTWrapper import FFTWrapper  # noqa: TC001
from utils.KalmanForecasterWrapper import KalmanForecasterWrapper  # noqa: TC001
from utils.CrostonWrapper import CrostonWrapper  # noqa: TC001

# Regression Models
from utils.LinearRegressionModelWrapper import LinearRegressionModelWrapper  # noqa: TC001
from utils.RandomForestWrapper import RandomForestWrapper  # noqa: TC001
from utils.LightGBMModelWrapper import LightGBMModelWrapper  # noqa: TC001
from utils.XGBModelWrapper import XGBModelWrapper  # noqa: TC001
from utils.CatBoostModelWrapper import CatBoostModelWrapper  # noqa: TC001

# Naive Models
from utils.NaiveMeanWrapper import NaiveMeanWrapper  # noqa: TC001
from utils.NaiveSeasonalWrapper import NaiveSeasonalWrapper  # noqa: TC001
from utils.NaiveDriftWrapper import NaiveDriftWrapper  # noqa: TC001
from utils.NaiveMovingAverageWrapper import NaiveMovingAverageWrapper  # noqa: TC001

__all__ = ["load_best_points", "load_json_log_to_dfp", "print_memory_usage", "run_bayesian_opt"]


# isort: on

# When showing warnings ignore everything except the message
# https://stackoverflow.com/a/2187390
warnings.formatwarning = lambda msg, *args, **kwargs: f"{msg}\n"  # noqa: U100

WrapperTypes: TypeAlias = type[
    # Prophet
    ProphetWrapper
    # PyTorch NN Models
    | NBEATSModelWrapper
    | NHiTSModelWrapper
    | TCNModelWrapper
    | TransformerModelWrapper
    | TFTModelWrapper
    | DLinearModelWrapper
    | NLinearModelWrapper
    | TiDEModelWrapper
    | RNNModelWrapper
    | BlockRNNModelWrapper
    # Statistical Models
    | AutoARIMAWrapper
    | BATSWrapper
    | TBATSWrapper
    | FourThetaWrapper
    | StatsForecastAutoThetaWrapper
    | FFTWrapper
    | KalmanForecasterWrapper
    | CrostonWrapper
    # Regression Models
    | LinearRegressionModelWrapper
    | RandomForestWrapper
    | LightGBMModelWrapper
    | XGBModelWrapper
    | CatBoostModelWrapper
    # Naive Models
    | NaiveMeanWrapper
    | NaiveSeasonalWrapper
    | NaiveDriftWrapper
    | NaiveMovingAverageWrapper
]

BAYESIAN_OPT_PREFIX: Final = "bayesian_opt_"

BAD_METRICS: Final = {str(k): -BAD_TARGET for k in METRICS_KEYS}

BAYES_OPT_LOG_COLS_FIXED: Final = [
    "datetime",
    "i_iter",
    "i_point",
    "is_clean",
    "represents_iter",
    "target",
] + [f"{_}_val_loss" for _ in METRICS_KEYS]


def clean_log_dfp(
    dfp: pd.DataFrame | None, *, exclude_outlier_minutes_elapsed_iteration: float = 100.0
) -> None | pd.DataFrame:
    """Clean and augment log dataframe.

    Args:
        dfp: Log as pandas dataframe.
        exclude_outlier_minutes_elapsed_iteration: Flag to exclude outliers minutes_elapsed_iteration from runs in different sessions.
            0: Do not exclude outliers
            -1: Exclude outliers greater than the 99th percentile
            x: Exclude outliers greater than Q3+x*IQR. Default is 100.0, i.e. extreme outliers in comparison to the standard 1.5.

    Returns:
        Log as pandas dataframe, cleaned and augmented.

    Raises:
        ValueError: Bad configuration.
    """
    if dfp is None:
        return None

    if "is_clean" in dfp.columns:
        dfp["is_clean"] = dfp["is_clean"].astype(bool)

    # Setup dfp_minutes for calculations. The datetime format here is set by bayes_opt.
    dfp["datetime"] = pd.to_datetime(dfp["datetime"], format="%Y-%m-%d %H:%M:%S")

    dfp_minutes = pd.DataFrame(dfp)

    dfp_minutes["minutes_elapsed_total"] = (
        dfp_minutes["datetime"] - dfp_minutes["datetime"].min()
    ) / pd.Timedelta(minutes=1)

    dfp_minutes = (
        dfp_minutes.groupby(["datetime"])
        .agg({"minutes_elapsed_total": "max"})
        .reset_index()
        .sort_values(by="datetime", ascending=True)
        .reset_index(drop=True)
    )

    if len(dfp["i_iter"].unique()) == 1:
        # Everything has i_iter=0, i.e. ran via the shell script calling python multiple times,
        # recompute i_iter based on datetime field.
        dfp_minutes["i_iter"] = dfp_minutes.index

        dfp = dfp.drop("i_iter", axis=1)
        dfp = dfp.merge(dfp_minutes[["datetime", "i_iter"]], how="left", on="datetime")
        dfp_minutes = dfp_minutes.drop("i_iter", axis=1)

    # Use is_clean if available to select the representative i_point per i_iter
    # Otherwise, take the last i_point per i_iter
    dfp["represents_iter"] = (
        dfp.sort_values(
            by=["is_clean", "i_iter"] if "is_clean" in dfp.columns else ["i_iter"],
            ascending=[False, True] if "is_clean" in dfp.columns else [False],
        )
        .groupby("i_iter", sort=False)
        .cumcount()
        .add(1)
        == 1
    )

    # compute minutes_elapsed_ columns
    dfp_minutes["minutes_elapsed_iteration"] = (
        dfp_minutes["minutes_elapsed_total"].diff().fillna(0.0)
    )

    if exclude_outlier_minutes_elapsed_iteration != 0:
        if exclude_outlier_minutes_elapsed_iteration == -1:
            outlier_threshold = dfp_minutes["minutes_elapsed_iteration"].quantile(0.99)
        else:
            outlier_threshold = dfp_minutes["minutes_elapsed_iteration"].quantile(0.75)
            iqr = dfp_minutes["minutes_elapsed_iteration"].quantile(0.75) - dfp_minutes[
                "minutes_elapsed_iteration"
            ].quantile(0.25)

            outlier_threshold += exclude_outlier_minutes_elapsed_iteration * iqr

        dfp_minutes["minutes_elapsed_iteration_full"] = dfp_minutes["minutes_elapsed_iteration"]

        dfp_minutes.loc[
            outlier_threshold < dfp_minutes["minutes_elapsed_iteration"],
            "minutes_elapsed_iteration",
        ] = np.nan

        # recompute minutes_elapsed_total
        dfp_minutes["minutes_elapsed_total"] = (
            dfp_minutes["minutes_elapsed_iteration"].fillna(0.0).cumsum()
        )

    dfp = (
        dfp.merge(dfp_minutes, how="left", on="datetime")
        .sort_values(by="datetime", ascending=True)
        .reset_index(drop=True)
    )

    return dfp[
        [_ for _ in BAYES_OPT_LOG_COLS_FIXED if _ in dfp.columns]
        + [_ for _ in dfp.columns if _ not in BAYES_OPT_LOG_COLS_FIXED]
    ]


def load_json_log_to_dfp(f_path: pathlib.Path) -> None | pd.DataFrame:
    """Load prior bayes_opt log from json file as a pandas dataframe.

    Args:
        f_path: Path to json log file.

    Returns:
        Log as pandas dataframe.
    """
    # Adapted from:
    # https://github.com/bayesian-optimization/BayesianOptimization/blob/129caac02177b146ce315e177d4d88950b75253a/bayes_opt/util.py#L214-L241
    with f_path.open("r", encoding="utf-8") as f_json:
        rows = []
        while True:
            try:
                iteration = next(f_json)
            except StopIteration:
                break

            row = {}
            for _k0, _v0 in dict(sorted(json.loads(iteration).items())).items():
                if isinstance(_v0, dict):
                    for _k1, _v1 in dict(sorted(_v0.items())).items():
                        if _k0 == "datetime":
                            if _k1 != "datetime":
                                continue

                            _prefix = ""
                        else:
                            _prefix = f"{_k0}_"

                        row[f"{_prefix}{_k1}"] = _v1
                else:
                    row[_k0] = _v0

            rows.append(row)

        f_json.close()

        if rows:
            dfp = pd.DataFrame(rows)
            dfp["i_point"] = dfp.index

            return clean_log_dfp(dfp)

        return None


def load_csv_log_to_dfp(f_path: pathlib.Path) -> None | pd.DataFrame:
    """Load prior bayes_opt log from csv file as a pandas dataframe.

    Args:
        f_path: Path to csv log file.

    Returns:
        Log as pandas dataframe.
    """
    with f_path.open("r", encoding="utf-8") as f_csv:
        dfp = pd.read_csv(f_csv, header=0)

        return clean_log_dfp(dfp)

    return None


def load_best_points(
    dir_path: pathlib.Path, *, use_csv: bool = True
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load best points from all bayes_opt, csv or json, log files in the dir_path.

    Args:
        dir_path: Path to search recursively for csv, or json, log files.
        use_csv: Flag to load csv files, rather than json, log files.

    Returns:
        Best points with metadata as pandas dataframe, and dict of all logs as pandas dataframes.

    Raises:
        ValueError: Could not load from disk, or found duplicate model_name.
    """
    dfp_runs_dict = {}
    rows = []
    for f_path in sorted(dir_path.glob(f"**/*.{'csv' if use_csv else 'json'}")):
        model_name = f_path.stem.replace(BAYESIAN_OPT_PREFIX, "")

        if use_csv:
            dfp = load_csv_log_to_dfp(f_path)
        else:
            dfp = load_json_log_to_dfp(f_path)

        if dfp is None:
            raise ValueError(f"Could load {f_path}!")

        if TYPE_CHECKING:
            assert isinstance(dfp, pd.DataFrame)  # noqa: SCS108 # nosec assert_used

        if model_name in dfp_runs_dict:
            raise ValueError(
                f"Already loaded log for {model_name}! Please clean the dir structure of {dir_path} and try again."
            )

        dfp_runs_dict[model_name] = pd.DataFrame(dfp)

        dfp_best_points = dfp.loc[dfp["target"] != BAD_TARGET]

        if "represents_iter" in dfp_best_points.columns:
            dfp_best_points = dfp_best_points.loc[
                (dfp["target"] == dfp["target"].max()) & dfp["represents_iter"]
            ]
        else:
            dfp_best_points = dfp_best_points.loc[dfp["target"] == dfp["target"].max()]

        if not dfp_best_points.index.size:
            dfp_best_points = pd.DataFrame(dfp)
            warnings.warn(
                f"Could not find a best point for {model_name} in {f_path}, just taking them all!",
                stacklevel=1,
            )

        dfp_best_points = dfp_best_points.sort_values(
            by=["is_clean", "i_point"] if "is_clean" in dfp_best_points.columns else ["i_point"],
            ascending=[False, True] if "is_clean" in dfp_best_points.columns else [True],
        )

        best_dict = dfp_best_points.iloc[0].to_dict()

        best_params = []
        for k, v in best_dict.items():
            if k.startswith("params_"):
                best_params.append(f'{k.replace("params_", "")}: {v}')

        rows.append(
            {
                "model_name": model_name,
                "target_best": best_dict["target"],
                "n_points": dfp["i_point"].max() + 1,
                "n_points_bad_target": dfp.loc[dfp["target"] == BAD_TARGET].index.size,
                "n_points_representative": dfp.loc[dfp["represents_iter"]].index.size,
                "n_points_representative_bad_target": dfp.loc[
                    (dfp["target"] == BAD_TARGET) & dfp["represents_iter"]
                ].index.size,
                "minutes_elapsed": dfp["minutes_elapsed_total"].max(),
                "minutes_elapsed_iteration_best": best_dict["minutes_elapsed_iteration"],
                "i_point_best": best_dict["i_point"],
                "datetime_best": best_dict["datetime"],
                "params_best": ", ".join(best_params),
            }
        )

    dfp_best_points = pd.DataFrame(rows)
    dfp_best_points = dfp_best_points.sort_values(
        by=["target_best", "model_name", "datetime_best"], ascending=[False, True, False]
    ).reset_index(drop=True)

    # Sort dfp_runs_dict in the same order as dfp_best_points
    # https://stackoverflow.com/a/21773891
    index_map = {v: i for i, v in enumerate(dfp_best_points["model_name"].to_list())}
    dfp_runs_dict = dict(sorted(dfp_runs_dict.items(), key=lambda pair: index_map[pair[0]]))

    return dfp_best_points, dfp_runs_dict


def write_search_results(  # noqa: C901
    f_excel: pathlib.Path,
    dfp_best_points: pd.DataFrame,
    dfp_runs_dict: dict[str, pd.DataFrame],
    *,
    bad_points_frac_thr: float = 0.2,
) -> None:
    """Load prior bayes_opt log from json file as a pandas dataframe.

    Args:
        f_excel: Path to output xlsx file.
        dfp_best_points: Best points dataframe created by load_best_points().
        dfp_best_points: Dict of all logs as pandas dataframes load_best_points().
        bad_points_frac_thr: Bad points fraction threshold for red formatting.
    """
    with pd.ExcelWriter(f_excel, engine="xlsxwriter") as xlsx_writer:
        workbook = xlsx_writer.book
        # Setup formats
        elapsed_minutes_fmt = workbook.add_format({"num_format": "0.00"})
        elapsed_minutes_fmt_bar = {
            "type": "data_bar",
            "bar_solid": True,
            "bar_no_border": True,
            "bar_direction": "right",
            "bar_color": "#4a86e8",
        }
        boolean_fmt = workbook.add_format({"num_format": "BOOLEAN"})
        loss_fmt = workbook.add_format({"num_format": "0.000000"})
        loss_color_fmt = {
            "type": "3_color_scale",
            "min_color": "#57bb8a",
            "mid_color": "#ffffff",
            "max_color": "#e67c73",
        }
        target_color_fmt = {
            "type": "3_color_scale",
            "min_value": -0.015,
            "min_color": loss_color_fmt["max_color"],
            "mid_value": -0.01,
            "mid_color": loss_color_fmt["mid_color"],
            "max_value": -0.005,
            "max_color": loss_color_fmt["min_color"],
        }
        red_format = workbook.add_format({"bg_color": "#e67c73"})
        bad_points_color_fmt = {
            "type": "cell",
            "criteria": ">=",
            "format": red_format,
        }
        for k in ["min_type", "mid_type", "max_type"]:
            loss_color_fmt[k] = "num"
            target_color_fmt[k] = "num"

        def _fmt_worksheet(
            worksheet: xlsxwriter.worksheet.Worksheet,
            dfp_source: pd.DataFrame,
            *,
            hide_non_represents_iter: bool = False,
        ) -> None:
            """Format a log worksheet for this project

            Args:
                worksheet: Input worksheet.
                dfp_source: Orignal dataframe.
                hide_non_represents_iter: Hide non-represents_iter columns.
            """
            # Format loss columns
            for i_col, col_str in enumerate(dfp_source.columns):
                if not re.match(r"^.*?_val_loss$", col_str):
                    continue

                _dfp = dfp_source.loc[dfp_source[col_str] != -BAD_TARGET]
                _min = _dfp[col_str].min()
                _max = _dfp[col_str].max()
                loss_color_fmt["min_value"] = _min
                loss_color_fmt["mid_value"] = _min + (_max - _min) / 2.0
                loss_color_fmt["max_value"] = _max

                worksheet.set_column(i_col, i_col, None, loss_fmt)
                worksheet.conditional_format(1, i_col, dfp_source.shape[0], i_col, loss_color_fmt)

            # Format target columns
            for i_col, col_str in enumerate(dfp_source.columns):
                if not re.match(r"^target.*$", col_str):
                    continue

                worksheet.set_column(i_col, i_col, None, loss_fmt)
                worksheet.conditional_format(1, i_col, dfp_source.shape[0], i_col, target_color_fmt)

            # Format minutes elapsed columns
            for i_col, col_str in enumerate(dfp_source.columns):
                if not re.match(r"^minutes_elapsed.*$", col_str):
                    continue

                worksheet.set_column(i_col, i_col, None, elapsed_minutes_fmt)

                elapsed_minutes_fmt_bar["min_value"] = dfp_source[col_str].min()
                elapsed_minutes_fmt_bar["max_value"] = dfp_source[col_str].max()

                worksheet.conditional_format(
                    1, i_col, dfp_source.shape[0], i_col, elapsed_minutes_fmt_bar
                )

            # Format n_points_ based on percent of n_points
            for col_str, col_denom in {
                "n_points_bad_target": "n_points",
                "n_points_representative_bad_target": "n_points_representative",
            }.items():
                if {col_str, col_denom}.issubset(set(dfp_source.columns)):
                    _i_col = list(dfp_source.columns).index(col_str)
                    for i_row in range(1, dfp_source.shape[0] + 1):
                        bad_points_color_fmt["value"] = (
                            bad_points_frac_thr * dfp_source[col_denom].iloc[i_row - 1]
                        )
                        worksheet.conditional_format(
                            i_row, _i_col, i_row, _i_col, bad_points_color_fmt
                        )

            for i_col, col_str in enumerate(dfp_source.columns):
                if col_str not in ["is_clean", "represents_iter"]:
                    continue

                worksheet.set_column(i_col, i_col, None, boolean_fmt)

            # Filter columns
            worksheet.autofilter(0, 0, dfp_source.shape[0], dfp_source.shape[1] - 1)

            if "represents_iter" in dfp_source.columns:
                _i_col = list(dfp_source.columns).index("represents_iter")
                worksheet.filter_column(_i_col, "x == TRUE")

                # Hide rows which do not match the filter criteria
                for i_row, row in dfp_source.iterrows():  # type: ignore[assignment]
                    if not row["represents_iter"]:
                        worksheet.set_row(i_row + 1, options={"hidden": True})

            # Autofit column widths
            worksheet.autofit()

            # hide non-represents_iter columns
            if hide_non_represents_iter:
                for i_col, col_str in enumerate(dfp_source.columns):
                    if col_str not in ["n_points", "n_points_bad_target"]:
                        continue

                    worksheet.set_column(i_col, i_col, None, options={"hidden": True})

        # Write and format sheets
        dfp_best_points.to_excel(
            xlsx_writer, sheet_name="Best Points", freeze_panes=(1, 1), index=False
        )
        _fmt_worksheet(
            xlsx_writer.sheets["Best Points"], dfp_best_points, hide_non_represents_iter=True
        )

        for model_name, dfp in dfp_runs_dict.items():
            dfp.to_excel(xlsx_writer, sheet_name=model_name, freeze_panes=(1, 1), index=False)
            _fmt_worksheet(xlsx_writer.sheets[model_name], dfp)


def print_memory_usage(*, header: str | None = None) -> None:
    """Print system memory usage statistics.

    Args:
        header: Header to print before the rest of the memory usage.
    """
    ram_info = psutil.virtual_memory()
    process = psutil.Process()
    if header is not None and header != "":
        header = f"{header}\n"
    else:
        header = ""

    memory_usage_str = (
        header
        + f"RAM Available: {humanize.naturalsize(ram_info.available)}, "
        + f"System Used: {humanize.naturalsize(ram_info.used)}, {ram_info.percent:.2f}%, "
        + f"Process Used: {humanize.naturalsize(process.memory_info().rss)}"
    )

    if torch.cuda.is_available():
        gpu_memory_stats = {}
        with suppress(Exception):
            gpu_memory_stats = torch.cuda.memory_stats()

        def get_gpu_mem_key(key: str) -> str:
            """Print system memory usage statistics.

            Args:
                key: Key to get from gpu_memory_stats.

            Returns:
                Clean humanized string for printing.
            """
            _v = gpu_memory_stats.get(key)
            if _v is not None:
                return str(humanize.naturalsize(_v))

            return "MISSING"

        memory_usage_str += (
            f", GPU RAM Current: {get_gpu_mem_key('allocated_bytes.all.current')}, "
            + f"Peak: {get_gpu_mem_key('allocated_bytes.all.peak')}"
        )

    print(memory_usage_str)


n_points = 0  # # pylint: disable=invalid-name


def run_bayesian_opt(  # noqa: C901 # pylint: disable=too-many-statements,too-many-locals
    *,
    parent_wrapper: TSModelWrapper,
    model_wrapper_class: WrapperTypes,
    model_wrapper_kwargs: dict | None = None,
    hyperparams_to_opt: list[str] | None = None,
    n_iter: int = 100,
    allow_duplicate_points: bool = False,
    utility_kind: str = "ucb",
    utility_kappa: float = 2.576,
    verbose: int = 3,
    model_verbose: int = -1,
    enable_torch_warnings: bool = False,
    enable_torch_model_summary: bool = True,
    enable_torch_progress_bars: bool = False,
    disregard_training_exceptions: bool = False,
    max_time_per_model: datetime.timedelta | None = None,
    accelerator: str | None = "auto",
    fixed_hyperparams_to_alter: dict | None = None,
    enable_json_logging: bool = True,
    enable_reloading: bool = True,
    enable_model_saves: bool = False,
    bayesian_opt_work_dir_name: str = "bayesian_optimization",
) -> tuple[dict, bayes_opt.BayesianOptimization, int]:
    """Run Bayesian optimization for this model wrapper.

    Args:
        parent_wrapper: TSModelWrapper object containing all parent configs.
        model_wrapper_class: TSModelWrapper class to optimize.
        model_wrapper_kwargs: kwargs to passs to model_wrapper.
        hyperparams_to_opt: List of hyperparameters to optimize.
            If None, use all configurable hyperparameters.
        n_iter: How many iterations of Bayesian optimization to perform.
            This is the number of new models to train, in addition to any duplicated or reloaded points.
        allow_duplicate_points: If True, the optimizer will allow duplicate points to be registered.
            This behavior may be desired in high noise situations where repeatedly probing
            the same point will give different answers. In other situations, the acquisition
            may occasionally generate a duplicate point.
        utility_kind: {'ucb', 'ei', 'poi'}
            * 'ucb' stands for the Upper Confidence Bounds method
            * 'ei' is the Expected Improvement method
            * 'poi' is the Probability Of Improvement criterion.
        utility_kappa: Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is the highest.
        verbose: Optimizer verbosity
            7 prints memory usage
            6 prints points before training
            5 prints point count at each iteration
            4 prints full stack traces
            3 prints basic workflow messages
            2 prints all iterations
            1 prints only when a maximum is observed
            0 is silent
        model_verbose: Verbose level of model_wrapper, default is -1 to silence LightGBMModel.
        enable_torch_warnings: Enable torch warning messages about training devices and CUDA, globally, via the logging module.
        enable_torch_model_summary: Enable torch model summary.
        enable_torch_progress_bars: Enable torch progress bars.
        disregard_training_exceptions: Flag to disregard all exceptions raised when training a model, and return BAD_TARGET instead.
        max_time_per_model: Set the maximum amount of training time for each iteration.
            Torch models will use max_time_per_model as the max time per epoch,
            while non-torch models will use it for the whole iteration if signal is avaliable e.g. Linux, Darwin.
        accelerator: Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto")
        fixed_hyperparams_to_alter: Dict of fixed hyperparameters to alter, but not optimize.
        enable_json_logging: Enable JSON logging of points.
        enable_reloading: Enable reloading of prior points from JSON log.
        enable_model_saves: Save the trained model at each iteration.
        bayesian_opt_work_dir_name: Directory name to save logs and models in, within the parent_wrapper.work_dir_base.

    Returns:
        optimal_values: Optimal hyperparameter values.
        optimizer: bayes_opt.BayesianOptimization object for further details.
        exception_status: Int exception status, to pass on to bash scripts.

    Raises:
        ValueError: Bad configuration.
    """
    global n_points

    exception_status = 0

    if model_wrapper_kwargs is None:
        model_wrapper_kwargs = {}

    # Setup hyperparameters
    model_wrapper = model_wrapper_class(TSModelWrapper=parent_wrapper, **model_wrapper_kwargs)
    configurable_hyperparams = model_wrapper.get_configurable_hyperparams()
    if hyperparams_to_opt is None:
        hyperparams_to_opt = list(configurable_hyperparams.keys())

    # Setup hyperparameter bounds
    hyperparam_bounds = {}
    for hyperparam in hyperparams_to_opt:
        hyperparam_min = configurable_hyperparams.get(hyperparam, {}).get("min")
        hyperparam_max = configurable_hyperparams.get(hyperparam, {}).get("max")
        if hyperparam_min is None or hyperparam_max is None:
            raise ValueError(f"Could not load hyperparameter definition for {hyperparam = }!")

        hyperparam_bounds[hyperparam] = (hyperparam_min, hyperparam_max)

    # Setup Bayesian optimization objects
    # https://github.com/bayesian-optimization/BayesianOptimization/blob/11a0c6aba1fcc6b5d2716052da5222a84259c5b9/bayes_opt/util.py#L113
    utility = bayes_opt.UtilityFunction(kind=utility_kind, kappa=utility_kappa)

    optimizer = bayes_opt.BayesianOptimization(
        f=None,
        pbounds=hyperparam_bounds,
        random_state=model_wrapper.get_random_state(),
        verbose=verbose,
        allow_duplicate_points=allow_duplicate_points,
    )

    # Setup Logging
    generic_model_name: Final = model_wrapper.get_generic_model_name()
    bayesian_opt_work_dir: Final = pathlib.Path(
        model_wrapper.work_dir_base, bayesian_opt_work_dir_name, generic_model_name
    ).expanduser()
    fname_json_log: Final = (
        bayesian_opt_work_dir / f"{BAYESIAN_OPT_PREFIX}{generic_model_name}.json"
    )
    fname_csv_log: Final = bayesian_opt_work_dir / f"{BAYESIAN_OPT_PREFIX}{generic_model_name}.csv"

    # Reload prior points, must be done before json_logger is recreated to avoid duplicating past runs
    n_points = 0
    if enable_reloading and fname_json_log.is_file():
        if 3 <= verbose:
            print(f"Resuming Bayesian optimization from:\n{fname_json_log}\n")

        optimizer.dispatch(Events.OPTIMIZATION_START)
        load_logs(optimizer, logs=str(fname_json_log))
        n_points = len(optimizer.space)
        if 3 <= verbose:
            print(f"Loaded {n_points} existing points.\n")

    # Continue to setup logging
    if enable_json_logging:
        bayesian_opt_work_dir.mkdir(parents=True, exist_ok=True)
        json_logger = JSONLogger(path=str(fname_json_log), reset=False)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)

    if verbose:
        screen_logger = ScreenLogger(verbose=verbose)
        for event in DEFAULT_EVENTS:
            optimizer.subscribe(event, screen_logger)

    # Function to write additional metadata per point to a csv
    # Easier than modifying the json_logger from bayes_opt
    def _write_csv_row(
        *,
        datetime_str: str | None,
        i_iter: int,
        i_point: int,
        is_clean: bool,
        target: float,
        metrics_val: dict[str, float],
        point: dict,
    ) -> None:
        """Save validation metrics and other metadata for this iteration and point to csv.

        Args:
            datetime_str: Datetime string of this iteration, from json log.
            i_iter: Index of this iteration.
            i_point: Index of this point.
            is_clean: Flag for if these are cleaned or raw hyperparameters.
            target: Target value.
            metrics_val: Metrics on the validation set.
            point: Hyperparameter point.
        """
        if not enable_json_logging:
            return

        if datetime_str is None:
            datetime_str = "NULL"

        new_row = [datetime_str, i_iter, i_point, int(is_clean), target]
        metrics_val_sorted = {k: metrics_val[str(k)] for k in METRICS_KEYS}
        new_row += list(metrics_val_sorted.values())
        point = dict(sorted(point.items()))
        new_row += list(point.values())

        with fname_csv_log.open("a", encoding="utf-8") as f_csv:
            m_writer = writer(f_csv)
            if f_csv.tell() == 0:
                # empty file, create header
                m_writer.writerow(
                    [_ for _ in BAYES_OPT_LOG_COLS_FIXED if _ != "represents_iter"]
                    + [f"params_{_}" for _ in point.keys()]
                )

            m_writer.writerow(new_row)
            f_csv.close()

    def _get_datetime_str_from_json() -> str | None:
        """Load datatime str from last row in json log.

            The {"datetime: {"datetime": "..."}} timestamp in the json log created by the optimizer.register() call
            is a good field to have, but is only created in in the json logger here:
            https://github.com/bayesian-optimization/BayesianOptimization/blob/129caac02177b146ce315e177d4d88950b75253a/bayes_opt/logger.py#L153C50-L158
            We need to load last line of the json from disk and extract the datatime string.

        Returns:
            datetime_str: Datetime as str.
        """
        last_datetime_str = None
        if enable_json_logging:
            with fname_json_log.open("rb") as f_json:
                # https://stackoverflow.com/a/54278929
                try:  # catch OSError in case of a one line file
                    f_json.seek(-2, os.SEEK_END)
                    while f_json.read(1) != b"\n":
                        f_json.seek(-2, os.SEEK_CUR)
                except OSError:
                    f_json.seek(0)

                last_line = json.loads(f_json.readline().decode())
                last_datetime_str = last_line.get("datetime", {}).get("datetime")

        return last_datetime_str  # noqa: R504

    # Define function to complete an iteration
    def complete_iter(
        i_iter: int,
        model_wrapper: TSModelWrapper,
        target: float,
        metrics_val: dict[str, float],
        point_to_probe: dict,
        *,
        probed_point: dict | None = None,
    ) -> None:
        """Complete this iteration, register point(s) and clean up.

        Args:
            i_iter: Index of this iteration.
            model_wrapper: Model wrapper object to reset.
            target: Target value to register.
            metrics_val: Metrics on the validation set.
            point_to_probe: Raw point to probe.
            probed_point: Point that was actually probed.
        """
        global n_points
        optimizer.register(params=point_to_probe, target=target)
        _write_csv_row(
            datetime_str=_get_datetime_str_from_json(),
            i_iter=i_iter,
            i_point=n_points,
            is_clean=False,
            target=target,
            metrics_val=metrics_val,
            point=point_to_probe,
        )

        n_points += 1
        if probed_point:
            # translate strange hyperparam_values back to original representation
            probed_point = model_wrapper.translate_hyperparameters_to_numeric(probed_point)

            if not (
                # point_to_probe is exactly the same as probed_point on this iter,
                # i.e. the optimizer chose a point that required no cleanup in _assemble_hyperparams().
                # Do not register the probed_point.
                np.array_equiv(
                    optimizer.space.params_to_array(point_to_probe),
                    optimizer.space.params_to_array(probed_point),
                )
            ):
                optimizer.register(params=probed_point, target=target)
                _write_csv_row(
                    datetime_str=_get_datetime_str_from_json(),
                    i_iter=i_iter,
                    i_point=n_points,
                    is_clean=True,
                    target=target,
                    metrics_val=metrics_val,
                    point=probed_point,
                )

                n_points += 1

        model_wrapper.reset_wrapper()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        if 7 <= verbose:
            print_memory_usage()

        if 3 <= verbose:
            print(f"Completed {i_iter = }, with {n_points = }")

    # Setup signal_handler to kill iteration if it runs too long
    def signal_handler(
        dummy_signal: int,  # noqa: U100
        dummy_frame: FrameType | None,  # noqa: U100
    ) -> None:
        """Stop iteration gracefuly.

        https://medium.com/@chamilad/timing-out-of-long-running-methods-in-python-818b3582eed6

        Args:
            dummy_signal: signal number.
            dummy_frame: Frame object.

        Raises:
            RuntimeError: Out of Time!
        """
        raise RuntimeError("Out of Time!")

    max_time_per_model_flag = (
        max_time_per_model is not None
        and not model_wrapper.is_nn
        and platform.system() in ["Linux", "Darwin"]
    )
    if max_time_per_model_flag:
        signal.signal(signal.SIGALRM, signal_handler)

    next_point_to_probe = None
    next_point_to_probe_cleaned = None

    def _build_error_msg(error_msg: str, error: Exception) -> str:
        if 3 <= verbose:
            error_msg = f"""{error_msg}
{error = }"""

        if 4 <= verbose:
            error_msg = f"""{error_msg}
{type(error) = }
{traceback.format_exc()}"""

            if next_point_to_probe is not None:
                error_msg = f"""{error_msg}
next_point_to_probe = {pprint.pformat(next_point_to_probe)}"""

            if next_point_to_probe is not None:
                error_msg = f"""{error_msg}
next_point_to_probe_cleaned = {pprint.pformat(next_point_to_probe_cleaned)}"""

        return error_msg  # noqa: R504

    # Run Bayesian optimization iterations
    try:
        for i_iter in range(n_iter):
            if i_iter == 0:
                optimizer.dispatch(Events.OPTIMIZATION_START)

            if 3 <= verbose:
                print(f"\nStarting {i_iter = }, with {n_points = }")

            next_point_to_probe = optimizer.suggest(utility)

            # Setup model_wrapper
            model_wrapper.alter_fixed_hyperparams(
                fixed_hyperparams_to_alter=fixed_hyperparams_to_alter
            )
            model_wrapper.set_work_dir(work_dir_absolute=bayesian_opt_work_dir)
            model_wrapper.set_enable_torch_messages(
                enable_torch_warnings=enable_torch_warnings,
                enable_torch_model_summary=enable_torch_model_summary,
                enable_torch_progress_bars=enable_torch_progress_bars,
            )
            model_wrapper.set_max_time(max_time=max_time_per_model)
            model_wrapper.set_accelerator(accelerator=accelerator)
            model_wrapper.verbose = model_verbose

            try:
                # Check if we already tested this chosen_hyperparams point
                # If it has been tested, save the raw next_point_to_probe with the same target and continue
                chosen_hyperparams = model_wrapper.preview_hyperparameters(**next_point_to_probe)

                next_point_to_probe_cleaned = {k: chosen_hyperparams[k] for k in hyperparams_to_opt}
                if 6 <= verbose:
                    print(f"next_point_to_probe = {pprint.pformat(next_point_to_probe)}")
                    print(
                        f"next_point_to_probe_cleaned = {pprint.pformat(next_point_to_probe_cleaned)}"
                    )

                is_duplicate_point = False
                for i_param in range(optimizer.space.params.shape[0]):
                    if np.array_equiv(
                        optimizer.space.params_to_array(next_point_to_probe_cleaned),
                        optimizer.space.params[i_param],
                    ):
                        target = optimizer.space.target[i_param]
                        if 3 <= verbose:
                            print(
                                f"On iteration {i_iter} testing prior point {i_param}, returning prior {target = } for the raw next_point_to_probe."
                            )

                        complete_iter(
                            i_iter, model_wrapper, target, BAD_METRICS, next_point_to_probe
                        )
                        is_duplicate_point = True
                        break

                if is_duplicate_point:
                    continue

                # set model_name_tag for this iteration
                model_wrapper.set_model_name_tag(model_name_tag=f"iteration_{n_points}")

                # Setup iteration kill timer
                if max_time_per_model_flag:
                    if TYPE_CHECKING:
                        assert isinstance(  # noqa: SCS108 # nosec assert_used
                            max_time_per_model, datetime.timedelta
                        )

                    signal.alarm(max_time_per_model.seconds)

                # Actually train the model!
                loss_val, metrics_val = model_wrapper.train_model(**next_point_to_probe_cleaned)

                # make the target the negative loss, as we want to maximize the target
                target = -loss_val

                # Put a lower bound on target at BAD_TARGET.
                # This is in case a NN is interrupted mid-epoch and returns a target of -float("inf") or is np.nan.
                if np.isnan(target) or target < BAD_TARGET:
                    target = BAD_TARGET

                # clean any nan metrics
                metrics_val = {
                    k: v if not np.isnan(v) else -BAD_TARGET for k, v in metrics_val.items()
                }

            except KeyboardInterrupt:
                print("KeyboardInterrupt: Ending now!")
                optimizer.dispatch(Events.OPTIMIZATION_END)
                raise
            except Exception as error:
                error_msg = None
                # Expected exceptions
                if "Out of Time!" in str(error):
                    error_msg = "Ran out of time"
                elif "out of memory" in str(error):
                    error_msg = "Ran out of memory"
                elif re.match(
                    r"^Hyperparameter .*? with value .*? is not allowed",
                    str(error),
                ):
                    error_msg = "Bad hyperparameter value, likely caused by additional conditions adjusting the value beyond its limits"
                elif (
                    "Multiplicative seasonality is not appropriate for zero and negative values"
                    in str(error)
                ):
                    error_msg = (
                        "Multiplicative seasonality is not appropriate for zero and negative values"
                    )
                elif re.match(
                    r"^The expanded size of the tensor \(\d*?\) must match the existing size \(\d*?\) at non-singleton dimension",
                    str(error),
                ):
                    error_msg = "Bad value of d_model for this input_chunk_length"
                elif "embed_dim must be divisible by num_heads" in str(error):
                    error_msg = "Bad value of d_model for this nheads"
                elif "Dimension out of range" in str(error):
                    error_msg = str(error)
                # Unexpected exceptions
                elif disregard_training_exceptions:
                    error_msg = (
                        "Unexpected error while training, disregard_training_exceptions is set"
                    )

                # use BAD_TARGET as target and continue
                if error_msg is not None:
                    error_msg = _build_error_msg(error_msg, error)
                    print(
                        f"""{error_msg}
Returning {BAD_TARGET:.3g} as target and continuing"""
                    )
                    complete_iter(
                        i_iter,
                        model_wrapper,
                        BAD_TARGET,
                        BAD_METRICS,
                        next_point_to_probe,
                        probed_point=next_point_to_probe_cleaned,
                    )
                    continue

                # Raise the exception, kill the iterations
                raise
            finally:
                if max_time_per_model_flag:
                    signal.alarm(0)

            if enable_model_saves:
                fname_model = (
                    bayesian_opt_work_dir / f"iteration_{n_points}_{generic_model_name}.pt"
                )
                model_wrapper.get_model().save(fname_model)

            # Register the point
            complete_iter(
                i_iter,
                model_wrapper,
                target,
                metrics_val,
                next_point_to_probe,
                probed_point=next_point_to_probe_cleaned,
            )

    except KeyboardInterrupt:
        exception_status = 1
        print(f"KeyboardInterrupt: Returning with current objects and {exception_status = }.")
    except bayes_opt.util.NotUniqueError as error:
        error_msg = (
            str(error).replace(
                '. You can set "allow_duplicate_points=True" to avoid this error', ""
            )
            + ", stopping optimization here."
        )
        if disregard_training_exceptions:
            error_msg = f"""{error_msg}i
Disregard_training_exceptions is set, continuing!"""
        else:
            exception_status = 2

        error_msg = f"""{error_msg}
Returning with current objects and {exception_status = }."""
    except Exception as error:
        error_msg = _build_error_msg("Unexpected error in run_bayesian_opt():", error)
        if disregard_training_exceptions:
            error_msg = f"""{error_msg}
Disregard_training_exceptions is set, continuing!"""
        else:
            exception_status = 3

        error_msg = f"""{error_msg}
Returning with current objects and {exception_status = }."""
        print(error_msg)

    optimizer.dispatch(Events.OPTIMIZATION_END)

    return optimizer.max, optimizer, exception_status
