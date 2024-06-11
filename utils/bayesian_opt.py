"""Setup Bayesian optimization."""

import datetime
import gc
import hashlib
import json
import os
import pathlib
import platform
import pprint
import re
import signal
import traceback
import warnings
import zoneinfo
from contextlib import suppress
from csv import writer
from types import FrameType  # noqa: TC003
from typing import TYPE_CHECKING, Any, Final, TypeAlias

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
from TSModelWrappers.TSModelWrapper import TSModelWrapper, BAD_TARGET, METRICS_KEYS

# Prophet
from TSModelWrappers.ProphetWrapper import ProphetWrapper  # noqa: TC001

# PyTorch NN Models
from TSModelWrappers.NBEATSModelWrapper import NBEATSModelWrapper  # noqa: TC001
from TSModelWrappers.NHiTSModelWrapper import NHiTSModelWrapper  # noqa: TC001
from TSModelWrappers.TCNModelWrapper import TCNModelWrapper  # noqa: TC001
from TSModelWrappers.TransformerModelWrapper import TransformerModelWrapper  # noqa: TC001
from TSModelWrappers.TFTModelWrapper import TFTModelWrapper  # noqa: TC001
from TSModelWrappers.TSMixerModelWrapper import TSMixerModelWrapper  # noqa: TC001
from TSModelWrappers.DLinearModelWrapper import DLinearModelWrapper  # noqa: TC001
from TSModelWrappers.NLinearModelWrapper import NLinearModelWrapper  # noqa: TC001
from TSModelWrappers.TiDEModelWrapper import TiDEModelWrapper  # noqa: TC001
from TSModelWrappers.RNNModelWrapper import RNNModelWrapper  # noqa: TC001
from TSModelWrappers.BlockRNNModelWrapper import BlockRNNModelWrapper  # noqa: TC001

# Statistical Models
from TSModelWrappers.AutoARIMAWrapper import AutoARIMAWrapper  # noqa: TC001
from TSModelWrappers.BATSWrapper import BATSWrapper  # noqa: TC001
from TSModelWrappers.TBATSWrapper import TBATSWrapper  # noqa: TC001
from TSModelWrappers.FourThetaWrapper import FourThetaWrapper  # noqa: TC001
from TSModelWrappers.StatsForecastAutoThetaWrapper import (  # noqa: TC001
    StatsForecastAutoThetaWrapper,
)
from TSModelWrappers.FFTWrapper import FFTWrapper  # noqa: TC001
from TSModelWrappers.KalmanForecasterWrapper import KalmanForecasterWrapper  # noqa: TC001
from TSModelWrappers.CrostonWrapper import CrostonWrapper  # noqa: TC001

# Regression Models
from TSModelWrappers.LinearRegressionModelWrapper import LinearRegressionModelWrapper  # noqa: TC001
from TSModelWrappers.RandomForestWrapper import RandomForestWrapper  # noqa: TC001
from TSModelWrappers.LightGBMModelWrapper import LightGBMModelWrapper  # noqa: TC001
from TSModelWrappers.XGBModelWrapper import XGBModelWrapper  # noqa: TC001
from TSModelWrappers.CatBoostModelWrapper import CatBoostModelWrapper  # noqa: TC001

# Naive Models
from TSModelWrappers.NaiveMeanWrapper import NaiveMeanWrapper  # noqa: TC001
from TSModelWrappers.NaiveSeasonalWrapper import NaiveSeasonalWrapper  # noqa: TC001
from TSModelWrappers.NaiveDriftWrapper import NaiveDriftWrapper  # noqa: TC001
from TSModelWrappers.NaiveMovingAverageWrapper import NaiveMovingAverageWrapper  # noqa: TC001

__all__ = ["load_best_points", "load_json_log_to_dfp", "print_memory_usage", "run_bayesian_opt"]


# isort: on

# When showing warnings ignore everything except the message
# https://stackoverflow.com/a/2187390
warnings.formatwarning = lambda msg, *args, **kwargs: f"{msg}\n"  # noqa: U100

PointType: TypeAlias = dict[str, Any]
HyperParamType: TypeAlias = dict[str, Any]

WrapperTypes: TypeAlias = type[
    # Prophet
    ProphetWrapper
    # PyTorch NN Models
    | NBEATSModelWrapper
    | NHiTSModelWrapper
    | TCNModelWrapper
    | TransformerModelWrapper
    | TFTModelWrapper
    | TSMixerModelWrapper
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
    "datetime_start",
    "datetime_end",
    "id_point",
    "rank_point",
    "represents_point",
    "is_clean",
    "target",
    "model_name",
    "model_type",
] + [f"{_}_val_loss" for _ in METRICS_KEYS]

NON_CSV_COLS: Final = [
    "rank_point",
    "represents_point",
]

# The datetime format used by bayes_opt.
BAYES_OPT_DATETIME_FMT: Final = "%Y-%m-%d %H:%M:%S"


def clean_log_dfp(dfp: pd.DataFrame | None) -> None | pd.DataFrame:
    """Clean and augment log dataframe.

    Args:
        dfp (pd.DataFrame | None): Log dataframe.

    Returns:
        None | pd.DataFrame: Log cleaned and augmented.
    """
    if dfp is None:
        return None

    has_is_clean = "is_clean" in dfp.columns
    if has_is_clean:
        dfp["is_clean"] = dfp["is_clean"].astype(bool)

    # Setup dfp_minutes for calculations
    dfp["datetime_end"] = pd.to_datetime(dfp["datetime_end"], format=BAYES_OPT_DATETIME_FMT)

    has_measured_datetime_start = "datetime_start" in dfp.columns
    if has_measured_datetime_start:
        dfp["datetime_start"] = pd.to_datetime(dfp["datetime_start"], format=BAYES_OPT_DATETIME_FMT)

    dfp_minutes = pd.DataFrame(dfp)

    if not has_measured_datetime_start:
        dfp_minutes["datetime_start"] = dfp_minutes["datetime_end"]

    dfp_minutes = (
        dfp_minutes.groupby("datetime_end", sort=False)
        .agg({"datetime_start": "min"})
        .reset_index()
        .sort_values(by="datetime_end", ascending=True)
        .reset_index(drop=True)
    )

    if has_measured_datetime_start:
        dfp_minutes["minutes_elapsed_point"] = (
            dfp_minutes["datetime_end"] - dfp_minutes["datetime_start"]
        ) / pd.Timedelta(  # type: ignore[operator]
            minutes=1
        )
        dfp_minutes["minutes_elapsed_total"] = dfp_minutes["minutes_elapsed_point"].cumsum()
    else:
        dfp_minutes["minutes_elapsed_total"] = (
            dfp_minutes["datetime_end"] - dfp_minutes["datetime_end"].min()
        ) / pd.Timedelta(minutes=1)
        dfp_minutes["minutes_elapsed_point"] = (
            dfp_minutes["minutes_elapsed_total"].diff().fillna(0.0)
        )

    dfp = dfp.merge(dfp_minutes.drop("datetime_start", axis=1), how="left", on="datetime_end")

    # Add represents_point
    dfp["row_number"] = (
        dfp.sort_values(
            by=["is_clean", "datetime_end"] if has_is_clean else "datetime_end",
            ascending=[False, True] if has_is_clean else True,
        )
        .groupby("id_point", sort=False)
        .cumcount()
    )
    dfp["represents_point"] = dfp["row_number"] == 0
    dfp = dfp.drop("row_number", axis=1)

    # Add rank_point
    if "id_point" in dfp.columns:
        dfp_id_to_rank = (
            dfp.groupby("id_point", sort=False)
            .agg({"target": "max", "datetime_end": "min"})
            .reset_index()
            .sort_values(by=["target", "datetime_end"], ascending=[False, True])
            .reset_index(drop=True)
        )
        dfp_id_to_rank["rank_point"] = dfp_id_to_rank.index
        dfp_id_to_rank = dfp_id_to_rank[["id_point", "rank_point"]]

        dfp = dfp.merge(dfp_id_to_rank, how="left", on="id_point")

    dfp = dfp.sort_values(
        by=(
            ["datetime_end", "is_clean", "represents_point"]
            if has_is_clean
            else ["datetime_end", "represents_point"]
        ),
        ascending=True,
    ).reset_index(drop=True)

    return dfp[
        [_ for _ in BAYES_OPT_LOG_COLS_FIXED if _ in dfp.columns]
        + [_ for _ in dfp.columns if _ not in BAYES_OPT_LOG_COLS_FIXED]
    ]


def load_json_log_to_dfp(f_path: pathlib.Path) -> None | pd.DataFrame:
    """Load prior bayes_opt log from JSON file as a pandas dataframe.

    Args:
        f_path (pathlib.Path): Path to JSON log file.

    Returns:
        None | pd.DataFrame: Log.
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
                            _postfix = "_end"
                        else:
                            _prefix = f"{_k0}_"
                            _postfix = ""

                        row[f"{_prefix}{_k1}{_postfix}"] = _v1
                else:
                    row[_k0] = _v0

            rows.append(row)

        f_json.close()

        if rows:
            dfp = pd.DataFrame(rows)
            dfp["id_point"] = dfp.index
            dfp["id_point"] = dfp["id_point"].astype(str)

            return clean_log_dfp(dfp)

        return None


def load_csv_log_to_dfp(f_path: pathlib.Path) -> None | pd.DataFrame:
    """Load prior bayes_opt log from CSV file as a pandas dataframe.

    Args:
        f_path (pathlib.Path): Path to CSV log file.

    Returns:
        None | pd.DataFrame: Log.
    """
    with f_path.open("r", encoding="utf-8") as f_csv:
        dfp = pd.read_csv(f_csv, header=0)

        return clean_log_dfp(dfp)


def load_best_points(
    dir_path: pathlib.Path, *, use_csv: bool = True
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load best points from all bayes_opt, CSV or JSON, log files in the dir_path.

    Args:
        dir_path (pathlib.Path): Path to search recursively for CSV, or JSON, log files.
        use_csv (bool): Flag to load CSV files, rather than JSON, log files. (Default value = True)

    Returns:
        tuple[pd.DataFrame, dict[str, pd.DataFrame]]: Best points with metadata as pandas dataframe, and dict of all logs as pandas dataframes.

    Raises:
        ValueError: Could not load from disk, or found duplicate generic_model_name.
    """
    dfp_runs_dict = {}
    rows = []
    for f_path in sorted(dir_path.glob(f"**/*.{'csv' if use_csv else 'json'}")):
        generic_model_name = f_path.stem.replace(BAYESIAN_OPT_PREFIX, "")

        dfp = load_csv_log_to_dfp(f_path) if use_csv else load_json_log_to_dfp(f_path)

        if dfp is None:
            msg = f"Could load {f_path}!"
            raise ValueError(msg)

        if TYPE_CHECKING:
            assert isinstance(dfp, pd.DataFrame)  # noqa: SCS108 # nosec: B101

        if generic_model_name in dfp_runs_dict:
            msg = f"Already loaded log for {generic_model_name}! Please clean the dir structure of {dir_path} and try again."
            raise ValueError(msg)

        dfp_runs_dict[generic_model_name] = pd.DataFrame(dfp)

        dfp_best_points = dfp.loc[dfp["target"] != BAD_TARGET]

        if "represents_point" in dfp_best_points.columns:
            dfp_best_points = dfp_best_points.loc[
                (dfp["target"] == dfp["target"].max()) & dfp["represents_point"]
            ]
        else:
            dfp_best_points = dfp_best_points.loc[dfp["target"] == dfp["target"].max()]

        if not dfp_best_points.index.size:
            dfp_best_points = pd.DataFrame(dfp)
            warnings.warn(
                f"Could not find a best point for {generic_model_name} in {f_path}, just taking them all!",
                stacklevel=1,
            )

        has_is_clean = "is_clean" in dfp_best_points.columns
        dfp_best_points = dfp_best_points.sort_values(
            by=["is_clean", "datetime_end"] if has_is_clean else "datetime_end",
            ascending=[False, True] if has_is_clean else True,
        )

        best_dict = dfp_best_points.iloc[0].to_dict()

        best_params = []
        for k, v in best_dict.items():
            if k.startswith("params_"):
                best_params.append(f'{k.replace("params_", "")}: {v}')

        rows.append(
            {
                "generic_model_name": generic_model_name,
                "target_best": best_dict["target"],
                "model_type": best_dict.get("model_type"),
                "n_points": dfp.index.size,
                "n_points_bad_target": dfp.loc[dfp["target"] == BAD_TARGET].index.size,
                "n_points_representative": dfp.loc[dfp["represents_point"]].index.size,
                "n_points_representative_bad_target": dfp.loc[
                    (dfp["target"] == BAD_TARGET) & dfp["represents_point"]
                ].index.size,
                "minutes_elapsed_total": dfp["minutes_elapsed_total"].max(),
                "minutes_elapsed_point_best": best_dict["minutes_elapsed_point"],
                "minutes_elapsed_mean": dfp.loc[dfp["model_name"] != "manual_bad_point"][
                    "minutes_elapsed_point"
                ].mean(),
                "minutes_elapsed_stddev": dfp.loc[dfp["model_name"] != "manual_bad_point"][
                    "minutes_elapsed_point"
                ].std(),
                "id_point_best": best_dict["id_point"],
                "datetime_end_best": best_dict["datetime_end"],
                "params_best": ", ".join(best_params),
            }
        )

    dfp_best_points = pd.DataFrame(rows)
    dfp_best_points = dfp_best_points.sort_values(
        by=["target_best", "generic_model_name", "datetime_end_best"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    # Sort dfp_runs_dict in the same order as dfp_best_points
    # https://stackoverflow.com/a/21773891
    index_map = {v: i for i, v in enumerate(dfp_best_points["generic_model_name"].to_list())}
    dfp_runs_dict = dict(sorted(dfp_runs_dict.items(), key=lambda pair: index_map[pair[0]]))

    return dfp_best_points, dfp_runs_dict


def write_search_results(  # noqa: C901
    f_excel: pathlib.Path,
    dfp_best_points: pd.DataFrame,
    dfp_runs_dict: dict[str, pd.DataFrame],
    *,
    bad_points_frac_thr: float = 0.2,
) -> None:
    """Write search results to excel file.

    Args:
        f_excel (pathlib.Path): Path to output xlsx file.
        dfp_best_points (pd.DataFrame): Best points dataframe created by load_best_points().
        dfp_runs_dict (dict[str, pd.DataFrame]): Dict of all logs as pandas dataframes created by load_best_points().
        bad_points_frac_thr (float): Bad points fraction threshold for red formatting. (Default value = 0.2)
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

        def _fmt_worksheet(  # type: ignore[no-any-unimported]
            worksheet: xlsxwriter.worksheet.Worksheet,
            dfp_source: pd.DataFrame,
            *,
            hide_debug_cols: bool = True,
        ) -> None:
            """Format a log worksheet for this project.

            Args:
                worksheet (xlsxwriter.worksheet.Worksheet): Input worksheet.
                dfp_source (pd.DataFrame): Original dataframe.
                hide_debug_cols (bool): Hide low level debugging columns. (Default value = True)
            """
            # Format loss columns
            for i_col, col_str in enumerate(dfp_source.columns):
                if not re.match(r"^.*?_val_loss$", col_str):
                    continue

                _loss = dfp_source.loc[dfp_source[col_str] != -BAD_TARGET][col_str].to_numpy()
                _loss = _loss[np.isfinite(_loss)]
                if len(_loss) == 0:
                    continue

                _min = np.min(_loss)
                _q1 = np.quantile(_loss, 0.25)
                _median = np.quantile(_loss, 0.50)
                _q3 = np.quantile(_loss, 0.75)
                _max = np.max(_loss)

                loss_color_fmt["min_value"] = max(_min, _q1 - 1.5 * (_q3 - _q1))
                loss_color_fmt["mid_value"] = _median
                loss_color_fmt["max_value"] = min(_max, _q3 + 1.5 * (_q3 - _q1))

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

                elapsed_minutes_fmt_bar["min_value"] = 0.0
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
                if col_str not in ["is_clean", "represents_point"]:
                    continue

                worksheet.set_column(i_col, i_col, None, boolean_fmt)

            # Filter columns
            worksheet.autofilter(0, 0, dfp_source.shape[0], dfp_source.shape[1] - 1)

            if "represents_point" in dfp_source.columns:
                _i_col = list(dfp_source.columns).index("represents_point")
                worksheet.filter_column(_i_col, "x == TRUE")

                # Hide rows which do not match the filter criteria
                for i_row, row in dfp_source.iterrows():  # type: ignore[assignment]
                    if not row["represents_point"]:
                        worksheet.set_row(i_row + 1, options={"hidden": True})

            # Autofit column widths
            worksheet.autofit()

            # Hide columns
            for i_col, col_str in enumerate(dfp_source.columns):
                if hide_debug_cols and (
                    col_str
                    in [
                        "datetime_start",
                        "n_points",
                        "n_points_bad_target",
                        "id_point",
                        "model_name",
                        "id_point_best",
                        "minutes_elapsed_point_best",
                        "datetime_end_best",
                    ]
                    or (col_str == "model_type" and "minutes_elapsed_point" in dfp_source.columns)
                ):
                    worksheet.set_column(i_col, i_col, None, options={"hidden": True})

        # Write and format sheets
        dfp_best_points.to_excel(
            xlsx_writer, sheet_name="Best Points", freeze_panes=(1, 1), index=False
        )
        _fmt_worksheet(xlsx_writer.sheets["Best Points"], dfp_best_points)

        for generic_model_name, dfp in dfp_runs_dict.items():
            dfp.to_excel(
                xlsx_writer, sheet_name=generic_model_name, freeze_panes=(1, 2), index=False
            )
            _fmt_worksheet(xlsx_writer.sheets[generic_model_name], dfp)


def print_memory_usage(*, header: str | None = None) -> None:
    """Print system memory usage statistics.

    Args:
        header (str | None): Header to print before the rest of the memory usage. (Default value = None)
    """
    ram_info = psutil.virtual_memory()
    process = psutil.Process()
    header = f"{header}\n" if header is not None and header != "" else ""

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
                key (str): Key to get from gpu_memory_stats.

            Returns:
                str: Clean humanized string for printing.
            """
            _v = gpu_memory_stats.get(key)
            if _v is not None:
                return str(humanize.naturalsize(_v))

            return "MISSING"

        memory_usage_str += (
            f", GPU RAM Current: {get_gpu_mem_key('allocated_bytes.all.current')}, "  # noqa: ISC003
            + f"Peak: {get_gpu_mem_key('allocated_bytes.all.peak')}"
        )

    print(memory_usage_str)


n_points = 0  # pylint: disable=invalid-name


# Easier than modifying the json_logger from bayes_opt
def write_csv_row(  # pylint: disable=too-many-arguments
    *,
    enable_csv_logging: bool,
    fname_csv_log: pathlib.Path,
    datetime_start_str: str,
    datetime_end_str: str,
    id_point: str,
    target: float,
    metrics_val: dict[str, float],
    point: PointType,
    is_clean: bool,
    model_name: str,
    model_type: str,
) -> None:
    """Save validation metrics and other metadata for this point to CSV.

    Args:
        enable_csv_logging (bool): Enable CSV logging of points.
        fname_csv_log (pathlib.Path): Path to CSV log file.
        datetime_start_str (str): Starting datetime string of this iteration.
        datetime_end_str (str): Ending datetime string of this iteration, from JSON log.
        id_point (str): ID of this point.
        target (float): Target value.
        metrics_val (dict[str, float]): Metrics on the validation set.
        point (PointType): Hyperparameter point.
        is_clean (bool): Flag for if these are cleaned or raw hyperparameters.
        model_name (str): Model name with training time stamp.
        model_type (str): General type of model; prophet, torch, statistical, regression, or naive.
    """
    if not enable_csv_logging:
        return

    new_row = [
        datetime_start_str,
        datetime_end_str,
        id_point,
        int(is_clean),
        target,
        model_name,
        model_type,
    ]
    metrics_val_sorted = {k: metrics_val[str(k)] for k in METRICS_KEYS}
    new_row += list(metrics_val_sorted.values())
    point = dict(sorted(point.items()))
    new_row += list(point.values())

    with fname_csv_log.open("a", encoding="utf-8") as f_csv:
        m_writer = writer(f_csv)
        if f_csv.tell() == 0:
            # empty file, create header
            m_writer.writerow(
                [_ for _ in BAYES_OPT_LOG_COLS_FIXED if _ not in NON_CSV_COLS]
                + [f"params_{_}" for _ in point]
            )

        m_writer.writerow(new_row)
        f_csv.close()


def get_datetime_str_from_json(*, enable_json_logging: bool, fname_json_log: pathlib.Path) -> str:
    """Load datatime str from last row in JSON log.

    The {"datetime": {"datetime": "..."}} timestamp in the JSON log created by the optimizer.register() call
        is a good field to have, but is only created in in the JSON logger here:
        https://github.com/bayesian-optimization/BayesianOptimization/blob/129caac02177b146ce315e177d4d88950b75253a/bayes_opt/logger.py#L153C50-L158
        We need to load last line of the JSON from disk and extract the datatime string.

    Args:
        enable_json_logging (bool): Enable JSON logging of points.
        fname_json_log (pathlib.Path): Path to JSON log file.

    Returns:
        str: Datetime.
    """
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
            return str(last_line.get("datetime", {}).get("datetime", "NULL"))

    return "NULL"


def get_i_point_duplicate(  # type: ignore[no-any-unimported]
    point: PointType, optimizer: bayes_opt.BayesianOptimization
) -> int:
    """Get index of duplicate prior point from optimizer, if one exists.

    Args:
        point (PointType): The point to check.
        optimizer (bayes_opt.BayesianOptimization): The optimizer to search.

    Returns:
        int: The index of the first duplicate prior point from optimizer, if one exists, otherwise returns -1.
    """
    for i_param in range(optimizer.space.params.shape[0]):
        if np.array_equiv(optimizer.space.params_to_array(point), optimizer.space.params[i_param]):
            return i_param

    return -1


def get_point_hash(point: PointType) -> str:
    """Get hash of prior.

    Args:
        point (PointType): The point to hash.

    Returns:
        str: The SHA-256 hash of the point.
    """
    return hashlib.sha256(
        ", ".join([f"{k}: {v}" for k, v in dict(sorted(point.items())).items()]).encode("utf-8")
    ).hexdigest()


def signal_handler_for_stopping(
    dummy_signal: int,  # noqa: U100 # pylint: disable=unused-argument
    dummy_frame: FrameType | None,  # noqa: U100 # pylint: disable=unused-argument
) -> None:
    """Stop iteration gracefully.

    https://medium.com/@chamilad/timing-out-of-long-running-methods-in-python-818b3582eed6

    Args:
        dummy_signal (int): signal number.
        dummy_frame (FrameType | None): Frame object.

    Raises:
        RuntimeError: Out of Time!
    """
    msg = "Out of Time!"
    raise RuntimeError(msg)


def run_bayesian_opt(  # type: ignore[no-any-unimported] # noqa: C901 # pylint: disable=too-many-statements,too-many-locals,too-many-arguments
    *,
    parent_wrapper: TSModelWrapper,
    model_wrapper_class: WrapperTypes,
    model_wrapper_kwargs: dict[str, Any] | None = None,
    hyperparams_to_opt: list[str] | None = None,
    n_iter: int = 100,
    max_points: int | None = None,
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
    fixed_hyperparams_to_alter: HyperParamType | None = None,
    enable_json_logging: bool = True,
    enable_reloading: bool = True,
    enable_model_saves: bool = False,
    bayesian_opt_work_dir_name: str = "bayesian_optimization",
    local_timezone: zoneinfo.ZoneInfo | None = None,
) -> tuple[dict[str, float], bayes_opt.BayesianOptimization, int]:
    """Run Bayesian optimization for this model wrapper.

    Args:
        parent_wrapper (TSModelWrapper): TSModelWrapper object containing all parent configs.
        model_wrapper_class (WrapperTypes): TSModelWrapper class to optimize.
        model_wrapper_kwargs (dict[str, Any] | None): kwargs to pass to model_wrapper. (Default value = None)
        hyperparams_to_opt (list[str] | None): List of hyperparameters to optimize.
            If None, use all configurable hyperparameters. (Default value = None)
        n_iter (int): How many iterations of Bayesian optimization to perform.
            This is the number of new models to train, in addition to any duplicated or reloaded points. (Default value = 100)
        max_points (int | None): The maximum number of points to train.
            If this number of points has been reached, stop optimizing even without finishing n_iter. (Default value = None)
        allow_duplicate_points (bool): If True, the optimizer will allow duplicate points to be registered.
            This behavior may be desired in high noise situations where repeatedly probing
            the same point will give different answers. In other situations, the acquisition
            may occasionally generate a duplicate point. (Default value = False)
        utility_kind (str): {'ucb', 'ei', 'poi'}
            * 'ucb' stands for the Upper Confidence Bounds method
            * 'ei' is the Expected Improvement method
            * 'poi' is the Probability Of Improvement criterion. (Default value = 'ucb')
        utility_kappa (float): Parameter to indicate how closed are the next parameters sampled.
            Higher value = favors spaces that are least explored.
            Lower value = favors spaces where the regression function is the highest. (Default value = 2.576)
        verbose (int): Optimizer verbosity
            7 prints memory usage
            6 prints points before training
            5 prints point count at each iteration
            4 prints full stack traces
            3 prints basic workflow messages
            2 prints all iterations
            1 prints only when a maximum is observed
            0 is silent (Default value = 3)
        model_verbose (int): Verbose level of model_wrapper, -1 silences LightGBMModel. (Default value = -1)
        enable_torch_warnings (bool): Enable torch warning messages about training devices and CUDA, globally, via the logging module. (Default value = False)
        enable_torch_model_summary (bool): Enable torch model summary. (Default value = True)
        enable_torch_progress_bars (bool): Enable torch progress bars. (Default value = False)
        disregard_training_exceptions (bool): Flag to disregard all exceptions raised when training a model, and return BAD_TARGET instead. (Default value = False)
        max_time_per_model (datetime.timedelta | None): Set the maximum amount of training time for each iteration.
            Torch models will use max_time_per_model as the max time per epoch,
            while non-torch models will use it for the whole iteration if signal is available e.g. Linux, Darwin. (Default value = None)
        accelerator (str | None): Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "auto") (Default value = 'auto')
        fixed_hyperparams_to_alter (HyperParamType | None): Fixed hyperparameters to alter, but not optimize. (Default value = None)
        enable_json_logging (bool): Enable JSON logging of points. (Default value = True)
        enable_reloading (bool): Enable reloading of prior points from JSON log. (Default value = True)
        enable_model_saves (bool): Save the trained model at each iteration. (Default value = False)
        bayesian_opt_work_dir_name (str): Directory name to save logs and models in, within the parent_wrapper.work_dir_base. (Default value = 'bayesian_optimization')
        local_timezone (zoneinfo.ZoneInfo | None): Local timezone. (Default value = None)

    Returns:
        tuple[dict[str, float], bayes_opt.BayesianOptimization, int]: optimal_values - Optimal hyperparameter values,
            optimizer - bayes_opt.BayesianOptimization object for further details,
            exception_status - Int exception status to pass on to bash scripts.

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
            msg = f"Could not load hyperparameter definition for {hyperparam = }!"
            raise ValueError(msg)

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
    model_type: Final = model_wrapper.get_model_type()
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
        # _iterations and _previous_max are not reloaded correctly by default
        # https://github.com/bayesian-optimization/BayesianOptimization/blob/c7e5c3926944fc6011ae7ace29f7b5ed0f9c983b/bayes_opt/observer.py#L9
        # pylint: disable=protected-access
        screen_logger._iterations = n_points
        screen_logger._previous_max = max(optimizer.space.target, default=BAD_TARGET)
        # pylint: enable=protected-access
        for event in DEFAULT_EVENTS:
            if (verbose < 3) and event in [Events.OPTIMIZATION_START, Events.OPTIMIZATION_END]:
                continue

            optimizer.subscribe(event, screen_logger)

    # Define function to complete an iteration
    def complete_iter(
        datetime_start_str: str,
        i_iter: int,
        model_wrapper: TSModelWrapper,
        target: float,
        metrics_val: dict[str, float],
        *,
        point_to_probe: HyperParamType,
        point_to_probe_is_clean: bool,
        point_to_probe_clean: HyperParamType,
    ) -> None:
        """Complete this iteration, register point(s) and clean up.

        Args:
            datetime_start_str (str): Starting datetime string of this iteration.
            i_iter (int): Index of this iteration.
            model_wrapper (TSModelWrapper): Model wrapper object to reset.
            target (float): Target value to register.
            metrics_val (dict[str, float]): Metrics on the validation set.
            point_to_probe (HyperParamType): Raw point to probe.
            point_to_probe_is_clean (bool): If point_to_probe is clean.
            point_to_probe_clean (HyperParamType): Point that was actually probed.
        """
        global n_points

        id_point = get_point_hash(point_to_probe_clean)

        model_name = model_wrapper.get_model_name()
        if model_name is None:
            model_name = "reusing_prior_point"

        if get_i_point_duplicate(point_to_probe, optimizer) == -1:
            optimizer.register(params=point_to_probe, target=target)
            datetime_end_str = get_datetime_str_from_json(
                enable_json_logging=enable_json_logging, fname_json_log=fname_json_log
            )

            write_csv_row(
                enable_csv_logging=enable_json_logging,
                fname_csv_log=fname_csv_log,
                datetime_start_str=datetime_start_str,
                datetime_end_str=datetime_end_str,
                id_point=id_point,
                target=target,
                metrics_val=metrics_val,
                point=point_to_probe,
                is_clean=point_to_probe_is_clean,
                model_name=model_name,
                model_type=model_type,
            )

            n_points += 1

            if get_i_point_duplicate(point_to_probe_clean, optimizer) == -1:
                optimizer.register(params=point_to_probe_clean, target=target)

                write_csv_row(
                    enable_csv_logging=enable_json_logging,
                    fname_csv_log=fname_csv_log,
                    datetime_start_str=datetime_start_str,
                    datetime_end_str=datetime_end_str,
                    id_point=id_point,
                    target=target,
                    metrics_val=metrics_val,
                    point=point_to_probe_clean,
                    is_clean=True,
                    model_name=model_name,
                    model_type=model_type,
                )

                n_points += 1

        model_wrapper.reset_wrapper()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        if 7 <= verbose:
            print_memory_usage()

        if 3 <= verbose:
            print(f"Completed {i_iter = }, {id_point = }, with {n_points = }")

    max_time_per_model_flag = (
        max_time_per_model is not None
        and model_type != "torch"
        and platform.system() in ["Linux", "Darwin"]
    )
    if max_time_per_model_flag:
        signal.signal(signal.SIGALRM, signal_handler_for_stopping)

    next_point_to_probe = None
    next_point_to_probe_is_clean = False
    next_point_to_probe_cleaned = None

    def _build_error_msg(error_msg: str, error: Exception) -> str:
        """Build error message from user string and exception.

        Args:
            error_msg (str): Custom error message from user.
            error (Exception): Exception from process.

        Returns:
            str: Nicely formatted error message with full context.
        """
        if 3 <= verbose:
            error_msg = f"""{error_msg}
{error = }"""

        if 4 <= verbose:
            error_msg = f"""{error_msg}
{type(error) = }
{traceback.format_exc()}"""

            if next_point_to_probe is not None:
                error_msg = f"{error_msg}\nnext_point_to_probe = {pprint.pformat(next_point_to_probe)}"  # type: ignore[unreachable]

            if next_point_to_probe_cleaned is not None:
                error_msg = f"{error_msg}\nnext_point_to_probe_cleaned = {pprint.pformat(next_point_to_probe_cleaned)}"  # type: ignore[unreachable]

        return error_msg  # noqa: R504

    # Run Bayesian optimization iterations
    try:
        for i_iter in range(n_iter):
            if max_points is not None and max_points <= n_points:
                if 3 <= verbose:
                    print(f"Have {max_points = } <= {n_points = }, stopping here.")

                break

            if i_iter == 0:
                optimizer.dispatch(Events.OPTIMIZATION_START)

            if 3 <= verbose:
                print(f"\nStarting {i_iter = }, with {n_points = }")

            datetime_start_str = datetime.datetime.now(local_timezone).strftime(
                BAYES_OPT_DATETIME_FMT
            )

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
                # Construct next_point_to_probe_cleaned to be in the same format as next_point_to_probe
                chosen_hyperparams = model_wrapper.translate_hyperparameters_to_numeric(
                    model_wrapper.preview_hyperparameters(**next_point_to_probe)
                )
                next_point_to_probe_cleaned = {k: chosen_hyperparams[k] for k in hyperparams_to_opt}

                # Check if next_point_to_probe is clean
                next_point_to_probe_is_clean = np.array_equiv(
                    optimizer.space.params_to_array(next_point_to_probe),
                    optimizer.space.params_to_array(next_point_to_probe_cleaned),
                )

                if 6 <= verbose:
                    print(f"{next_point_to_probe_is_clean = }")
                    print(f"next_point_to_probe = {pprint.pformat(next_point_to_probe)}")
                    print(
                        f"next_point_to_probe_cleaned = {pprint.pformat(next_point_to_probe_cleaned)}"
                    )

                # Check if we already tested this next_point_to_probe_cleaned point
                # If it has been tested, save the raw next_point_to_probe with the same target and continue
                i_point_duplicate = get_i_point_duplicate(next_point_to_probe_cleaned, optimizer)
                if 0 <= i_point_duplicate:
                    target = optimizer.space.target[i_point_duplicate]
                    if 3 <= verbose:
                        print(
                            f"On iteration {i_iter} testing prior i_point = {i_point_duplicate}, returning prior {target = :.9f} for the next_point_to_probe, which is a {'clean' if next_point_to_probe_is_clean else 'raw'} point."
                        )

                    complete_iter(
                        datetime_start_str,
                        i_iter,
                        model_wrapper,
                        target,
                        BAD_METRICS,
                        point_to_probe=next_point_to_probe,
                        point_to_probe_is_clean=next_point_to_probe_is_clean,
                        point_to_probe_clean=next_point_to_probe_cleaned,
                    )

                    continue

                # set model_name_tag for this iteration
                model_wrapper.set_model_name_tag(model_name_tag=f"iteration_{n_points}")

                # Setup iteration kill timer
                if max_time_per_model_flag:
                    if TYPE_CHECKING:
                        assert isinstance(  # noqa: SCS108 # nosec: B101
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

            # Handle training exceptions
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
                        datetime_start_str,
                        i_iter,
                        model_wrapper,
                        BAD_TARGET,
                        BAD_METRICS,
                        point_to_probe=next_point_to_probe,
                        point_to_probe_is_clean=next_point_to_probe_is_clean,
                        point_to_probe_clean=(
                            # use next_point_to_probe_cleaned to create id_point if it exists, otherwise fall back to next_point_to_probe
                            next_point_to_probe_cleaned
                            if next_point_to_probe_cleaned is not None
                            else next_point_to_probe
                        ),
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

            complete_iter(
                datetime_start_str,
                i_iter,
                model_wrapper,
                target,
                metrics_val,
                point_to_probe=next_point_to_probe,
                point_to_probe_is_clean=next_point_to_probe_is_clean,
                point_to_probe_clean=next_point_to_probe_cleaned,
            )

    # Handle optimizer exceptions
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
    except Exception as error:  # pylint: disable=broad-exception-caught
        error_msg = _build_error_msg("Unexpected error in run_bayesian_opt():", error)
        if disregard_training_exceptions:
            error_msg = f"""{error_msg}
Disregard_training_exceptions is set, continuing!"""
        else:
            exception_status = 3

        error_msg = f"""{error_msg}
Returning with current objects and {exception_status = }."""
        print(error_msg)

    with suppress(Exception):
        optimizer.dispatch(Events.OPTIMIZATION_END)

    return optimizer.max, optimizer, exception_status


def write_manual_bad_point(
    *,
    bad_point_to_write: PointType,
    bad_point_to_write_clean: PointType,
    parent_wrapper: TSModelWrapper,
    model_wrapper_class: WrapperTypes,
    bayesian_opt_work_dir_name: str = "bayesian_optimization",
) -> None:
    """Manually write a point, raw and clean, as a failed point to the JSON and CSV logs.

    This is useful when an iteration is killed by the OS with an uncatchable SIGKILL.

    Args:
        bad_point_to_write (PointType): Bad hyperparameter point to write, raw.
        bad_point_to_write_clean (PointType): Bad hyperparameter point to write, clean.
        parent_wrapper (TSModelWrapper): TSModelWrapper object containing all parent configs.
        model_wrapper_class (WrapperTypes): TSModelWrapper class to optimize.
        bayesian_opt_work_dir_name (str): Directory name to save logs and models in, within the parent_wrapper.work_dir_base. (Default value = 'bayesian_optimization')
    """
    model_wrapper = model_wrapper_class(TSModelWrapper=parent_wrapper)
    optimizer = bayes_opt.BayesianOptimization(
        f=None, pbounds={k: (None, None) for k, v in bad_point_to_write.items()}
    )

    # Setup Logging
    generic_model_name: Final = model_wrapper.get_generic_model_name()
    model_type: Final = model_wrapper.get_model_type()
    bayesian_opt_work_dir: Final = pathlib.Path(
        model_wrapper.work_dir_base, bayesian_opt_work_dir_name, generic_model_name
    ).expanduser()
    fname_json_log: Final = (
        bayesian_opt_work_dir / f"{BAYESIAN_OPT_PREFIX}{generic_model_name}.json"
    )
    fname_csv_log: Final = bayesian_opt_work_dir / f"{BAYESIAN_OPT_PREFIX}{generic_model_name}.csv"

    # Reload prior points, must be done before json_logger is recreated to avoid duplicating past runs
    json_logger = JSONLogger(path=str(fname_json_log), reset=False)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, json_logger)

    id_point = get_point_hash(bad_point_to_write_clean)
    model_name = "manual_bad_point"

    optimizer.register(params=bad_point_to_write, target=BAD_TARGET)
    datetime_end_str = get_datetime_str_from_json(
        enable_json_logging=True, fname_json_log=fname_json_log
    )

    write_csv_row(
        enable_csv_logging=True,
        fname_csv_log=fname_csv_log,
        datetime_start_str=datetime_end_str,
        datetime_end_str=datetime_end_str,
        id_point=id_point,
        target=BAD_TARGET,
        metrics_val=BAD_METRICS,
        point=bad_point_to_write,
        is_clean=False,
        model_name=model_name,
        model_type=model_type,
    )

    optimizer.register(params=bad_point_to_write_clean, target=BAD_TARGET)

    write_csv_row(
        enable_csv_logging=True,
        fname_csv_log=fname_csv_log,
        datetime_start_str=datetime_end_str,
        datetime_end_str=datetime_end_str,
        id_point=id_point,
        target=BAD_TARGET,
        metrics_val=BAD_METRICS,
        point=bad_point_to_write_clean,
        is_clean=True,
        model_name=model_name,
        model_type=model_type,
    )
