"""Run ETL script.

Cleans and combines raw_data/*.csv files into a single parquet file.
"""
################################################################################
# python imports
import datetime
import glob
import os
import traceback
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import hydra
import polars as pl

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="..", config_name="config")
def etl(cfg: DictConfig) -> None:  # pylint: disable=used-before-assignment
    """Run ETL script.

    Args:
        cfg: Hydra configuration.

    Raises:
        OSError: An error occurred reading a csv.
        ValueError: A data quality check failed.
    """
    # setup variables
    package_path = cfg["general"]["package_path"]
    raw_data = cfg["daq"]["raw_data"]
    saved_data = cfg["etl"]["saved_data"]

    date_fmt = cfg["general"]["date_fmt"]
    time_fmt = cfg["general"]["time_fmt"]
    datetime_fmt = f"{date_fmt} {time_fmt}"

    # when the issue of drifting seconds was fixed by replacing t_start's second and microsecond with 0
    dt_end_of_drifting_seconds = datetime.datetime.strptime(
        cfg["daq"]["end_of_drifting_seconds"], datetime_fmt
    ).replace(tzinfo=ZoneInfo("UTC"))

    # when web threading was fixed, eliminating duplicate records from multiple threads
    dt_end_of_threading_duplicates = datetime.datetime.strptime(
        cfg["daq"]["end_of_threading_duplicates"], datetime_fmt
    ).replace(tzinfo=ZoneInfo("UTC"))

    # load raw csv files
    dfpl_list = []

    for f_csv in glob.glob(os.path.expanduser(os.path.join(package_path, raw_data, "*.csv"))):
        try:
            dfpl = pl.scan_csv(f_csv)
            dfpl = dfpl.with_columns(pl.lit(f_csv.split("/")[-1]).alias("fname"))
            dfpl_list.append(dfpl)
        except Exception as error:
            raise OSError(
                f"Error loading file {f_csv}!\n{error=}\n{type(error)=}\n{traceback.format_exc()}"
            ) from error

    dfpl = pl.concat(dfpl_list)
    dfpl = (
        # convert date columns
        dfpl.with_columns(
            pl.col("datetime_utc").str.to_datetime(datetime_fmt).dt.replace_time_zone("UTC")
        ).with_columns(
            pl.col("datetime_utc").dt.convert_time_zone("US/Eastern").alias("datetime_est")
        )
        # Add more date columns
        .with_columns(
            pl.col("datetime_est").dt.weekday().alias("day_of_week_int"),
            pl.col("datetime_est").dt.to_string("%A").alias("day_of_week_str"),
        )
    )

    # remove any datetimes that have more than 1 row, caused by historical bug in DAQ threading
    dfpl_duplicate_datetime = (
        dfpl.groupby(by=["datetime_utc"]).agg(pl.count()).filter(1 < pl.col("count"))
    )
    dfpl = dfpl.join(
        dfpl_duplicate_datetime.select("datetime_utc"),
        on="datetime_utc",
        how="anti",
    )

    # make sure none of the duplicates are post threading fix
    dfpl_duplicate_datetime_post_threading_fix = dfpl_duplicate_datetime.filter(
        dt_end_of_threading_duplicates < pl.col("datetime_utc")
    )

    n_duplicate_datetime_post_threading_fix = (
        dfpl_duplicate_datetime_post_threading_fix.select(pl.count()).collect(streaming=True).item()
    )
    if 0 < n_duplicate_datetime_post_threading_fix:
        duplicate_datetime_post_fix_min = (
            dfpl_duplicate_datetime_post_threading_fix.select(pl.col("datetime_utc").min())
            .collect(streaming=True)
            .item()
            .strftime(datetime_fmt)
        )
        duplicate_datetime_post_fix_max = (
            dfpl_duplicate_datetime_post_threading_fix.select(pl.col("datetime_utc").max())
            .collect(streaming=True)
            .item()
            .strftime(datetime_fmt)
        )
        raise ValueError(
            f"Found {n_duplicate_datetime_post_threading_fix} datetimes with multiple entries"
            + f" after {dt_end_of_threading_duplicates.strftime(datetime_fmt)} UTC!"
            + f"\nDuplicates are between {duplicate_datetime_post_fix_min} and {duplicate_datetime_post_fix_max}."
        )

    # check for drifting seconds after fix implemented at dt_end_of_drifting_seconds
    dfpl_drift_seconds_records = dfpl.filter(
        (dt_end_of_drifting_seconds < pl.col("datetime_utc"))
        & (pl.col("datetime_utc").dt.second() != 0)
    )
    n_rows_drift_seconds = (
        dfpl_drift_seconds_records.select(pl.count()).collect(streaming=True).item()
    )
    if 0 < n_rows_drift_seconds:
        print(dfpl_drift_seconds_records.collect(streaming=True))
        raise ValueError(
            f"Found {n_rows_drift_seconds = } after {dt_end_of_drifting_seconds.strftime(datetime_fmt)} UTC!"
        )

    print(
        "\nETL Summary:",
        dfpl.select("datetime_utc", "mean_pressure_value").collect(streaming=True).describe(),
    )

    os.makedirs(os.path.expanduser(os.path.join(package_path, saved_data)), exist_ok=True)

    f_parquet = os.path.expanduser(os.path.join(package_path, saved_data, "etl_data.parquet"))
    dfpl.collect(streaming=True).write_parquet(f_parquet)

    print(f"\nCombined parquet saved to {f_parquet}\n")


if __name__ == "__main__":
    etl()  # pylint: disable=no-value-for-parameter
