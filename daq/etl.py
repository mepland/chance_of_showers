"""Run ETL script.

Cleans and combines raw_data/*.csv files into a single parquet file.
"""

import datetime
import pathlib
import traceback
import zoneinfo
from typing import Final

import humanize
import hydra
import polars as pl
from omegaconf import DictConfig  # noqa: TC002

__all__: list[str] = []


@hydra.main(version_base=None, config_path="..", config_name="config")
def etl(cfg: DictConfig) -> None:  # pylint: disable=too-many-locals
    """Run ETL script.

    Args:
        cfg (DictConfig): Hydra configuration.

    Raises:
        OSError: An error occurred reading a csv.
        ValueError: A data quality check failed.
    """
    # setup variables
    # pylint: disable=duplicate-code,invalid-name
    PACKAGE_PATH: Final = pathlib.Path(cfg["general"]["package_path"]).expanduser()
    RAW_DATA_RELATIVE_PATH: Final = cfg["daq"]["raw_data_relative_path"]
    SAVED_DATA_RELATIVE_PATH: Final = cfg["etl"]["saved_data_relative_path"]

    DATE_FMT: Final = cfg["general"]["date_fmt"]
    TIME_FMT: Final = cfg["general"]["time_fmt"]
    FNAME_DATETIME_FMT: Final = cfg["general"]["fname_datetime_fmt"]
    DATETIME_FMT: Final = f"{DATE_FMT} {TIME_FMT}"

    LOCAL_TIMEZONE_STR: Final = cfg["general"]["local_timezone"]

    if LOCAL_TIMEZONE_STR not in zoneinfo.available_timezones():
        AVAILABLE_TIMEZONES: Final = "\n".join(list(zoneinfo.available_timezones()))
        msg = f"Unknown {LOCAL_TIMEZONE_STR = }, choose from:\n{AVAILABLE_TIMEZONES}"
        raise ValueError(msg)

    UTC_TIMEZONE: Final = zoneinfo.ZoneInfo("UTC")
    # pylint: enable=duplicate-code

    # when the issue of drifting seconds was fixed by replacing t_start's second and microsecond with 0
    DT_END_OF_DRIFTING_SECONDS: Final = datetime.datetime.strptime(
        cfg["daq"]["end_of_drifting_seconds"], DATETIME_FMT
    ).replace(tzinfo=UTC_TIMEZONE)

    # when web threading was fixed, eliminating duplicate records from multiple threads
    DT_END_OF_THREADING_DUPLICATES: Final = datetime.datetime.strptime(
        cfg["daq"]["end_of_threading_duplicates"], DATETIME_FMT
    ).replace(tzinfo=UTC_TIMEZONE)

    # When sticking flow variable was fixed
    DT_END_OF_STICKING_FLOW: Final = datetime.datetime.strptime(
        cfg["daq"]["end_of_sticking_flow"], DATETIME_FMT
    ).replace(tzinfo=UTC_TIMEZONE)
    # pylint: enable=invalid-name

    # load raw csv files
    dfpl_list = []

    csv_total_bytes = 0
    for f_csv_str in (PACKAGE_PATH / RAW_DATA_RELATIVE_PATH).glob("*.csv"):
        f_csv = pathlib.Path(f_csv_str)
        try:
            dfpl = pl.scan_csv(f_csv)
            dfpl = dfpl.with_columns(pl.lit(f_csv.name).alias("fname"))
            dfpl_list.append(dfpl)
            csv_total_bytes += f_csv.stat().st_size
        except Exception as error:
            msg = f"Error loading file {f_csv}!\n{error = }\n{type(error) = }\n{traceback.format_exc()}"
            raise OSError(msg) from error

    dfpl = pl.concat(dfpl_list)
    # set UTC timezone
    dfpl = dfpl.with_columns(
        pl.col("datetime_utc").str.to_datetime(DATETIME_FMT).dt.replace_time_zone("UTC")
    )

    # remove any datetimes that have more than 1 row, caused by historical bug in DAQ threading
    DFPL_DUPLICATE_DATETIME: Final = (  # pylint: disable=invalid-name
        dfpl.group_by("datetime_utc").agg(pl.len()).filter(1 < pl.col("len"))
    )
    dfpl = dfpl.join(
        DFPL_DUPLICATE_DATETIME.select("datetime_utc"),
        on="datetime_utc",
        how="anti",
    )

    # make sure none of the duplicates are post threading fix
    DFPL_DUPLICATE_DATETIME_POST_THREADING_FIX: Final = (  # pylint: disable=invalid-name
        DFPL_DUPLICATE_DATETIME.filter(DT_END_OF_THREADING_DUPLICATES < pl.col("datetime_utc"))
    )

    N_DUPLICATE_DATETIME_POST_THREADING_FIX: Final = (  # pylint: disable=invalid-name
        DFPL_DUPLICATE_DATETIME_POST_THREADING_FIX.select(pl.len()).collect(streaming=True).item()
    )
    if 0 < N_DUPLICATE_DATETIME_POST_THREADING_FIX:
        DUPLICATE_DATETIME_POST_FIX_MIN: Final = (  # pylint: disable=invalid-name
            DFPL_DUPLICATE_DATETIME_POST_THREADING_FIX.select(pl.col("datetime_utc").min())
            .collect(streaming=True)
            .item()
            .strftime(DATETIME_FMT)
        )
        DUPLICATE_DATETIME_POST_FIX_MAX: Final = (  # pylint: disable=invalid-name
            DFPL_DUPLICATE_DATETIME_POST_THREADING_FIX.select(pl.col("datetime_utc").max())
            .collect(streaming=True)
            .item()
            .strftime(DATETIME_FMT)
        )
        raise ValueError(
            f"Found {N_DUPLICATE_DATETIME_POST_THREADING_FIX} datetimes with multiple entries"  # noqa: ISC003
            + f" after {DT_END_OF_THREADING_DUPLICATES.strftime(DATETIME_FMT)} UTC!"
            + f"\nDuplicates are between {DUPLICATE_DATETIME_POST_FIX_MIN} and {DUPLICATE_DATETIME_POST_FIX_MAX}."
        )

    # check for drifting seconds after fix implemented at DT_END_OF_DRIFTING_SECONDS
    DFPL_DRIFT_SECONDS_RECORDS: Final = dfpl.filter(  # pylint: disable=invalid-name
        (DT_END_OF_DRIFTING_SECONDS < pl.col("datetime_utc"))
        & (pl.col("datetime_utc").dt.second() != 0)
    )
    N_ROWS_DRIFT_SECONDS: Final = (  # pylint: disable=invalid-name
        DFPL_DRIFT_SECONDS_RECORDS.select(pl.len()).collect(streaming=True).item()
    )
    if 0 < N_ROWS_DRIFT_SECONDS:
        print(DFPL_DRIFT_SECONDS_RECORDS.collect(streaming=True))
        msg = f"Found {N_ROWS_DRIFT_SECONDS = } after {DT_END_OF_DRIFTING_SECONDS.strftime(DATETIME_FMT)} UTC!"
        raise ValueError(msg)

    # set had_flow to -1 for dates before DT_END_OF_STICKING_FLOW
    dfpl = dfpl.with_columns(pl.col("had_flow").alias("had_flow_original")).with_columns(
        pl.when(pl.col("datetime_utc") <= DT_END_OF_STICKING_FLOW)
        .then(pl.lit(-1))
        .otherwise(pl.col("had_flow_original"))
        .alias("had_flow")
    )
    # check for invalid had_flow values
    DFPL_INVALID_HAD_FLOW_RECORDS: Final = dfpl.filter(  # pylint: disable=invalid-name
        ~pl.col("had_flow").is_in([-1, 0, 1])
    )
    N_INVALID_HAD_FLOW_ROWS: Final = (  # pylint: disable=invalid-name
        DFPL_INVALID_HAD_FLOW_RECORDS.select(pl.len()).collect(streaming=True).item()
    )
    if 0 < N_INVALID_HAD_FLOW_ROWS:
        print(DFPL_INVALID_HAD_FLOW_RECORDS.collect(streaming=True))
        msg = f"Found {N_INVALID_HAD_FLOW_ROWS = }!"
        raise ValueError(msg)

    dfpl = dfpl.sort(pl.col("datetime_utc"), descending=False)

    print(
        "\nETL Summary:",
        dfpl.select("datetime_utc", "mean_pressure_value").collect(streaming=True).describe(),
    )

    (PACKAGE_PATH / SAVED_DATA_RELATIVE_PATH).mkdir(parents=True, exist_ok=True)

    PARQUET_DATETIME_MIN: Final = (  # pylint: disable=invalid-name
        dfpl.select(pl.col("datetime_utc").min())
        .collect(streaming=True)
        .item()
        .strftime(FNAME_DATETIME_FMT)
    )

    PARQUET_DATETIME_MAX: Final = (  # pylint: disable=invalid-name
        dfpl.select(pl.col("datetime_utc").max())
        .collect(streaming=True)
        .item()
        .strftime(FNAME_DATETIME_FMT)
    )

    f_parquet = (
        PACKAGE_PATH
        / SAVED_DATA_RELATIVE_PATH
        / f"data_{PARQUET_DATETIME_MIN}_to_{PARQUET_DATETIME_MAX}.parquet"
    )

    dfpl.collect(streaming=True).write_parquet(f_parquet)
    parquet_total_bytes = f_parquet.stat().st_size

    print(
        f"\nCombined parquet saved to {f_parquet}"  # noqa: ISC003
        + f"\n\nInput CSVs: {humanize.naturalsize(csv_total_bytes)}"
        + f", Output parquet: {humanize.naturalsize(parquet_total_bytes)}"
        + f", a reduction of {(csv_total_bytes - parquet_total_bytes) / csv_total_bytes:.0%}\n"
    )


if __name__ == "__main__":
    etl()  # pylint: disable=no-value-for-parameter
